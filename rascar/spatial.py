"""DINOv2-B occupancy heads. All targets come from manual training boxes."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from .core import DAYS, CLASSES, BBOX, CROP, write_json

SEEDS = (3101, 3102, 3103)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(768, 64, 1), nn.GELU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                                 nn.Dropout2d(.1), nn.Conv2d(64, 4, 1))

    def forward(self, x):
        return self.net(F.normalize(x, dim=1) * np.sqrt(768))


def filename(day, seed):
    return f'spatial_day{day[-2:]}_seed{seed}.pt'


def occupancy(record, manual):
    c = record[CROP].to_numpy(float)
    cell = (c[2:] - c[:2]) / 32
    xx = c[0] + np.arange(32) * cell[0]
    yy = c[1] + np.arange(32) * cell[1]
    result = np.zeros((4, 32, 32), np.float32)
    for k, cls in enumerate(CLASSES):
        boxes = manual[manual.frame_id.eq(record.frame_id) & manual.class_name.eq(cls)]
        for box in boxes[BBOX].to_numpy(float):
            wx = np.maximum(0, np.minimum(xx + cell[0], box[2]) - np.maximum(xx, box[0])) / cell[0]
            wy = np.maximum(0, np.minimum(yy + cell[1], box[3]) - np.maximum(yy, box[1])) / cell[1]
            result[k] = np.maximum(result[k], wy[:, None] * wx[None, :])
    return result


def load_grids(cache, records):
    cache = Path(cache)
    marker = cache / 'dense_complete.json'
    if not marker.exists():
        raise FileNotFoundError('Complete DINO grids required. Run prepare.py --stage dense first.')
    from .core import read_json
    info = read_json(marker)
    if info['ids'] != records.id.tolist():
        raise ValueError('Dense grid record order changed')
    original = np.load(cache / 'dense.npy', mmap_mode='r')
    flipped = np.load(cache / 'dense_flip.npy', mmap_mode='r')
    expected = (len(records), 768, 32, 32)
    if original.shape != expected or flipped.shape != expected:
        raise ValueError('Incomplete dense feature cache')
    return original, flipped


def train(data, cache, models, device):
    torch.set_num_threads(2)
    rec = pd.read_csv(Path(data) / 'dense_records.csv')
    h = pd.read_csv(Path(data) / 'manual_boxes.csv')
    x, xf = load_grids(cache, rec)
    # Float16 matches the original stored target grid.
    targets = np.stack([occupancy(r, h) if r.split != 'test' else np.zeros((4, 32, 32))
                        for _, r in rec.iterrows()]).astype(np.float16)
    models = Path(models)
    models.mkdir(parents=True, exist_ok=True)
    ledger = []
    for seed in SEEDS:
        for day in DAYS:
            tr = rec.index[rec.split.eq('manual') & rec.day.eq(day)].to_numpy()
            counts = rec.loc[tr].groupby('frame_id').size()
            weights = np.array([1/counts[f] for f in rec.loc[tr, 'frame_id']])
            weights /= weights.sum()
            torch.manual_seed(seed)
            rng, aug = np.random.default_rng(seed), np.random.default_rng(seed+15000)
            model = Head().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.03)
            for _ in range(450):
                ix = rng.choice(tr, 8, replace=True, p=weights)
                xb, yb = [], []
                for i in ix:
                    flip = aug.random() < .5
                    xb.append(xf[i] if flip else x[i])
                    yb.append(targets[i, :, :, ::-1] if flip else targets[i])
                xx = torch.from_numpy(np.array(xb, np.float32)).to(device)
                yy = torch.from_numpy(np.array(yb, np.float32)).to(device)
                loss = F.binary_cross_entropy_with_logits(model(xx), yy, weight=1+3*yy)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            name = filename(day, seed)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, models / name)
            ledger.append(dict(file=name, fit_day=day, seed=seed, steps=450,
                               training_frames=sorted(rec.loc[tr, 'frame_id'].unique())))
            print(f'Spatial: fit {day}, seed {seed}, loss {float(loss):.5f}', flush=True)
    write_json(models / 'spatial_training.json', ledger)


def overlap(maps, rows):
    result = []
    for p, (_, r) in zip(maps, rows.iterrows()):
        c, box = r[CROP].to_numpy(float), r[BBOX].to_numpy(float)
        mask = (p[CLASSES.index(r.class_name)] >= .5).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        boxes = [np.array([x, y, x+w, y+h])/32*np.tile(c[2:]-c[:2], 2)+np.tile(c[:2], 2)
                 for x, y, w, h, area in stats[1:] if area >= 4]
        if not boxes:
            result.append(np.nan)
            continue
        b = np.array(boxes)
        inter = np.maximum(0, np.minimum(box[2:], b[:, 2:])-np.maximum(box[:2], b[:, :2])).prod(1)
        union = (box[2:]-box[:2]).prod() + (b[:, 2:]-b[:, :2]).prod(1)-inter
        result.append(float((inter/np.maximum(union, 1e-9)).max()))
    return np.array(result)


def predict(data, cache, models, device):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    rec = pd.read_csv(Path(data) / 'dense_records.csv')
    x, xf = load_grids(cache, rec)
    outputs = []
    for day in DAYS:
        ev = rec.index[rec.split.eq('test') | (rec.split.eq('oof') & rec.day.ne(day))].to_numpy()
        original, flipped = [], []
        for seed in SEEDS:
            head = Head().to(device)
            head.load_state_dict(torch.load(Path(models) / filename(day, seed), map_location=device, weights_only=True))
            head.eval()
            om, fm = [], []
            with torch.inference_mode():
                for start in range(0, len(ev), 16):
                    ix = ev[start:start+16]
                    om.append(head(torch.from_numpy(np.array(x[ix], np.float32)).to(device)).sigmoid().cpu().numpy())
                    fm.append(head(torch.from_numpy(np.array(xf[ix], np.float32)).to(device)).sigmoid().cpu().numpy()[:, :, :, ::-1])
            original.append(np.concatenate(om))
            flipped.append(np.concatenate(fm))
        maps = .5*np.mean(original, axis=0)+.5*np.mean(flipped, axis=0)
        outputs.append(rec.loc[ev, ['id', 'split', 'day']].assign(fit_day=day, overlap=overlap(maps, rec.loc[ev])))
        print(f'Spatial inference: fit {day}', flush=True)
    return pd.concat(outputs, ignore_index=True)
