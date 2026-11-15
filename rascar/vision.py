"""Regenerate ROI embeddings and dense grids directly from competition JPGs."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import cv2
import torch
from .core import BBOX, CROP, queries, write_json


def resize_without_ipp(image, size):
    """Match OpenCV cubic resize used by the original macOS feature runs."""
    previous = cv2.ipp.useIPP()
    try:
        cv2.ipp.setUseIPP(False)
        return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    finally:
        cv2.ipp.setUseIPP(previous)


def image_reader(images):
    paths = {p.stem: p for p in Path(images).rglob('*.jpg')}
    @lru_cache(maxsize=32)
    def read(frame):
        if frame not in paths:
            raise FileNotFoundError(f'Missing competition image: {frame}.jpg')
        im = cv2.imread(str(paths[frame]))
        if im is None:
            raise ValueError(f'Unreadable image: {paths[frame]}')
        return cv2.resize(im, (1280, 720))[:, :, ::-1].copy()
    return read


def prototype_records(data):
    rows = queries(data)
    h = pd.read_csv(Path(data) / 'manual_boxes.csv')
    meta = pd.read_csv(Path(data) / 'prototypes.csv')
    records = []
    for _, p in meta.iterrows():
        index = int(p.id.split('_')[1])
        r = h.loc[index]
        if r.frame_id != p.frame_id or r.class_name != p.class_name:
            raise ValueError('Manual box order differs from prototype IDs')
        records.append(dict(id=p.id, frame_id=p.frame_id, **{c: r[c] for c in BBOX}))
    result = pd.concat([rows[['id', 'frame_id']+BBOX], pd.DataFrame(records)], ignore_index=True)
    b = result[BBOX].to_numpy(float)
    wh = np.maximum(b[:, 2:]-b[:, :2], 4)
    center = (b[:, :2]+b[:, 2:])/2
    crop = np.c_[np.maximum(0, np.floor(center-wh*.95)),
                 np.minimum([1280, 720], np.ceil(center+wh*.95))]
    result[CROP] = crop
    return result, len(rows)


def roi_mask(records, grid):
    b, c = records[BBOX].to_numpy(float), records[CROP].to_numpy(float)
    rel = (b-np.tile(c[:, :2], 2))/np.tile(c[:, 2:]-c[:, :2], 2)
    low, high = np.arange(grid)/grid, (np.arange(grid)+1)/grid
    wx = np.maximum(0, np.minimum(rel[:, 2, None], high)-np.maximum(rel[:, 0, None], low))*grid
    wy = np.maximum(0, np.minimum(rel[:, 3, None], high)-np.maximum(rel[:, 1, None], low))*grid
    return torch.tensor((wy[:, :, None]*wx[:, None, :]).reshape(len(records), -1, 1), dtype=torch.float32)


def extract_roi(data, images, weights, output, device):
    from transformers import AutoModel, CLIPVisionModel
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    records, n_queries = prototype_records(data)
    read = image_reader(images)
    output = Path(output)/'embeddings'
    output.mkdir(parents=True, exist_ok=True)
    for key in ['clip', 'vit_b', 'vit_l']:
        cls = CLIPVisionModel if key == 'clip' else AutoModel
        model = cls.from_pretrained(str(Path(weights)/key), local_files_only=True,
                                    attn_implementation='eager').eval().to(device)
        mean = [.48145466, .4578275, .40821073] if key == 'clip' else [.485, .456, .406]
        std = [.26862954, .26130258, .27577711] if key == 'clip' else [.229, .224, .225]
        mean = torch.tensor(mean, device=device)[None, :, None, None]
        std = torch.tensor(std, device=device)[None, :, None, None]
        masks = roi_mask(records, 7 if key == 'clip' else 16)
        features = []
        for start in range(0, len(records), 8):
            batch = []
            for _, r in records.iloc[start:start+8].iterrows():
                c = r[CROP].to_numpy(int)
                im = read(r.frame_id)[c[1]:c[3], c[0]:c[2]]
                batch.append(resize_without_ipp(im, (224, 224)).transpose(2, 0, 1).copy())
            x = torch.from_numpy(np.stack(batch)).to(device).float()/255
            with torch.inference_mode():
                z = model(pixel_values=(x-mean)/std).last_hidden_state
                if key == 'clip':
                    z = model.vision_model.post_layernorm(z)
                mask = masks[start:start+len(batch)].to(device)
                inside = (z[:, 1:]*mask).sum(1)/mask.sum(1).clamp_min(.01)
            features.append(inside.cpu().numpy())
            if start % 160 == 0:
                print(f'ROI {key}: {start}/{len(records)}', flush=True)
        features = np.concatenate(features).astype(np.float32)
        np.save(output/f'{key}_queries.npy', features[:n_queries])
        np.save(output/f'{key}_prototypes.npy', features[n_queries:])
        del model
    write_json(output/'extraction.json', dict(record_ids=records.id.tolist(),
               labels_used=False, crop_scale=1.9, input_size=224, device=device, resize_ipp=False))


def extract_dense(data, images, weights, cache, device, small=False):
    from transformers import AutoModel
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    rec = pd.read_csv(Path(data)/'dense_records.csv')
    read = image_reader(images)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache/('dense_small_complete.json' if small else 'dense_complete.json')
    dim = 384 if small else 768
    marker.unlink(missing_ok=True)
    model = AutoModel.from_pretrained(str(Path(weights)/('vit_s' if small else 'vit_b')), local_files_only=True).to(device).eval()
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    views = [(False, 'dense_small.npy')] if small else [(False, 'dense.npy'), (True, 'dense_flip.npy')]
    for flip, name in views:
        temp = cache/(name+'.partial')
        dst = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float16, shape=(len(rec), dim, 32, 32))
        for start in range(0, len(rec), 2):
            batch = []
            for _, r in rec.iloc[start:start+2].iterrows():
                c = r[CROP].to_numpy(int)
                im = read(r.frame_id)[c[1]:c[3], c[0]:c[2]]
                if flip:
                    im = im[:, ::-1]
                resized = resize_without_ipp(im, (448, 448)) if small else cv2.resize(
                    im, (448, 448), interpolation=cv2.INTER_CUBIC)
                batch.append(resized.transpose(2, 0, 1).copy())
            x = torch.from_numpy(np.stack(batch)).to(device).float()/255
            with torch.inference_mode():
                z = model(pixel_values=(x-mean)/std).last_hidden_state[:, 1:].transpose(1, 2).reshape(-1, dim, 32, 32)
            dst[start:start+len(batch)] = z.cpu().numpy().astype(np.float16)
            if start % 100 == 0:
                print(f'{name}: {start}/{len(rec)}', flush=True)
        dst.flush()
        del dst
        temp.replace(cache/name)
    write_json(marker, dict(ids=rec.id.tolist(), device=device, rgb_flip=not small,
                           resize_ipp=False if small else cv2.ipp.useIPP()))
