"""Recompute query-dependent features on fixed frozen image contexts.

Inner occupancy heads never saw the parent frame. Prototype banks are training-day
only and exclude every frame within 120 seconds of the augmented parent.
"""
from common54 import *
import torch, cv2, time
from torch import nn
import torch.nn.functional as F

KEYS = ['small_control', 'siglip2', 'vit_b', 'vit_l']


class Dense(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim; self.net = nn.Conv2d(dim, 4, 1)
    def forward(self, x):
        return self.net(F.normalize(x, dim=1) * np.sqrt(self.dim))


def timestamps(rows):
    return pd.to_datetime(rows.frame_id.str.rsplit('_', n=2).str[-2:].str.join(''), format='%Y%m%d%H%M%S').astype('int64').to_numpy() / 1e9


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def geometry(rows, original):
    rawpath = ROOT / 'rascar_v16/inputs/v15/previous/rascar_v14/stage1'
    raw = pd.concat([pd.read_csv(rawpath / f'raw_frozen_{s}_features.csv') for s in ['train', 'test']]).set_index('id')
    owl = {}
    for split in ['oof', 'test']:
        for line in (ROOT / f'rascar_v50/features/{split}.jsonl').read_text().splitlines():
            z = json.loads(line); owl[z['id']] = z['references']
    blocks, audit = [], []
    for parent, group in rows.groupby('parent_id', sort=False):
        boxes = group[B].to_numpy(float); source = original.loc[parent, B].to_numpy(float)
        r = raw.loc[parent]; parts = []
        if r.no_reference:
            refs = []
        else:
            wh = (source[2:] - source[:2]) / np.exp([r.log_width_ratio, r.log_height_ratio])
            start = source[:2] - np.array([r.left_offset, r.top_offset]) * wh
            refs = [np.r_[start, start + wh]]
        f = reference_features(boxes, refs, [r.ref_confidence] if refs else None)
        reconstructed = f.iloc[0]['iou']
        audit.append(dict(id=parent, original_iou=r.best_iou, reconstructed_iou=reconstructed))
        parts.append(f.add_prefix('raw_'))
        for cut in [.05, .1, .2]:
            rr = [x for x in owl[parent] if x['positive'] >= cut]
            parts.append(reference_features(boxes, [x['xyxy'] for x in rr], [x['positive'] for x in rr]).add_prefix(f'owl{int(cut*100):02}_'))
        wh = np.maximum(boxes[:, 2:] - boxes[:, :2], 1)
        parts.append(pd.DataFrame(dict(log_area=np.log(wh.prod(1)), log_aspect=np.log(wh[:, 0]/wh[:, 1]), width=wh[:, 0]/1280, height=wh[:, 1]/720)))
        block = pd.concat(parts, axis=1); block.index = group.index; blocks.append(block)
    audit = pd.DataFrame(audit)
    assert abs(audit.original_iou - audit.reconstructed_iou).max() < 1e-5
    audit.to_csv(A / 'audit/raw_geometry_reconstruction.csv', index=False)
    return pd.concat(blocks).sort_index()


def repool(rows):
    rec = pd.read_csv(ROOT / 'rascar_v41/inputs/records.csv'); lookup = rec.reset_index().set_index('id')['index']
    for key in ['clip', 'vit_b', 'vit_l']:
        out = A / f'features/{key}_roi.npy'
        if out.exists():
            continue
        patches = np.load(ROOT / f'rascar_v41/features/{key}_final_patches.npy', mmap_mode='r')
        result = np.zeros((len(rows), patches.shape[1]), np.float32)
        for parent, group in rows.groupby('parent_id', sort=False):
            i = lookup[parent]; crop = rec.loc[i, CB].to_numpy(float)
            weights = grid_weights(group[B].to_numpy(float), crop, patches.shape[2]).astype(np.float32)
            weights /= np.maximum(weights.sum(1, keepdims=True), 1e-8)
            result[group.index] = weights @ np.asarray(patches[i], np.float32).reshape(patches.shape[1], -1).T
        np.save(out, result); print('REPOOL', key, result.shape, flush=True)


def prototype_features(rows, fit, allowed_frames=None):
    proto = pd.read_csv(ROOT / 'rascar_v52/inputs/prototypes.csv')
    qt, pt = timestamps(rows), timestamps(proto)
    output = {}; ledger = {}
    for key in ['clip', 'vit_b', 'vit_l']:
        x = normalize(np.load(A / f'features/{key}_roi.npy'))
        p = np.load(ROOT / f'rascar_v52/features/{key}_prototypes.npy'); dim = p.shape[1] // 3
        p = normalize(p[:, dim:2*dim]); own = []
        for cls in CL:
            use = proto.day.eq(fit) & proto.class_name.eq(cls)
            if allowed_frames is not None:
                use &= proto.frame_id.isin(allowed_frames)
            sim = x @ p[use].T; valid = abs(qt[:, None] - pt[None, use]) > 120
            top = np.sort(np.where(valid, sim, -10), axis=1)[:, -min(3, use.sum()):]
            best = np.maximum(top, 0).sum(1) / np.maximum((top > -9).sum(1), 1)
            output[f'proto_{key}_{cls}_top3'] = best
            output[f'proto_{key}_{cls}_mean'] = np.where(valid, sim, 0).sum(1) / np.maximum(valid.sum(1), 1)
            own.append(best)
            ledger[cls] = sorted(proto.loc[use, 'frame_id'].unique())
        own = np.stack(own, 1); ci = rows.class_name.map(CL.index).to_numpy()
        output[f'proto_{key}_margin'] = own[np.arange(len(rows)), ci] - np.max(np.where(np.arange(4)[None, :] == ci[:, None], -10, own), axis=1)
    if allowed_frames is None:
        (A / f'audit/prototype_sources_fit{fit[-2:]}.json').write_text(json.dumps(ledger, indent=2))
    return pd.DataFrame(output)


def dense_evidence(prob, boxes, crop, cls):
    side = prob.shape[-1]; ci = CL.index(cls); p = prob[ci]
    weights = grid_weights(boxes, crop, side).astype(np.float32)
    inside = weights / np.maximum(weights.sum(1, keepdims=True), 1e-8)
    outside = 1 - weights; outside /= np.maximum(outside.sum(1, keepdims=True), 1e-8)
    means = inside @ prob.reshape(4, -1).T
    _, labels, stats, _ = cv2.connectedComponentsWithStats((p >= .5).astype(np.uint8), 8)
    refs, conf = [], []
    for j, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if area >= 4:
            refs.append(np.array([x,y,x+w,y+h])/side * np.tile(crop[2:]-crop[:2], 2) + np.tile(crop[:2], 2))
            conf.append(float(p[labels == j].mean()))
    f = reference_features(boxes, refs, conf)
    f['mean'] = means[:, ci]; f['coverage'] = inside @ (p.ravel() >= .5)
    f['outside'] = outside @ p.ravel(); f['inside_minus_outside'] = f['mean'] - f['outside']
    f['other_mean'] = means[:, np.arange(4) != ci].max(1)
    center = (boxes[:, :2] + boxes[:, 2:]) / 2; wh = (boxes[:, 2:] - boxes[:, :2]) * .5
    center_weights = grid_weights(np.c_[center-wh/2, center+wh/2], crop, side)
    center_weights /= np.maximum(center_weights.sum(1, keepdims=True), 1e-8)
    f['center'] = center_weights @ p.ravel()
    return f


def dense_features(rows, original, fit, key):
    rec = pd.read_csv(ROOT / 'rascar_v31/inputs/dense_records.csv'); lookup = rec.reset_index().set_index('id')['index']
    manifest = json.loads((ROOT / 'rascar_v48/models/inner_manifest.json').read_text())
    path = ROOT / ('rascar_v31/features/dense_grids.npy' if key == 'small_control' else f'rascar_v46/features/{key}_grids.npy')
    features = np.load(path, mmap_mode='r'); heads = {}; ledger = []
    for m in manifest:
        if m['key'] != key or m['outer_fit_day'] != fit:
            continue
        assert not set(m['fit_frames']) & set(m['held_frames'])
        assert all('_' + fit.replace('-', '') + '_' in f for f in m['fit_frames'])
        model = Dense(features.shape[1]); block = m['inner_block']
        path = ROOT / (f'rascar_v45/models/inner_dense_fit{fit[-2:]}_block{block}.pt' if key == 'small_control' else f'rascar_v48/models/{key}_fit{fit[-2:]}_block{block}.pt')
        model.load_state_dict(torch.load(path, weights_only=True)); model.eval()
        for frame in m['held_frames']:
            heads[frame] = model
        ledger.append(dict(key=key, fit_day=fit, **{k:m[k] for k in ['inner_block','fit_frames','held_frames']}))
    data = np.load(ROOT / f'rascar_v46/features/{key}_s3101_fit{fit[-2:]}_maps.npz')
    maps = dict(zip(data['ids'], data['maps'])); blocks = []
    for parent, group in rows.groupby('parent_id', sort=False):
        query = original.loc[parent]; i = lookup[parent]; crop = rec.loc[i, CB].to_numpy(float)
        if query.day == fit:
            assert query.frame_id in heads
            with torch.inference_mode():
                prob = heads[query.frame_id](torch.from_numpy(np.array(features[i:i+1], np.float32)))[0].sigmoid().numpy()
        else:
            prob = maps[parent].astype(np.float32)
        f = dense_evidence(prob, group[B].to_numpy(float), crop, query.class_name).add_prefix(f'dense_{key}_')
        f.index = group.index; blocks.append(f)
    (A / f'audit/occupancy_{key}_fit{fit[-2:]}.json').write_text(json.dumps(ledger, indent=2))
    return pd.concat(blocks).sort_index()


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1)
    rows = pd.read_csv(A / 'inputs/records.csv')
    original = rows[rows.family.eq('original')].set_index('id')
    static = geometry(rows, original); static.to_pickle(A / 'features/static.pkl')
    print('GEOMETRY', static.shape, flush=True); repool(rows)
    for fit in DAYS:
        parts = [static, prototype_features(rows, fit)]
        for key in KEYS:
            path = A / f'features/{key}_fit{fit[-2:]}.pkl'
            if path.exists():
                f = pd.read_pickle(path)
            else:
                f = dense_features(rows, original, fit, key); f.to_pickle(path)
            parts.append(f); print('DENSE', fit, key, f.shape, flush=True)
        bank = pd.concat(parts, axis=1)
        assert np.isfinite(bank.to_numpy()).all()
        bank['class_name'] = rows.class_name
        bank.to_pickle(A / f'features/bank_fit{fit[-2:]}.pkl')
        print('BANK', fit, bank.shape, flush=True)
    groups = dict(geometry=list(static), proto=[c for c in bank if c.startswith('proto_')],
                  dense=[c for c in bank if c.startswith('dense_')],
                  dense_small=[c for c in bank if c.startswith('dense_small_control_')],
                  dense_siglip=[c for c in bank if c.startswith('dense_siglip2_')])
    (A / 'features/groups.json').write_text(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
