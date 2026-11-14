"""Perturb one official query at a time; recompute full one-to-one assignment."""
from common54 import *
from scipy.optimize import linear_sum_assignment


def errors(boxes, gt):
    overlaps = iou(boxes, gt)
    labels = np.ones(len(boxes), int)
    if overlaps.size:
        r, c = linear_sum_assignment(-((overlaps >= .6) * (max(overlaps.shape) + 1) + overlaps * (overlaps >= .6)))
        labels[r[overlaps[r, c] >= .6]] = 0
    return labels


def main():
    q, t, h = inputs()
    clip_records = pd.read_csv(ROOT / 'rascar_v41/inputs/records.csv').set_index('id')
    rng = np.random.default_rng(5401)
    output = []
    matching = {}
    for (frame, cls), group in q.groupby(['frame_id', 'class_name']):
        gt = h[h.frame_id.eq(frame) & h.class_name.eq(cls)][B].to_numpy()
        assert np.array_equal(errors(group[B], gt), group.error)
        matching[frame, cls] = (group, gt)
    for split, data in [('oof', q), ('test', t)]:
        for _, query in data.iterrows():
            original = query[B].to_numpy(float)
            crop = clip_records.loc[query.id, CB].to_numpy(float)
            variants = [('original', 0., original)]
            if split == 'oof':
                wh = original[2:] - original[:2]
                center = (original[:2] + original[2:]) / 2
                for axis in [0, 1]:
                    for shift in [-.4, -.2, -.08, .08, .2, .4]:
                        b = original.copy(); b[[axis, axis + 2]] += shift * wh[axis]
                        variants.append(('shift', abs(shift), b))
                for scale in [.65, .8, .92, 1.08, 1.2, 1.4]:
                    variants.append(('scale', abs(scale - 1), np.r_[center - wh * scale / 2, center + wh * scale / 2]))
                for side in range(4):
                    for move in [-.32, -.12, .12, .32]:
                        b = original.copy(); b[side] += move * wh[side % 2]
                        variants.append(('edge', abs(move), b))
                for magnitude in [.08, .18, .35]:
                    for _ in range(4):
                        variants.append(('jitter', magnitude, original + rng.uniform(-magnitude, magnitude, 4) * np.tile(wh, 2)))
                group, gt = matching[query.frame_id, query.class_name]
                if len(gt):
                    target = gt[iou(original, gt)[0].argmax()]
                    for amount in [.35, .7, 1.]:
                        variants.append(('toward_gt', amount, (1 - amount) * original + amount * target))
            seen = set()
            for family, strength, box in variants:
                # Fixed image context is shared; no pixels/feature grid are copied as if moved.
                if family != 'original':
                    box = np.minimum(np.maximum(box, np.tile(crop[:2], 2)), np.tile(crop[2:], 2))
                    if (box[2:] - box[:2] < 4).any():
                        continue
                fingerprint = tuple(np.round(box, 4))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                label = np.nan
                overlap = np.nan
                if split == 'oof':
                    group, gt = matching[query.frame_id, query.class_name]
                    boxes = group[B].to_numpy().copy()
                    position = np.flatnonzero(group.id.eq(query.id))[0]
                    boxes[position] = box
                    label = int(errors(boxes, gt)[position])
                    overlap = float(iou(box, gt).max(initial=0))
                    if family == 'original':
                        assert label == query.error
                identifier = query.id if family == 'original' else f'aug_{query.id}_{len(seen)-1:02}'
                output.append(dict(id=identifier, parent_id=query.id, frame_id=query.frame_id, day=query.day,
                                   class_name=query.class_name, split=split if family == 'original' else 'augmentation',
                                   family=family, strength=strength, error=label, gt_iou=overlap,
                                   **dict(zip(B, box))))
    rows = pd.DataFrame(output)
    assert rows.id.is_unique
    assert rows.loc[rows.split.eq('test'), ['error', 'gt_iou']].isna().all().all()
    rows.to_csv(A / 'inputs/records.csv', index=False)
    original = rows[rows.split.eq('oof')].set_index('id')
    aug = rows[rows.split.eq('augmentation')].copy()
    aug['label_changed'] = aug.error.to_numpy() != original.loc[aug.parent_id, 'error'].to_numpy()
    summary = aug.groupby('family').agg(rows=('id', 'size'), errors=('error', 'sum'), changed_labels=('label_changed', 'sum'))
    summary.to_csv(A / 'audit/augmentation_summary.csv')
    protocol = dict(seed=5401, official_queries=len(q), test_queries=len(t), augmented_queries=len(aug),
                    changed_labels=int(aug.label_changed.sum()), frozen_image_context=True,
                    target='Replace exactly one query in its original frame/class set; rerun maximum-cardinality then max-IoU Hungarian at IoU>=0.6.',
                    validation='Original queries from the other day only; no synthetic validation rows.',
                    feature_policy='Repool cached spatial embeddings for moved ROI and recompute reference/occupancy geometry. SAM/WeVis scalar scores are excluded because they cannot be correctly inherited after moving a box.',
                    pixel_augmentation=False, test_targets_used=False,
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'train.csv', D/'test.csv', D/'human_boxes_train.csv']})
    (A / 'inputs/protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps(protocol, indent=2)); print(summary.to_string())


if __name__ == '__main__':
    main()
