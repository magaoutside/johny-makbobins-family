"""Deterministic metadata preparation from the original competition tables."""
from pathlib import Path
import numpy as np
import pandas as pd
from .core import BBOX, CROP, CLASSES


def prepare(raw, out):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    def table(name):
        matches = list(raw.rglob(name))
        if len(matches) != 1:
            raise ValueError(f'Expected one {name} below {raw}; found {len(matches)}')
        return pd.read_csv(matches[0])
    q, t, h = table('train.csv'), table('test.csv'), table('human_boxes_train.csv')
    for d in [q, t, h]:
        d['class_name'] = d.class_name.replace({'nude_hand': 'nude_hands'})
        d['day'] = pd.to_datetime(d.frame_id.str.split('_').str[-2], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    q = q[q.class_name.isin(CLASSES)].reset_index(drop=True)
    rows = pd.concat([q.assign(split='oof'), t.assign(split='test')], ignore_index=True)
    rows.loc[rows.split.eq('oof'), ['id', 'error']].astype({'error': int}).to_csv(out/'train_labels.csv', index=False)
    rows.drop(columns='error').to_csv(out/'queries.csv', index=False)
    h[BBOX] = np.c_[(h.cx-h.w/2)*1280, (h.cy-h.h/2)*720, (h.cx+h.w/2)*1280, (h.cy+h.h/2)*720]
    h.to_csv(out/'manual_boxes.csv', index=False)
    proto = h[h.class_name.isin(CLASSES)][['frame_id', 'day', 'class_name']].copy()
    proto.insert(0, 'id', [f'synth_{i}_0' for i in proto.index])
    proto.to_csv(out/'prototypes.csv', index=False)
    dense = []
    rng = np.random.default_rng(3101)
    for j, r in h.iterrows():
        if r.class_name not in list(CLASSES)+['phone', 'full_plate', 'empty_plate', 'hair', 'personal_items']:
            continue
        for aug in range(2 if r.class_name in CLASSES else 1):
            b = r[BBOX].to_numpy(float)
            wh = np.maximum(b[2:]-b[:2], 16)
            center = (b[:2]+b[2:])/2+rng.uniform(-.35, .35, 2)*wh
            size = wh*rng.uniform(2., 3.2)
            crop = np.r_[np.maximum(0, np.floor(center-size/2)), np.minimum([1280, 720], np.ceil(center+size/2))]
            dense.append(dict(id=f'manual_{j}_{aug}', frame_id=r.frame_id, day=r.day,
                              class_name=r.class_name, split='manual', origin_id=f'manual_{j}',
                              **dict(zip(BBOX, b)), **dict(zip(CROP, crop))))
    for _, r in rows.iterrows():
        b = r[BBOX].to_numpy(float)
        center = (b[:2]+b[2:])/2
        size = np.maximum(b[2:]-b[:2], 16)*2.4
        crop = np.r_[np.maximum(0, np.floor(center-size/2)), np.minimum([1280, 720], np.ceil(center+size/2))]
        dense.append(dict(id=r.id, frame_id=r.frame_id, day=r.day, class_name=r.class_name,
                          split=r.split, origin_id=r.id, **dict(zip(BBOX, b)), **dict(zip(CROP, crop))))
    dense = pd.DataFrame(dense)
    dense['feature_index'] = np.arange(len(dense))
    dense.to_csv(out/'dense_records.csv', index=False)
    print(f'Metadata: {len(q)} train queries, {len(t)} test queries, {len(proto)} prototypes, {len(dense)} dense crops')
