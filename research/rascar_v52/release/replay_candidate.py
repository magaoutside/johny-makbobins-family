"""Replay the frozen V52 candidate on its 198 OOF and 362 test queries.

This is cached-feature inference, not an extractor for arbitrary new images.
It deliberately never reads error labels, fits a model, or selects parameters.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

CLASSES = ['person', 'nude_hands', 'cup', 'tray']


def replay(bundle, output):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == expected, name
    rows = pd.read_csv(bundle / 'inputs/rows.csv')
    proto = pd.read_csv(bundle / 'inputs/prototypes.csv')
    static = pd.read_csv(bundle / 'inputs/static.csv')
    assert rows.id.is_unique and static.id.tolist() == rows.id.tolist()
    assert not {'error', 'score'} & set(static.columns)
    seconds = lambda d: pd.to_datetime(
        d.frame_id.str.rsplit('_', n=2).str[-2:].str.join(''),
        format='%Y%m%d%H%M%S',
    ).astype('int64').to_numpy() / 1e9
    qt, pt = seconds(rows), seconds(proto)
    predictions = []
    for fit in manifest['fit_days']:
        train = rows.split.eq('oof') & rows.day.eq(fit)
        evaluate = ~train
        bank = static.drop(columns='id').copy()
        for backbone in ['clip', 'vit_b', 'vit_l']:
            x = np.load(bundle / f'inputs/{backbone}_roi.npy')
            p = np.load(bundle / f'inputs/{backbone}_prototype_roi.npy')
            assert len(x) == len(rows) and len(p) == len(proto)
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
            p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
            own = []
            for cls in CLASSES:
                use = proto.day.eq(fit) & proto.class_name.eq(cls)
                assert use.any(), (fit, cls)
                sim = x @ p[use].T
                valid = np.ones(sim.shape, bool)
                valid[train] = abs(qt[train, None] - pt[None, use]) > 120
                top = np.sort(np.where(valid, sim, -10), axis=1)[:, -min(3, use.sum()):]
                best = np.maximum(top, 0).sum(1) / np.maximum((top > -9).sum(1), 1)
                mean = np.where(valid, sim, 0).sum(1) / np.maximum(valid.sum(1), 1)
                bank[f'proto_{backbone}_{cls}_top3'] = best
                bank[f'proto_{backbone}_{cls}_mean'] = mean
                own.append(best)
            own = np.stack(own, 1)
            ci = rows.class_name.map(CLASSES.index).to_numpy()
            bank[f'proto_{backbone}_margin'] = own[np.arange(len(rows)), ci] - np.max(
                np.where(np.arange(4)[None, :] == ci[:, None], -10, own), axis=1)
        for entry in manifest['models']:
            if entry['fit_day'] != fit:
                continue
            assert entry['train_ids'] == rows.loc[train, 'id'].tolist()
            model = CatBoostClassifier()
            model.load_model(str(bundle / entry['path']))
            assert model.feature_names_ == entry['features']
            probability = model.predict_proba(bank.loc[evaluate, entry['features']])[:, 1]
            predictions.append(rows.loc[evaluate, ['id', 'split']].assign(
                p_error=probability, fit_day=fit, seed=entry['seed']))
    predictions = pd.concat(predictions, ignore_index=True)
    counts = predictions.groupby('id').size()
    assert counts.loc[rows.loc[rows.split.eq('oof'), 'id']].eq(3).all()
    assert counts.loc[rows.loc[rows.split.eq('test'), 'id']].eq(6).all()
    mean = predictions.groupby('id').p_error.mean()
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / 'model_predictions.csv', index=False)
    checks = {}
    for split in ['oof', 'test']:
        r = rows[rows.split.eq(split)]
        base = pd.read_csv(bundle / f'inputs/baseline_{split}.csv')
        assert base.id.tolist() == r.id.tolist()
        p = mean.loc[r.id].to_numpy()
        for alpha in manifest['alpha']:
            score = base.score.to_numpy().copy()
            mask = r.class_name.eq('person').to_numpy()
            score[mask] = (score[mask] + alpha * p[mask]) / (1 + alpha)
            assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
            name = f'{split}_a{round(100 * alpha):02}.csv'
            actual = pd.DataFrame({'id': r.id, 'score': score})
            expected = pd.read_csv(bundle / 'expected' / name)
            assert actual.id.tolist() == expected.id.tolist()
            error = float(np.max(abs(actual.score.to_numpy() - expected.score.to_numpy())))
            assert error < 1e-10, (name, error)
            actual.to_csv(output / name, index=False)
            checks[name] = {'rows': len(actual), 'maximum_absolute_error': error}
    (output / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    replay(args.bundle, args.output or args.bundle / 'replayed')
