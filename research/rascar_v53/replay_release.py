"""Replay V53's 50% full-data refit using frozen features and nine weights."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.bundle
    output = args.output or root / 'replayed'
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, sha in manifest['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    rows = pd.read_csv(root / 'inputs/rows.csv')
    static = pd.read_csv(root / 'inputs/static.csv')
    proto = pd.read_csv(root / 'inputs/prototypes.csv')
    assert static.id.tolist() == rows.id.tolist()
    seconds = lambda d: pd.to_datetime(d.frame_id.str.rsplit('_', n=2).str[-2:].str.join(''),
                                      format='%Y%m%d%H%M%S').astype('int64').to_numpy() / 1e9
    qt, pt = seconds(rows), seconds(proto)
    classes = ['person', 'nude_hands', 'cup', 'tray']
    predictions = []
    for fit in ['2026-10-25', '2026-10-26', 'full']:
        days = ['2026-10-25', '2026-10-26'] if fit == 'full' else [fit]
        train = rows.split.eq('oof') & rows.day.isin(days)
        evaluate = rows.split.eq('test') if fit == 'full' else ~train
        bank = static.drop(columns='id').copy()
        for e in ['clip', 'vit_b', 'vit_l']:
            x = np.load(root / f'inputs/{e}_roi.npy')
            p = np.load(root / f'inputs/{e}_prototype_roi.npy')
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
            p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
            own = []
            for cls in classes:
                use = proto.day.isin(days) & proto.class_name.eq(cls)
                sim = x @ p[use].T
                valid = abs(qt[:, None] - pt[None, use]) > 120
                top = np.sort(np.where(valid, sim, -10), axis=1)[:, -min(3, use.sum()):]
                best = np.maximum(top, 0).sum(1) / np.maximum((top > -9).sum(1), 1)
                mean = np.where(valid, sim, 0).sum(1) / np.maximum(valid.sum(1), 1)
                for prefix in ['proto', 'gt']:
                    bank[f'{prefix}_{e}_{cls}_top3'] = best
                    bank[f'{prefix}_{e}_{cls}_mean'] = mean
                own.append(best)
            own = np.stack(own, 1)
            ci = rows.class_name.map(classes.index).to_numpy()
            margin = own[np.arange(len(rows)), ci] - np.max(np.where(np.arange(4)[None, :] == ci[:, None], -10, own), axis=1)
            for prefix in ['proto', 'gt']:
                bank[f'{prefix}_{e}_margin'] = margin
        for entry in manifest['models']:
            if entry['fit_day'] != fit:
                continue
            model = CatBoostClassifier()
            model.load_model(str(root / entry['path']))
            assert model.feature_names_ == entry['features']
            assert entry['train_ids'] == rows.loc[train, 'id'].tolist()
            p = model.predict_proba(bank.loc[evaluate, entry['features']])[:, 1]
            predictions.append(rows.loc[evaluate, ['id', 'split']].assign(p_error=p, fit_day=fit, seed=entry['seed']))
    predictions = pd.concat(predictions, ignore_index=True)
    test = rows[rows.split.eq('test')]
    fold_p = predictions[predictions.split.eq('test') & predictions.fit_day.ne('full')].groupby('id').p_error.mean().loc[test.id].to_numpy()
    full_p = predictions[predictions.split.eq('test') & predictions.fit_day.eq('full')].groupby('id').p_error.mean().loc[test.id].to_numpy()
    base = pd.read_csv(root / 'inputs/baseline_test.csv')
    assert base.id.tolist() == test.id.tolist()
    score = base.score.to_numpy().copy()
    mask = test.class_name.eq('person').to_numpy()
    p = .5 * fold_p + .5 * full_p
    score[mask] = (score[mask] + .2 * p[mask]) / 1.2
    expected = pd.read_csv(root / 'submission_v53_refit50.csv')
    assert expected.id.tolist() == test.id.tolist()
    error = float(abs(score - expected.score.to_numpy()).max())
    assert error < 1e-10, error
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
    # The opposite-day outputs validate the original recipe; full models have no OOF.
    q = rows[rows.split.eq('oof')]
    oo = predictions[predictions.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy()
    reference = pd.read_csv(root / 'inputs/baseline_oof.csv')
    sm = q.class_name.eq('person').to_numpy()
    ss = reference.score.to_numpy().copy()
    ss[sm] = (ss[sm] + .2 * oo[sm]) / 1.2
    old = pd.read_csv(root / 'inputs/v52_oof_reference.csv')
    assert old.id.tolist() == q.id.tolist()
    oof_error = float(abs(ss - old.score.to_numpy()).max())
    assert oof_error < 1e-10
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': test.id, 'score': score}).to_csv(output / 'submission_v53_refit50.csv', index=False)
    predictions.to_csv(output / 'model_predictions.csv', index=False)
    checks = dict(test_rows=len(test), test_replay_max_error=error,
                  original_recipe_oof_replay_max_error=oof_error,
                  full_refit_has_no_independent_oof=True, loaded_models=len(manifest['models']))
    (output / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
