"""Replay the V23 cup candidate from frozen CV features and two fitted heads.

This is a ranking replay, not raw-image detector inference. To regenerate the
features/models, run features23.py and evaluate23.py with their saved V22 inputs.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

A = Path(__file__).resolve().parent
R = A / 'release'
DAYS = ['2026-10-25', '2026-10-26']


def predict(split):
    rows = pd.read_csv(R / f'{split}_inputs.csv')
    probabilities = []
    for fit in DAYS:
        bundle = joblib.load(R / 'models' / f'cup_{fit}.joblib')
        features = pd.read_csv(R / f'{split}_features_{fit}.csv').set_index('id')
        raw = bundle['model'].predict_proba(features[bundle['columns']])[:, 1]
        logits = np.log(np.clip(raw, 1e-5, 1-1e-5) / np.clip(1-raw, 1e-5, 1))[:, None]
        calibrated = bundle['calibrator'].predict_proba(logits)[:, 1]
        probabilities.append(pd.Series(calibrated, index=features.index).reindex(rows.id).to_numpy())
    if split == 'oof':
        p = np.where(rows.day.eq(DAYS[1]), probabilities[0], probabilities[1])
        agreement = np.ones(len(rows), dtype=bool)
    else:
        p = np.mean(probabilities, axis=0)
        agreement = np.maximum(*probabilities) < .5
    base = rows.baseline.to_numpy()
    gate = (rows.class_name.eq('cup').to_numpy() & (p < base) & (p < .4)
            & rows.semantic_probability.ge(.8).to_numpy()
            & rows.distinct_instances.le(1).to_numpy()
            & ~rows.ambiguous_duplicate.astype(bool).to_numpy() & agreement)
    result = base.copy()
    result[gate] += .4 * (p[gate] - base[gate])
    assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    return rows[['id']].assign(score=result)


if __name__ == '__main__':
    report = {}
    for split in ['oof', 'test']:
        result = predict(split)
        reference = pd.read_csv(R / f'{split}_expected.csv')
        assert result.id.equals(reference.id)
        difference = float(np.max(np.abs(result.score - reference.score)))
        assert difference < 1e-12, (split, difference)
        result.to_csv(R / f'{split}_replayed.csv', index=False)
        report[split] = dict(rows=len(result), max_abs_difference=difference, verified=True)
    (R / 'replay_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
