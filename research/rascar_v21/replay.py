"""Replay V21 ranking from cached model outputs and geometric gates.

This is an inference replay, not model training. It needs only this file,
replay_inputs/*.csv, and replay_inputs/manifest.json. No old project folders,
GPU, images, model downloads, or network access are used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

VARIANTS = {
    'main': (.5, .75, 0.),
    'conservative': (.3, .5, 0.),
    'seat_experiment': (.5, .75, .25),
}

def scores(rows, weights):
    base = rows.baseline.to_numpy(float)
    result = base.copy()
    for component, weight, direction in zip(['cup', 'hand', 'person'], weights, [-1, 1, 1]):
        gate = rows[component + '_gate'].astype(str).str.lower().isin(['true', '1'])
        target = rows[component + '_target'].to_numpy(float)
        result += direction * weight * np.maximum(direction * (target - base), 0) * gate.to_numpy()
    assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).resolve().parent / 'replay_inputs')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'replayed')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    for filename, digest in manifest['input_sha256'].items():
        assert hashlib.sha256((args.inputs / filename).read_bytes()).hexdigest() == digest, filename
    verification = {}
    for split in ['oof', 'test']:
        rows = pd.read_csv(args.inputs / f'{split}.csv')
        assert len(rows) == (198 if split == 'oof' else 362) and rows.id.is_unique
        for name, weights in VARIANTS.items():
            prediction = scores(rows, weights)
            error = float(np.max(np.abs(prediction - rows['expected_' + name].to_numpy())))
            assert error < 1e-12, (split, name, error)
            filename = f'submission_v21_{name}.csv' if split == 'test' else f'oof_v21_{name}.csv'
            rows[['id']].assign(score=prediction).to_csv(args.output / filename, index=False)
            verification[filename] = {'rows': len(rows), 'max_abs_error': error}
    (args.output / 'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))

if __name__ == '__main__':
    main()
