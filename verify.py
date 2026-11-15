"""Validate release artifacts and generated predictions against the original run."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rascar.core import ROOT, queries, aligned_csv, check_predictions, verify_artifacts

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--allow-drift', action='store_true', help='Report GPU/retraining drift instead of requiring exact release scores')
    args = parser.parse_args()
    count = verify_artifacts()
    rows = queries(ROOT / 'data')
    frames = []
    for split, name in [('oof', 'oof.csv'), ('test', 'submission.csv')]:
        subset = rows[rows.split.eq(split)]
        frame = pd.read_csv(args.out / name)
        if frame.columns.tolist() != ['id', 'score'] or frame.id.duplicated().any():
            raise ValueError('Expected unique id,score columns')
        if frame.id.tolist() != subset.id.tolist():
            raise ValueError('Wrong prediction order')
        frames.append(frame)
    scores = pd.concat(frames).set_index('id').loc[rows.id, 'score'].to_numpy()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid score range')
    report = check_predictions(rows, scores, ROOT / 'data', args.out, exact=not args.allow_drift)
    print(f'OK: {count} artifacts; OOF AP={report["pooled_oof_ap"]:.9f}; max score difference={report["max_absolute_difference_from_release"]:.3g}')
    print(f'Отчёт: {args.out / "verification.json"}')

if __name__ == '__main__':
    main()
