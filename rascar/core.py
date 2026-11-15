"""Paths, input contracts and release validation shared by the command-line tools."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAYS = ('2026-10-25', '2026-10-26')
CLASSES = ('person', 'nude_hands', 'cup', 'tray')
BBOX = ['x1', 'y1', 'x2', 'y2']
CROP = ['crop_x1', 'crop_y1', 'crop_x2', 'crop_y2']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def queries(data_dir):
    rows = pd.read_csv(Path(data_dir) / 'queries.csv')
    if not rows.id.is_unique or not set(rows.class_name) <= set(CLASSES):
        raise ValueError('queries.csv: invalid/duplicate IDs or unknown classes')
    if set(rows.split) != {'oof', 'test'} or 'error' in rows:
        raise ValueError('queries.csv must contain oof/test splits and no error labels')
    return rows


def labels(data_dir, rows):
    table = pd.read_csv(Path(data_dir) / 'train_labels.csv').set_index('id')
    train = rows.split.eq('oof')
    if not table.index.is_unique or set(table.index) != set(rows.loc[train, 'id']):
        raise ValueError('train_labels.csv must contain exactly the OOF IDs')
    y = pd.Series(np.nan, index=rows.index)
    y.loc[train] = table.loc[rows.loc[train, 'id'], 'error'].to_numpy()
    if not set(y[train]) <= {0, 1}:
        raise ValueError('Training labels must be 0 or 1')
    return y


def aligned_csv(path, ids):
    table = pd.read_csv(path)
    if not table.id.is_unique or set(table.id) != set(ids):
        raise ValueError(f'ID mismatch: {path}')
    return table.set_index('id').loc[ids].reset_index()


def verify_artifacts(root=ROOT):
    expected = read_json(root / 'checksums.json')
    for relative, checksum in expected.items():
        path = root / relative
        if not path.is_file() or sha256(path) != checksum:
            raise ValueError(f'Missing or changed release artifact: {relative}')
    return len(expected)


def save_predictions(rows, scores, out):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError('Prediction shape mismatch or non-finite scores')
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError('Scores must be in [0, 1]')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for split, filename in [('test', 'submission.csv'), ('oof', 'oof.csv')]:
        use = rows.split.eq(split)
        pd.DataFrame({'id': rows.loc[use, 'id'], 'score': scores[use]}).to_csv(
            out / filename, index=False, float_format='%.12f')
    return out / 'submission.csv'


def check_predictions(rows, scores, data_dir, out, exact=True):
    """Evaluation is separate from inference: only this function reads held-out labels."""
    from sklearn.metrics import average_precision_score
    y = labels(data_dir, rows)
    expected = aligned_csv(ROOT / 'reference/predictions.csv', rows.id).score.to_numpy()
    difference = float(np.max(np.abs(np.asarray(scores) - expected)))
    train = rows.split.eq('oof').to_numpy()
    report = {'rows_test': int((~train).sum()), 'rows_oof': int(train.sum()),
              'max_absolute_difference_from_release': difference,
              'pooled_oof_ap': float(average_precision_score(y[train], np.asarray(scores)[train])),
              'test_labels_used': False, 'exact_release_check': exact,
              'submission_sha256': sha256(Path(out) / 'submission.csv')}
    for day in DAYS:
        mask = train & rows.day.eq(day).to_numpy()
        report[f'oof_ap_{day}'] = float(average_precision_score(y[mask], np.asarray(scores)[mask]))
    report['matches_release'] = difference <= 1e-10
    write_json(Path(out) / 'verification.json', report)
    if exact and not report['matches_release']:
        raise ValueError(f'Release mismatch: {difference:.3g}; see {out}/verification.json')
    return report
