from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'rascar_v41'))
from common import inputs, iou, metrics, DAYS, CL, B, CB, D
sys.path.pop(0)

for folder in ['inputs', 'features', 'models', 'results/oof', 'results/submissions', 'audit', 'logs']:
    (A / folder).mkdir(parents=True, exist_ok=True)


def incumbent(q, t):
    o = pd.read_csv(ROOT / 'rascar_v52/results/oof/purged_ensemble_d2_person_a20.csv').set_index('id').loc[q.id, 'score'].to_numpy()
    z = pd.read_csv(ROOT / 'rascar_v52/release/submission_v52_candidate_purged_a20.csv').set_index('id').loc[t.id, 'score'].to_numpy()
    return o, z


def grid_weights(boxes, crop, side):
    rel = (np.asarray(boxes) - np.tile(crop[:2], 2)) / np.tile(crop[2:] - crop[:2], 2)
    lo, hi = np.arange(side) / side, (np.arange(side) + 1) / side
    wx = np.maximum(0, np.minimum(rel[:, 2, None], hi) - np.maximum(rel[:, 0, None], lo)) * side
    wy = np.maximum(0, np.minimum(rel[:, 3, None], hi) - np.maximum(rel[:, 1, None], lo)) * side
    return (wy[:, :, None] * wx[:, None, :]).reshape(len(boxes), -1)


def reference_features(boxes, references, confidence=None):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    references = np.asarray(references, float).reshape(-1, 4)
    names = ['iou', 'second_iou', 'cover_query', 'cover_ref', 'left', 'top', 'right', 'bottom',
             'log_width', 'log_height', 'center_x', 'center_y', 'confidence', 'missing', 'nearby_count']
    result = np.zeros((len(boxes), len(names)), float)
    if not len(references):
        result[:, names.index('missing')] = 1
        return pd.DataFrame(result, columns=names)
    overlaps = iou(boxes, references)
    best = overlaps.argmax(1)
    ref = references[best]
    wh = np.maximum(ref[:, 2:] - ref[:, :2], 1)
    qw = np.maximum(boxes[:, 2:] - boxes[:, :2], 1)
    intersection = np.maximum(0, np.minimum(boxes[:, 2:], ref[:, 2:]) - np.maximum(boxes[:, :2], ref[:, :2])).prod(1)
    result[:, 0] = overlaps[np.arange(len(boxes)), best]
    result[:, 1] = np.sort(overlaps, axis=1)[:, -2] if len(references) > 1 else 0
    result[:, 2] = intersection / qw.prod(1)
    result[:, 3] = intersection / wh.prod(1)
    result[:, 4:8] = (boxes - ref) / np.tile(wh, 2)
    result[:, 8:10] = np.log(qw / wh)
    result[:, 10:12] = ((boxes[:, :2] + boxes[:, 2:]) - (ref[:, :2] + ref[:, 2:])) / (2 * wh)
    result[:, 12] = np.asarray(confidence)[best] if confidence is not None else 1
    result[:, 14] = (overlaps > .1).sum(1)
    return pd.DataFrame(result, columns=names)
