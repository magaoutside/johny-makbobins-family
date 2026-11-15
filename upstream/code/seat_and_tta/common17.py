from pathlib import Path
import sys, json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP
A = Path(__file__).resolve().parent
V16 = A.parent / 'geometry_guards'
V15 = A.parent / 'local_evidence'
sys.path.insert(0, str(V16))
import common16 as prev
V15 = prev.V15
B, CL, DAYS = (prev.B, prev.CL, prev.DAYS)
q, te, h, fr = (prev.q.copy(), prev.te.copy(), prev.h.copy(), prev.fr.copy())
iou = prev.iou
baseline15 = q.score.to_numpy().copy()
test15 = prev.current.copy()
q['score'] = pd.read_csv(V16 / 'results/oof/final_targeted_short_with_contradiction.csv').set_index('id').loc[q.id, 'score'].to_numpy()
current = pd.read_csv(V16 / 'submission_v16_targeted_short.csv').set_index('id').loc[te.id, 'score'].to_numpy()
R = A / 'results'
R.mkdir(exist_ok=True)
for folder in ['oof', 'submissions', 'reviews']:
    (R / folder).mkdir(exist_ok=True)

def align(path, rows, cols=None):
    df = pd.read_csv(path).set_index('id').loc[rows.id]
    return df if cols is None else df[cols].to_numpy()

def metric(rows, s):
    s = np.asarray(s)
    out = {'pooled_ap': float(AP(rows.error, s))}
    for d, g in rows.groupby('day'):
        out['day_' + d[-2:]] = float(AP(g.error, s[g.index]))
    for c, g in rows.groupby('class_name'):
        out['ap_' + c] = float(AP(g.error, s[g.index]))
    return out

def register(name, s, t, details=None):
    s = np.asarray(s, float)
    t = np.asarray(t, float)
    assert s.shape == (198,) and t.shape == (362,) and np.isfinite(s).all() and np.isfinite(t).all()
    assert ((s >= 0) & (s <= 1)).all() and ((t >= 0) & (t <= 1)).all()
    result = A / 'results' if 'A' in globals() else V16 / 'results'
    (result / 'oof').mkdir(parents=True, exist_ok=True)
    (result / 'submissions').mkdir(parents=True, exist_ok=True)
    q.assign(score=s).to_csv(result / 'oof' / f'{name}.csv', index=False)
    path = result / 'submissions' / f'submission_{name}.csv'
    te[['id']].assign(score=t).to_csv(path, index=False)
    row = dict(experiment=name, **metric(q, s), details=details or {}, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (result / f'{name}.json').write_text(json.dumps(row, indent=2))
    print('RESULT', name, round(row['pooled_ap'], 9), flush=True)
    return row
