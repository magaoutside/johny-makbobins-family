from pathlib import Path
import sys, json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP
V16 = Path(__file__).resolve().parent
V15 = V16 / 'inputs/v15' if (V16 / 'inputs/v15').is_dir() else V16.parent / 'local_evidence'
sys.path.insert(0, str(V15))
import common as old
B, CL, DAYS = (old.B, old.CL, old.DAYS)
q, te, h, fr = (old.q.copy(), old.te.copy(), old.h.copy(), old.fr.copy())
iou = old.iou
q['score'] = pd.read_csv(V15 / 'results/oof/final_hands0_pose0.2.csv').set_index('id').loc[q.id, 'score'].to_numpy()
current = pd.read_csv(V16 / 'baseline_submission_0.92822.csv').set_index('id').loc[te.id, 'score'].to_numpy()
R = V16 / 'results'
R.mkdir(exist_ok=True)
for folder in ['oof', 'submissions', 'reviews']:
    (R / folder).mkdir(exist_ok=True)

def metric(rows, s):
    out = {'pooled_ap': float(AP(rows.error, s))}
    for d, g in rows.groupby('day'):
        out['day_' + d[-2:]] = float(AP(g.error, np.asarray(s)[g.index]))
    for c, g in rows.groupby('class_name'):
        out['ap_' + c] = float(AP(g.error, np.asarray(s)[g.index]))
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
