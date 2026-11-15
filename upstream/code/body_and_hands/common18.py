from pathlib import Path
import sys, json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP
A = Path(__file__).resolve().parent
V17 = A.parent / 'seat_and_tta'
V16 = A.parent / 'geometry_guards'
sys.path.insert(0, str(V17))
import common17 as old
V15 = old.V15
B, CL, DAYS = (old.B, old.CL, old.DAYS)
iou = old.iou
align = old.align
q, te, h, fr = (old.q.copy(), old.te.copy(), old.h.copy(), old.fr.copy())
q['score'] = align(V17 / 'results/oof/release_balanced.csv', q, 'score')
current = align(A / 'baseline_lb095180.csv', te, 'score')
R = A / 'results'
R.mkdir(exist_ok=True)
for f in ['oof', 'submissions', 'reviews']:
    (R / f).mkdir(exist_ok=True)

def boolean(values):
    """CSV columns can mix True/False with 0; never use truthiness of strings."""
    return pd.Series(values).astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy()

def metric(rows, s):
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

def duplicate_veto(rows, test):
    split = 'test' if test else 'oof'
    g = pd.read_csv(V16 / f'hand_rescue_group_guard_{split}.csv').set_index('id').reindex(rows.id)
    names = [V15 / 'hand_group_test.csv'] if test else [V15 / f'hand_group_oof_{d}.csv' for d in DAYS]
    losers = set(pd.concat([pd.read_csv(p) for p in names]).query("reason=='duplicate_loser'").id)
    return rows.id.isin(losers).to_numpy() | g.veto.fillna(False).to_numpy(bool)
