"""Frozen detection matching and the original semantic/retrieval blend.

No model is fitted to test labels. Training queries retrieve only the other day.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import linear_sum_assignment
BBOX = ['x1', 'y1', 'x2', 'y2']
SCORED_CLASSES = ['person', 'cup', 'nude_hands', 'tray']
def box_iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)

def match_from_iou(iou, threshold=0.6):
    iou = np.asarray(iou, dtype=float)
    n, m = iou.shape
    if n == 0 or m == 0:
        return (np.array([], dtype=int), np.array([], dtype=int))
    reward = np.zeros((n, m + n))
    valid = iou >= threshold
    reward[:, :m] = np.where(valid, min(n, m) + 1 + iou, 0)
    rows, cols = linear_sum_assignment(-reward)
    real = cols < m
    rows, cols = (rows[real], cols[real])
    valid = iou[rows, cols] >= threshold
    return (rows[valid], cols[valid])

def compare_proposals(proposals, references, min_conf=0.25):
    out = proposals.copy().reset_index(drop=True)
    refs = references[references.confidence.ge(min_conf)].copy()
    out['rule_score'] = 0.5
    out['reference_error'] = 1
    out['reason'] = 'unverified_no_reference'
    out['best_same_iou'] = 0.0
    out['assigned_iou'] = np.nan
    out['reference_confidence'] = 0.0
    out['reference_id'] = ''
    out['other_class_overlap'] = 0.0
    out['other_class_name'] = ''
    for (frame, cls), q in out.groupby(['frame_id', 'class_name'], sort=True):
        ref = refs[(refs.frame_id == frame) & (refs.class_name == cls)].reset_index(drop=True)
        other = refs[(refs.frame_id == frame) & (refs.class_name != cls) & (refs.class_name != 'table')].reset_index(drop=True)
        iou = box_iou(q[BBOX], ref[BBOX])
        rr, cc = match_from_iou(iou, 0.6)
        pairs = dict(zip(rr, cc))
        for local_i, index in enumerate(q.index):
            if len(ref):
                j = int(iou[local_i].argmax())
                best = float(iou[local_i, j])
                chosen = pairs.get(local_i, j)
                overlap = float(iou[local_i, chosen])
                trust = float(ref.iloc[chosen].confidence)
                out.loc[index, ['best_same_iou', 'reference_confidence', 'reference_id']] = [best, trust, str(ref.iloc[chosen].reference_id)]
                if local_i in pairs:
                    out.loc[index, ['reference_error', 'reason', 'assigned_iou']] = [0, 'matched_one_to_one', overlap]
                    risk = expit((0.6 - overlap) / 0.08)
                elif best >= 0.6:
                    out.loc[index, 'reason'] = 'duplicate_one_to_one'
                    risk = 1.0
                elif best > 0:
                    out.loc[index, 'reason'] = 'geometry_disagreement'
                    risk = expit((0.6 - best) / 0.08)
                else:
                    risk = 0.5
                out.loc[index, 'rule_score'] = 0.5 + trust * (risk - 0.5)
            if len(other):
                cross = box_iou(q.loc[[index], BBOX], other[BBOX])[0]
                j = int(cross.argmax())
                out.loc[index, ['other_class_overlap', 'other_class_name']] = [float(cross[j]), other.iloc[j].class_name]
    out['class_conflict_flag'] = out.other_class_overlap.ge(0.6) & out.other_class_overlap.gt(out.best_same_iou + 0.2)
    assert out.rule_score.between(0, 1).all() and np.isfinite(out.rule_score).all()
    return out

def score_refs_v4(refs, proposals):
    z = compare_proposals(proposals, refs.assign(confidence=1.0))
    z['score'] = expit((0.6 - z.assigned_iou.fillna(z.best_same_iou)) / 0.08)
    z.loc[z.reason.eq('duplicate_one_to_one'), 'score'] = 1.0
    z = z[z.class_name.isin(SCORED_CLASSES)].copy()
    if 'day' in z:
        z['day'] = z.day.astype(str)
    return z

def retrieval_scores7(queries, qfeatures, records, features, cross_day):
    bank = records[records.kind.isin(['input', 'gt'])].copy()
    bank['label'] = np.where(bank.kind.eq('gt'), 0, bank.error)
    out = []
    for qi, (_, q) in enumerate(queries.iterrows()):
        b = bank[bank.class_name.eq(q.class_name)]
        if cross_day:
            b = b[b.day.ne(q.day)]
        assert q.frame_id not in set(b.frame_id)
        r = {'id': q.id, 'class_name': q.class_name}
        for label, key in [(0, 'good'), (1, 'bad')]:
            n = b[b.label.eq(label)].copy()
            n['similarity'] = features[n.index] @ qfeatures[qi]
            n = n.sort_values('similarity', ascending=False).drop_duplicates('frame_id').head(3)
            assert len(n) > 0
            r[key + '_similarity'] = float(n.similarity.mean())
            r[key + '_frames'] = '|'.join(n.frame_id)
            r[key + '_records'] = '|'.join(n.record_id)
        r['knn_score'] = float(expit((r['bad_similarity'] - r['good_similarity']) / 0.05))
        out.append(r)
    return pd.DataFrame(out)

def blend_scores7(baseline, wevis, knn):
    z = baseline.copy()
    s = z.score.to_numpy().copy()
    k = knn.set_index('id').loc[z.id, 'knn_score'].to_numpy()
    z['sam_score'] = s
    z['knn_score'] = k
    z['score'] = 0.9 * s + 0.1 * k
    person = z.class_name.eq('person')
    w = wevis.set_index('id').loc[z.loc[person, 'id'], 'score'].to_numpy()
    z.loc[person, 'wevis_score'] = w
    z.loc[person, 'score'] = 0.65 * s[person] + 0.25 * w + 0.1 * k[person]
    z.loc[z.reason.eq('duplicate_one_to_one'), 'score'] = 1.0
    z['score'] = z.score.clip(0, 1)
    assert np.isfinite(z.score).all()
    return z

def fuse_person8(sam, wevis, weight, threshold=0.3):
    result = sam.copy()
    audit = []
    for fid, g in sam[sam.class_name.eq('person')].groupby('frame_id'):
        w = wevis[wevis.frame_id.eq(fid) & wevis.class_name.eq('person')]
        if len(w) == 0:
            continue
        iou = box_iou(g[BBOX], w[BBOX])
        rr, cc = match_from_iou(iou, threshold)
        for i, j in zip(rr, cc):
            old = g.iloc[i][BBOX].to_numpy(float)
            new = w.iloc[j][BBOX].to_numpy(float)
            result.loc[g.index[i], BBOX] = (1 - weight) * old + weight * new
            audit.append(dict(frame_id=fid, reference_id=g.iloc[i].reference_id, wevis_id=w.iloc[j].reference_id, iou=float(iou[i, j]), weight=weight))
    return (result, pd.DataFrame(audit))
