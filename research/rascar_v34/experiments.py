"""V34: isolated postcorrections of frozen, public-confirmed V31.

No test labels or manual test-audit categories are read. Existing cross-day
CV predictions are reused. Class priors use ONLY the other day's labels,
not the reverse fold's OOF predictions (which can leak the evaluation day).
These are development experiments, not an untouched evaluation.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'rascar_v34'
sys.path.insert(0, str(ROOT/'rascar_v23'))
sys.path.insert(0, str(ROOT/'rascar_v25'))
from common23 import q, te, B, CL, DAYS, base_oof, current, align, iou, metric
from common25 import guard

V31_OOF = ROOT/'rascar_v31/results/oof/linear_mask_cut0.5_all_two_sided_w0.25.csv'
V31_TEST = ROOT/'rascar_v31/submission_v31_dense_oof098784.csv'
BASE = {'oof': align(V31_OOF, q, 'score'), 'test': align(V31_TEST, te, 'score')}
FLOOR = {'oof': base_oof, 'test': current}
ROWS = {'oof': q, 'test': te}
for sub in ['results/oof', 'results/submissions', 'audit', 'inputs', 'logs']:
    (A/sub).mkdir(parents=True, exist_ok=True)

def score_hashes():
    paths = [V31_TEST, ROOT/'rascar_v22/submission_v22_candidate.csv',
             ROOT/'rascar_v32/submission_v32_research_oof098800.csv',
             ROOT/'rascar_v33/submission_v33_visual_control.csv']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def side_gate(rows, split, fit):
    evidence = [align(ROOT/f'rascar_v33/features/side_{kind}_fit{fit[-2:]}.csv', rows)
                for kind in ['pool_regional_balanced', 'pool_regional_balanced_rare']]
    refs = [d[['ref_'+c for c in B]].to_numpy(float) for d in evidence]
    mutual = np.array([iou(a,b)[0,0] for a,b in zip(*refs)])
    present = np.logical_and.reduce([d.valid.to_numpy(bool) & d.presence.ge(.9).to_numpy() for d in evidence])
    confirmed = present & (mutual >= .65)
    confirmed &= np.logical_and.reduce([d.ref_iou.ge(.65).to_numpy() for d in evidence])
    # Keep the semantic/duplicate vetoes from the established pipeline.
    for cls in CL:
        ids = np.flatnonzero(rows.class_name.eq(cls).to_numpy())
        if len(ids): confirmed[ids] &= guard(split, rows.iloc[ids], cls)
    return confirmed, refs

def confirmation(split):
    rows = ROWS[split]
    if split == 'oof':
        out = np.zeros(len(rows), bool)
        for fit in DAYS:
            ids = np.flatnonzero(rows.day.ne(fit).to_numpy())
            out[ids], _ = side_gate(rows.iloc[ids], split, fit)
        return out
    gates, refs = [], []
    for fit in DAYS:
        g, r = side_gate(rows, split, fit); gates.append(g); refs += r
    out = np.logical_and.reduce(gates)
    # Reject reference identity changes between days as well as within a pair.
    for i in range(len(refs)):
        for j in range(i):
            out &= np.array([iou(a,b)[0,0] >= .65 for a,b in zip(refs[i], refs[j])])
    return out

def hand_stability(split):
    rows = ROWS[split]; all_stable=[]; records=[]
    for tail in (['oof'] if split == 'oof' else ['test_fit25','test_fit26']):
        ds = [align(ROOT/f'rascar_v31/features/linear_mask_cut{cut}_{tail}_evidence.csv', rows)
              for cut in [.35,.5,.65]]
        refs = np.stack([d[['ref_'+c for c in B]].to_numpy(float) for d in ds])
        overlaps = np.stack([d.overlap.to_numpy(float) for d in ds])
        valid = np.isfinite(refs).all(axis=(0,2))
        mutual = np.ones(len(rows))
        for i in range(3):
            for j in range(i):
                pair = np.array([iou(a,b)[0,0] for a,b in zip(refs[i],refs[j])])
                mutual = np.minimum(mutual, np.nan_to_num(pair, nan=0.))
        stable = valid & (mutual >= .7)
        stable &= (np.all(overlaps >= .65, axis=0) | np.all(overlaps < .55, axis=0))
        all_stable.append(stable)
        records.append(pd.DataFrame({'id':rows.id, 'split':split, 'fit':tail,
                         'stable':stable, 'min_ref_agreement':mutual,
                         'min_query_iou':np.min(overlaps,axis=0),
                         'max_query_iou':np.max(overlaps,axis=0)}))
    return np.logical_and.reduce(all_stable), pd.concat(records,ignore_index=True)

def priors(train, strength):
    # Each frame has equal weight, preventing busy frames from dominating.
    weights = 1 / train.groupby('frame_id').id.transform('size').to_numpy(float)
    weights *= len(train) / weights.sum()
    y = train.error.to_numpy(float)
    global_p = (np.sum(weights*y)+1)/(weights.sum()+2)
    offsets={}; details=[]
    for cls in CL:
        use=train.class_name.eq(cls).to_numpy()
        p=(np.sum(weights[use]*y[use])+strength*global_p)/(weights[use].sum()+strength)
        offsets[cls]=float(logit(p)-logit(global_p))
        details.append(dict(class_name=cls,n=int(use.sum()),weighted_n=float(weights[use].sum()),
                            error_prior=float(p),global_prior=float(global_p),offset=offsets[cls]))
    return offsets,details

def save(name, scores, family, policy):
    s,t = scores['oof'],scores['test']
    for split,v in scores.items():
        assert v.shape == (len(ROWS[split]),) and np.isfinite(v).all()
        assert (v>=0).all() and (v<=1).all()
    q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False)
    path=A/f'results/submissions/submission_{name}.csv'
    te[['id']].assign(score=t).to_csv(path,index=False)
    audits=[]
    for split,rows in ROWS.items():
        d=rows.copy(); d['split']=split; d['v31_score']=BASE[split]
        d['new_score']=scores[split];d['delta']=d.new_score-d.v31_score
        audits.append(d[abs(d.delta)>1e-10])
    pd.concat(audits,ignore_index=True).to_csv(A/f'audit/{name}_changes.csv',index=False)
    result=dict(experiment=name,family=family,**metric(q,s),
                delta_ap=metric(q,s)['pooled_ap']-metric(q,BASE['oof'])['pooled_ap'],
                changed_oof=int((abs(s-BASE['oof'])>1e-10).sum()),
                changed_test=int((abs(t-BASE['test'])>1e-10).sum()),
                policy=policy,public_ap=None,submitted=False,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    print(name,round(result['pooled_ap'],8),result['changed_oof'],result['changed_test'],flush=True)
    return result

def main():
    frozen=score_hashes(); results=[]
    results.append(save('v31_frozen',BASE,'baseline','Public AP .96239 reported by user.'))
    confirmed={split:confirmation(split) for split in ROWS}
    for split,mask in confirmed.items():
        ROWS[split][['id','class_name']].assign(confirmed=mask).to_csv(A/f'audit/confirmation_{split}.csv',index=False)
    # Person-only changes leave V31 hands, cups and trays byte-equivalent as floats.
    for ceiling in [.2,.4,1.]:
        for fraction in [.5,1.]:
            scores={}
            for split,rows in ROWS.items():
                active=confirmed[split]&rows.class_name.eq('person').to_numpy()
                active &= (FLOOR[split]<=ceiling)&(BASE[split]>FLOOR[split])
                scores[split]=BASE[split] - fraction*np.maximum(BASE[split]-FLOOR[split],0)*active
                assert np.array_equal(scores[split][~rows.class_name.eq('person')],BASE[split][~rows.class_name.eq('person')])
            results.append(save(f'person_confirm_floor{ceiling}_restore{fraction}',scores,'person_confirmation',
                f'Undo {fraction} of V31-only person penalty with V22<={ceiling}, two side-head agreement and presence >=.9. Test also requires cross-day agreement.'))
    stable={}; logs=[]
    for split in ROWS:
        stable[split],df=hand_stability(split);logs.append(df)
    pd.concat(logs,ignore_index=True).to_csv(A/'audit/hand_stability.csv',index=False)
    for direction in ['penalty','both']:
        for fraction in [.5,1.]:
            scores={}
            for split,rows in ROWS.items():
                active=rows.class_name.eq('nude_hands').to_numpy() & ~stable[split]
                if direction=='penalty':active &= BASE[split]>FLOOR[split]
                scores[split]=BASE[split]-fraction*(BASE[split]-FLOOR[split])*active
            results.append(save(f'hand_unstable_{direction}_rollback{fraction}',scores,'hand_stability',
                'Threshold sensitivity check on existing mask predictions, not a calibrated confidence interval. Missing/unstable mask only rolls back a V31 change.'))
    # A genuinely label-separated prior experiment; no claim of fitted probability calibration.
    prior_log=[]
    for strength in [30.,100.]:
        offsets={}
        for fit in DAYS:
            offsets[fit],details=priors(q[q.day.eq(fit)],strength)
            prior_log += [dict(fit_day=fit,strength=strength,**d) for d in details]
        for weight in [.25,.5]:
            scores={}
            for split,rows in ROWS.items():
                delta=np.zeros(len(rows))
                if split=='oof':
                    for fit in DAYS:
                        ix=rows.day.ne(fit);delta[ix]=rows.loc[ix,'class_name'].map(offsets[fit])
                else:
                    daily=np.stack([rows.class_name.map(offsets[fit]).to_numpy() for fit in DAYS])
                    # Same formula family on both folds; average prior rather than refit on test.
                    delta=daily.mean(0)
                scores[split]=expit(logit(np.clip(BASE[split],1e-6,1-1e-6))+weight*delta)
            results.append(save(f'class_prior_strength{strength}_w{weight}',scores,'class_prior',
                'Frame-weighted smoothed class prevalence, fit labels from opposite day only. No reverse-OOF scores fitted; not calibrated probabilities.'))
    pd.DataFrame(prior_log).to_csv(A/'audit/class_prior_parameters.csv',index=False)
    pd.DataFrame(results).to_csv(A/'results/experiments.csv',index=False)
    assert score_hashes()==frozen
    (A/'results/protocol.json').write_text(json.dumps(dict(
        parent='V31',public_ap=.96239,frozen_hashes=frozen,experiments=len(results),
        primary_validation='198 official queries; fit25/predict26 and reverse; no tables',
        selection='Development OOF reused many times; no untouched holdout claimed.',
        calibration_leakage='Class-prior estimates use labels only; do not fit on reverse-fold OOF predictions.',
        test_manual_labels_used=False,model_training=False,leaderboard_upload=False),indent=2))

if __name__=='__main__':main()
