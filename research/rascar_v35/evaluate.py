"""Small fixed experiment grid; temporal evidence substitutes V31 geometry.

All trials remain development results. A temporal gain must exceed a placebo
using the current-frame reference on exactly the same eligible queries.
"""
from temporal_context import *

def aligned(path,rows):
    return pd.read_csv(path).set_index('id').loc[rows.id]

def aggregate(rows,split,fit,evidence,support):
    old=aligned(V31/f'features/linear_mask_cut0.5_{"oof" if split=="oof" else "test_fit"+fit[-2:]}_evidence.csv',rows)
    sub=evidence[evidence.fit_day.eq(fit)&evidence.split.eq(split)]
    grouped={k:d for k,d in sub.groupby('id')};out=[]
    for j,(_,r) in enumerate(rows.iterrows()):
        d=grouped.get(r.id,pd.DataFrame());cur=old.iloc[j]
        entry=dict(id=r.id,split=split,fit_day=fit,support_mode=support,eligible=False,neighbors=0,agreement=np.nan,
                   old_iou=cur.overlap,context_iou=np.nan,old_target=np.nan,context_target=np.nan,
                   **{f'ref_{b}':np.nan for b in B})
        if d.empty or not np.isfinite(cur.overlap):out.append(entry);continue
        valid=d[d.valid]
        if len(valid)<(2 if support=='two' else 1):out.append(entry);continue
        refs=valid[['ref_'+b for b in B]].to_numpy(float)
        current_ref=cur[['ref_'+b for b in B]].to_numpy(float)
        agreement=float(iou(refs[:1],refs[1:])[0,0]) if len(refs)==2 else float(iou(current_ref,refs)[0,0])
        # A single neighbor must agree with the current-frame reference; two
        # neighbors may correct it if they agree with each other.
        if agreement<.65:out.append(entry);continue
        if support=='current_and_neighbor' and iou(current_ref,refs).min()<.65:out.append(entry);continue
        ref=np.median(refs,axis=0);ov=float(iou(r[B],ref)[0,0])
        entry.update(eligible=True,neighbors=len(refs),agreement=agreement,context_iou=ov,
                     old_target=float(np.clip((.85-cur.overlap)/.5,0,1)),context_target=float(np.clip((.85-ov)/.5,0,1)),
                     **dict(zip(['ref_'+b for b in B],ref)))
        out.append(entry)
    return pd.DataFrame(out)

def gates():
    evidence=pd.read_csv(A/'features/context_evidence.csv');out={};diagnostics=[]
    for support in ['two','current_and_neighbor']:
        out[support]={}
        for split,rows in ROWS.items():
            parts=[]
            for fit in DAYS:
                rr=rows[rows.day.ne(fit)] if split=='oof' else rows
                d=aggregate(rr,split,fit,evidence,support);parts.append(d);diagnostics.append(d)
            if split=='oof':
                merged=pd.concat(parts).set_index('id').loc[rows.id]
                good=merged.eligible.to_numpy();delta=(merged.context_target-merged.old_target).to_numpy()
                placebo=(merged.old_target-BASE[split]).to_numpy()
            else:
                parts=[d.set_index('id').loc[rows.id] for d in parts]
                delta_raw=np.stack([(d.context_target-d.old_target).to_numpy() for d in parts])
                good=np.logical_and.reduce([d.eligible.to_numpy() for d in parts])
                good &= (np.all(delta_raw>=0,0)|np.all(delta_raw<=0,0))
                refs=[d[['ref_'+b for b in B]].to_numpy(float) for d in parts]
                good &= np.array([iou(x,y)[0,0]>=.65 for x,y in zip(*refs)])
                delta=np.where(np.all(delta_raw>=0,0),np.min(delta_raw,0),np.max(delta_raw,0))
                pr=np.stack([d.old_target.to_numpy()-BASE[split] for d in parts])
                same=np.all(pr>=0,0)|np.all(pr<=0,0)
                placebo=np.where(same,np.where(np.all(pr>=0,0),np.min(pr,0),np.max(pr,0)),0)
            out[support][split]=(good,np.nan_to_num(delta,nan=0.),np.nan_to_num(placebo,nan=0.))
    pd.concat(diagnostics,ignore_index=True).to_csv(A/'audit/aggregated_evidence.csv',index=False)
    return out

def save(name,scores,policy):
    records=[]
    for split,rows in ROWS.items():
        s=scores[split]
        assert s.shape==(len(rows),) and np.isfinite(s).all() and (s>=0).all() and (s<=1).all()
        d=rows.copy();d['split']=split;d['v31_score']=BASE[split];d['new_score']=s;d['delta']=s-BASE[split]
        records.append(d[abs(d.delta)>1e-10])
        dest=A/f'results/oof/{name}.csv' if split=='oof' else A/f'results/submissions/submission_{name}.csv'
        (rows.assign(score=s) if split=='oof' else rows[['id']].assign(score=s)).to_csv(dest,index=False)
    pd.concat(records,ignore_index=True).to_csv(A/f'audit/{name}_changes.csv',index=False)
    m=metrics(q,scores['oof']);m.update(experiment=name,delta_ap=m['pooled_ap']-metrics(q,BASE['oof'])['pooled_ap'],
            changed_oof=int((abs(scores['oof']-BASE['oof'])>1e-10).sum()),changed_test=int((abs(scores['test']-BASE['test'])>1e-10).sum()),policy=policy,submitted=False,public_ap=None)
    print(name,round(m['pooled_ap'],8),m['changed_oof'],m['changed_test'],flush=True)
    return m

def main():
    before={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in PATHS.items()}
    policies=gates();results=[save('v31_frozen',BASE,'Frozen public-confirmed V31')]
    for support in policies:
        for family,classes in [('person',['person']),('tableware',['cup','tray']),('nonhands',['person','cup','tray'])]:
            for direction in ['rescue','both']:
                for weight in [.125,.25]:
                    scores={}
                    for split,rows in ROWS.items():
                        good,delta,placebo=policies[support][split]
                        active=good & rows.class_name.isin(classes).to_numpy()
                        if direction=='rescue':delta=np.minimum(delta,0)
                        scores[split]=np.clip(BASE[split]+weight*delta*active,0,1)
                        assert np.array_equal(scores[split][rows.class_name.eq('nude_hands')],BASE[split][rows.class_name.eq('nude_hands')])
                    name=f'temporal_{support}_{family}_{direction}_w{weight}'
                    results.append(save(name,scores,'Replace half/full dense geometry term by +/-30sec context consensus; failed tracking abstains. Both fit days must agree on test.'))
        # Placebo: same eligibility, reinforce old current-frame evidence only.
        for weight in [.125,.25]:
            scores={}
            for split,rows in ROWS.items():
                good,_,placebo=policies[support][split]
                active=good & rows.class_name.ne('nude_hands').to_numpy()
                scores[split]=np.clip(BASE[split]+weight*placebo*active,0,1)
            results.append(save(f'placebo_{support}_w{weight}',scores,'Same eligible rows; more current-frame V31 evidence, no temporal geometry substitution.'))
    pd.DataFrame(results).sort_values('pooled_ap',ascending=False).to_csv(A/'results/experiments.csv',index=False)
    assert before=={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in PATHS.items()}
    (A/'inputs/frozen_hashes.json').write_text(json.dumps(before,indent=2))

if __name__=='__main__':main()
