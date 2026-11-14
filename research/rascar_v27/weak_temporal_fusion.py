"""Low-confidence hand evidence is usable only with strong independent support.

A weak WiLoR prediction cannot rescue a query alone. Require an independent
reference, pixel-boundary expert, and a consistent match in another frame.
"""
from common27 import *
from evaluate_temporal import prepare
import joblib

def weak_geometry(rows,source_ids=None):
    raw=prepare();out=[]
    for i,r in enumerate(rows.itertuples()):
        key=r.id if source_ids is None else source_ids[i];d=raw[raw.id.eq(key)]
        cur=d[d.seconds.eq(0)&d.confidence.ge(.2)]
        prev=d[d.seconds.ne(0)&d.confidence.ge(.15)&d.ncc.ge(.9)&d.tracked_fraction.ge(.7)&d.inlier_fraction.ge(.8)&d.fb_error.le(.5)]
        f=dict(id=r.id,current_iou=0.,temporal_iou=0.,mutual_iou=0.,confidence=0.,ncc=0.,source_frame='')
        f.update({'ref_'+c:0. for c in B})
        if len(cur) and len(prev):
            b=np.array([getattr(r,c) for c in B]);a=iou(b,cur[B])[0];bb=iou(b,prev[B])[0];mutual=iou(cur[B],prev[B])
            eligible=(a[:,None]>=.55)&(bb[None]>=.45)&(mutual>=.6)
            if eligible.any():
                rank=np.where(eligible,a[:,None]+bb[None]+mutual,-1);j,k=np.unravel_index(rank.argmax(),rank.shape)
                z=prev.iloc[k];f.update(current_iou=float(a[j]),temporal_iou=float(bb[k]),mutual_iou=float(mutual[j,k]),confidence=float(cur.iloc[j].confidence),ncc=float(z.ncc),source_frame=z.source_frame,
                    **dict(zip(['ref_'+c for c in B],(cur.iloc[j][B].to_numpy(float)+z[B].to_numpy(float))/2)))
        out.append(f)
    return pd.DataFrame(out).set_index('id').loc[rows.id]

def main():
    outputs=[]
    for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
        weak=weak_geometry(rows);weak.to_csv(A/f'features/weak_temporal_{split}.csv')
        old=align(ROOT/f'rascar_v24/features/instances_{split}.csv',rows)
        p=align(ROOT/f'rascar_v24/features/context_trees_{split}.csv',rows)
        gate=rows.class_name.eq('nude_hands').to_numpy()&guard(split,rows,'nude_hands')
        gate&=weak.current_iou.ge(.55).to_numpy()&weak.temporal_iou.ge(.45).to_numpy()&weak.mutual_iou.ge(.6).to_numpy()
        gate&=old.reference_iou.ge(.65).to_numpy()&old.ref_conf.ge(.55).to_numpy()&old.assigned.eq(1).to_numpy()&old.rival_margin.ge(.1).to_numpy()&p.probability.lt(.4).to_numpy()
        if split=='test':gate&=p[['p25','p26']].max(axis=1).lt(.5).to_numpy()
        score=base-.5*gate*np.maximum(base-p.probability.to_numpy(),0)
        pd.DataFrame(dict(id=rows.id,gate=gate,target=p.probability.to_numpy(),score=score)).to_csv(A/f'features/weak_temporal_fusion_{split}.csv',index=False)
        outputs.append(score)
    register('weak_temporal_fusion',*outputs,dict(type='weak_detection_with_strong_reference_boundary_and_temporal_confirmation',weight=.5,
        current_min_conf=.2,neighbour_min_conf=.15,ncc_min=.9,current_iou_min=.55,neighbour_iou_min=.45,mutual_iou_min=.6,
        caveat='No new threshold fit to test labels; thresholds are development choices after inspecting hard cases. Main OOF coverage is reported separately.'))

def auxiliary():
    # Reuse synthetic queries from V24, but predictions are always by the opposite
    # day model. WiLoR crops are GT-centred here: this is an auxiliary geometric
    # stress test, NOT an end-to-end detection benchmark or extra independent data.
    d=pd.read_csv(ROOT/'rascar_v24/inputs/boundary_training.csv')
    d=d[d.class_name.eq('nude_hands')&~d.kind.eq('real_query')].reset_index(drop=True)
    source=[]
    for r in d.itertuples():
        gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq('nude_hands')]
        source.append('manual_'+str(gt.index[iou([getattr(r,c) for c in B],gt[B])[0].argmax()]) if len(gt) else 'none')
    weak=weak_geometry(d,source);x=align(ROOT/'rascar_v24/features/boundary_train.csv',d)
    probabilities=np.ones(len(d))
    for fit,held in zip(DAYS,DAYS[::-1]):
        bundle=joblib.load(ROOT/f'rascar_v24/models/context_trees_nude_hands_{fit}.joblib');m=d.day.eq(held)
        raw=bundle['model'].predict_proba(x.loc[m.to_numpy(),bundle['columns']])[:,1]
        logit=np.log(np.clip(raw,1e-5,1-1e-5)/np.clip(1-raw,1e-5,1))[:,None]
        probabilities[m]=bundle['calibrator'].predict_proba(logit)[:,1]
    # Include the independent geometric reference. Semantic and duplicate gates
    # are unavailable for synthetic queries, so this remains a broader screen.
    refs=pd.read_csv(ROOT/'rascar_v24/features/instance_references_oof.csv')
    ref_iou=[]
    for r in d.itertuples():
        rr=refs[refs.frame_id.eq(r.frame_id)&refs.class_name.eq('nude_hands')&refs.confidence.ge(.55)]
        ref_iou.append(float(iou([getattr(r,c) for c in B],rr[B]).max()) if len(rr) else 0.)
    basic=weak.current_iou.ge(.55).to_numpy()&weak.temporal_iou.ge(.45).to_numpy()&(probabilities<.4)
    gate=basic&(np.asarray(ref_iou)>=.65)
    d['gate_without_independent_reference']=basic;d['gate']=gate;d['independent_iou']=ref_iou;d['boundary_probability']=probabilities;d.to_csv(A/'results/weak_temporal_auxiliary.csv',index=False)
    summary=[]
    for (day,kind),g in d.groupby(['day','kind']):
        summary.append(dict(day=day,kind=kind,rows=len(g),eligible=int(g.gate.sum()),correct_rescues=int((g.gate&g.error.eq(0)).sum()),bad_rescues=int((g.gate&g.error.eq(1)).sum())))
    pd.DataFrame(summary).to_csv(A/'results/weak_temporal_auxiliary_summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':
    main();auxiliary()
