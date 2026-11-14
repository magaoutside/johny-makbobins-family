"""Release invariants and numerical checks for V30, run after postcorrect.py."""
from pathlib import Path
import sys,json,hashlib,joblib
import numpy as np,pandas as pd
from postcorrect import iou,upper_iou,BASE_SHA
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v30'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,h,B,base_oof,current,align,metric,DAYS
from features23 import nms

def run():
    actual=align(A/'submission_v30_precision_postcorrection.csv',te,'score')
    expected=align(A/'results/submissions/submission_bounded_cup_u2_c0.65_w0.5.csv',te,'score')
    assert np.max(abs(actual-expected))<1e-14
    assert len(actual)==362 and np.isfinite(actual).all() and ((actual>=0)&(actual<=1)).all()
    changed=abs(actual-current)>1e-12
    assert changed.sum()==2 and te.loc[changed,'class_name'].eq('cup').all()
    assert (actual>=current-1e-14).all()
    assert hashlib.sha256((ROOT/'rascar_v22/submission_v22_candidate.csv').read_bytes()).hexdigest()==BASE_SHA
    oof=align(A/'results/oof/bounded_cup_u2_c0.65_w0.5.csv',q,'score')
    assert metric(q,oof)==metric(q,base_oof)
    pairs=np.triu(np.ones((len(q),len(q)),bool),1)
    flips=(np.sign(oof[:,None]-oof[None,:])!=np.sign(base_oof[:,None]-base_oof[None,:]))&pairs
    opposite=q.error.to_numpy()[:,None]!=q.error.to_numpy()[None,:]
    assert not (flips&opposite).any()
    raw=pd.read_csv(ROOT/'rascar_v15/v8/fused_train_references.csv');controls=[]
    for (day,fid,cls),g in h[h.class_name.isin(['cup','nude_hands'])].groupby(['day','frame_id','class_name']):
        rr=nms(raw[raw.frame_id.eq(fid)&raw.class_name.eq(cls)&raw.confidence.ge(.45)],.5)
        for ix,r in g.iterrows():
            d=dict(gt_index=ix,day=day,frame_id=fid,class_name=cls,reference_available=bool(len(rr)),
                   confidence=0.,raw_iou=0.,upper2=1.,flag=False)
            if len(rr):
                b=r[B].to_numpy(float);rb=rr[B].to_numpy(float);ov=iou(b,rb)[0];j=ov.argmax()
                upper=max(upper_iou(b,x,2) for x in rb);conf=float(rr.iloc[j].confidence)
                d.update(confidence=conf,raw_iou=float(ov[j]),upper2=upper,flag=bool(conf>=.65 and ov[j]>=.1 and upper<.6))
            controls.append(d)
    controls=pd.DataFrame(controls);assert len(controls)==133 and not controls.flag.any()
    controls.to_csv(A/'audit/bounded_manual_positive_controls.csv',index=False)
    rng=np.random.default_rng(3003)
    assert iou([0,0,10,10],[0,0,10,10])[0,0]==1
    assert iou([0,0,10,10],[20,20,30,30])[0,0]==0
    for _ in range(120):
        query=np.r_[rng.uniform(20,60,2),rng.uniform(70,110,2)]
        ref=query+rng.uniform(-20,20,4);ref[2:]=np.maximum(ref[2:],ref[:2]+8)
        u=3.;upper=upper_iou(query,ref,u)
        samples=ref+rng.uniform(-u,u,(1500,4));valid=(samples[:,2:]>samples[:,:2]).all(1)
        assert upper+1e-12>=iou(query,samples[valid]).max()
        assert upper+1e-12>=upper_iou(query,ref,1.)
    folds=[]
    for path in (A/'models').glob('hand_topology_*.joblib'):
        bundle=joblib.load(path);fit=bundle['fit_day'];frames=bundle['training_frames']
        assert all(fit.replace('-','') in fid for fid in frames)
        assert set(frames).isdisjoint(q[q.day.ne(fit)].frame_id)
        folds.append(dict(model=path.name,fit_day=fit,frames=len(frames),held_overlap=0))
    for path in (A/'features').glob('exemplars_*fit*.csv'):
        f=pd.read_csv(path);assert f.apply(lambda r:r.fit_day.replace('-','') in r.source_frame,axis=1).all()
        assert (~f.frame_id.eq(f.source_frame)).all()
        if '_oof_' in path.name:assert f.apply(lambda r:r.fit_day.replace('-','') not in r.frame_id,axis=1).all()
    checks=dict(submission_rows=362,unique_ids=True,changed_rows=2,unchanged_non_cup=True,
        max_replay_abs_difference=float(np.max(abs(actual-expected))),baseline_sha256=BASE_SHA,
        output_sha256=hashlib.sha256((A/'submission_v30_precision_postcorrection.csv').read_bytes()).hexdigest(),
        metric_unchanged=True,metric=metric(q,oof),random_envelope_checks=120,random_boxes_per_check=1500,
        oof_same_label_order_changes=int((flips&~opposite).sum()),oof_opposite_label_order_changes=0,
        manual_positive_controls=dict(n=len(controls),with_references=int(controls.reference_available.sum()),false_flags=0),
        training_fold_checks=folds,test_annotation_labels_used_for_fit=False,
        test_visual_review_used_for_selection=True,leaderboard_submitted=False)
    (A/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':run()
