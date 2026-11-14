"""Ablation: raise error only when two localizers agree on a displaced hand."""
from common17 import *
for weight in [.25,.5]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        split='test' if test else 'oof';a=align(A/f'tta_v15_{split}.csv',rows);b=align(A/f'tta_refined_{split}.csv',rows)
        wf=align(V16/f'wevis_features_{split}.csv',rows)
        agreeing=[];distance=[]
        for j,r in enumerate(rows.itertuples()):
            x=a.iloc[j][['ref_'+c for c in B]].to_numpy(float);y=b.iloc[j][['ref_'+c for c in B]].to_numpy(float)
            agreeing.append(float(iou(x,y)[0,0]));bb=np.array([getattr(r,c) for c in B]);wh=np.maximum(bb[2:]-bb[:2],1)
            distance.append(float(np.linalg.norm(((y[:2]+y[2:])-(bb[:2]+bb[2:]))/2/wh)))
        m=rows.class_name.eq('nude_hands').to_numpy()&(a.overlap.to_numpy()<.4)&(b.overlap.to_numpy()<.4)&(wf.hand_iou.to_numpy()<.4)&(b.confidence.to_numpy()>=.45)&(np.array(agreeing)>=.7)&(np.array(distance)<1.2)&(a.views.to_numpy()>=3)&(b.views.to_numpy()>=3)
        outputs.append(base+weight*(1-base)*m)
    register(f'tta_local_contradiction_w{weight}',*outputs,details=dict(weight=weight,maximum_query_iou=.4,reference_agreement_iou=.7,maximum_normalized_center_distance=1.2,confidence_min=.45))
