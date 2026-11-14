"""Require an independent localizer to support YOLO rescues, not just semantics."""
from common17 import *
for checkpoint in ['v15','refined','agreement']:
    for weight in [.25,.5]:
        name=f'tta_{checkpoint}_iou0.6_w{weight}';outputs=[]
        for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
            split='test' if test else 'oof';wf=align(V16/f'wevis_features_{split}.csv',rows);cf=align(V16/f'calibrated_hand_{split}.csv',rows)
            src=R/('submissions' if test else 'oof')/(('submission_' if test else '')+name+'.csv')
            candidate=align(src,rows,'score');m=(wf.hand_iou.to_numpy()>=.3)&cf.assigned.to_numpy(bool)
            s=base.copy();s[m]=candidate[m];outputs.append(s)
        register('guarded_'+name,*outputs,details=dict(source=name,independent_localizer_min_iou=.3,independent_one_to_one_match=True))

for weight in [.25,.5]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        folder=R/('submissions' if test else 'oof');prefix='submission_' if test else ''
        outputs.append(np.minimum(*[align(folder/f'{prefix}guarded_tta_{ck}_iou0.6_w{weight}.csv',rows,'score') for ck in ['v15','refined']]))
    register(f'guarded_hand_tta_union_w{weight}',*outputs,details=dict(weight=weight,combination='maximum supported downward correction',independent_localizer_min_iou=.3))
