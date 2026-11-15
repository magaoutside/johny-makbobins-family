"""Optional structural ablation for a query covering multiple distinct hands."""
from common17 import *
for weight in [.25,.5]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        split='test' if test else 'oof';old=align(A/f'tta_v15_{split}.csv',rows);new=align(A/f'tta_refined_{split}.csv',rows);wv=align(V16/f'wevis_features_{split}.csv',rows)
        signal=rows.class_name.eq('nude_hands').to_numpy()&(old.instances.to_numpy()>=2)&(new.instances.to_numpy()>=2)&(new.overlap.to_numpy()<.6)&(wv.hand_iou.to_numpy()<.6)
        outputs.append(base+weight*(1-base)*signal)
    register(f'merged_hand_multiview_w{weight}',*outputs,details=dict(weight=weight,distinct_centers_min=2,refined_and_wevis_max_iou=.6,old_yolo_counts_also_required=True))
