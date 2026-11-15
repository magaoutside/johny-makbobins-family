"""Agreement branch plus stricter independent geometry for new-only detections."""
from common17 import *

def cached(name,rows,test):
    return align(R/('submissions' if test else 'oof')/(('submission_' if test else '')+name+'.csv'),rows,'score')

for weight in [.25,.5]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        split='test' if test else 'oof';ref=align(A/f'tta_refined_{split}.csv',rows);ind=align(V16/f'calibrated_hand_{split}.csv',rows)
        supported=(ref.overlap.to_numpy()>=.7)&(ind.calibrated_hand_iou.to_numpy()>=.6)&ind.assigned.to_numpy(bool)
        agreement=cached(f'guarded_tta_agreement_iou0.6_w{weight}',rows,test)
        new=cached('guarded_tta_refined_iou0.6_w0.25',rows,test)
        strict=base.copy();strict[supported]=new[supported];outputs.append(np.minimum(agreement,strict))
    register(f'hand_tta_hybrid_w{weight}',*outputs,details=dict(agreement_weight=weight,new_only_weight=.25,new_only_yolo_iou_min=.7,new_only_calibrated_independent_iou_min=.6,combination='maximum downward correction'))
    for seatweight in [.4,.65]:
        predictions=[cached(f'seat_union_w{seatweight}',rows,test)+s-base for rows,base,s,test in [(q,q.score.to_numpy(),outputs[0],False),(te,current,outputs[1],True)]]
        register(f'final_seat{seatweight}_hands{weight}',*predictions,details=dict(seat=f'seat_union_w{seatweight}',hands=f'hand_tta_hybrid_w{weight}',combination='disjoint person and hand corrections'))
