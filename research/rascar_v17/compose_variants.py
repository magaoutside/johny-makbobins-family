from common17 import *

def cached(name,test=False):
    return align(R/('submissions' if test else 'oof')/(('submission_' if test else '')+name+'.csv'),te if test else q,'score')

for weight in [.25,.5]:
    hand_names=[f'tta_{checkpoint}_iou0.6_w{weight}' for checkpoint in ['v15','refined']]
    register(f'hand_tta_union_w{weight}',*[np.minimum(*[cached(n,test) for n in hand_names]) for test in [False,True]],
        details=dict(components=hand_names,combination='maximum supported downward correction, not additive'))

for seat in ['seat_union_w0.4','seat_union_w0.65']:
    for hand in ['tta_v15_iou0.6_w0.25','tta_refined_iou0.6_w0.25','hand_tta_union_w0.25','hand_tta_union_w0.5']:
        suffix=seat.removeprefix('seat_union_w')+'_'+hand
        predictions=[cached(seat,test)+cached(hand,test)-base for test,base in [(False,q.score.to_numpy()),(True,current)]]
        register('combined_seat'+suffix,*predictions,details=dict(components=[seat,hand],combination='disjoint class-specific corrections'))
