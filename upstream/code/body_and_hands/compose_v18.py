"""Saved-score composition; max related corrections, disjoint class branches."""
from common18 import *

def cached(name,rows,test):
    return align(R/('submissions' if test else 'oof')/(('submission_' if test else '')+name+'.csv'),rows,'score')

register('baseline_v17_lb095180',q.score,current,details=dict(user_reported_public_lb=.95180))
for seat in [.4,.55]:
    for hand in [0.,.25,.5]:
        outputs=[]
        for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
            people=cached('body_backbench_w0.5',rows,test)
            if seat>.4:people=np.maximum(people,cached(f'seat_weight{seat}',rows,test))
            hands=cached('pose_distal_hand_w0.5',rows,test)
            if hand>0:hands=np.minimum(hands,cached(f'hand_geometry_catboost_w{hand}',rows,test))
            # Forearm penalty and palm rescue are mutually exclusive by pose direction.
            forearm=cached('forearm_direction_w0.25',rows,test)-base
            assert not ((forearm>1e-9)&(hands<base-1e-9)).any()
            output=base+(people-base)+(hands-base)+forearm
            outputs.append(output)
        name=f'combined_seat{seat}_geometry{hand}'
        register(name,*outputs,details=dict(person_body='guarded ridge w0.5',seat_weight=seat,hand_geometry_weight=hand,pose_distal_hand_weight=.5,forearm_weight=.25,combination='max related person corrections; max related hand rescues'))
        without=align(V17/'results/oof/release_balanced_without_v16_contradiction.csv',q,'score')
        register(name+'_no_inherited_contradiction',outputs[0]-(q.score.to_numpy()-without),outputs[1],details=dict(ablation='remove inherited OOF-only V16 contradiction; identical test CSV',same_submission=name))
