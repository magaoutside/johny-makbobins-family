from common18 import *

def cached(name,rows,test):
    return align(R/('submissions' if test else 'oof')/(('submission_' if test else '')+name+'.csv'),rows,'score')

for name,source,absence,merged in [
    ('release_main','combined_seat0.55_geometry0.25',.5,.5),
    ('release_conservative','combined_seat0.4_geometry0.0',.25,.5)]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        selected=cached(source,rows,test)
        # Structural penalties are applied only when no rescue/seat branch changed that row.
        penalty=cached(f'person_absence_w{absence}',rows,test)-base+cached(f'merged_hands_w{merged}',rows,test)-base
        assert not ((penalty>1e-9)&(abs(selected-base)>1e-9)).any()
        outputs.append(np.clip(selected+penalty,0,1))
    register(name,*outputs,details=dict(base=source,person_absence_weight=absence,merged_hand_replacement_weight=merged))
    without=align(V17/'results/oof/release_balanced_without_v16_contradiction.csv',q,'score')
    register(name+'_no_inherited_contradiction',outputs[0]-(q.score.to_numpy()-without),outputs[1],details=dict(ablation='subtract inherited OOF-only correction; same test CSV',same_submission=name))
