"""Ablations of existing seat evidence, replacing (not stacking) its V17 weight."""
from common18 import *
for weight in [.55,.7,.85]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        folder=V17/'results'/('submissions' if test else 'oof');prefix='submission_' if test else ''
        prior=align(folder/f'{prefix}seat_union_w0.4.csv',rows,'score');base16=old.q.score.to_numpy() if not test else old.current
        initial=prior-base16
        # Hand changes and other V17 branches are preserved exactly.
        outputs.append(np.clip(base+initial*(weight/.4-1),0,1))
    register(f'seat_weight{weight}',*outputs,details=dict(new_weight=weight,old_weight=.4,replaces_previous_weight=True))
