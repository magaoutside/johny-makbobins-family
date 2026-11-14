from common17 import *
for name,source in [('release_balanced','final_seat0.4_hands0.5'),('release_stronger_seats','final_seat0.65_hands0.5')]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        folder=R/('submissions' if test else 'oof');prefix='submission_' if test else ''
        selected=align(folder/f'{prefix}{source}.csv',rows,'score')
        merged=align(folder/f'{prefix}merged_hand_multiview_w0.25.csv',rows,'score')
        # Positive structural evidence cannot rescue another duplicate member.
        fraction=np.maximum(merged-base,0)/np.maximum(1-base,1e-12)
        outputs.append(selected+fraction*(1-selected))
    register(name,*outputs,details=dict(base=source,structural_component='merged_hand_multiview_w0.25',combination='bounded positive correction of remaining score range'))
    if name=='release_balanced':
        without=align(V16/'results/oof/final_targeted_short.csv',q,'score')
        delta=q.score.to_numpy()-without
        register('release_balanced_without_v16_contradiction',outputs[0]-delta,outputs[1],
            details=dict(ablation='remove inherited V16 contradiction correction; this correction changes zero test rows',identical_submission='release_balanced'))
