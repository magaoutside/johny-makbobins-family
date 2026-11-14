"""Independent absence evidence and replacement of existing merged-hand weight."""
from common18 import *

for mode in ['person_absence','merged_hands']:
    for weight in [.25,.5] if mode=='person_absence' else [.5,.65]:
        outputs=[]
        for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
            split='test' if test else 'oof';wv=align(V16/f'wevis_features_{split}.csv',rows)
            if mode=='person_absence':
                tight=align(V16/f'semantic_tight_{split}.csv',rows,'p_person');context=align(V16/f'semantic_context_{split}.csv',rows,'p_person');pose=align(V16/f'pose_features_{split}.csv',rows)
                signal=rows.class_name.eq('person').to_numpy()&(wv.head_count.to_numpy()==0)&(pose.pose_best_iou.to_numpy()<.2)&(pose.pose_confirmed_views.to_numpy()==0)&(tight<.5)&(context<.5)
                output=base+weight*(1-base)*signal
            else:
                a=align(V17/f'tta_v15_{split}.csv',rows);b=align(V17/f'tta_refined_{split}.csv',rows)
                signal=rows.class_name.eq('nude_hands').to_numpy()&(a.instances.to_numpy()>=2)&(b.instances.to_numpy()>=2)&(b.overlap.to_numpy()<.6)&(wv.hand_iou.to_numpy()<.6)
                # V17 already applied weight .25. Invert and replace it exactly.
                before=np.where(signal,(base-.25)/.75,base)
                output=np.where(signal,before+weight*(1-before),base)
            outputs.append(output)
            if weight==.5:pd.DataFrame(dict(id=rows.id,eligible=signal)).to_csv(A/f'{mode}_guard_{split}.csv',index=False)
        register(f'{mode}_w{weight}',*outputs,details=dict(mode=mode,weight=weight,prior_weight=.25 if mode=='merged_hands' else 0.,requires='independent semantic and localizer agreement'))
