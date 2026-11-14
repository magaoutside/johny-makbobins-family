from common19 import *
for weight in [.25,.5]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        split='test' if test else 'oof';f=align(A/f'rtm_v2_features_{split}.csv',rows)
        yo=align(V17/f'tta_v15_{split}.csv',rows);yn=align(V17/f'tta_refined_{split}.csv',rows)
        merged=(yo.instances.to_numpy()>=2)&(yn.instances.to_numpy()>=2)
        eligible=rows.class_name.eq('nude_hands').to_numpy()&(f.assigned.to_numpy()==1)&(f.iou.to_numpy()>=.5)&(f.views.to_numpy()>=4)&(f.turns.to_numpy()>=3)&(f.coherence.to_numpy()>=.65)&(f.point_fraction.to_numpy()>=.85)&(f.palm_fraction.to_numpy()>=.8)&(f.instances.to_numpy()<=1)&~merged&~duplicate_veto(rows,test)
        target=np.clip((.8-f.iou.to_numpy())/.6,0,1)
        output=base-weight*np.maximum(base-target,0)*eligible;outputs.append(output)
        if weight==.5:pd.DataFrame(dict(id=rows.id,eligible=eligible,target=target)).to_csv(A/f'rtm_v2_guard_{split}.csv',index=False)
    register(f'rtm_v2_rescue_w{weight}',*outputs,details=dict(weight=weight,reference='frozen MediaPipe, calibrated on fitting day',minimum_views=4,minimum_turns=3,minimum_iou=.5,minimum_point_fraction=.85,duplicate_and_merged_veto=True))
