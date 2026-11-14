from common19 import *
for weight in [.25,.5]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        split='test' if test else 'oof';f=align(V16/f'wevis_features_{split}.csv',rows)
        source=align(R/('submissions' if test else 'oof')/(('submission_' if test else '')+f'landmark_joint_rescue_w{weight}.csv'),rows,'score')
        # Independent multi-hand evidence takes precedence over one palm skeleton.
        outputs.append(np.where(f.hand_instances.to_numpy()>=2,base,source))
    register(f'landmark_group_guard_w{weight}',*outputs,details=dict(weight=weight,source='landmark_joint',additional_veto='WeVis has >=2 hand centers inside supplied box'))
