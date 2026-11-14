from common16 import *

def cached(name,test=False):
    p=R/'submissions'/f'submission_{name}.csv' if test else R/'oof'/f'{name}.csv'
    return pd.read_csv(p).set_index('id').loc[te.id if test else q.id,'score'].to_numpy()

# Require new localized evidence before a DINO-only rescue of highly penalized hands.
preds=[];counts=[]
for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
    candidate=cached('hand_rescue_guarded_s0.9_w0.2',test)
    f=pd.read_csv(V16/f'wevis_features_{"test" if test else "oof"}.csv').set_index('id').loc[rows.id]
    support=(f.hand_iou.to_numpy()>=.3)&(f.hand_instances.to_numpy()==1)
    s=base.copy();s[support]=candidate[support];preds.append(s);counts.append(int(np.sum(s<base-1e-9)))
register('hand_dino_local_supported',*preds,details=dict(dino_variant='hand_rescue_guarded_s0.9_w0.2',local_iou_min=.3,local_instances=1,changed_counts=counts))

configs={
    'heads_and_hands':['wevis_head_near_outside_0.5','wevis_two_heads_0.25','semantic_both_person_0.5','hand_calibrated_iou0.6_w0.2','hand_dino_local_supported'],
    'head_geometry_and_hands':['wevis_head_near_outside_0.5','wevis_two_heads_0.25','semantic_both_person_0.5','head_geometry_0.25','hand_calibrated_iou0.6_w0.2','hand_dino_local_supported'],
    'strong_head_geometry':['wevis_head_near_outside_0.5','wevis_two_heads_0.25','semantic_both_person_0.5','head_geometry_0.5','hand_calibrated_iou0.6_w0.2','hand_dino_local_supported'],
    'heads_only':['wevis_head_near_outside_0.5','wevis_two_heads_0.25','semantic_both_person_0.5'],
}
for name,parts in configs.items():
    predictions=[]
    for base,test in [(q.score.to_numpy(),False),(current,True)]:
        s=base.copy()
        for component in parts:
            delta=cached(component,test)-base
            positive=delta>0
            s[positive]+=delta[positive]*(1-s[positive])/np.maximum(1-base[positive],1e-12)
            s[~positive]+=delta[~positive]
        predictions.append(np.clip(s,0,1))
    register('combined_'+name,*predictions,details=dict(components=parts))
    # This rule fixes an OOF elbow false positive; on this test it changes zero rows.
    s=predictions[0].copy();delta=cached('hand_contradiction_0.6')-q.score.to_numpy();m=delta>0
    s[m]+=delta[m]*(1-s[m])/np.maximum(1-q.score.to_numpy()[m],1e-12)
    assert np.max(np.abs(cached('hand_contradiction_0.6',True)-current))<1e-12
    register('combined_'+name+'_with_contradiction',s,predictions[1],details=dict(components=parts+['hand_contradiction_0.6'],contradiction_changes_test_rows=0,identical_test_submission_to='combined_'+name))
