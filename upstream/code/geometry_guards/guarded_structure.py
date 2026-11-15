"""General guards derived from visual failure modes, never from test labels/IDs."""
from common16 import *

def cached(name, test=False):
    path=R/'submissions'/f'submission_{name}.csv' if test else R/'oof'/f'{name}.csv'
    return pd.read_csv(path).set_index('id').loc[te.id if test else q.id,'score'].to_numpy()

# A second head can belong to a passer-by. An independently matched full body
# above the official IoU threshold vetoes this penalty.
signals=[]
for rows,test in [(q,False),(te,True)]:
    split='test' if test else 'oof'
    w=pd.read_csv(V16/f'wevis_features_{split}.csv').set_index('id').loc[rows.id]
    p=pd.read_csv(V16/f'pose_features_{split}.csv').set_index('id').loc[rows.id]
    signals.append(rows.class_name.eq('person').to_numpy()&(w.head_count.to_numpy()>=2)&(p.pose_best_iou.to_numpy()<.6))
for weight in [.25,.5]:
    register(f'guarded_two_heads_{weight}',q.score+weight*(1-q.score)*signals[0],current+weight*(1-current)*signals[1],dict(weight=weight,body_iou_veto=.6))

# A small localization gain must not overturn a strongly preferred group member.
outputs=[]
for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
    split='test' if test else 'oof'
    f=pd.read_csv(V16/f'calibrated_hand_{split}.csv').set_index('id').loc[rows.id].reset_index()
    s=np.minimum(cached('hand_calibrated_iou0.6_w0.2',test),cached('hand_dino_local_supported',test))
    # One rescue per connected group of overlapping queries for the same hand.
    # Uncertain group losers retain their previous score instead of being rescued.
    group_losers=set();group_owner={}
    for fid,g in rows[rows.class_name.eq('nude_hands')].groupby('frame_id'):
        ids=list(g.index);adj={i:set() for i in ids}
        for ii,i in enumerate(ids):
            for j in ids[ii+1:]:
                rb=f.loc[i,['ref_'+c for c in B]].to_numpy(float);ob=f.loc[j,['ref_'+c for c in B]].to_numpy(float)
                if iou(rows.loc[i,B],rows.loc[j,B])[0,0]>=.35 and np.isfinite(rb).all() and np.isfinite(ob).all() and iou(rb,ob)[0,0]>.8:
                    adj[i].add(j);adj[j].add(i)
        seen=set()
        for i in ids:
            if i in seen:continue
            component=set();stack=[i]
            while stack:
                j=stack.pop()
                if j in component:continue
                component.add(j);stack.extend(adj[j]-component)
            seen.update(component)
            if len(component)<2:continue
            winner=max(component,key=lambda j:(bool(f.loc[j,'assigned']),f.loc[j,'calibrated_hand_iou']+.1*(1-base[j]),str(rows.loc[j,'id'])))
            for j in component-{winner}:group_losers.add(j);group_owner[j]=rows.loc[winner,'id']
    decisions=[]
    for idx,r in rows.iterrows():
        if r.class_name!='nude_hands':continue
        g=rows[rows.frame_id.eq(r.frame_id)&rows.class_name.eq('nude_hands')&rows.id.ne(r.id)]
        veto=idx in group_losers;competitor=group_owner.get(idx,'')
        for j,other in g.iterrows():
            if iou(r[B],other[B])[0,0]<.35 or base[j]>base[idx]-.2:continue
            # Both boxes refer to the same detected hand, with nearly tied geometry.
            rb=f.loc[idx,['ref_'+c for c in B]].to_numpy(float)
            ob=f.loc[j,['ref_'+c for c in B]].to_numpy(float)
            if np.isfinite(rb).all() and np.isfinite(ob).all() and iou(rb,ob)[0,0]>.8 and f.loc[j,'calibrated_hand_iou']>=.45 and f.loc[idx,'calibrated_hand_iou']-f.loc[j,'calibrated_hand_iou']<.1:
                veto=True;competitor=other.id;break
        proposed=s[idx]
        if veto:s[idx]=base[idx]
        decisions.append(dict(id=r.id,competing_query=competitor,veto=veto,baseline=base[idx],proposed=proposed,final=s[idx]))
    pd.DataFrame(decisions).to_csv(V16/f'hand_rescue_group_guard_{split}.csv',index=False)
    outputs.append(s)
register('hand_rescue_group_guarded',*outputs,details=dict(near_tie_iou_margin=.1,baseline_preference_margin=.2,query_overlap_min=.35,one_rescue_per_reference_group=True,combination='max individual downward correction'))

# Geometry mismatch alone also penalizes valid, different sitting positions.
# Restrict the template evidence to a substantial vertical deficit shared by
# several nearby manually annotated examples.
bank=pd.read_csv(V16/'head_geometry_templates.csv')
heads=pd.read_csv(V16/'wevis_heads_references.csv')
def short_evidence(rows,fit=None):
    pool=bank if fit is None else bank[bank.day.eq(fit)]
    boxes=pool[B].to_numpy();centers=pool[['head_x','head_y']].to_numpy();sizes=pool.head_w.to_numpy()
    result=[]
    for _,r in rows.iterrows():
        rec=dict(id=r.id,height_deficit=0.,eligible=False,expected_min_height=0.,height_ratio=1.,top_alignment=1.)
        b=r[B].to_numpy(float);wh=b[2:]-b[:2]
        if r.class_name!='person':result.append(rec);continue
        hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy();cc=(hh[:,:2]+hh[:,2:])/2
        inside=((cc>=b[:2])&(cc<=b[2:])).all(1)
        if inside.sum()!=1:result.append(rec);continue
        hb=hh[inside][0];center=(hb[:2]+hb[2:])/2
        dist=np.linalg.norm((centers-center)/[1280,720],axis=1)
        m=(dist<.085)&(np.abs(np.log(np.maximum(sizes,1)/max(hb[2]-hb[0],1)))<.55)
        ix=np.where(m)[0];ix=ix[np.argsort(dist[ix])[:5]]
        if len(ix)<2:result.append(rec);continue
        translated=boxes[ix]+np.tile(center-centers[ix],(1,2))
        heights=translated[:,3]-translated[:,1];expected=np.quantile(heights,.2)
        top=float(np.median(np.abs(translated[:,1]-b[1])/heights))
        ratio=float(wh[1]/expected)
        eligible=(ratio<.75) and (top<.2) and (wh[0]/wh[1]<1.6)
        rec.update(height_deficit=float(np.clip((.85-ratio)/.35,0,1)),eligible=bool(eligible),expected_min_height=float(expected),height_ratio=ratio,top_alignment=top)
        result.append(rec)
    return pd.DataFrame(result)
f=pd.concat([short_evidence(q[q.day.eq(held)],fit) for fit,held in zip(DAYS,DAYS[::-1])]).set_index('id').loc[q.id].reset_index()
ft=short_evidence(te)
f.to_csv(V16/'short_height_oof.csv',index=False);ft.to_csv(V16/'short_height_test.csv',index=False)
outputs=[]
for rows,base,feat,test in [(q,q.score.to_numpy(),f,False),(te,current,ft,True)]:
    geometry=pd.read_csv(V16/f'head_geometry_{"test" if test else "oof"}.csv').set_index('id').loc[rows.id]
    signal=feat.eligible.to_numpy()*geometry.head_anchor_risk.to_numpy()*feat.height_deficit.to_numpy()
    outputs.append(base+.25*(1-base)*signal)
register('guarded_short_height_0.25',*outputs,details=dict(weight=.25,height_ratio_max=.75,top_alignment_max=.2,aspect_max=1.6,minimum_templates=2))

configs={
    'targeted':['wevis_head_near_outside_0.5','guarded_two_heads_0.5','semantic_both_person_0.5','hand_rescue_group_guarded'],
    'targeted_short':['wevis_head_near_outside_0.5','guarded_two_heads_0.5','semantic_both_person_0.5','hand_rescue_group_guarded','guarded_short_height_0.25'],
}
for name,parts in configs.items():
    predictions=[]
    for base,test in [(q.score.to_numpy(),False),(current,True)]:
        s=base.copy();down=np.zeros_like(base)
        for component in parts:
            delta=cached(component,test)-base;m=delta>0
            s[m]+=delta[m]*(1-s[m])/np.maximum(1-base[m],1e-12)
            down=np.minimum(down,delta)
        predictions.append(np.clip(s+down,0,1))
    register('final_'+name,*predictions,details=dict(components=parts,negative_delta_combination='max individual downward correction'))
    delta=cached('hand_contradiction_0.6')-q.score.to_numpy();m=delta>0;s=predictions[0].copy()
    s[m]+=delta[m]*(1-s[m])/np.maximum(1-q.score.to_numpy()[m],1e-12)
    assert np.allclose(cached('hand_contradiction_0.6',True),current,atol=1e-12)
    register('final_'+name+'_with_contradiction',s,predictions[1],details=dict(components=parts+['hand_contradiction_0.6'],contradiction_changes_test_rows=0,identical_test_submission_to='final_'+name))
