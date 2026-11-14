"""Use distinct object instances before applying a crop-level quality expert."""
from common21 import *
from guard_quality import duplicate_veto

def object_structure(rows,refs):
    result=[]
    for (fid,cls),g in rows.groupby(['frame_id','class_name']):
        ref=refs[refs.frame_id.eq(fid)&refs.class_name.eq(cls)&refs.confidence.ge(.45)].sort_values('confidence',ascending=False)
        keep=[]
        for ix,r in ref.iterrows():
            if not keep or iou(r[B],ref.loc[keep,B]).max()<.5:keep.append(ix)
        ref=ref.loc[keep];bb=ref[B].to_numpy();cent=(bb[:,:2]+bb[:,2:])/2
        ov=iou(g[B],bb);between=iou(g[B],g[B])
        for j,r in enumerate(g.itertuples()):
            b=np.array([getattr(r,c) for c in B]);inside=((cent>=b[:2])&(cent<=b[2:])).all(1)
            cover=np.maximum(0,np.minimum(b[2:],bb[:,2:])-np.maximum(b[:2],bb[:,:2])).prod(1)/np.maximum((bb[:,2:]-bb[:,:2]).prod(1),1e-9)
            count=int((inside&(cover>=.5)).sum());rivals=np.flatnonzero((between[j]>=.5)&(np.arange(len(g))!=j));veto=False
            if len(rivals):
                if not len(ref):veto=True
                else:
                    best=int(ov[j].argmax());competitor=ov[rivals,best].max()
                    # A unique geometric winner can be rescued. Abstain when
                    # matching is ambiguous; never rescue both duplicate boxes.
                    veto=bool(ov[j,best]<.6 or ov[j,best]-competitor<.08)
            result.append(dict(id=r.id,distinct_instances=count,rivals=len(rivals),ambiguous_duplicate=veto,best_reference_iou=float(ov[j].max()) if len(ref) else 0.))
    return pd.DataFrame(result).set_index('id').loc[rows.id].reset_index()

if __name__=='__main__':
    fs={}
    for split,rows in [('oof',q),('test',te)]:
        ref=pd.read_csv(ROOT/f'rascar_v15/v8/fused_{"train" if split=="oof" else "test"}_references.csv')
        f=object_structure(rows,ref);f.to_csv(A/f'object_structure_{split}.csv',index=False);fs[split]=f
    for weight in [.3,.5]:
        out=[]
        for split,rows,base,test in [('oof',q,q.score.to_numpy(),False),('test',te,current,True)]:
            f=fs[split];p=align(A/f'rbf2_{split}.csv',rows,'probability');sem=align(ROOT/f'rascar_v16/semantic_both_{split}.csv',rows,'p_cup')
            m=rows.class_name.eq('cup').to_numpy()&(sem>=.8)&(p<=.65)&f.distinct_instances.le(1).to_numpy()&~boolean(f.ambiguous_duplicate)&~duplicate_veto(rows,test)
            s=base.copy();s[m]-=weight*np.maximum(s[m]-p[m],0);out.append(s)
            pd.DataFrame(dict(id=rows.id.to_numpy(),gate=m,target=p,score=s)).to_csv(A/f'cup_instance_quality_w{weight}_{split}.csv',index=False)
        register(f'cup_instance_quality_w{weight}',*out,dict(weight=weight,model='rbf2',semantic_min=.8,quality_max=.65,instances='frozen multi-view references confidence>=.45, NMS .5, center and coverage within query',duplicates='unique reference winner or abstain'))
    for weight in [.5,.75]:
        for branch in ['merged','conflict','both']:
            out=[]
            for split,rows,base,test in [('oof',q,q.score.to_numpy(),False),('test',te,current,True)]:
                k=align(A/f'rbf5_{split}.csv',rows,'probability');z=align(A/f'ridge2_{split}.csv',rows,'probability')
                geo=align(ROOT/f'rascar_v18/hand_geometry_features_{split}.csv',rows)
                hand=align(ROOT/f'rascar_v20/hand_expanded_sahi_{split}.csv',rows)
                eligible=rows.class_name.eq('nude_hands').to_numpy()&(k>=.75)&(z>=.75)
                merged=eligible&geo.center_instances.ge(2).to_numpy()&geo.reference_iou.between(.25,.6).to_numpy()&geo.iou_q75.lt(.6).to_numpy()
                conflict=eligible&geo.reference_iou.lt(.2).to_numpy()&geo.center_instances.eq(0).to_numpy()&hand.confidence.ge(.7).to_numpy()&hand.overlap.between(.4,.6).to_numpy()
                if test:conflict &= hand.folds.ge(2).to_numpy()
                m=merged if branch=='merged' else conflict if branch=='conflict' else merged|conflict
                target=(k+z)/2;s=base.copy();s[m]+=weight*np.maximum(target[m]-s[m],0);out.append(s)
                pd.DataFrame(dict(id=rows.id.to_numpy(),gate=m,merged=merged,conflict=conflict,target=target,score=s)).to_csv(A/f'hand_quality_{branch}_w{weight}_{split}.csv',index=False)
            register(f'hand_quality_{branch}_w{weight}',*out,dict(weight=weight,branch=branch,agreement='SVC and kernel ridge >=.75; independent geometric contradiction or multiple hand centers',action='increase error only'))
