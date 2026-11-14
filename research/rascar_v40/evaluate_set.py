"""Fixed geometry repair and duplicate ablations with consistent cross-day OOF."""
from common40 import *
import argparse

def majority_instances(ev):
    """Keep a geometrically consistent majority within EACH fitting day.
    A day cannot disappear from the test ensemble; no labels/scores select seeds.
    """
    selected=[]
    for fit,g in ev.groupby('fit_day'):
        refs=g[['ref_'+b for b in B]].to_numpy(float);valid=np.isfinite(refs).all(1);g=g.loc[valid];refs=refs[valid]
        if len(g)<2:return ev.iloc[:0]
        overlap=iou(refs,refs);pairs=[(overlap[i,j],i,j) for i in range(len(g)) for j in range(i+1,len(g)) if overlap[i,j]>=.7]
        if not pairs:return ev.iloc[:0]
        _,i,j=max(pairs);keep=[i,j]
        for k in range(len(g)):
            if k not in keep and (overlap[k,keep]>=.7).all():keep.append(k)
        selected.append(g.iloc[keep])
    return pd.concat(selected)

def person_identity(query,refs,heads,poses,new_penalty):
    hh=heads[heads.frame_id.eq(query.frame_id)][B].to_numpy(float);centers=(hh[:,:2]+hh[:,2:])/2;b=query[B].to_numpy(float)
    inside=((centers>=b[:2])&(centers<=b[2:])).all(1)
    if inside.sum()!=1:return False,'query_head_count'
    head=centers[inside][0];diag=np.linalg.norm(hh[inside][0,2:]-hh[inside][0,:2])
    for ref in refs:
        contains=((centers>=ref[:2])&(centers<=ref[2:])).all(1)
        if contains.sum()!=1 or not ((head>=ref[:2])&(head<=ref[2:])).all() or (head[1]-ref[1])/max(ref[3]-ref[1],1)>.5:return False,'reference_identity'
    if new_penalty:
        selected=[]
        for turn in [0,2]:
            candidates=[]
            for _,p in poses[poses.frame_id.eq(query.frame_id)&poses.turn.eq(turn)].iterrows():
                xy=np.array(json.loads(p.body_xy));co=np.array(json.loads(p.body_conf));dist=np.linalg.norm(np.median(xy[:5],axis=0)-head)
                if dist<=.75*diag:candidates.append((dist,xy,co))
            if not candidates:break
            _,xy,co=min(candidates,key=lambda x:x[0]);selected.append((xy,co))
        if len(selected)==2:
            xy0,c0=selected[0];xy1,c1=selected[1];tol=max(12,.1*np.linalg.norm(b[2:]-b[:2]));margin=max(6,.025*(b[3]-b[1]))
            for chain in [[13,15],[14,16]]:
                pts=(xy0[chain]+xy1[chain])/2
                supported=(c0[chain]>=.5*c0.max()).all() and (c1[chain]>=.5*c1.max()).all() and (np.linalg.norm(xy0[chain]-xy1[chain],axis=1)<=tol).all()
                if supported and ((pts>=b[:2])&(pts<=b[2:])).all() and np.linalg.norm(pts[1]-pts[0])>=3:
                    if any(((pts<ref[:2]-margin)|(pts>ref[2:]+margin)).any() for ref in refs):return False,'visible_leg_cut'
    return True,''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-name',default='refine32_identity');parser.add_argument('--consensus',choices=['strict','majority'],default='strict');args=parser.parse_args();source=args.run_name;run=source+('_majority' if args.consensus=='majority' else '')
    path=A/f'features/{source}_matching_evidence.csv'
    if source=='set' and not path.exists():path=A/'features/matching_evidence.csv'
    evidence=pd.read_csv(path);heads=pd.read_csv(ROOT/'rascar_v16/wevis_heads_references.csv');poses=pd.read_csv(ROOT/'rascar_v25/gpu/results/rtmw/keypoints.csv').drop_duplicates(['frame_id','turn','person'])
    inherited=pd.read_csv(ROOT/'rascar_v18/results/oof/release_main.csv').set_index('id').loc[q.id].score.to_numpy()-pd.read_csv(ROOT/'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv').set_index('id').loc[q.id].score.to_numpy()
    floor={'oof':np.clip(pd.read_csv(ROOT/'rascar_v22/results/oof/final_reviewed.csv').set_index('id').loc[q.id].score.to_numpy()-inherited,0,1),'test':pd.read_csv(ROOT/'rascar_v22/submission_v22_candidate.csv').set_index('id').loc[te.id].score.to_numpy()}
    old_dup={split:pd.read_csv(ROOT/f'rascar_v22/replay_inputs/{split}.csv').set_index('id').loc[rows.id,'old_duplicate_veto'].astype(str).str.lower().isin(['true','1','1.0']).to_numpy() for split,rows in [('oof',q),('test',te)]}
    candidates={};audit=[]
    for split,rows in [('oof',q),('test',te)]:
        test_fits=set(evidence.loc[evidence.split.eq('test'),'fit_day'])
        assert test_fits==set(DAYS) or test_fits=={'all'}
        expected=3 if split=='oof' or test_fits=={'all'} else 6
        for mode in ['geometry','merged_only','duplicate_only','combined']:
            scores=BASE[split].copy()
            for i,r in rows.iterrows():
                original=evidence[evidence.split.eq(split)&evidence.id.eq(r.id)];assert len(original)==expected
                ev=majority_instances(original) if args.consensus=='majority' else original
                ref=ev[['ref_'+b for b in B]].to_numpy(float);valid=len(ev)>0 and np.isfinite(ref).all();reason='';new=scores[i]
                if not valid:reason='missing_reference'
                elif iou(ref,ref).min()<.7:reason='instance_disagreement'
                else:
                    target=np.clip((.85-ev.overlap.to_numpy())/.5,0,1);direction=target-floor[split][i]
                    up=(direction>=0).all();down=(direction<=0).all();dup=ev.duplicate_loser.all();merged=ev.old_reference_multi_instance.all()
                    if mode=='duplicate_only':
                        if original.duplicate_loser.all() and original.max_overlap.ge(.7).all() and original.duplicate_margin.ge(.03).all():new=max(scores[i],.95)
                        else:reason='no_consistent_duplicate'
                    elif original.duplicate_loser.any():
                        if mode=='combined' and original.duplicate_loser.all() and original.max_overlap.ge(.7).all() and original.duplicate_margin.ge(.03).all():new=max(scores[i],.95)
                        else:reason='duplicate_not_geometry_rescue'
                    elif mode=='merged_only' and not merged:reason='old_reference_not_consistently_merged'
                    elif up or down:
                        conservative=target.min() if up else target.max();new=.75*floor[split][i]+.25*conservative
                        if mode=='combined' and dup and ev.max_overlap.ge(.7).all() and ev.duplicate_margin.ge(.03).all():new=max(new,.95)
                    else:reason='score_direction_disagreement'
                    if old_dup[split][i] and new<scores[i]:reason='preserved_duplicate_veto';new=scores[i]
                    if r.class_name=='person' and abs(new-scores[i])>1e-10:
                        allowed,why=person_identity(r,ref,heads,poses,new>scores[i])
                        if not allowed:reason=why;new=scores[i]
                if reason:new=scores[i]
                scores[i]=new;audit.append(dict(mode=mode,split=split,id=r.id,changed=abs(new-BASE[split][i])>1e-10,models_used=len(ev),reason=reason))
            candidates[(mode,split)]=scores
    results=[]
    for mode in ['geometry','merged_only','duplicate_only','combined']:
        for cls in ['all']+CL:
            scores={split:np.where(rows.class_name.eq(cls) if cls!='all' else np.ones(len(rows),bool),candidates[(mode,split)],BASE[split]) for split,rows in [('oof',q),('test',te)]}
            name=f'v40_{run}_{mode}_{cls}_RESEARCH';result=save(name,scores,'One-to-one set prediction, 3 seeds; cross-crop fusion and global matching. Fixed .5 confidence/.7 instance agreement/.25 geometry blend; independent person identity and visible-leg veto. No new test labels.')
            result.update(mode=mode,classes=cls,run_name=run,consensus=args.consensus,test_fit_mode='full_fit_cross_day_proxy' if test_fits=={'all'} else 'daily_ensemble');results.append(result)
    pd.DataFrame(results).to_csv(A/f'results/{run}_experiments.csv',index=False);pd.DataFrame(audit).to_csv(A/f'audit/{run}_decision_gates.csv',index=False)
    print(pd.DataFrame(results)[['experiment','ap','2026-10-25','2026-10-26','changed_test']].to_string(index=False))

if __name__=='__main__':main()
