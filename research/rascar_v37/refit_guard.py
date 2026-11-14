"""Apply established head association / supported-leg veto to the refit.
No frame IDs or test-review categories enter the rule.
"""
from common37 import *

def main():
    heads=pd.read_csv(ROOT/'rascar_v16/wevis_heads_references.csv')
    poses=pd.read_csv(ROOT/'rascar_v25/gpu/results/rtmw/keypoints.csv').drop_duplicates(['frame_id','turn','person'])
    predictions=pd.read_csv(A/'features/refit_predictions.csv')
    def head_and_legs(r):
        hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy(float);centers=(hh[:,:2]+hh[:,2:])/2;b=r[B].to_numpy(float)
        inside=((centers>=b[:2])&(centers<=b[2:])).all(1)
        if inside.sum()!=1:return None,centers,[]
        head=centers[inside][0];diag=np.linalg.norm(hh[inside][0,2:]-hh[inside][0,:2]);selected=[]
        for turn in [0,2]:
            cand=[]
            for _,p in poses[poses.frame_id.eq(r.frame_id)&poses.turn.eq(turn)].iterrows():
                xy=np.array(json.loads(p.body_xy));co=np.array(json.loads(p.body_conf));dist=np.linalg.norm(np.median(xy[:5],axis=0)-head)
                if dist<=.75*diag:cand.append((dist,xy,co))
            if not cand:return head,centers,[]
            _,xy,co=min(cand,key=lambda x:x[0]);selected.append((xy,co))
        xy0,c0=selected[0];xy1,c1=selected[1];keep=[];tol=max(12,.1*np.linalg.norm(b[2:]-b[:2]))
        for chain in [[13,15],[14,16]]:
            pts=(xy0[chain]+xy1[chain])/2
            if not ((c0[chain]>=.5*c0.max()).all() and (c1[chain]>=.5*c1.max()).all()):continue
            if not (np.linalg.norm(xy0[chain]-xy1[chain],axis=1)<=tol).all():continue
            if not ((pts>=b[:2])&(pts<=b[2:])).all() or np.linalg.norm(pts[1]-pts[0])<3:continue
            keep.extend(pts.tolist())
        return head,centers,keep
    geometry={r.id:head_and_legs(r) for rows in ROWS.values() for _,r in rows[rows.class_name.eq('person')].iterrows()}
    results=[];audit=[]
    for mode in ['daily_consensus','full_fit']:
        name=f'refit_{mode}_median_person';scores={}
        for split,rows in ROWS.items():
            path=A/(f'results/oof/{name}.csv' if split=='oof' else f'results/submissions/submission_{name}.csv')
            s=pd.read_csv(path).set_index('id').loc[rows.id].score.to_numpy()
            for i,r in rows.iterrows():
                if r.class_name!='person' or abs(s[i]-BASE[split][i])<1e-10:continue
                head,centers,leglist=geometry[r.id];d=predictions[predictions.id.eq(r.id)&predictions.split.eq(split)]
                if split=='test':d=d[d.fit_day.eq('all') if mode=='full_fit' else d.fit_day.ne('all')]
                reasons=[]
                if head is None:reasons.append('query_without_single_head')
                else:
                    for ref in d[['ref_'+b for b in B]].to_numpy():
                        if not np.isfinite(ref).all():reasons.append('missing_reference');continue
                        contains=((centers>=ref[:2])&(centers<=ref[2:])).all(1)
                        if contains.sum()!=1 or not ((head>=ref[:2])&(head<=ref[2:])).all() or (head[1]-ref[1])/max(ref[3]-ref[1],1)>.5:reasons.append('reference_identity_or_head_position')
                        if s[i]>BASE[split][i] and leglist:
                            pts=np.array(leglist);margin=max(6,.025*(r.y2-r.y1))
                            if ((pts<ref[:2]-margin)|(pts>ref[2:]+margin)).any():reasons.append('supported_leg_outside_reference')
                audit.append(dict(mode=mode,split=split,id=r.id,before=s[i],v31=BASE[split][i],veto=bool(reasons),reasons=';'.join(sorted(set(reasons)))))
                if reasons:s[i]=BASE[split][i]
            scores[split]=s
        results.append(save(f'{name}_guarded',scores,'Three-seed refit plus one-head identity, upper-half head reference gate, and RTMW supported-leg veto, same as earlier V36 protections.'))
    pd.DataFrame(results).to_csv(A/'results/refit_guard_experiments.csv',index=False);pd.DataFrame(audit).to_csv(A/'audit/refit_quality_vetoes.csv',index=False)

if __name__=='__main__':main()
