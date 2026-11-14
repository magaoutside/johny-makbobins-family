"""Compare checkpoints on identical TTA; no manual-held crops in query evidence."""
from common17 import *
from scipy.optimize import linear_sum_assignment

G=A/'gpu_results'/'hand_refinement'

def clusters(raw):
    out=[]
    for fid,rr in raw.groupby('frame_id'):
        rr=rr[rr.confidence.ge(.2)].copy()
        if rr.empty:continue
        b=rr[B].to_numpy();edge=np.c_[b[:,0]-rr.roi_x1,b[:,1]-rr.roi_y1,rr.roi_x2-b[:,2],rr.roi_y2-b[:,3]].min(1)
        rr=rr[edge>=2].sort_values('confidence',ascending=False)
        while len(rr):
            ov=iou(rr.iloc[0][B],rr[B])[0];m=ov>=.55;part=rr[m].copy();rr=rr[~m]
            part=part.drop_duplicates(['query_id','scale','turn'])
            bb=part[B].median().to_numpy();coherence=float(np.median(iou(bb,part[B])))
            out.append(dict(frame_id=fid,views=len(part),turns=part.turn.nunique(),scales=part.scale.nunique(),
                confidence=float(part.confidence.median()),max_confidence=float(part.confidence.max()),coherence=coherence,
                **dict(zip(B,bb))))
    result=pd.DataFrame(out)
    # Fused centers can move relative to their original cluster seed. Suppress
    # duplicate fused hypotheses again before one-to-one query assignment.
    keep=[]
    for _,g in result.groupby('frame_id'):
        chosen=[]
        for idx,row in g.sort_values(['views','confidence'],ascending=False).iterrows():
            if not chosen or iou(row[B],g.loc[chosen,B]).max()<.5:chosen.append(idx)
        keep.extend(chosen)
    return result.loc[keep].reset_index(drop=True)

def evidence(rows,refs):
    out=[]
    for fid,g in rows.groupby('frame_id'):
        rr=refs[refs.frame_id.eq(fid)].reset_index(drop=True)
        rr=rr[(rr.views>=3)&(rr.turns>=2)&(rr.coherence>=.65)].reset_index(drop=True)
        if not len(rr):
            out += [dict(id=r.id,overlap=0.,assigned=False,instances=0,views=0,turns=0,confidence=0.,coherence=0.,competitor_iou=0.) for r in g.itertuples()];continue
        ov=iou(g[B],rr[B]);reward=np.c_[np.where(ov>=.4,min(ov.shape)+1+ov,0),np.zeros((len(g),len(g)))];ii,jj=linear_sum_assignment(-reward)
        match={i:j for i,j in zip(ii,jj) if j<len(rr) and ov[i,j]>=.4};centers=(rr[B].to_numpy()[:,:2]+rr[B].to_numpy()[:,2:])/2
        for j,r in enumerate(g.itertuples()):
            k=match.get(j,int(ov[j].argmax()));b=np.array([getattr(r,c) for c in B]);ref=rr.iloc[k]
            competitor=float(np.delete(ov[:,k],j).max()) if len(g)>1 else 0.
            out.append(dict(id=r.id,overlap=float(ov[j,k]),assigned=j in match,instances=int(((centers>=b[:2])&(centers<=b[2:])).all(1).sum()),
                views=ref.views,turns=ref.turns,confidence=ref.confidence,coherence=ref.coherence,competitor_iou=competitor,
                **{'ref_'+c:ref[c] for c in B}))
    return pd.DataFrame(out)

held=[]
for checkpoint in ['v15','refined']:
    oo=[]
    for fit,day in zip(DAYS,DAYS[::-1]):
        raw=pd.read_csv(G/fit/f'tta_{checkpoint}.csv')
        refs=clusters(raw[raw.kind.eq('query_oof')]);refs.to_csv(A/f'tta_{checkpoint}_refs_{day}.csv',index=False)
        rows=q[q.class_name.eq('nude_hands')&q.day.eq(day)];oo.append(evidence(rows,refs))
        # Conditional localization diagnostic: GT-centered crops, not detector AP.
        gt=h[h.class_name.eq('nude_hands')&h.day.eq(day)]
        for ix,r in gt.iterrows():
            p=raw[raw.kind.eq('manual_held')&raw.query_id.eq('gt_'+str(ix))&raw.confidence.ge(.2)]
            overlap=float(iou(r[B],p[B]).max()) if len(p) else 0.
            held.append(dict(checkpoint=checkpoint,day=day,gt_index=ix,frame_id=r.frame_id,best_iou=overlap,found_iou06=overlap>=.6,confidence_min=.2))
    f=pd.concat(oo).set_index('id').reindex(q.id).fillna(0).reset_index();f.to_csv(A/f'tta_{checkpoint}_oof.csv',index=False)
    raw=pd.read_csv(G/'all'/f'tta_{checkpoint}.csv');refs=clusters(raw[raw.kind.eq('query_test')]);refs.to_csv(A/f'tta_{checkpoint}_refs_test.csv',index=False)
    ft=evidence(te[te.class_name.eq('nude_hands')],refs).set_index('id').reindex(te.id).fillna(0).reset_index();ft.to_csv(A/f'tta_{checkpoint}_test.csv',index=False)
pd.DataFrame(held).to_csv(A/'tta_held_manual_diagnostic.csv',index=False)

for checkpoint in ['v15','refined','agreement']:
    for threshold in [.6,.7]:
        for weight in [.25,.5]:
            outputs=[]
            for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
                split='test' if test else 'oof'
                old=align(A/f'tta_v15_{split}.csv',rows);new=align(A/f'tta_refined_{split}.csv',rows)
                f=old if checkpoint=='v15' else new
                overlap=np.minimum(old.overlap,new.overlap).to_numpy() if checkpoint=='agreement' else f.overlap.to_numpy()
                sem=align(V16/f'semantic_tight_{split}.csv',rows,'p_nude_hands')
                group=pd.read_csv(V16/f'hand_rescue_group_guard_{split}.csv').set_index('id').reindex(rows.id)
                loser_paths=[V15/'hand_group_test.csv'] if test else [V15/f'hand_group_oof_{d}.csv' for d in DAYS]
                losers=set(pd.concat([pd.read_csv(p) for p in loser_paths]).query("reason=='duplicate_loser'").id)
                m=rows.class_name.eq('nude_hands').to_numpy()&(overlap>=threshold)&f.assigned.to_numpy(bool)&(f.instances.to_numpy()==1)&(sem>=.85)&~rows.id.isin(losers).to_numpy()&~group.veto.fillna(False).to_numpy(bool)
                if checkpoint=='agreement':m &= old.assigned.to_numpy(bool)&(old.instances.to_numpy()==1)
                # Strong geometric support changes ranking gradually. No audit labels.
                target=np.clip((.85-overlap)/.5,0,1);s=base.copy();s[m]-=weight*np.maximum(s[m]-target[m],0);outputs.append(s)
            register(f'tta_{checkpoint}_iou{threshold}_w{weight}',*outputs,details=dict(checkpoint=checkpoint,threshold=threshold,weight=weight,
                detection_confidence_min=.2,minimum_views=3,minimum_rotations=2,minimum_coherence=.65,semantic_min=.85,
                group_assignment='one-to-one',manual_held_crops_excluded=True))
