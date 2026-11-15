"""One-to-one geometric evidence; fixed old ranking target; strict cross-day split."""
from common20 import *
from scipy.optimize import linear_sum_assignment

def clusters(raw,minimum_scales=2):
    out=[]
    if not len(raw):return pd.DataFrame(columns=['frame_id','class_name','confidence','scales','folds','coherence']+B)
    for (fid,c),g in raw.groupby(['frame_id','class_name']):
        g=g[g.confidence.ge(.25)].sort_values('confidence',ascending=False).copy()
        while len(g):
            ov=iou(g.iloc[0][B],g[B])[0];m=ov>=.55;part=g[m].copy();g=g[~m]
            part=part.drop_duplicates(['tile','fit']);bb=part[B].median().to_numpy(float)
            if part.tile.nunique()<minimum_scales:continue
            coherence=float(np.median(iou(bb,part[B])))
            if coherence<.65:continue
            out.append(dict(frame_id=fid,class_name=c,confidence=float(part.confidence.median()),scales=part.tile.nunique(),folds=part['fit'].nunique(),coherence=coherence,**dict(zip(B,bb))))
    result=pd.DataFrame(out)
    if not len(result):return pd.DataFrame(columns=['frame_id','class_name','confidence','scales','folds','coherence']+B)
    keep=[]
    for _,g in result.groupby(['frame_id','class_name']):
        chosen=[]
        for ix,r in g.sort_values(['folds','scales','confidence'],ascending=False).iterrows():
            if not chosen or iou(r[B],g.loc[chosen,B]).max()<.5:chosen.append(ix)
        keep+=chosen
    return result.loc[keep].reset_index(drop=True)

def evidence(rows,refs):
    out=[]
    for (fid,c),g in rows.groupby(['frame_id','class_name']):
        rr=refs[refs.frame_id.eq(fid)&refs.class_name.eq(c)].reset_index(drop=True);ov=iou(g[B],rr[B])
        if len(rr):
            reward=np.c_[np.where(ov>=.4,min(ov.shape)+1+ov,0),np.zeros((len(g),len(g)))];ii,jj=linear_sum_assignment(-reward);match={i:j for i,j in zip(ii,jj) if j<len(rr) and ov[i,j]>=.4};centers=(rr[B].to_numpy()[:,:2]+rr[B].to_numpy()[:,2:])/2
        for j,r in enumerate(g.itertuples()):
            rec=dict(id=r.id,overlap=0.,assigned=False,instances=0,confidence=0.,coherence=0.,folds=0,cover_query=0.,cover_ref=0.,duplicate_loser=False,competitor_iou=0.,winner_id='',ref_x1=0.,ref_y1=0.,ref_x2=0.,ref_y2=0.)
            if not len(rr):out.append(rec);continue
            k=match.get(j,int(ov[j].argmax()));ref=rr.iloc[k];b=np.array([getattr(r,c) for c in B]);bb=ref[B].to_numpy(float)
            competitor=float(np.delete(ov[:,k],j).max()) if len(g)>1 else 0.;winner=next((i for i,x in match.items() if x==k),None)
            lost=False;wid=''
            if j not in match and winner is not None and competitor>=.65 and competitor-ov[j,k]>=.08:
                other=g.iloc[winner][B].to_numpy(float);lost=bool(iou(b,other)[0,0]>=.4);wid=str(g.iloc[winner].id)
            inter=np.maximum(0,np.minimum(b[2:],bb[2:])-np.maximum(b[:2],bb[:2])).prod()
            rec.update(overlap=float(ov[j,k]),assigned=j in match,instances=int(((centers>=b[:2])&(centers<=b[2:])).all(1).sum()),confidence=float(ref.confidence),coherence=float(ref.coherence),folds=int(ref.folds),cover_query=inter/np.prod(b[2:]-b[:2]),cover_ref=inter/np.prod(bb[2:]-bb[:2]),duplicate_loser=lost,competitor_iou=competitor,winner_id=wid,**{'ref_'+c:float(ref[c]) for c in B});out.append(rec)
    return pd.DataFrame(out).set_index('id').loc[rows.id].reset_index()

def old_veto(rows,test):
    split='test' if test else 'oof';old=align(ROOT/f'body_and_hands/replay_inputs/{split}.csv',rows)
    dup=boolean(old.duplicate_veto)&rows.class_name.eq('nude_hands').to_numpy()
    groups=pd.read_csv(ROOT/f'local_evidence/duplicate_groups_{split}.csv');dup|=rows.id.isin(groups.loc[groups.reason.eq('duplicate_loser'),'id']).to_numpy()
    return dup

def detection_metrics(refs,rows,classes):
    metrics=[]
    for day in DAYS:
        fids=set(rows[rows.day.eq(day)].frame_id)
        for c in classes:
            gt=h[h.day.eq(day)&h.frame_id.isin(fids)&h.class_name.eq(c)];pr=refs[refs.frame_id.isin(fids)&refs.class_name.eq(c)].sort_values('confidence',ascending=False);seen=set();hits=[]
            for r in pr.itertuples():
                g=gt[gt.frame_id.eq(r.frame_id)];ov=iou([getattr(r,k) for k in B],g[B])[0] if len(g) else [];valid=[(v,ix) for v,ix in zip(ov,g.index) if v>=.6 and ix not in seen]
                if valid:seen.add(max(valid)[1]);hits.append(1)
                else:hits.append(0)
            precision=np.cumsum(hits)/np.arange(1,len(hits)+1) if hits else np.array([])
            metrics.append(dict(day=day,class_name=c,gt_boxes=len(gt),references=len(pr),matched=len(seen),recall=len(seen)/max(len(gt),1),det_ap06=float((precision*np.asarray(hits)).sum()/max(len(gt),1)),scope='manual ground truth on official query frames only; not all 84 manual frames'))
    return metrics

def run(branch,folder,filename,classes,minimum_scales=2):
    raw={fit:pd.read_csv(A/f'gpu_results/{folder}/{fit}/{filename}').assign(fit=fit) for fit in DAYS}
    oo=[];ro=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        rows=q[q.day.eq(held)];rr=clusters(raw[fit][raw[fit].frame_id.isin(rows.frame_id)],minimum_scales);oo.append(evidence(rows,rr));ro.append(rr)
    fo=pd.concat(oo).set_index('id').loc[q.id].reset_index();rrt=clusters(pd.concat([d[d.frame_id.isin(te.frame_id)] for d in raw.values()],ignore_index=True),minimum_scales);ft=evidence(te,rrt)
    fo.to_csv(A/f'{branch}_oof.csv',index=False);ft.to_csv(A/f'{branch}_test.csv',index=False);pd.concat(ro).to_csv(A/f'{branch}_refs_oof.csv',index=False);rrt.to_csv(A/f'{branch}_refs_test.csv',index=False)
    dm=detection_metrics(pd.concat(ro),q,classes)
    (A/f'{branch}_detection_metrics.json').write_text(json.dumps(dm,indent=2))
    if minimum_scales==1:return
    for threshold in [.6,.7]:
        for weight in [.25,.5]:
            out=[]
            for rows,base,f,test in [(q,q.score.to_numpy(),fo,False),(te,current,ft,True)]:
                sem=align(ROOT/f'geometry_guards/semantic_tight_{"test" if test else "oof"}.csv',rows)
                semantic=np.array([sem.iloc[j]['p_'+c] for j,c in enumerate(rows.class_name)])
                m=rows.class_name.isin(classes).to_numpy()&f.assigned.to_numpy(bool)&f.instances.eq(1).to_numpy()&f.overlap.ge(threshold).to_numpy()&f.confidence.ge(.3).to_numpy()&(semantic>=.75)&~old_veto(rows,test)&~f.duplicate_loser.to_numpy(bool)
                target=np.clip((.85-f.overlap.to_numpy())/.5,0,1);s=base.copy();s[m]-=weight*np.maximum(s[m]-target[m],0);out.append(s)
            register(f'{branch}_rescue_iou{threshold}_w{weight}',*out,dict(weight=weight,overlap_min=threshold,semantic_min=.75,reference='independent SAHI tiles; cross-day detector; one-to-one assignment',ranking_target='same linear IoU target used by V17'))
    for weight in [.25,.5]:
        out=[]
        for rows,base,f,test in [(q,q.score.to_numpy(),fo,False),(te,current,ft,True)]:
            m=rows.class_name.isin(classes).to_numpy()&f.confidence.ge(.5).to_numpy()&f.coherence.ge(.75).to_numpy()
            merged=m&f.instances.ge(2).to_numpy()&f.overlap.lt(.5).to_numpy()
            under=m&f.cover_query.ge(.9).to_numpy()&f.cover_ref.lt(.5).to_numpy()&f.overlap.lt(.5).to_numpy()
            target=np.clip((.85-f.overlap.to_numpy())/.5,0,1);s=base.copy();flag=merged|under;s[flag]+=weight*np.maximum(target[flag]-s[flag],0)
            s[f.duplicate_loser.to_numpy(bool)&m]=np.maximum(s[f.duplicate_loser.to_numpy(bool)&m],.95)
            out.append(s)
        register(f'{branch}_structural_w{weight}',*out,dict(weight=weight,confidence_min=.5,undercoverage='query cover >=.9, ref cover <.5',duplicates='unique winner IoU>=.65 and margin>=.08'))

if __name__=='__main__':
    import sys
    selections=sys.argv[1:] or ['expanded']
    for variant in selections:
        if variant=='cup_tray':run('cup_tray_sahi','cup_tray','references.csv',['cup','tray'])
        else:
            path=A/f'gpu_results/yolo_hands/2026-10-25/{variant}_sahi_references.csv'
            if path.exists():run(f'hand_{variant}_sahi','yolo_hands',f'{variant}_sahi_references.csv',['nude_hands'])
