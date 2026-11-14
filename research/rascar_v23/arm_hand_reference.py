"""Infer expected palm boxes from the elbow-wrist chain, calibrated on fit-day GT."""
from common23 import *
from features23 import frame_pose
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from functools import lru_cache

def wrists(fid):
    output=[]
    for pr,kp in frame_pose(fid):
        for e,k in [(7,9),(8,10)]:
            if min(kp[e,2],kp[k,2])<.65:continue
            axis=kp[k,:2]-kp[e,:2];length=np.linalg.norm(axis)
            if not 12<length<180:continue
            unit=axis/length
            descriptor=np.r_[kp[k,0]/220,kp[k,1]/140,unit*.8,np.log(length)]
            output.append(dict(view=pr.view,k=k,wrist=kp[k,:2],unit=unit,length=length,confidence=min(kp[e,2],kp[k,2]),x=descriptor))
    return output

def calibration():
    out=[]
    for (fid,day),g in h[h.class_name.eq('nude_hands')].groupby(['frame_id','day']):
        gt=g[B].to_numpy();centers=(gt[:,:2]+gt[:,2:])/2;sizes=gt[:,2:]-gt[:,:2];ww=wrists(fid)
        for view in sorted({v['view'] for v in ww}):
            pp=[v for v in ww if v['view']==view]
            cost=cdist(centers,np.array([v['wrist'] for v in pp]))/np.sqrt(sizes.prod(1))[:,None]
            ii,jj=linear_sum_assignment(cost)
            for i,j in zip(ii,jj):
                if cost[i,j]>1.2:continue
                v=pp[j];u=v['unit'];perp=np.array([-u[1],u[0]]);delta=(centers[i]-v['wrist'])/v['length']
                y=np.r_[delta@u,delta@perp,np.log(sizes[i]/v['length'])]
                if not -.3<y[0]<1.2 or abs(y[1])>.6:continue
                out.append(dict(frame_id=fid,day=day,view=view,**{f'x{k}':a for k,a in enumerate(v['x'])},**{f'y{k}':a for k,a in enumerate(y)}))
    return pd.DataFrame(out)

bank=calibration();bank.to_csv(A/'inputs/arm_hand_calibration.csv',index=False)

@lru_cache(None)
def references(fid,fit,mode):
    tr=bank[bank.day.eq(fit)&bank.frame_id.ne(fid)];x=tr[[f'x{k}' for k in range(5)]].to_numpy();yy=tr[[f'y{k}' for k in range(4)]].to_numpy()
    out=[]
    for v in wrists(fid):
        distance=np.linalg.norm(x-v['x'],axis=1)
        # Select one representative observation per training frame, avoiding TTA duplication.
        order=np.argsort(distance);chosen=[];frames=set()
        for j in order:
            fid2=tr.iloc[j].frame_id
            if fid2 in frames:continue
            frames.add(fid2);chosen.append(j)
            if len(chosen)==7:break
        if len(chosen)<3:continue
        dist=distance[chosen];chosen=np.asarray(chosen)
        if mode=='local':
            chosen=chosen[dist<2.2];dist=dist[dist<2.2]
            if len(chosen)<3:continue
            w=np.exp(-dist**2/2);w/=w.sum();target=w@yy[chosen]
        else:target=np.median(yy,axis=0)
        u=v['unit'];perp=np.array([-u[1],u[0]]);cc=v['wrist']+v['length']*(target[0]*u+target[1]*perp);wh=np.exp(target[2:])*v['length']
        bb=np.r_[cc-wh/2,cc+wh/2];out.append(dict(box=bb,view=v['view'],confidence=v['confidence'],support=len(chosen)))
    # Merge repeated views of each wrist, preserving two neighboring hands as separate instances.
    clusters=[]
    while out:
        ov=iou([r['box'] for r in out],[r['box'] for r in out]);seed=int((ov>.45).sum(1).argmax());m=ov[seed]>.45
        group=[r for r,yes in zip(out,m) if yes];out=[r for r,yes in zip(out,m) if not yes]
        bb=np.median([r['box'] for r in group],axis=0)
        clusters.append(dict(box=bb,views=len({r['view'] for r in group}),confidence=float(np.mean([r['confidence'] for r in group])),
            coherence=float(np.median(iou(bb,[r['box'] for r in group]))),support=min(r['support'] for r in group)))
    return clusters

def evidence(rows,fit,mode):
    out=[]
    for fid,g in rows.groupby('frame_id'):
        rr=references(fid,fit,mode);rr=[r for r in rr if r['views']>=2 and r['coherence']>=.6]
        rb=np.asarray([r['box'] for r in rr]).reshape(-1,4);hands=g[g.class_name.eq('nude_hands')]
        ov=iou(hands[B],rb);assigned={}
        if ov.size:
            ii,jj=linear_sum_assignment(-ov);assigned={hands.iloc[i].id:j for i,j in zip(ii,jj) if ov[i,j]>.1}
        for r in g.itertuples():
            v=dict(id=r.id,overlap=0.,views=0.,coherence=0.,instances=0.,assigned=False,ref_x1=0.,ref_y1=0.,ref_x2=0.,ref_y2=0.)
            if r.class_name!='nude_hands' or not len(rr):out.append(v);continue
            b=np.array([getattr(r,c) for c in B]);o=iou(b,rb)[0];j=int(o.argmax());cent=(rb[:,:2]+rb[:,2:])/2
            v.update(overlap=float(o[j]),views=rr[j]['views'],coherence=rr[j]['coherence'],assigned=assigned.get(r.id,-1)==j,
                instances=int(((cent>=b[:2])&(cent<=b[2:])).all(1).sum()),**dict(zip(['ref_'+c for c in B],rb[j])))
            out.append(v)
    return pd.DataFrame(out).set_index('id').loc[rows.id]

if __name__=='__main__':
    print('ARM_CALIBRATION',bank.groupby('day').agg(n=('frame_id','size'),frames=('frame_id','nunique')).to_dict(),flush=True)
    for mode in ['global','local']:
        oo=[];tt=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            oo.append(evidence(q[q.day.eq(held)],fit,mode));tt.append(evidence(te,fit,mode))
        fo=pd.concat(oo).loc[q.id];ft=(tt[0].astype(float)+tt[1].astype(float))/2
        fo.reset_index().to_csv(A/f'features/arm_{mode}_oof.csv',index=False);ft.reset_index().to_csv(A/f'features/arm_{mode}_test.csv',index=False)
        for overlap in [.55,.65]:
            for weight in [.25,.45]:
                output=[]
                for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                    z=align(V22/f'replay_inputs/{split}.csv',rows)
                    truth=lambda s:s.astype(str).str.lower().isin(['true','1','1.0']).to_numpy()
                    gate=rows.class_name.eq('nude_hands').to_numpy()&f.assigned.eq(1).to_numpy()&f.overlap.ge(overlap).to_numpy()&f.instances.le(1).to_numpy()&f.views.ge(2).to_numpy()
                    gate &= z.semantic_hand.ge(.8).to_numpy()&z.independent_hand_centers.lt(2).to_numpy()&~truth(z.old_duplicate_veto)&~truth(z.zoom_duplicate_loser)
                    gate &= ~(np.minimum(z.quality_logreg,z.quality_svm).to_numpy()>=.75)&~(z.structural_hand.to_numpy()>z.baseline.to_numpy()+1e-9)
                    if split=='test':gate &= np.minimum(tt[0].overlap,tt[1].overlap).to_numpy()>=overlap-.05
                    target=np.clip((.85-f.overlap.to_numpy())/.5,0,1);score=base-weight*np.maximum(base-target,0)*gate;output.append(score)
                    pd.DataFrame(dict(id=rows.id,gate=gate,target=target,score=score)).to_csv(A/f'features/arm_{mode}_{overlap}_{weight}_{split}.csv',index=False)
                register(f'arm_{mode}_iou{overlap}_w{weight}',*output,dict(type='forearm_to_palm_reference',mode=mode,min_iou=overlap,weight=weight,
                    calibration='fit-day manually annotated hands; one nearest observation per frame; no held/test labels'))
