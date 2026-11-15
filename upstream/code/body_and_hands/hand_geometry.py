"""Uncertainty in frozen localizer boundaries, learned solely from fit-day GT."""
from common18 import *
from scipy.optimize import linear_sum_assignment
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import joblib

raw=pd.read_csv(V16/'wevis_hands_references.csv');cache={r.id: {'roi': np.r_[np.maximum(0, (np.array([r.x1,r.y1])+np.array([r.x2,r.y2]))/2-np.maximum((np.array([r.x2,r.y2])-np.array([r.x1,r.y1]))*4,160)/2), np.minimum([1280,720], (np.array([r.x1,r.y1])+np.array([r.x2,r.y2]))/2+np.maximum((np.array([r.x2,r.y2])-np.array([r.x1,r.y1]))*4,160)/2)].astype(int).tolist()} for r in pd.concat([q,te],ignore_index=True).itertuples() if r.class_name=='nude_hands'}
keep=[]
for r in raw.itertuples():
    b=np.array([getattr(r,c) for c in B]);roi=np.array(cache[r.query_id]['roi'])
    if min(b[0]-roi[0],b[1]-roi[1],roi[2]-b[2],roi[3]-b[3])>=2:keep.append(r.Index)
raw=raw.loc[keep];parts=[]
for fid,g in raw.groupby('frame_id'):
    left=g.copy()
    while len(left):
        m=iou(left.iloc[0][B],left[B])[0]>=.6;part=left[m];left=left[~m]
        row=part.iloc[0].copy();row[B]=part[B].median();row['observations']=len(part);parts.append(row)
refs=pd.DataFrame(parts).reset_index(drop=True);refs.to_csv(A/'frozen_hand_references.csv',index=False)
pairs=[]
for (fid,day),g in h[h.class_name.eq('nude_hands')].groupby(['frame_id','day']):
    rr=refs[refs.frame_id.eq(fid)];ov=iou(g[B],rr[B])
    if not ov.size:continue
    ii,jj=linear_sum_assignment(-ov)
    for i,j in zip(ii,jj):
        gt=g.iloc[i][B].to_numpy(float);bb=rr.iloc[j][B].to_numpy(float);wh=bb[2:]-bb[:2];ratio=(gt[2:]-gt[:2])/wh;offset=((gt[:2]+gt[2:])-(bb[:2]+bb[2:]))/2/wh
        if ov[i,j]<.2 or (ratio<.5).any() or (ratio>3).any() or (abs(offset)>.75).any():continue
        pairs.append(dict(frame_id=fid,day=day,initial_iou=float(ov[i,j]),dx=offset[0],dy=offset[1],log_sx=np.log(ratio[0]),log_sy=np.log(ratio[1]),ref_aspect=np.log(wh[0]/wh[1]),ref_scale=np.log(np.sqrt(wh.prod())),**{'ref_'+c:bb[k] for k,c in enumerate(B)},**{'gt_'+c:gt[k] for k,c in enumerate(B)}))
pairs=pd.DataFrame(pairs);pairs.to_csv(A/'hand_boundary_pairs.csv',index=False)

def extract(rows,fit,leave_frame=False):
    pool=pairs if fit=='all' else pairs[pairs.day.eq(fit)];out=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=np.maximum(b[2:]-b[:2],1);rr=refs[refs.frame_id.eq(r.frame_id)]
        rec=dict(id=r.id,reference_iou=0.,reference_count=len(rr),observations=0,center_dx=9.,center_dy=9.,log_area_ratio=0.,log_aspect_ratio=0.,ref_width=0.,ref_height=0.,q_width=wh[0],q_height=wh[1],prob_valid=0.,iou_q25=0.,iou_q50=0.,iou_q75=0.,iou_q90=0.,center_instances=0)
        pp=pool[pool.frame_id.ne(r.frame_id)] if leave_frame else pool
        if not len(rr) or len(pp)<5:out.append(rec);continue
        ov=iou(b,rr[B])[0];j=int(ov.argmax());bb=rr.iloc[j][B].to_numpy(float);rw=bb[2:]-bb[:2];center=(bb[:2]+bb[2:])/2;cc=(rr[B].to_numpy()[:,:2]+rr[B].to_numpy()[:,2:])/2
        # Shape-near examples are more informative; retain a broad prior to avoid
        # unsupported extrapolation in the front of the café.
        distance=abs(pp.ref_aspect-np.log(rw[0]/rw[1]))+.35*abs(pp.ref_scale-np.log(np.sqrt(rw.prod())))
        weights=np.exp(-distance.to_numpy()/.7)+.15;weights/=weights.sum()
        centers=center+pp[['dx','dy']].to_numpy()*rw;size=rw*np.exp(pp[['log_sx','log_sy']].to_numpy());hyp=np.c_[centers-size/2,centers+size/2]
        overlap=iou(b,hyp)[0];order=np.argsort(overlap);cw=np.cumsum(weights[order])
        quant=lambda level:float(overlap[order[min(np.searchsorted(cw,level),len(order)-1)]])
        delta=((b[:2]+b[2:])/2-center)/rw
        rec.update(reference_iou=float(ov[j]),observations=float(rr.iloc[j].observations),center_dx=delta[0],center_dy=delta[1],log_area_ratio=float(np.log(wh.prod()/rw.prod())),log_aspect_ratio=float(np.log((wh[0]/wh[1])/(rw[0]/rw[1]))),ref_width=rw[0],ref_height=rw[1],prob_valid=float(weights@(overlap>=.6)),center_instances=int(((cc>=b[:2])&(cc<=b[2:])).all(1).sum()),iou_q25=quant(.25),iou_q50=quant(.5),iou_q75=quant(.75),iou_q90=quant(.9))
        out.append(rec)
    return pd.DataFrame(out)

records=pd.read_csv(V16/'dino_records.csv');training={};features={}
for fit,held in zip(DAYS,DAYS[::-1]):
    tr=records[records.class_name.eq('nude_hands')&records.day.eq(fit)&records.frame_id.isin(refs.frame_id)].copy().reset_index(drop=True)
    x=extract(tr,fit,True);v=extract(q[q.class_name.eq('nude_hands')&q.day.eq(held)],fit);t=extract(te[te.class_name.eq('nude_hands')],fit)
    tr.to_csv(A/f'hand_geometry_training_{fit}.csv',index=False);x.to_csv(A/f'hand_geometry_train_features_{fit}.csv',index=False)
    v.to_csv(A/f'hand_geometry_held_features_{fit}.csv',index=False);t.to_csv(A/f'hand_geometry_test_features_{fit}.csv',index=False)
    training[fit]=tr;features[fit]=(x,v,t)
fo=pd.concat([features[d][1] for d in DAYS]).set_index('id').reindex(q.id).fillna(0).reset_index();ft=extract(te,'all')
fo.to_csv(A/'hand_geometry_features_oof.csv',index=False);ft.to_csv(A/'hand_geometry_features_test.csv',index=False)

for name in ['logreg','catboost']:
    oof=np.ones(len(q));test=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        tr=training[fit];x,v,t=features[fit];cols=[c for c in x if c!='id'];weights=tr.weight.to_numpy().copy();weights[tr.kind.eq('query_train')]*=2
        if name=='logreg':
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000));model.fit(x[cols],tr.error,logisticregression__sample_weight=weights)
        else:
            model=CatBoostClassifier(iterations=300,depth=3,learning_rate=.03,l2_leaf_reg=15,random_seed=181,verbose=False,thread_count=2,allow_writing_files=False);model.fit(x[cols],tr.error,sample_weight=weights)
        mask=q.class_name.eq('nude_hands')&q.day.eq(held);oof[mask]=pd.Series(model.predict_proba(v[cols])[:,1],index=v.id).loc[q.loc[mask,'id']];test.append(pd.Series(model.predict_proba(t[cols])[:,1],index=t.id).reindex(te.id).fillna(1).to_numpy());joblib.dump(model,A/f'hand_geometry_{name}_{fit}.joblib')
    pt=np.mean(test,axis=0);pd.DataFrame({'id':q.id,'prediction':oof}).to_csv(A/f'hand_geometry_{name}_oof.csv',index=False);pd.DataFrame({'id':te.id,'prediction':pt}).to_csv(A/f'hand_geometry_{name}_test.csv',index=False)
    for weight in [.25,.5]:
        outputs=[]
        for rows,base,pred,f,testflag in [(q,q.score.to_numpy(),oof,fo,False),(te,current,pt,ft,True)]:
            split='test' if testflag else 'oof';sem=align(V16/f'semantic_tight_{split}.csv',rows,'p_nude_hands');cal=align(V16/f'calibrated_hand_{split}.csv',rows)
            eligible=rows.class_name.eq('nude_hands').to_numpy()&(f.reference_iou.to_numpy()>=.3)&(f.center_instances.to_numpy()==1)&(sem>=.75)&cal.assigned.to_numpy(bool)&~duplicate_veto(rows,testflag)
            s=base.copy();s[eligible]-=weight*np.maximum(s[eligible]-pred[eligible],0);outputs.append(s)
        register(f'hand_geometry_{name}_w{weight}',*outputs,details=dict(weight=weight,model=name,train='fit-day synthetic plus real queries; fit-day boundary residuals leave own frame out',reference='frozen WeVis, no test fitting labels',semantic_min=.75))

for weight in [.25,.5]:
    outputs=[]
    for rows,base,f,testflag in [(q,q.score.to_numpy(),fo,False),(te,current,ft,True)]:
        split='test' if testflag else 'oof';sem=align(V16/f'semantic_tight_{split}.csv',rows,'p_nude_hands');cal=align(V16/f'calibrated_hand_{split}.csv',rows)
        eligible=rows.class_name.eq('nude_hands').to_numpy()&(f.prob_valid.to_numpy()>=.5)&(f.reference_iou.to_numpy()>=.3)&(f.center_instances.to_numpy()==1)&(sem>=.75)&cal.assigned.to_numpy(bool)&~duplicate_veto(rows,testflag)
        pred=1-f.prob_valid.to_numpy();s=base.copy();s[eligible]-=weight*np.maximum(s[eligible]-pred[eligible],0);outputs.append(s)
    register(f'hand_boundary_distribution_w{weight}',*outputs,details=dict(weight=weight,probability_valid_min=.5,fit='opposite-day manual boundary residuals'))
