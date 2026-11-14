"""Head/pose-conditioned body extent trained only on opposite-day manual people."""
from common18 import *
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
bank=pd.read_csv(V16/'head_geometry_templates.csv');heads=pd.read_csv(V16/'wevis_heads_references.csv');poses=pd.read_csv(V16/'multiview_pose.csv')

def descriptor(fid,hc,hw):
    rec=dict(hx=hc[0]/1280,hy=hc[1]/720,hw=hw[0]/1280,hh=hw[1]/720,head_aspect=np.log(hw[0]/hw[1]),pose_found=0,pose_conf=0.)
    for k in range(5,17):
        for c in ['dx','dy','conf']:rec[f'kp{k}_{c}']=0.
    for c in ['left','top','right','bottom']:rec['pose_'+c]=0.
    pp=poses[poses.frame_id.eq(fid)&poses.confidence.ge(.35)];candidates=[]
    for rr in pp.itertuples():
        kp=np.array([[getattr(rr,f'kp{k}_x'),getattr(rr,f'kp{k}_y'),getattr(rr,f'kp{k}_conf')] for k in range(17)])
        m=kp[:5,2]>=.3
        if m.sum()<2:continue
        center=np.average(kp[:5,:2][m],weights=kp[:5,2][m],axis=0);dist=np.linalg.norm((center-hc)/hw)
        if dist<1.2:candidates.append((dist-.25*rr.confidence,rr,kp))
    if candidates:
        _,rr,kp=min(candidates,key=lambda x:x[0]);rec.update(pose_found=1,pose_conf=rr.confidence)
        for k in range(5,17):
            delta=(kp[k,:2]-hc)/hw
            if kp[k,2]>=.3:rec.update({f'kp{k}_dx':float(np.clip(delta[0],-12,12)),f'kp{k}_dy':float(np.clip(delta[1],-12,12)),f'kp{k}_conf':kp[k,2]})
        bb=(np.array([getattr(rr,c) for c in B])-np.tile(hc,2))/np.tile(hw,2)
        rec.update(dict(zip(['pose_left','pose_top','pose_right','pose_bottom'],bb)))
    return rec

train=pd.DataFrame([descriptor(r.frame_id,np.array([r.head_x,r.head_y]),np.array([r.head_w,r.head_h])) for r in bank.itertuples()])
train.to_csv(A/'person_body_training_features.csv',index=False)
target=(bank[B].to_numpy()-np.tile(bank[['head_x','head_y']].to_numpy(),(1,2)))/np.tile(bank[['head_w','head_h']].to_numpy(),(1,2))
def queries(rows):
    out=[];matched=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);hh=heads[heads.frame_id.eq(r.frame_id)];cc=(hh[B].to_numpy()[:,:2]+hh[B].to_numpy()[:,2:])/2;inside=((cc>=b[:2])&(cc<=b[2:])).all(1)
        if r.class_name!='person' or inside.sum()!=1:
            out.append({c:0. for c in train});matched.append(dict(id=r.id,eligible=False,hx=0.,hy=0.,hw=1.,hh=1.));continue
        hb=hh.loc[inside,B].to_numpy()[0];hc=(hb[:2]+hb[2:])/2;hw=hb[2:]-hb[:2];out.append(descriptor(r.frame_id,hc,hw));matched.append(dict(id=r.id,eligible=True,hx=hc[0],hy=hc[1],hw=hw[0],hh=hw[1]))
    return pd.DataFrame(out),pd.DataFrame(matched)
xq,mq=queries(q);xt,mt=queries(te)
xq.to_csv(A/'person_body_query_features_oof.csv',index=False);xt.to_csv(A/'person_body_query_features_test.csv',index=False)
mq.to_csv(A/'person_body_query_heads_oof.csv',index=False);mt.to_csv(A/'person_body_query_heads_test.csv',index=False)
pd.DataFrame(target,columns=B).assign(day=bank.day.to_numpy(),frame_id=bank.frame_id.to_numpy()).to_csv(A/'person_body_training_targets.csv',index=False)
for name in ['ridge','catboost']:
    oof=np.zeros((len(q),4));pt=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        m=bank.day.eq(fit);v=q.day.eq(held)
        if name=='ridge':model=make_pipeline(StandardScaler(),Ridge(alpha=100))
        else:model=CatBoostRegressor(loss_function='MultiRMSE',iterations=400,depth=3,learning_rate=.025,l2_leaf_reg=20,random_seed=181,verbose=False,thread_count=2,allow_writing_files=False)
        model.fit(train[m],target[m]);oof[v]=model.predict(xq[v]);pt.append(model.predict(xt));joblib.dump(model,A/f'person_body_{name}_{fit}.joblib')
    test=np.mean(pt,axis=0);evidence=[]
    for rows,pred,matches,split in [(q,oof,mq,'oof'),(te,test,mt,'test')]:
        centers=matches[['hx','hy']].to_numpy();sizes=matches[['hw','hh']].to_numpy();bb=pred*np.tile(sizes,(1,2))+np.tile(centers,(1,2));bb[:,2:]=np.maximum(bb[:,2:],bb[:,:2]+1)
        out=[]
        for i,r in enumerate(rows.itertuples()):
            b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];pw=bb[i,2:]-bb[i,:2]
            out.append(dict(id=r.id,eligible=bool(matches.iloc[i].eligible),body_iou=float(iou(b,bb[i])[0,0]),height_ratio=float(wh[1]/pw[1]),top_error=float(abs(b[1]-bb[i,1])/pw[1]),width_ratio=float(wh[0]/pw[0]),**{'ref_'+c:bb[i,j] for j,c in enumerate(B)}))
        f=pd.DataFrame(out);f.to_csv(A/f'person_body_{name}_{split}.csv',index=False);evidence.append(f)
    for weight in [.25,.5]:
        outputs=[]
        for rows,base,f in [(q,q.score.to_numpy(),evidence[0]),(te,current,evidence[1])]:
            eligible=f.eligible.to_numpy()&(f.height_ratio.to_numpy()<.75)&(f.top_error.to_numpy()<.2)&(f.width_ratio.to_numpy()>.65)&(f.width_ratio.to_numpy()<1.5)&(f.body_iou.to_numpy()<.6)
            strength=np.clip((.85-f.height_ratio.to_numpy())/.35,0,1);outputs.append(base+weight*(1-base)*strength*eligible)
        register(f'person_body_{name}_w{weight}',*outputs,details=dict(model=name,weight=weight,training='manual person boxes; held day excluded',head_count=1,max_height_ratio=.75,max_top_offset=.2))
