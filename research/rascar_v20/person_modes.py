"""Three person modes, fit-day templates, leave-frame-out training evidence.

Mode labels were visually reviewed only on 77 manual TRAIN people. Test audit
labels never enter this module. Mode uncertainty is retained in the features.
"""
from common20 import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from catboost import CatBoostClassifier
from functools import lru_cache
import joblib

V16=ROOT/'rascar_v16';V18=ROOT/'rascar_v18'
bank=pd.read_csv(V16/'head_geometry_templates.csv')
train_pose=pd.read_csv(V18/'person_body_training_features.csv')
heads=pd.read_csv(V16/'wevis_heads_references.csv')
poses=pd.read_csv(V16/'multiview_pose.csv')
records=pd.read_csv(V16/'dino_records.csv')
maps=json.loads((ROOT/'rascar_v17/seat_maps.json').read_text())
front={1,6,8,10,16,18,23,26,27,28,31,35,36,42,60,73,74}
standing={2,11,12,13,24,43,44,47,48,76}
bank['mode']=['standing' if i in standing else 'front_seated' if i in front else 'back_seated' for i in range(len(bank))]
bank['mode_source']='visual_review_manual_train_only_2026-11-13'
bank.to_csv(A/'inputs/person_mode_labels.csv',index=False)
MODES=['back_seated','front_seated','standing']

@lru_cache(None)
def head_descriptor(fid,head_index):
    hh=heads[heads.frame_id.eq(fid)].reset_index(drop=True);hb=hh.iloc[head_index][B].to_numpy(float)
    hc=(hb[:2]+hb[2:])/2;hw=hb[2:]-hb[:2]
    rec=dict(hx=hc[0]/1280,hy=hc[1]/720,hw=hw[0]/1280,hh=hw[1]/720,head_aspect=np.log(hw[0]/hw[1]),pose_found=0,pose_conf=0.)
    for k in range(5,17):
        for c in ['dx','dy','conf']:rec[f'kp{k}_{c}']=0.
    for c in ['left','top','right','bottom']:rec['pose_'+c]=0.
    candidates=[]
    for rr in poses[poses.frame_id.eq(fid)&poses.confidence.ge(.35)].itertuples():
        kp=np.array([[getattr(rr,f'kp{k}_x'),getattr(rr,f'kp{k}_y'),getattr(rr,f'kp{k}_conf')] for k in range(17)])
        m=kp[:5,2]>=.3
        if m.sum()<2:continue
        center=np.average(kp[:5,:2][m],weights=kp[:5,2][m],axis=0);dist=np.linalg.norm((center-hc)/hw)
        if dist<1.2:candidates.append((dist-.25*rr.confidence,rr,kp))
    if candidates:
        _,rr,kp=min(candidates,key=lambda x:x[0]);rec.update(pose_found=1,pose_conf=rr.confidence)
        for k in range(5,17):
            if kp[k,2]>=.3:
                delta=(kp[k,:2]-hc)/hw
                rec.update({f'kp{k}_dx':float(np.clip(delta[0],-12,12)),f'kp{k}_dy':float(np.clip(delta[1],-12,12)),f'kp{k}_conf':kp[k,2]})
        bb=(np.array([getattr(rr,c) for c in B])-np.tile(hc,2))/np.tile(hw,2)
        rec.update(dict(zip(['pose_left','pose_top','pose_right','pose_bottom'],bb)))
    return rec,hc,hw

def mode_x(x,fit):
    x=x.copy();t=np.asarray(maps[fit]['tables']);w=t[:,2]-t[:,0];ht=t[:,3]-t[:,1];cx=(t[:,2]+t[:,0])/2
    anchors=np.vstack([np.c_[cx,t[:,1]-.45*ht],np.c_[cx,t[:,3]+.1*ht]])
    scales=np.vstack([np.c_[w,ht],np.c_[w,ht]])
    dist=np.linalg.norm((x[['hx','hy']].to_numpy()[:,None,:]*[1280,720]-anchors[None,:,:])/scales[None,:,:],axis=2)
    for j in range(12):x[f'seat_distance_{j}']=dist[:,j].clip(0,6)
    for side in [(11,13,15),(12,14,16)]:
        a,b,c=side;u=x[[f'kp{a}_dx',f'kp{a}_dy']].to_numpy()-x[[f'kp{b}_dx',f'kp{b}_dy']].to_numpy();v=x[[f'kp{c}_dx',f'kp{c}_dy']].to_numpy()-x[[f'kp{b}_dx',f'kp{b}_dy']].to_numpy()
        conf=x[[f'kp{k}_conf' for k in side]].min(axis=1).to_numpy()
        x[f'knee_{b}_cos']=(u*v).sum(1)/np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),1e-5)*conf
    return x

@lru_cache(None)
def mode_model(fit,exclude=''):
    mask=bank.day.eq(fit)&bank.frame_id.ne(exclude)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.12,max_iter=1500,class_weight='balanced'))
    model.fit(mode_x(train_pose,fit)[mask],bank.loc[mask,'mode'])
    return model

def extract(rows,fit,leave_frame_out=False,mode_override=None):
    template_mask=bank.day.eq(fit);pool=bank[template_mask].copy();pool_pose=train_pose[template_mask].copy()
    out=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=np.maximum(b[2:]-b[:2],1)
        rec=dict(id=r.id,eligible=0,head_count=0,mode_confidence=0.,mode_entropy=1.,mode_back=0.,mode_front=0.,mode_standing=0.,
                 support=0,valid_fraction=0.,iou_q25=0.,iou_q50=0.,iou_q75=0.,iou_max=0.,height_ratio=1.,height_cv=1.,
                 bottom_gap=0.,top_gap=0.,width_ratio=1.,position_distance=9.,visible_outside=0.,legs_outside=0.,
                 pose_conf=0.,query_aspect=np.log(wh[0]/wh[1]),query_area=np.log(wh.prod()),head_fraction=0.,edge=0.)
        hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy();cc=(hh[:,:2]+hh[:,2:])/2
        inside=((cc>=b[:2])&(cc<=b[2:])).all(1);rec['head_count']=int(inside.sum())
        if r.class_name!='person' or inside.sum()!=1:out.append(rec);continue
        desc,hc,hw=head_descriptor(r.frame_id,int(np.flatnonzero(inside)[0]));dx=pd.DataFrame([desc],columns=train_pose.columns)
        if mode_override is None:
            model=mode_model(fit,r.frame_id if leave_frame_out else '')
            probs=dict(zip(model.classes_,model.predict_proba(mode_x(dx,fit))[0]))
        else:probs=mode_override[r.id]
        pp=np.array([probs.get(k,0.) for k in MODES])
        rec.update(eligible=1,mode_confidence=pp.max(),mode_entropy=float(-(pp*np.log(pp+1e-9)).sum()/np.log(3)),
                   mode_back=pp[0],mode_front=pp[1],mode_standing=pp[2],pose_conf=desc['pose_conf'],head_fraction=hw.prod()/wh.prod(),edge=float(b[0]<5 or b[2]>1275 or b[1]<5 or b[3]>715))
        kp=np.array([[hc[0]+desc[f'kp{k}_dx']*hw[0],hc[1]+desc[f'kp{k}_dy']*hw[1],desc[f'kp{k}_conf']] for k in range(5,17)])
        outside=np.maximum(b[:2]-kp[:,:2],0)+np.maximum(kp[:,:2]-b[2:],0)
        margin=np.linalg.norm(outside/wh,axis=1);usable=kp[:,2]>=.6
        rec['visible_outside']=float((margin*kp[:,2]*usable).sum()/max((kp[:,2]*usable).sum(),1))
        rec['legs_outside']=float((margin[6:]*kp[6:,2]*(kp[6:,2]>=.75)).max())
        pc=pool[['head_x','head_y']].to_numpy();ps=pool[['head_w','head_h']].to_numpy()
        loc=np.linalg.norm((pc-hc)/np.array([100,90]),axis=1)+abs(np.log(ps[:,0]/hw[0]))*.5
        ppose=pool_pose.to_numpy();query=dx.to_numpy()[0];pose_dist=np.zeros(len(pool));valid_count=np.zeros(len(pool))
        for k in range(5,17):
            c=pool_pose[f'kp{k}_conf'].to_numpy();cf=desc[f'kp{k}_conf'];weight=np.minimum(c,cf)*(c>.5)*(cf>.5)
            delta=pool_pose[[f'kp{k}_dx',f'kp{k}_dy']].to_numpy()-[desc[f'kp{k}_dx'],desc[f'kp{k}_dy']]
            pose_dist+=np.minimum(np.linalg.norm(delta,axis=1),4)*weight;valid_count+=weight
        pose_dist/=np.maximum(valid_count,1)
        mode_weight=np.array([probs.get(m,0.) for m in pool['mode']]);distance=loc+.4*pose_dist
        allowed=pool.frame_id.ne(r.frame_id).to_numpy() if leave_frame_out else np.ones(len(pool),bool)
        weights=np.exp(-distance**2/(2*.95**2))*mode_weight*allowed
        ix=np.argsort(weights)[-7:];ix=ix[(weights[ix]>.015)&(loc[ix]<2.5)]
        if len(ix)<2:out.append(rec);continue
        sub=pool.iloc[ix];ww=weights[ix]/weights[ix].sum();scale=np.clip(hw[0]/sub.head_w.to_numpy(),.7,1.4)
        pb=(sub[B].to_numpy()-np.tile(sub[['head_x','head_y']].to_numpy(),(1,2)))*scale[:,None]+np.tile(hc,(len(ix),2))
        ov=iou(b,pb)[0];hhp=pb[:,3]-pb[:,1];wwp=pb[:,2]-pb[:,0]
        def quantile(v,p):
            j=np.argsort(v);return float(v[j[min(np.searchsorted(np.cumsum(ww[j]),p),len(j)-1)]])
        rec.update(support=len(ix),valid_fraction=float(ww@(ov>=.6)),iou_max=ov.max(),height_ratio=wh[1]/quantile(hhp,.5),
                   height_cv=float(np.sqrt(ww@((hhp-ww@hhp)**2))/(ww@hhp)),bottom_gap=float(ww@((pb[:,3]-b[3])/hhp)),
                   top_gap=float(ww@((b[1]-pb[:,1])/hhp)),width_ratio=wh[0]/quantile(wwp,.5),position_distance=float(ww@loc[ix]))
        for p in [25,50,75]:rec[f'iou_q{p}']=quantile(ov,p/100)
        out.append(rec)
    return pd.DataFrame(out)

if __name__=='__main__':
    mode_predictions=[];fold_data={}
    for fit,held in zip(DAYS,DAYS[::-1]):
        mask=bank.day.eq(held);mo=mode_model(fit);pred=mo.predict(mode_x(train_pose,fit)[mask]);mode_predictions.extend(zip(bank.index[mask],pred));joblib.dump(mo,A/'models'/f'mode_{fit}.joblib')
        tr=records[records.day.eq(fit)&records.class_name.eq('person')&~records.kind.eq('query_test')].reset_index(drop=True)
        x=extract(tr,fit,True);v=extract(q[q.day.eq(held)],fit);t=extract(te,fit)
        for name,data in [('train',x),('held',v),('test',t)]:data.to_csv(A/f'person_mode_{name}_{fit}.csv',index=False)
        fold_data[fit]=(tr,x,v,t)
        print('MODE_FOLD',fit,'train records',len(tr),'held mode acc',accuracy_score(bank.loc[mask,'mode'],pred),flush=True)
    pd.DataFrame(mode_predictions,columns=['template_index','oof_mode']).to_csv(A/'person_mode_oof_labels.csv',index=False)
    for family in ['logreg','catboost']:
        so=np.zeros(len(q));pt=[];fo=[];ft=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            tr,x,v,t=fold_data[fit];cols=[c for c in x if c!='id'];m=x.eligible.eq(1)&x.support.ge(2)&tr.kind.ne('background')
            weights=tr.weight.to_numpy().copy();weights[tr.kind.eq('query_train')]*=2
            if family=='logreg':
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,class_weight='balanced'))
                model.fit(x.loc[m,cols],tr.loc[m,'error'],logisticregression__sample_weight=weights[m])
            else:
                model=CatBoostClassifier(iterations=300,depth=3,learning_rate=.025,l2_leaf_reg=20,verbose=False,thread_count=2,random_seed=201,allow_writing_files=False)
                model.fit(x.loc[m,cols],tr.loc[m,'error'],sample_weight=weights[m])
            so[q.day.eq(held)]=model.predict_proba(v[cols])[:,1];pt.append(model.predict_proba(t[cols])[:,1]);fo.append(v);ft.append(t)
            joblib.dump(model,A/'models'/f'person_{family}_{fit}.joblib')
        f=pd.concat(fo).set_index('id').loc[q.id].reset_index();tf=ft[0].copy();tf.iloc[:,1:]=(ft[0].iloc[:,1:].to_numpy()+ft[1].iloc[:,1:].to_numpy())/2
        pd.DataFrame(dict(id=q.id,probability=so)).to_csv(A/f'person_modes_{family}_oof.csv',index=False)
        pd.DataFrame(dict(id=te.id,probability=np.mean(pt,axis=0))).to_csv(A/f'person_modes_{family}_test.csv',index=False)
        for weight in [.15,.3]:
            outputs=[]
            for rows,base,pred,fe in [(q,q.score.to_numpy(),so,f),(te,current,np.mean(pt,axis=0),tf)]:
                gate=rows.class_name.eq('person').to_numpy()&fe.eligible.eq(1).to_numpy()&fe.support.ge(3).to_numpy()&fe.mode_confidence.ge(.6).to_numpy()&fe.edge.eq(0).to_numpy()
                score=base.copy();score[gate]=(1-weight)*score[gate]+weight*pred[gate];outputs.append(score)
            register(f'person_modes_{family}_w{weight}',*outputs,dict(training='fit-day manual visual mode labels; fit-day errors and synthetic box perturbations; leave-frame-out templates/mode classifier',weight=weight))
