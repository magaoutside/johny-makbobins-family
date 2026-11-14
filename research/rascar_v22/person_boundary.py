"""Fit-day boundary experts with dense near-threshold positive perturbations.

Mode and template evidence for training records excludes their entire frame.
Test visual labels and query scores never enter these model features or targets.
"""
from common22 import *
from functools import lru_cache
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
bank=pd.read_csv(ROOT/'rascar_v20/inputs/person_mode_labels.csv')
bp=pd.read_csv(ROOT/'rascar_v18/person_body_training_features.csv')
heads=pd.read_csv(ROOT/'rascar_v16/wevis_heads_references.csv')
poses=pd.read_csv(ROOT/'rascar_v16/multiview_pose.csv')
maps=json.loads((ROOT/'rascar_v17/seat_maps.json').read_text())
MODES=['back_seated','front_seated','standing']
hg={fid:g for fid,g in h[h.class_name.eq('person')].groupby('frame_id')}

@lru_cache(None)
def descriptor(fid,j):
    hh=heads[heads.frame_id.eq(fid)].reset_index(drop=True);bb=hh.iloc[j][B].to_numpy(float)
    center=(bb[:2]+bb[2:])/2;size=bb[2:]-bb[:2]
    v=dict(hx=center[0]/1280,hy=center[1]/720,hw=size[0]/1280,hh=size[1]/720,head_aspect=np.log(size[0]/size[1]),pose_found=0,pose_conf=0.)
    for k in range(5,17):
        for name in ['dx','dy','conf']:v[f'kp{k}_{name}']=0.
    for name in ['left','top','right','bottom']:v['pose_'+name]=0.
    candidates=[]
    for rr in poses[poses.frame_id.eq(fid)&poses.confidence.ge(.35)].itertuples():
        kp=np.array([[getattr(rr,f'kp{k}_x'),getattr(rr,f'kp{k}_y'),getattr(rr,f'kp{k}_conf')] for k in range(17)])
        valid=kp[:5,2]>=.3
        if valid.sum()<2:continue
        c=np.average(kp[:5,:2][valid],weights=kp[:5,2][valid],axis=0)
        distance=np.linalg.norm((c-center)/size)
        if distance<1.2:candidates.append((distance-.25*rr.confidence,rr,kp))
    if candidates:
        _,rr,kp=min(candidates,key=lambda a:a[0]);v.update(pose_found=1,pose_conf=rr.confidence)
        for k in range(5,17):
            if kp[k,2]>=.3:
                z=np.clip((kp[k,:2]-center)/size,-12,12)
                v.update({f'kp{k}_dx':z[0],f'kp{k}_dy':z[1],f'kp{k}_conf':kp[k,2]})
        v.update(dict(zip(['pose_left','pose_top','pose_right','pose_bottom'],(np.array([getattr(rr,c) for c in B])-np.tile(center,2))/np.tile(size,2))))
    return v,center,size

@lru_cache(None)
def mode(fit,exclude):
    m=bank.day.eq(fit)&bank.frame_id.ne(exclude)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.12,max_iter=1000,class_weight='balanced'))
    model.fit(bp[m],bank.loc[m,'mode'])
    return model

def features(rows,fit,lofo=False):
    outputs=[];tables=np.array(maps[fit]['tables']);tw=tables[:,2]-tables[:,0];th=tables[:,3]-tables[:,1]
    anchors=np.vstack([np.c_[(tables[:,0]+tables[:,2])/2,tables[:,1]-.45*th],np.c_[(tables[:,0]+tables[:,2])/2,tables[:,3]+.1*th]])
    scales=np.vstack([np.c_[tw,th],np.c_[tw,th]])
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=np.maximum(b[2:]-b[:2],1)
        hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy();cc=(hh[:,:2]+hh[:,2:])/2
        inside=((cc>=b[:2])&(cc<=b[2:])).all(1)
        f=dict(id=r.id,head_count=int(inside.sum()),eligible=int(inside.sum()==1),edge=int(b[0]<5 or b[2]>1275 or b[1]<5 or b[3]>715),aspect=np.log(wh[0]/wh[1]),area=np.log(wh.prod()))
        if inside.sum()!=1:outputs.append(f);continue
        d,hc,hw=descriptor(r.frame_id,int(np.flatnonzero(inside)[0]));f.update(d)
        normalized=(b-np.tile(hc,2))/np.tile(hw,2)
        for c,v in zip(B,normalized):f['query_'+c]=v
        model=mode(fit,r.frame_id if lofo else '')
        p=dict(zip(model.classes_,model.predict_proba(pd.DataFrame([d],columns=bp.columns))[0]))
        for name in MODES:f['mode_'+name]=p.get(name,0.)
        f['mode_entropy']=-sum(v*np.log(v+1e-9) for v in p.values())
        seat_dist=np.linalg.norm((anchors-hc)/scales,axis=1)
        for j,v in enumerate(seat_dist):f[f'seat_{j}']=min(v,6.)
        table_overlap=iou(b,tables)[0];f['max_table_iou']=table_overlap.max()
        for k in range(5,17):
            xy=hc+np.array([d[f'kp{k}_dx'],d[f'kp{k}_dy']])*hw;conf=d[f'kp{k}_conf']
            z=(xy-b[:2])/wh
            for name,v in zip(['x','y'],z):f[f'kp{k}_query_{name}']=float(np.clip(v,-3,4)) if conf>=.3 else 0.
            f[f'kp{k}_outside']=float(np.maximum(-z,0).sum()+np.maximum(z-1,0).sum())*conf
        f['upper_outside']=sum(f[f'kp{k}_outside'] for k in range(5,11))/6
        f['lower_outside']=sum(f[f'kp{k}_outside'] for k in range(11,17))/6
        template=bank[bank.day.eq(fit)&(bank.frame_id.ne(r.frame_id) if lofo else True)]
        dist=np.linalg.norm((template[['head_x','head_y']].to_numpy()-hc)/[100,90],axis=1)+.5*abs(np.log(template.head_w.to_numpy()/hw[0]))
        w=np.exp(-dist**2/(2*.95**2))*np.array([p.get(v,0.) for v in template['mode']])
        ix=np.argsort(w)[-7:];ix=ix[(w[ix]>.015)&(dist[ix]<2.5)]
        f['support']=len(ix)
        if len(ix)>=2:
            ss=template.iloc[ix];w=w[ix]/w[ix].sum();scale=np.clip(hw[0]/ss.head_w.to_numpy(),.7,1.4)
            ref=(ss[B].to_numpy()-np.tile(ss[['head_x','head_y']].to_numpy(),(1,2)))*scale[:,None]+np.tile(hc,(len(ix),2))
            ov=iou(b,ref)[0];heights=ref[:,3]-ref[:,1]
            f.update(template_iou_mean=float(w@ov),template_iou_max=float(ov.max()),template_valid=float(w@(ov>=.6)),template_height_ratio=float(wh[1]/(w@heights)),template_bottom_gap=float(w@((ref[:,3]-b[3])/heights)),template_top_gap=float(w@((b[1]-ref[:,1])/heights)),template_width_ratio=float(wh[0]/(w@(ref[:,2]-ref[:,0]))))
        outputs.append(f)
    return pd.DataFrame(outputs).fillna(0.)

def training(fit):
    rng=np.random.default_rng(2201);out=[]
    for i,r in h[h.class_name.eq('person')&h.day.eq(fit)].iterrows():
        b=r[B].to_numpy(float);wh=b[2:]-b[:2];variants=[('gt',b)]
        # Dense, realistic small border changes retain many correct short boxes.
        for side in range(4):
            for delta in [-.3,-.15,.15,.25,.35,.45,.6]:
                bb=b.copy();bb[side]+=delta*wh[side%2];variants.append(('border',bb))
        for j in range(36):variants.append(('random',b+rng.uniform(-.35,.35,4)*np.tile(wh,2)))
        for rr in hg[r.frame_id][B].to_numpy():
            if np.allclose(rr,b):continue
            variants.append(('merged',np.r_[np.minimum(b[:2],rr[:2]),np.maximum(b[2:],rr[2:])]))
        peers=hg[r.frame_id][B].to_numpy()
        for j,(kind,bb) in enumerate(variants):
            bb=np.clip(bb,0,[1280,720,1280,720])
            if min(bb[2:]-bb[:2])<20:continue
            target=float(iou(bb,peers).max());weight=2. if .6<=target<=.85 else 1.
            out.append(dict(id=f'person22_{i}_{j}',frame_id=r.frame_id,class_name='person',day=fit,error=int(target<.6),target_iou=target,kind=kind,weight=weight,**dict(zip(B,bb))))
    real=q[q.class_name.eq('person')&q.day.eq(fit)].copy();real['kind']='real';real['weight']=5.
    real['target_iou']=[float(iou(r[B].to_numpy(float),hg[r.frame_id][B]).max()) if r.frame_id in hg else 0 for _,r in real.iterrows()]
    return pd.concat([pd.DataFrame(out),real],ignore_index=True)

if __name__=='__main__':
    folds={};ledger=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        tr=training(fit);x=features(tr,fit,True);v=features(q[q.day.eq(held)],fit);t=features(te,fit)
        columns=sorted(set(x)-{'id'});x=x.reindex(columns=columns,fill_value=0);v=v.reindex(columns=columns,fill_value=0);t=t.reindex(columns=columns,fill_value=0)
        tr.to_csv(A/f'person_boundary_training_{fit}.csv',index=False)
        for name,rows,f in [('train',tr,x),('held',q[q.day.eq(held)],v),('test',te,t)]:f.assign(id=rows.id.to_numpy()).to_csv(A/f'person_boundary_features_{name}_{fit}.csv',index=False)
        folds[fit]=(tr,x,v,t,columns)
        ledger.append(dict(fit=fit,held=held,records=len(tr),frames=tr.frame_id.nunique(),borderline_correct=int(tr.target_iou.between(.6,.85).sum())))
        print('PERSON_FEATURES',ledger[-1],flush=True)
    pd.DataFrame(ledger).to_csv(A/'person_boundary_training_ledger.csv',index=False)
    for family in ['cb3','cb4','logreg']:
        so=np.zeros(len(q));pts=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            tr,x,v,t,columns=folds[fit];eligible=x.eligible.eq(1)
            if family=='logreg':
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=2000,class_weight='balanced'))
                model.fit(x[eligible],tr.loc[eligible,'error'],logisticregression__sample_weight=tr.loc[eligible,'weight'])
            else:
                model=CatBoostClassifier(iterations=450,depth=int(family[-1]),learning_rate=.025,l2_leaf_reg=30,verbose=False,thread_count=2,random_seed=2201,allow_writing_files=False)
                model.fit(x[eligible],tr.loc[eligible,'error'],sample_weight=tr.loc[eligible,'weight'])
            so[q.day.eq(held)]=model.predict_proba(v)[:,1];pts.append(model.predict_proba(t)[:,1]);joblib.dump(dict(model=model,columns=columns),A/'models'/f'person_boundary_{family}_{fit}.joblib')
        pt=np.mean(pts,axis=0)
        pd.DataFrame(dict(id=q.id,probability=so)).to_csv(A/f'person_boundary_{family}_oof.csv',index=False)
        pd.DataFrame(dict(id=te.id,probability=pt)).to_csv(A/f'person_boundary_{family}_test.csv',index=False)
        fo=pd.concat([pd.read_csv(A/f'person_boundary_features_held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id]
        tt=[pd.read_csv(A/f'person_boundary_features_test_{fit}.csv').set_index('id').loc[te.id] for fit in DAYS];ft=(tt[0]+tt[1])/2
        for rule in ['all','guarded','penalty']:
            for weight in [.25,.5]:
                out=[]
                for rows,base,pred,f,test in [(q,q.score.to_numpy(),so,fo,False),(te,current,pt,ft,True)]:
                    m=rows.class_name.eq('person').to_numpy()&f.eligible.eq(1).to_numpy()&f.edge.eq(0).to_numpy()
                    if rule!='all':
                        penalty=(pred>base)&(pred>.7)&f.support.ge(3).to_numpy()&f.template_valid.lt(.4).to_numpy()&f.template_height_ratio.lt(.85).to_numpy()
                        rescue=(pred<base)&(pred<.35)&f.upper_outside.lt(.04).to_numpy()&f.template_iou_max.ge(.65).to_numpy()
                        if test:
                            penalty &= (np.minimum(pts[0],pts[1])>.65)
                            rescue &= (np.maximum(pts[0],pts[1])<.45)
                        m &= penalty if rule=='penalty' else penalty|rescue
                    s=base.copy();s[m]=(1-weight)*base[m]+weight*pred[m];out.append(s)
                register(f'person_boundary_{family}_{rule}_w{weight}',*out,dict(family=family,weight=weight,guard=rule,training='dense boundary positives; fit-day-only; leave-frame-out modes and templates'))
