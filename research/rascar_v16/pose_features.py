from common16 import *
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import joblib

poses=pd.read_csv(V16/'multiview_pose.csv')
def features(rows):
    result=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2]
        rec=dict(id=r.id,pose_best_iou=0.,pose_head_outside=0.,pose_two_heads=0.,pose_body_outside=0.,pose_wrist_inside=0.,pose_forearm=0.,pose_confirmed_views=0.)
        pool=poses[(poses.frame_id==r.frame_id)&(poses.confidence>=.3)].copy()
        if pool.empty:result.append(rec);continue
        if r.class_name=='person':
            ov=iou(b,pool[B])[0];rec['pose_best_iou']=float(ov.max())
            rec['pose_confirmed_views']=int(pool.loc[ov>=.6,'view'].nunique())
            heads=[];outside=[];bodies=[]
            for j,rr in enumerate(pool.itertuples()):
                bb=np.array([getattr(rr,c) for c in B]);intersection=np.maximum(0,np.minimum(b[2:],bb[2:])-np.maximum(b[:2],bb[:2])).prod()
                qcover=intersection/np.prod(wh)
                k=np.array([[getattr(rr,f'kp{n}_x'),getattr(rr,f'kp{n}_y'),getattr(rr,f'kp{n}_conf')] for n in range(17)])
                valid=k[:5,2]>=.3
                if valid.sum()>=2:
                    head=np.average(k[:5,:2][valid],weights=k[:5,2][valid],axis=0)
                    inside=np.all(head>=b[:2]-4)&np.all(head<=b[2:]+4)
                    hc=float(k[:5,2][valid].mean())
                    if inside and ov[j]>=.1:heads.append((head,rr.view,hc,rr.confidence))
                    # Requires nearby, overlapping detected person, not a head elsewhere in scene.
                    if not inside and qcover>=.25 and ov[j]>=.1 and rr.confidence>=.45:
                        dist=np.linalg.norm((head-np.clip(head,b[:2],b[2:]))/wh)
                        if dist<=.8:outside.append((rr.view,hc))
                if ov[j]>=.3 and rr.confidence>=.4:
                    confident=k[5:,2]>=.55;out=(k[5:,0]<b[0]-8)|(k[5:,0]>b[2]+8)|(k[5:,1]<b[1]-8)|(k[5:,1]>b[3]+8)
                    bodies.append((rr.view,float(np.clip((confident&out).sum()/3,0,1))))
            # Require repetition from distinct views, preventing a single pose hallucination.
            if len(set(v for v,_ in outside))>=2:rec['pose_head_outside']=float(np.mean([p for _,p in outside]))
            byview={}
            for v,p in bodies:byview[v]=max(p,byview.get(v,0.))
            if len(byview)>=2:rec['pose_body_outside']=float(sorted(byview.values(),reverse=True)[1])
            clusters=[]
            for xy,view,hc,dc in sorted(heads,key=lambda z:z[2]*z[3],reverse=True):
                found=next((g for g in clusters if np.linalg.norm(xy-g['xy'])<max(10,min(wh)*.12)),None)
                if found is None:clusters.append(dict(xy=xy,views={view},score=hc*dc))
                else:found['views'].add(view)
            supported=[g for g in clusters if len(g['views'])>=2 and g['score']>=.15]
            rec['pose_two_heads']=float(len(supported)>=2)
        if r.class_name=='nude_hands':
            inside_views=set();forearm_views=set()
            for rr in pool.itertuples():
                if rr.confidence<.45:continue
                for e,w in [(7,9),(8,10)]:
                    wrist=np.array([getattr(rr,f'kp{w}_{c}') for c in ['x','y']]);elbow=np.array([getattr(rr,f'kp{e}_{c}') for c in ['x','y']])
                    if min(getattr(rr,f'kp{w}_conf'),getattr(rr,f'kp{e}_conf'))<.5:continue
                    if np.all(wrist>=b[:2]) and np.all(wrist<=b[2:]):inside_views.add(rr.view)
                    mid=(wrist+elbow)/2
                    if np.all(mid>=b[:2]) and np.all(mid<=b[2:]) and (np.any(wrist<b[:2]-5) or np.any(wrist>b[2:]+5)):forearm_views.add(rr.view)
            rec['pose_wrist_inside']=float(len(inside_views)>=2);rec['pose_forearm']=float(len(forearm_views)>=2)
        result.append(rec)
    return pd.DataFrame(result)

f,ft=features(q),features(te);f.to_csv(V16/'pose_features_oof.csv',index=False);ft.to_csv(V16/'pose_features_test.csv',index=False)
for mode,columns in [('head',['pose_head_outside']),('merged',['pose_two_heads']),('body',['pose_body_outside']),('structural',['pose_head_outside','pose_two_heads','pose_forearm'])]:
    sig=f[columns].max(axis=1).to_numpy();sigt=ft[columns].max(axis=1).to_numpy()
    for w in [.25,.5]:register(f'pose16_{mode}_{w}',q.score+w*(1-q.score)*sig,current+w*(1-current)*sigt,dict(weight=w,mode=mode,require_multiple_views=True))

raw=pd.read_csv(old.E/'stage1/raw_frozen_train_features.csv').set_index('id').loc[q.id].reset_index(drop=True)
rawt=pd.read_csv(old.E/'stage1/raw_frozen_test_features.csv').set_index('id').loc[te.id].reset_index(drop=True)
X=pd.concat([raw,f.drop(columns='id')],axis=1);XT=pd.concat([rawt,ft.drop(columns='id')],axis=1)
for name,ctor in [('logreg',lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=2000))),('catboost',lambda:CatBoostClassifier(iterations=250,depth=2,learning_rate=.03,l2_leaf_reg=10,random_seed=17,verbose=False,thread_count=2,allow_writing_files=False))]:
    oof=np.zeros(len(q));tt=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        model=ctor();m=q.day.eq(fit);v=q.day.eq(held);model.fit(X[m],q.loc[m,'error']);oof[v]=model.predict_proba(X[v])[:,1];tt.append(model.predict_proba(XT)[:,1]);joblib.dump(model,V16/f'pose16_{name}_{fit}.joblib')
    tt=np.mean(tt,axis=0)
    for w in [.15,.3]:
        s=q.score.to_numpy().copy();t=current.copy();m=q.class_name.eq('person');mt=te.class_name.eq('person');s[m]=(1-w)*s[m]+w*oof[m];t[mt]=(1-w)*t[mt]+w*tt[mt]
        register(f'pose16_ranker_{name}_{w}',s,t,dict(weight=w,model=name))
