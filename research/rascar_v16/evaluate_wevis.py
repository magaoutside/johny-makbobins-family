from common16 import *
import re
from scipy.optimize import linear_sum_assignment
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import joblib

def parse(task):
    boxes=[];coverage=[]
    for p in sorted((V16/f'wevis_{task}_cache').glob('*.json')):
        r=json.loads(p.read_text());parts=re.findall(r'<box>(.*?)</box>',r['response'],flags=re.S)
        n=0;valid=True
        for part in parts:
            values=[int(v) for v in re.findall(r'<(\d{1,4})>',part)]
            if len(values)%4:valid=False;continue
            for start in range(0,len(values),4):
                b=np.array(values[start:start+4],float)
                if (b<0).any() or (b>1000).any() or min(b[2:]-b[:2])<=0:valid=False;continue
                w,hh=r['crop_size'];b=b/1000*np.array([w,hh,w,hh])+np.tile(r['roi'][:2],2)
                boxes.append(dict(query_id=r['id'],frame_id=r['frame_id'],reference_id=f'{task}_{r["id"]}_{n}',**dict(zip(B,b))));n+=1
        coverage.append(dict(id=r['id'],frame_id=r['frame_id'],boxes=n,parsed=valid,truncated=r['new_tokens']>=384))
    pd.DataFrame(coverage).to_csv(V16/f'wevis_{task}_coverage.csv',index=False)
    df=pd.DataFrame(boxes);df.to_csv(V16/f'wevis_{task}_references.csv',index=False);return df
heads,hands=parse('heads'),parse('hands')
pose=pd.read_csv(V16/'multiview_pose.csv')
def extract(rows):
    out=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2]
        rec=dict(id=r.id,head_count=0.,head_near_outside=0.,head_iou=0.,head_query_cover=0.,hand_iou=0.,hand_area_ratio=0.,hand_query_coverage=0.,hand_instances=0.,head_cut_distance=0.)
        hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy()
        if len(hh):
            center=(hh[:,:2]+hh[:,2:])/2;inside=((center>=b[:2])&(center<=b[2:])).all(1)
            rec['head_count']=float(inside.sum());rec['head_iou']=float(iou(b,hh).max())
            inter=np.maximum(0,np.minimum(hh[:,2:],b[2:])-np.maximum(hh[:,:2],b[:2])).prod(1)
            rec['head_query_cover']=float((inter/max(wh.prod(),1)).max())
            pp=pose[(pose.frame_id==r.frame_id)&(pose.confidence>=.3)]
            body_iou=float(iou(b,pp[B]).max()) if len(pp) else 0.
            # Heads must lie nearby above the box; abstain if a good full-body match exists.
            dx=np.abs(center[:,0]-(b[0]+b[2])/2)/wh[0];dy=(b[1]-center[:,1])/wh[1]
            near=(dx<.5)&(dy>.08)&(dy<.8)
            if not inside.any() and near.any() and .05<body_iou<.55:
                rec['head_near_outside']=1.;rec['head_cut_distance']=float(dy[near].min())
        rr=hands[hands.query_id.eq(r.id)][B].to_numpy()
        if len(rr):
            ov=iou(b,rr)[0];j=int(ov.argmax());rec['hand_iou']=float(ov[j])
            rec['hand_area_ratio']=float(wh.prod()/max(np.prod(rr[j,2:]-rr[j,:2]),1))
            inter=np.maximum(0,np.minimum(rr[:,2:],b[2:])-np.maximum(rr[:,:2],b[:2])).prod(1)
            rec['hand_query_coverage']=float((inter/max(wh.prod(),1)).max())
            cc=(rr[:,:2]+rr[:,2:])/2;rec['hand_instances']=float(((cc>=b[:2])&(cc<=b[2:])).all(1).sum())
        out.append(rec)
    return pd.DataFrame(out)
f,ft=extract(q),extract(te);f.to_csv(V16/'wevis_features_oof.csv',index=False);ft.to_csv(V16/'wevis_features_test.csv',index=False)

for signal in ['head_near_outside','two_heads','hand_is_head']:
    preds=[]
    for rows,base,feat,test in [(q,q.score.to_numpy(),f,False),(te,current,ft,True)]:
        if signal=='head_near_outside':sig=feat.head_near_outside.to_numpy()*rows.class_name.eq('person').to_numpy()
        elif signal=='two_heads':sig=(feat.head_count.to_numpy()>=2)*rows.class_name.eq('person').to_numpy()
        else:
            sem=pd.read_csv(V16/f'semantic_tight_{"test" if test else "oof"}.csv').set_index('id').loc[rows.id,'p_nude_hands'].to_numpy()
            sig=(feat.head_iou.to_numpy()>=.5)*(sem<.8)*rows.class_name.eq('nude_hands').to_numpy()
        preds.append(sig)
    for w in [.25,.5]:register(f'wevis_{signal}_{w}',q.score+w*(1-q.score)*preds[0],current+w*(1-current)*preds[1],dict(signal=signal,weight=w))

def duplicate_ids(test):
    led=pd.read_csv(V15/'hand_group_test.csv') if test else pd.concat([pd.read_csv(V15/f'hand_group_oof_{d}.csv') for d in DAYS])
    return set(led.loc[led.reason.eq('duplicate_loser'),'id'])
for mode in ['geometry','dino_and_geometry']:
    for w in [.2,.35]:
        preds=[];counts=[]
        for rows,base,feat,test in [(q,q.score.to_numpy(),f,False),(te,current,ft,True)]:
            sem=pd.read_csv(V16/f'semantic_tight_{"test" if test else "oof"}.csv').set_index('id').loc[rows.id,'p_nude_hands'].to_numpy()
            m=rows.class_name.eq('nude_hands').to_numpy()&(feat.hand_iou.to_numpy()>=.65)&(feat.hand_instances.to_numpy()<=1)&(sem>=.8)&~rows.id.isin(duplicate_ids(test)).to_numpy()
            target=np.clip((.8-feat.hand_iou.to_numpy())/.4,0,1)
            if mode=='dino_and_geometry':
                quality=pd.read_csv(V16/f'quality_nude_hands_linear_tight_{"test" if test else "oof"}.csv').set_index('id').loc[rows.id,'quality'].to_numpy()
                m&=(quality<.6);target=np.maximum(target,quality)
            s=base.copy();s[m]=(1-w)*s[m]+w*np.minimum(s[m],target[m]);preds.append(s);counts.append(int(m.sum()))
        register(f'wevis_hand_rescue_{mode}_{w}',*preds,details=dict(weight=w,mode=mode,changed_counts=counts))

# A meta-model only gets frozen model features; no cross-day fitted head outputs.
raw=pd.read_csv(old.E/'stage1/raw_frozen_train_features.csv').set_index('id').loc[q.id].reset_index(drop=True)
rawt=pd.read_csv(old.E/'stage1/raw_frozen_test_features.csv').set_index('id').loc[te.id].reset_index(drop=True)
X=pd.concat([raw,f.drop(columns='id')],axis=1);XT=pd.concat([rawt,ft.drop(columns='id')],axis=1)
for name,ctor in [('logreg',lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=2000))),('catboost',lambda:CatBoostClassifier(iterations=250,depth=2,learning_rate=.03,l2_leaf_reg=10,random_seed=17,verbose=False,thread_count=2,allow_writing_files=False))]:
    pred=np.zeros(len(q));pt=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        m=q.day.eq(fit);v=q.day.eq(held);model=ctor();model.fit(X[m],q.loc[m,'error']);pred[v]=model.predict_proba(X[v])[:,1];pt.append(model.predict_proba(XT)[:,1]);joblib.dump(model,V16/f'wevis_meta_{name}_{fit}.joblib')
    pt=np.mean(pt,axis=0)
    for w in [.15,.3]:
        s=q.score.to_numpy().copy();t=current.copy();m=q.class_name.isin(['person','nude_hands']);mt=te.class_name.isin(['person','nude_hands']);s[m]=(1-w)*s[m]+w*pred[m];t[mt]=(1-w)*t[mt]+w*pt[mt]
        register(f'wevis_ranker_{name}_{w}',s,t,dict(weight=w,model=name))
