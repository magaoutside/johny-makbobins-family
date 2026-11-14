"""Hand extent quality from frozen appearance, local evidence and manual-box IoUs."""
from common17 import *
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
import joblib,cv2
from PIL import Image

records=pd.read_csv(V16/'dino_records.csv');emb=np.load(V16/'dino_features.npy').astype(np.float32)
chosen=records.class_name.eq('nude_hands');rec=records[chosen].copy().reset_index(drop=True);emb=emb[chosen]
gtvec=rec.kind.eq('gt').to_numpy()
images={fid:np.asarray(Image.open((V16/'test_images' if fid in set(te.frame_id) else V16/'images')/f'{fid}.jpg').convert('RGB')) for fid in rec.frame_id.unique()}
visual=[]
for r in rec.itertuples():
    b=np.rint([getattr(r,c) for c in B]).astype(int);im=images[r.frame_id];crop=im[max(b[1],0):min(b[3],720),max(b[0],0):min(b[2],1280)]
    crop=cv2.resize(crop,(64,64));lab=cv2.cvtColor(crop,cv2.COLOR_RGB2LAB).astype(float)/255;gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3);mag=np.sqrt(gx**2+gy**2)
    # Fixed spatial layout, including border strips: no test-label-dependent feature.
    values=[]
    for yi in range(4):
        for xi in range(4):
            a=lab[yi*16:(yi+1)*16,xi*16:(xi+1)*16];e=mag[yi*16:(yi+1)*16,xi*16:(xi+1)*16]
            values.extend(a.mean((0,1)));values.extend(a.std((0,1)));values.extend([e.mean(),e.std()])
    values.extend([mag[:5].mean(),mag[-5:].mean(),mag[:,:5].mean(),mag[:,-5:].mean(),mag[16:48,16:48].mean()])
    visual.append(values)
visual=np.asarray(visual);np.save(A/'hand_fixed_visual_features.npy',visual)
b=rec[B].to_numpy();wh=np.maximum(b[:,2:]-b[:,:2],1)
geo=np.c_[b/[1280,720,1280,720],np.log(wh),np.log(wh[:,0]/wh[:,1])]

target=rec.target_iou.to_numpy().copy()
for j,r in rec[rec.kind.eq('query_train')].iterrows():
    gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq('nude_hands')][B].to_numpy();target[j]=float(iou(r[B],gt).max()) if len(gt) else 0.
rec['regression_target_iou']=target;rec.to_csv(A/'hand_training_records.csv',index=False)

predictions={name:[np.zeros(len(q)),[]] for name in ['cat_iou','extra_iou','cat_error']}
retrieval_oof=np.zeros(len(q));retrieval_test=[]
for fit,held in zip(DAYS,DAYS[::-1]):
    tr=rec.day.eq(fit)&rec.kind.ne('query_test');va=rec.kind.eq('query_train')&rec.day.eq(held);tt=rec.kind.eq('query_test')
    # PCA only sees fit-day regions. Nearest manual-GT features are also fit-day only.
    pca=PCA(n_components=64,random_state=171);x=pca.fit_transform(emb[tr].reshape(tr.sum(),-1))
    allp=pca.transform(emb.reshape(len(emb),-1))
    prototypes=emb[gtvec&rec.day.eq(fit).to_numpy(),0];tight=emb[:,0]
    norms=np.linalg.norm(tight,axis=1)[:,None]*np.linalg.norm(prototypes,axis=1)[None]
    sim=tight@prototypes.T/np.maximum(norms,1e-8)
    # Leave-frame-out similarities for fitting, preventing self-GT identity leakage.
    pf=rec.loc[gtvec&rec.day.eq(fit).to_numpy(),'frame_id'].to_numpy()
    sim[tr.to_numpy()[:,None]&(rec.frame_id.to_numpy()[:,None]==pf[None])]=-1
    nearest=np.sort(sim,axis=1)[:,-3:];nearest_mean=nearest.mean(1)
    X=np.c_[allp,geo,visual,nearest]
    weights=rec.weight.to_numpy().copy();weights[rec.kind.eq('query_train')]*=2
    constructors={
        'cat_iou':lambda:CatBoostRegressor(iterations=450,depth=4,learning_rate=.03,l2_leaf_reg=12,random_seed=171,verbose=False,thread_count=2,allow_writing_files=False),
        'extra_iou':lambda:ExtraTreesRegressor(n_estimators=350,max_depth=12,min_samples_leaf=5,max_features=.7,random_state=171,n_jobs=2),
        'cat_error':lambda:CatBoostClassifier(iterations=400,depth=3,learning_rate=.03,l2_leaf_reg=15,random_seed=171,verbose=False,thread_count=2,allow_writing_files=False),
    }
    for name,ctor in constructors.items():
        model=ctor();y=rec.error if name=='cat_error' else target;model.fit(X[tr],np.asarray(y)[tr],sample_weight=weights[tr])
        pv=model.predict_proba(X[va])[:,1] if name=='cat_error' else model.predict(X[va])
        pt=model.predict_proba(X[tt])[:,1] if name=='cat_error' else model.predict(X[tt])
        mask=q.class_name.eq('nude_hands')&q.day.eq(held);predictions[name][0][mask]=pd.Series(pv,index=rec.loc[va,'id']).loc[q.loc[mask,'id']]
        predictions[name][1].append(pd.Series(pt,index=rec.loc[tt,'id']).reindex(te.id).fillna(0).to_numpy())
        joblib.dump(dict(pca=pca,model=model,prototype_vectors=prototypes,prototype_frames=pf),A/f'hand_{name}_{fit}.joblib')
    mask=q.class_name.eq('nude_hands')&q.day.eq(held);retrieval_oof[mask]=pd.Series(nearest_mean[va],index=rec.loc[va,'id']).loc[q.loc[mask,'id']]
    retrieval_test.append(pd.Series(nearest_mean[tt],index=rec.loc[tt,'id']).reindex(te.id).fillna(0).to_numpy())

for name,(oof,test) in predictions.items():
    pt=np.mean(test,axis=0)
    pd.DataFrame({'id':q.id,'prediction':oof,'manual_similarity':retrieval_oof}).to_csv(A/f'hand_{name}_oof.csv',index=False)
    pd.DataFrame({'id':te.id,'prediction':pt,'manual_similarity':np.mean(retrieval_test,axis=0)}).to_csv(A/f'hand_{name}_test.csv',index=False)
    for weight in [.25,.5]:
        outputs=[]
        for rows,base,pred,test_flag in [(q,q.score.to_numpy(),oof,False),(te,current,pt,True)]:
            split='test' if test_flag else 'oof'
            sem=align(V16/f'semantic_tight_{split}.csv',rows,'p_nude_hands')
            wf=align(V16/f'wevis_features_{split}.csv',rows)
            yf=align(V15/f'hand_specialist_{split}_evidence_0.35.csv',rows)
            group=pd.read_csv(V16/f'hand_rescue_group_guard_{split}.csv').set_index('id').reindex(rows.id)
            losers=set(pd.read_csv(V15/'hand_group_test.csv').query("reason=='duplicate_loser'").id) if test_flag else set(pd.concat([pd.read_csv(V15/f'hand_group_oof_{d}.csv') for d in DAYS]).query("reason=='duplicate_loser'").id)
            quality=pred if name=='cat_error' else np.clip((.8-pred)/.4,0,1)
            m=rows.class_name.eq('nude_hands').to_numpy()&(sem>=.8)&~rows.id.isin(losers).to_numpy()&~group.veto.fillna(False).to_numpy(bool)
            # Local support can be weak for tiny hands, but at least one independent
            # detector must overlap it. No ID-specific exceptions.
            m&=((wf.hand_iou.to_numpy()>=.2)|(yf.local_iou.to_numpy()>=.3))
            s=base.copy();s[m]=(1-weight)*s[m]+weight*np.minimum(s[m],quality[m]);outputs.append(s)
        register(f'hand_{name}_rescue_w{weight}',*outputs,details=dict(model=name,weight=weight,manual_gt_iou_target=name!='cat_error',synthetic_and_real_cross_day=True,duplicate_veto=True))

# Agreement across distinct regressors is tested separately, including cases
# where a tiny hand has no localized support but strongly matches manual GT.
for weight in [.4,.65]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        split='test' if test else 'oof';a=align(A/f'hand_cat_iou_{split}.csv',rows);b=align(A/f'hand_extra_iou_{split}.csv',rows);c=align(A/f'hand_cat_error_{split}.csv',rows)
        sem=align(V16/f'semantic_tight_{split}.csv',rows,'p_nude_hands');group=pd.read_csv(V16/f'hand_rescue_group_guard_{split}.csv').set_index('id').reindex(rows.id)
        loser_paths=[V15/'hand_group_test.csv'] if test else [V15/f'hand_group_oof_{d}.csv' for d in DAYS]
        losers=set(pd.concat([pd.read_csv(p) for p in loser_paths]).query("reason=='duplicate_loser'").id)
        consensus=np.minimum(a.prediction.to_numpy(),b.prediction.to_numpy())
        m=rows.class_name.eq('nude_hands').to_numpy()&(sem>=.9)&(consensus>=.6)&(c.prediction.to_numpy()<.4)&(a.manual_similarity.to_numpy()>=.75)&~rows.id.isin(losers).to_numpy()&~group.veto.fillna(False).to_numpy(bool)
        target=np.maximum(np.clip((.8-consensus)/.4,0,1),c.prediction.to_numpy());s=base.copy();s[m]=(1-weight)*s[m]+weight*np.minimum(s[m],target[m]);outputs.append(s)
    register(f'hand_consensus_w{weight}',*outputs,details=dict(weight=weight,minimum_regressed_iou=.6,maximum_classification_error=.4,minimum_manual_similarity=.75,semantic_min=.9))
