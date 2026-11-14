from common16 import *
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier
import joblib

records=pd.read_csv(V16/'dino_records.csv');features=np.load(V16/'dino_features.npy').astype(np.float32)
assert len(records)==len(features)
testidx=records.kind.eq('query_test');trainidx=records.kind.eq('query_train')
def align(pred,mask,ids):return pd.DataFrame(pred,index=records.loc[mask,'id']).loc[ids].to_numpy()
classes=CL+['background']
# Fitted heads always use just the opposite day, including PCA/scalers.
for view,X in [('tight',features[:,0]),('context',features[:,1]),('both',features.reshape(len(features),-1))]:
    oof=np.zeros((len(q),5));test=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        tr=records.day.eq(fit)&records.kind.isin(['gt','short_person','background']);va=trainidx&records.day.eq(held)
        model=LogisticRegression(C=10,max_iter=2000,class_weight='balanced')
        model.fit(X[tr],records.loc[tr,'semantic_class'],sample_weight=records.loc[tr,'weight'])
        order=[list(model.classes_).index(c) for c in classes]
        oof[q.day.eq(held)]=align(model.predict_proba(X[va])[:,order],va,q.loc[q.day.eq(held),'id'])
        test.append(align(model.predict_proba(X[testidx])[:,order],testidx,te.id))
        joblib.dump(model,V16/f'semantic_{view}_{fit}.joblib')
    st=np.mean(test,axis=0)
    pd.DataFrame(oof,columns=['p_'+c for c in classes]).assign(id=q.id).to_csv(V16/f'semantic_{view}_oof.csv',index=False)
    pd.DataFrame(st,columns=['p_'+c for c in classes]).assign(id=te.id).to_csv(V16/f'semantic_{view}_test.csv',index=False)
    for w in [.25,.5]:
        s=q.score.to_numpy().copy();t=current.copy();j=classes.index('person');bg=classes.index('background')
        sig=np.clip((oof[:,bg]-oof[:,j]-.1)/.7,0,1)*q.class_name.eq('person').to_numpy()
        sigt=np.clip((st[:,bg]-st[:,j]-.1)/.7,0,1)*te.class_name.eq('person').to_numpy()
        register(f'semantic_{view}_person_{w}',s+w*(1-s)*sig,t+w*(1-t)*sigt,dict(view=view,weight=w))

for c in ['person','nude_hands']:
    selected=records.class_name.eq(c)&~testidx
    for variant in ['linear_tight','linear_both','catboost_both']:
        X=features[:,0] if variant=='linear_tight' else features.reshape(len(features),-1)
        oof=np.zeros(len(q));tp=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            tr=selected&records.day.eq(fit);va=trainidx&records.day.eq(held)&records.class_name.eq(c);tt=testidx&records.class_name.eq(c)
            if variant.startswith('linear'):
                model=LogisticRegression(C=1,max_iter=3000,class_weight='balanced')
                model.fit(X[tr],records.loc[tr,'error'].astype(int),sample_weight=records.loc[tr,'weight'])
                pv=model.predict_proba(X[va])[:,1];pt=model.predict_proba(X[tt])[:,1]
                fitted=model
            else:
                pca=PCA(n_components=48,random_state=17);xx=pca.fit_transform(X[tr]);model=CatBoostClassifier(iterations=300,depth=3,learning_rate=.025,l2_leaf_reg=10,random_seed=17,verbose=False,allow_writing_files=False,thread_count=2)
                # Add relative and absolute geometry, identical for train/validation/test.
                b=records[B].to_numpy();wh=np.maximum(b[:,2:]-b[:,:2],1);geo=np.c_[b/np.array([1280,720,1280,720]),np.log(wh),np.log(wh[:,0]/wh[:,1])]
                model.fit(np.c_[xx,geo[tr]],records.loc[tr,'error'].astype(int),sample_weight=records.loc[tr,'weight'])
                pv=model.predict_proba(np.c_[pca.transform(X[va]),geo[va]])[:,1];pt=model.predict_proba(np.c_[pca.transform(X[tt]),geo[tt]])[:,1];fitted=(pca,model)
            mask=q.day.eq(held)&q.class_name.eq(c);oof[mask]=align(pv,va,q.loc[mask,'id']).ravel();tp.append(align(pt,tt,te.loc[te.class_name.eq(c),'id']).ravel())
            joblib.dump(fitted,V16/f'quality_{c}_{variant}_{fit}.joblib')
        pt=np.mean(tp,axis=0);tall=np.zeros(len(te));tall[te.class_name.eq(c)]=pt
        pd.DataFrame({'id':q.id,'quality':oof}).to_csv(V16/f'quality_{c}_{variant}_oof.csv',index=False)
        pd.DataFrame({'id':te.id,'quality':tall}).to_csv(V16/f'quality_{c}_{variant}_test.csv',index=False)
        for w in [.15,.3]:
            s=q.score.to_numpy().copy();t=current.copy();m=q.class_name.eq(c);mt=te.class_name.eq(c)
            s[m]=(1-w)*s[m]+w*oof[m];t[mt]=(1-w)*t[mt]+w*pt
            register(f'quality_{c}_{variant}_{w}',s,t,dict(class_name=c,variant=variant,weight=w,training='opposite day real queries + manual GT perturbations/background'))
