"""Learn which box is better for the SAME object, cancelling object identity.

Frozen DINOv3 features are reused. Pairwise examples are derived solely from
manual annotations and their perturbations, separately in each training day.
Calibration uses held-out frames inside that day. Test audit is never a target.
"""
from common27 import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from catboost import CatBoostRanker
import joblib

records=pd.read_csv(ROOT/'rascar_v16/dino_records.csv')
raw=np.load(ROOT/'rascar_v16/dino_features.npy').astype('float32')
# Tight crop, context, and their difference encode boundary changes.
X=np.concatenate([raw[:,0],raw[:,1],raw[:,0]-raw[:,1]],axis=1)/np.sqrt(3)
b=records[B].to_numpy();wh=np.maximum(b[:,2:]-b[:,:2],1)
G=np.c_[np.log(wh[:,0]/wh[:,1]),np.log(wh[:,0]*wh[:,1])/10,b/[1280,720,1280,720]].astype('float32')

def pairs(indices):
    df=records.loc[indices];out=[];rng=np.random.default_rng(2702)
    for origin,g in df.groupby('origin_id'):
        good=g[g.error.eq(0)].index.to_numpy();bad=g[g.error.eq(1)].index.to_numpy()
        p=np.array([(a,b) for a in bad for b in good],dtype=int).reshape(-1,2)
        if len(p)>60:p=p[rng.choice(len(p),60,replace=False)]
        out.extend(p.tolist())
    return np.asarray(out,dtype=int).reshape(-1,2)

def fit_model(indices,kind):
    if kind.startswith('pair'):
        p=pairs(indices)
        xx=np.c_[X,G*.3] if kind=='pair_geo' else X
        diff=xx[p[:,0]]-xx[p[:,1]]
        # Symmetry fixes the intercept at zero; weights equalize source objects.
        origins=records.loc[p[:,0],'origin_id'].to_numpy();_,inv,cnt=np.unique(origins,return_inverse=True,return_counts=True)
        weight=1/cnt[inv];weight*=len(weight)/weight.sum()
        clf=LogisticRegression(C=.5,fit_intercept=False,max_iter=800,solver='lbfgs')
        clf.fit(np.r_[diff,-diff],np.r_[np.ones(len(p)),np.zeros(len(p))],sample_weight=np.r_[weight,weight])
        return dict(model=clf,kind=kind,pairs=len(p))
    # PCA is refit for EVERY inner split. No held-day or test embedding enters it.
    pca=PCA(n_components=min(32,len(indices)-1),svd_solver='randomized',random_state=2702).fit(X[indices])
    xx=np.c_[pca.transform(X[indices]),G[indices]]
    # Same object is a ranking group. Background/real singleton groups add no pairs.
    origins=records.loc[indices,'origin_id'].to_numpy();order=np.argsort(origins,kind='stable')
    y=records.loc[indices,'error'].to_numpy(int)
    model=CatBoostRanker(iterations=180,depth=3,learning_rate=.035,l2_leaf_reg=20,
        loss_function='PairLogitPairwise',random_seed=2702,verbose=False,thread_count=2,allow_writing_files=False)
    model.fit(xx[order],y[order],group_id=origins[order])
    return dict(model=model,pca=pca,kind=kind,pairs=len(pairs(indices)))

def predict(bundle,indices):
    if bundle['kind'].startswith('pair'):
        xx=np.c_[X[indices],G[indices]*.3] if bundle['kind']=='pair_geo' else X[indices]
        return bundle['model'].decision_function(xx)
    return bundle['model'].predict(np.c_[bundle['pca'].transform(X[indices]),G[indices]])

def main():
    ledger=[]
    for kind in (sys.argv[1:] or ['pair_visual','pair_geo','rank_cb']):
        oo=np.zeros(len(q));tt=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            tp=np.zeros(len(te))
            for cls in CL:
                train=records.index[records.day.eq(fit)&records.class_name.eq(cls)&~records.kind.isin(['query_test','background','query_train'])].to_numpy()
                # Query-only rows participate in calibration if their frame is held
                # out internally, never as pairs or as the other-day fit target.
                calibration=records.index[records.day.eq(fit)&records.class_name.eq(cls)&~records.kind.eq('query_test')].to_numpy()
                frames=np.unique(records.loc[calibration,'frame_id']);pred=np.full(len(calibration),np.nan)
                splitter=GroupKFold(min(3,len(frames)))
                for _,vi in splitter.split(calibration,groups=records.loc[calibration,'frame_id']):
                    heldframes=set(records.loc[calibration[vi],'frame_id']);tr=train[~records.loc[train,'frame_id'].isin(heldframes)]
                    fitted=fit_model(tr,kind);pred[vi]=predict(fitted,calibration[vi])
                y=records.loc[calibration,'error'].to_numpy(int)
                weights=records.loc[calibration,'weight'].to_numpy(float)
                cal=LogisticRegression(C=.5,max_iter=800).fit(pred[:,None],y,sample_weight=weights)
                fitted=fit_model(train,kind);fitted.update(calibrator=cal,training_frames=sorted(records.loc[train,'frame_id'].unique()),fit_day=fit)
                joblib.dump(fitted,A/f'models/contrastive_{kind}_{cls}_{fit}.joblib',compress=3)
                def apply(rows,kindrow):
                    ix=records[records.kind.eq(kindrow)].set_index('id').loc[rows.id].index
                    inds=records.reset_index().set_index('id').loc[ix,'index'].to_numpy()
                    return cal.predict_proba(predict(fitted,inds)[:,None])[:,1]
                vm=q.day.eq(held)&q.class_name.eq(cls);tm=te.class_name.eq(cls)
                oo[vm]=apply(q[vm],'query_train');tp[tm]=apply(te[tm],'query_test')
                log=dict(kind=kind,fit=fit,held=held,cls=cls,frames=len(fitted['training_frames']),pairs=fitted['pairs'],
                    calibration_frames=len(frames),inner_ap=float(AP(y,pred)),held_ap=float(AP(q.loc[vm,'error'],oo[vm])))
                ledger.append(log);print('TRAIN',log,flush=True)
            tt.append(tp)
        mean=np.mean(tt,axis=0)
        pd.DataFrame(dict(id=q.id,probability=oo)).to_csv(A/f'features/contrastive_{kind}_oof.csv',index=False)
        pd.DataFrame(dict(id=te.id,probability=mean,p25=tt[0],p26=tt[1])).to_csv(A/f'features/contrastive_{kind}_test.csv',index=False)
        for cls in CL:
            for direction in ['both','rescue','penalty']:
                for weight in [.25,.5]:
                    outputs=[]
                    for split,rows,base,p in [('oof',q,base_oof,oo),('test',te,current,mean)]:
                        low=(p<base)&(p<.4)&guard(split,rows,cls);high=(p>base)&(p>.7)
                        if split=='test':low&=np.maximum(*tt)<.5;high&=np.minimum(*tt)>.6
                        gate=rows.class_name.eq(cls).to_numpy()&(low if direction=='rescue' else high if direction=='penalty' else low|high)
                        outputs.append(np.clip(base+weight*gate*(p-base),0,1))
                    register(f'contrastive_{kind}_{cls}_{direction}_w{weight}',*outputs,
                        dict(type='same_object_pairwise_quality',family=kind,class_name=cls,direction=direction,weight=weight,
                             validation='opposite day; frame-held inner calibration; frozen DINOv3; no test labels'))
    pd.DataFrame(ledger).to_csv(A/'results/contrastive_training.csv',index=False)
if __name__=='__main__':main()
