"""Nonlinear DINO quality; strict outer day split, inner frame-group calibration.

No test audit labels enter fitting or calibration. GT perturbations from the
same frame always stay together in internal calibration folds.
"""
from common21 import *
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import rbf_kernel
import joblib

records=pd.read_csv(ROOT/'geometry_guards/dino_records.csv')
raw=np.load(ROOT/'geometry_guards/dino_features.npy').astype(np.float64)
F=raw.reshape(len(raw),-1);F/=np.linalg.norm(F,axis=1,keepdims=True)
lookup={v:i for i,v in enumerate(records.id)}
variants={'rbf2':(5.,2.),'rbf5':(10.,5.),'ridge2':(.1,2.)}
ledger=[]
for name,(regular,gamma) in variants.items():
    all_o=q[['id','class_name']].copy();all_o['probability']=0.;all_t=te[['id','class_name']].copy();all_t['probability']=0.
    for cls in CL:
        testpred=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            tr=records[records.class_name.eq(cls)&records.day.eq(fit)&records.kind.ne('query_test')].copy()
            ix=tr.index.to_numpy();y=tr.error.astype(int).to_numpy();w=tr.weight.to_numpy().copy()
            # Real queried boxes matter more than many near-identical synthetic
            # variants; every inner split is grouped by the original frame.
            w[tr.kind.eq('query_train')]*=2
            K=rbf_kernel(F[ix],F[ix],gamma=gamma)
            inner=np.zeros(len(tr))
            for a,b in GroupKFold(n_splits=3).split(ix,y,tr.frame_id):
                if name.startswith('ridge'):m=KernelRidge(alpha=regular,kernel='precomputed')
                else:m=SVC(C=regular,kernel='precomputed',class_weight='balanced',probability=False,cache_size=400)
                m.fit(K[np.ix_(a,a)],y[a],sample_weight=w[a])
                inner[b]=m.predict(K[np.ix_(b,a)]) if name.startswith('ridge') else m.decision_function(K[np.ix_(b,a)])
            calibrator=LogisticRegression(C=1,max_iter=1000)
            calibrator.fit(inner[:,None],y,sample_weight=w)
            if name.startswith('ridge'):model=KernelRidge(alpha=regular,kernel='precomputed')
            else:model=SVC(C=regular,kernel='precomputed',class_weight='balanced',probability=False,cache_size=400)
            model.fit(K,y,sample_weight=w)
            v=q[q.class_name.eq(cls)&q.day.eq(held)];t=te[te.class_name.eq(cls)]
            def predict(rows):
                xx=F[[lookup[i] for i in rows.id]];kk=rbf_kernel(xx,F[ix],gamma=gamma)
                z=model.predict(kk) if name.startswith('ridge') else model.decision_function(kk)
                return calibrator.predict_proba(z[:,None])[:,1]
            pv=predict(v);pt=predict(t);all_o.loc[v.index,'probability']=pv;testpred.append(pt)
            joblib.dump(dict(model=model,calibrator=calibrator,train_features=F[ix],gamma=gamma,ids=tr.id.tolist()),A/'models'/f'{name}_{cls}_{fit}.joblib',compress=3)
            ledger.append(dict(variant=name,class_name=cls,fit=fit,held=held,train_records=len(tr),train_frames=tr.frame_id.nunique(),calibration='3-fold GroupKFold by frame on fitting day only',held_ap=float(AP(v.error,pv))))
            print('FIT',name,cls,fit,round(AP(v.error,pv),5),flush=True)
        all_t.loc[all_t.class_name.eq(cls),'probability']=np.mean(testpred,axis=0)
    all_o.to_csv(A/f'{name}_oof.csv',index=False);all_t.to_csv(A/f'{name}_test.csv',index=False)
    for cls in CL:
        for weight in [.15,.3]:
            out=[]
            for rows,base,pred in [(q,q.score.to_numpy(),all_o.probability.to_numpy()),(te,current,all_t.probability.to_numpy())]:
                m=rows.class_name.eq(cls).to_numpy();s=base.copy();s[m]=(1-weight)*base[m]+weight*pred[m];out.append(s)
            register(f'{name}_{cls}_w{weight}',*out,dict(model=name,weight=weight,grouped_calibration=True,class_name=cls))
pd.DataFrame(ledger).to_csv(A/'kernel_training_ledger.csv',index=False)
