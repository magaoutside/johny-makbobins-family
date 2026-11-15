"""Explicit palm completeness classifier with nearby partial-hand hard negatives.

Labels are derived only from official fit-day manual boxes, never test audit.
"""
from common18 import *
from sklearn.linear_model import LogisticRegression
import joblib
rec=pd.read_csv(V16/'dino_records.csv');x=np.load(V16/'dino_features.npy').astype(np.float32)
labels=[];weights=[];coverage=[];purity=[]
for r in rec.itertuples():
    if r.class_name!='nude_hands' or r.kind=='query_test':labels.append(-1);weights.append(0.);coverage.append(0.);purity.append(0.);continue
    b=np.array([getattr(r,c) for c in B]);gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq('nude_hands')][B].to_numpy();area=np.prod(b[2:]-b[:2])
    inter=np.maximum(0,np.minimum(gt[:,2:],b[2:])-np.maximum(gt[:,:2],b[:2])).prod(1) if len(gt) else np.array([])
    gtarea=np.prod(gt[:,2:]-gt[:,:2],axis=1) if len(gt) else np.array([])
    cov=inter/np.maximum(gtarea,1) if len(gt) else np.array([]);j=int(cov.argmax()) if len(gt) else None
    c=float(cov[j]) if j is not None else 0.;p=float(inter[j]/area) if j is not None else 0.
    label=1 if c>=.8 and p>=.5 else (0 if c<=.3 else -1)
    labels.append(label);weights.append(1. if r.kind in ['gt','query_train'] else (.6 if r.kind=='hand_displaced' else .3));coverage.append(c);purity.append(p)
labels=np.array(labels);weights=np.array(weights);rec.assign(palm_complete_label=labels,gt_coverage=coverage,crop_purity=purity,fit_weight=weights).to_csv(A/'palm_training_ledger.csv',index=False)
for view in ['tight','both']:
    features=x[:,0] if view=='tight' else x.reshape(len(x),-1)
    oof=np.zeros(len(q));tests=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        tr=rec.day.eq(fit).to_numpy()&(labels>=0);va=rec.kind.eq('query_train')&rec.day.eq(held)&rec.class_name.eq('nude_hands');tt=rec.kind.eq('query_test')&rec.class_name.eq('nude_hands')
        model=LogisticRegression(C=10,max_iter=3000,class_weight='balanced');model.fit(features[tr],labels[tr],sample_weight=weights[tr])
        m=q.class_name.eq('nude_hands')&q.day.eq(held);oof[m]=pd.Series(model.predict_proba(features[va])[:,1],index=rec.loc[va,'id']).loc[q.loc[m,'id']]
        tests.append(pd.Series(model.predict_proba(features[tt])[:,1],index=rec.loc[tt,'id']).reindex(te.id).fillna(0).to_numpy());joblib.dump(model,A/f'palm_{view}_{fit}.joblib')
    test=np.mean(tests,axis=0);pd.DataFrame({'id':q.id,'p_complete_palm':oof}).to_csv(A/f'palm_{view}_oof.csv',index=False);pd.DataFrame({'id':te.id,'p_complete_palm':test}).to_csv(A/f'palm_{view}_test.csv',index=False)
    for mode in ['penalty','rescue']:
        for weight in [.25,.5]:
            outputs=[]
            for rows,base,pred,testflag in [(q,q.score.to_numpy(),oof,False),(te,current,test,True)]:
                split='test' if testflag else 'oof';wf=align(V16/f'wevis_features_{split}.csv',rows);cf=align(V16/f'calibrated_hand_{split}.csv',rows)
                s=base.copy();hand=rows.class_name.eq('nude_hands').to_numpy()
                if mode=='penalty':
                    m=hand&(pred<.35)&(wf.hand_iou.to_numpy()<.3);strength=np.clip((.5-pred)/.5,0,1);s+=weight*(1-s)*strength*m
                else:
                    m=hand&(pred>=.75)&(cf.calibrated_hand_iou.to_numpy()>=.55)&cf.assigned.to_numpy(bool)&~duplicate_veto(rows,testflag)
                    # Appearance establishes palm presence; localizer geometry must
                    # separately support the query boundary.
                    geometry=np.clip((.8-cf.calibrated_hand_iou.to_numpy())/.5,0,1);target=np.maximum(1-pred,geometry)
                    s[m]-=weight*np.maximum(s[m]-target[m],0)
                outputs.append(s)
            register(f'palm_{view}_{mode}_w{weight}',*outputs,details=dict(view=view,mode=mode,weight=weight,positive_gt_coverage=.8,negative_gt_coverage=.3,training='fit-day manual GT-derived crop labels',new_test_labels_used=False))
