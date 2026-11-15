"""DINO mode labels are learned from visually reviewed manual TRAIN people only."""
from person_modes import *
X=np.load(V16/'dino_features.npy').astype(np.float32).reshape(len(records),-1)
gt=records[records.kind.eq('gt')&records.class_name.eq('person')];mode_labels={}
for r in bank.itertuples():
    g=gt[gt.frame_id.eq(r.frame_id)];err=((g[B].to_numpy()-[r.x1,r.y1,r.x2,r.y2])**2).sum(1);i=g.index[int(err.argmin())]
    assert err.min()<.01;mode_labels[i]=r.mode
lookup={v:i for i,v in enumerate(records.id)}
def probabilities(rows,fit,leave_frame_out=False):
    indices=np.array([lookup[i] for i in rows.id]);out={}
    groups=rows.groupby('frame_id') if leave_frame_out else [('',rows)]
    for fid,g in groups:
        tr=[i for i in mode_labels if records.loc[i,'day']==fit and records.loc[i,'frame_id']!=fid]
        model=LogisticRegression(C=10,max_iter=1000,class_weight='balanced').fit(X[tr],[mode_labels[i] for i in tr])
        p=model.predict_proba(X[[lookup[i] for i in g.id]])
        for j,id_ in enumerate(g.id):out[id_]=dict(zip(model.classes_,p[j]))
        if not leave_frame_out:joblib.dump(model,A/'models'/f'dino_mode_{fit}.joblib')
    return out
fold={};accuracy=[]
for fit,held in zip(DAYS,DAYS[::-1]):
    tr=records[records.day.eq(fit)&records.class_name.eq('person')&~records.kind.eq('query_test')].reset_index(drop=True)
    ptrain=probabilities(tr,fit,True);pheld=probabilities(q[q.day.eq(held)],fit);ptest=probabilities(te,fit)
    x=extract(tr,fit,True,ptrain);v=extract(q[q.day.eq(held)],fit,False,pheld);t=extract(te,fit,False,ptest)
    for name,d in [('train',x),('held',v),('test',t)]:d.to_csv(A/f'person_dino_mode_{name}_{fit}.csv',index=False)
    fold[fit]=(tr,x,v,t)
    gm=joblib.load(A/'models'/f'dino_mode_{fit}.joblib');ix=[i for i in mode_labels if records.loc[i,'day']==held]
    pred=gm.predict(X[ix]);accuracy.append(dict(fit=fit,held=held,n=len(ix),accuracy=float(accuracy_score([mode_labels[i] for i in ix],pred)),confusion=confusion_matrix([mode_labels[i] for i in ix],pred,labels=MODES).tolist()))
(A/'person_dino_mode_accuracy.json').write_text(json.dumps(accuracy,indent=2))
fo=pd.concat([v[2] for v in fold.values()]).set_index('id').loc[q.id].reset_index();tt=[fold[fit][3].set_index('id').loc[te.id] for fit in DAYS];ft=(tt[0]+tt[1])/2
for family in ['logreg','catboost']:
    so=np.zeros(len(q));testp=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        tr,x,v,t=fold[fit];cols=[c for c in x if c!='id'];m=x.eligible.eq(1)&x.support.ge(2)&tr.kind.ne('background');w=tr.weight.to_numpy().copy();w[tr.kind.eq('query_train')]*=2
        if family=='logreg':
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,class_weight='balanced'));model.fit(x.loc[m,cols],tr.loc[m,'error'],logisticregression__sample_weight=w[m])
        else:
            model=CatBoostClassifier(iterations=300,depth=3,learning_rate=.025,l2_leaf_reg=20,verbose=False,thread_count=2,random_seed=201,allow_writing_files=False);model.fit(x.loc[m,cols],tr.loc[m,'error'],sample_weight=w[m])
        so[q.day.eq(held)]=model.predict_proba(v[cols])[:,1];testp.append(model.predict_proba(t[cols])[:,1]);joblib.dump(model,A/'models'/f'person_dino_{family}_{fit}.joblib')
    pt=np.mean(testp,axis=0)
    pd.DataFrame(dict(id=q.id,probability=so)).to_csv(A/f'person_dino_modes_{family}_oof.csv',index=False);pd.DataFrame(dict(id=te.id,probability=pt)).to_csv(A/f'person_dino_modes_{family}_test.csv',index=False)
    for guard in [False,True]:
        for weight in [.15,.3]:
            outs=[]
            for rows,base,pred,f,test in [(q,q.score.to_numpy(),so,fo,False),(te,current,pt,ft,True)]:
                eligible=rows.class_name.eq('person').to_numpy()&f.eligible.eq(1).to_numpy()&f.support.ge(3).to_numpy()&f.edge.eq(0).to_numpy()
                if guard:
                    front=f.mode_front.ge(.6).to_numpy()
                    if test:front&=tt[0].mode_front.ge(.5).to_numpy()&tt[1].mode_front.ge(.5).to_numpy()
                    rescue=eligible&front&f.visible_outside.lt(.01).to_numpy()&(base>.6)&(pred<base)&f.position_distance.lt(1.5).to_numpy()
                    penalize=eligible&f.mode_back.ge(.6).to_numpy()&f.bottom_gap.gt(.18).to_numpy()&f.height_ratio.lt(.8).to_numpy()&f.top_gap.abs().lt(.1).to_numpy()&(pred>base)
                    m=rescue|penalize
                else:m=eligible&f.mode_confidence.ge(.6).to_numpy()
                s=base.copy();s[m]=(1-weight)*s[m]+weight*pred[m];outs.append(s)
            register(f'person_dino_modes_{family}_{"guarded" if guard else "all"}_w{weight}',*outs,dict(weight=weight,guarded=guard,mode_classifier='frozen DINOv3 + fit-day logistic classifier, 77 manually reviewed training modes; training records use leave-frame-out modes'))
