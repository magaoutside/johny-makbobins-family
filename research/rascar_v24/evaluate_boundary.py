from common24 import *
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import joblib

def run():
    tr=pd.read_csv(A/'inputs/boundary_training.csv')
    x=align(A/'features/boundary_train.csv',tr);xo=align(A/'features/boundary_oof.csv',q);xt=align(A/'features/boundary_test.csv',te)
    geo_o=pd.concat([pd.read_csv(ROOT/f'rascar_v23/features/held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id]
    geo_t=sum([align(ROOT/f'rascar_v23/features/test_{fit}.csv',te) for fit in DAYS])/2
    ledger=[]
    for family in ['border_cb','context_cb','context_trees']:
        cols=[c for c in x if family!='border_cb' or not c.startswith('hog_')]
        po=np.zeros(len(q));pts=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            pp=np.zeros(len(te))
            for cls in CL:
                m=tr.day.eq(fit)&tr.class_name.eq(cls);rows=tr[m];xx=x.loc[m.to_numpy(),cols];y=rows.error.to_numpy(int);w=rows.weight.to_numpy()
                assert not set(rows.frame_id)&set(q.loc[q.day.eq(held),'frame_id'])
                def make():
                    if family.endswith('trees'):return ExtraTreesClassifier(n_estimators=300,max_depth=12,min_samples_leaf=12,max_features=.6,n_jobs=2,random_state=2401)
                    return CatBoostClassifier(iterations=220,depth=3,learning_rate=.04,l2_leaf_reg=25,thread_count=2,verbose=False,random_seed=2401,allow_writing_files=False)
                raw=np.zeros(len(rows))
                for ii,jj in GroupKFold(3).split(xx,y,rows.frame_id):
                    model=make();model.fit(xx.iloc[ii],y[ii],sample_weight=w[ii]);raw[jj]=model.predict_proba(xx.iloc[jj])[:,1]
                logit=lambda p:np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1))[:,None]
                cal=LogisticRegression(C=.5,max_iter=1000);cal.fit(logit(raw),y,sample_weight=w)
                model=make();model.fit(xx,y,sample_weight=w)
                vm=q.day.eq(held)&q.class_name.eq(cls);tm=te.class_name.eq(cls)
                po[vm]=cal.predict_proba(logit(model.predict_proba(xo.loc[vm.to_numpy(),cols])[:,1]))[:,1]
                pp[tm]=cal.predict_proba(logit(model.predict_proba(xt.loc[tm.to_numpy(),cols])[:,1]))[:,1]
                joblib.dump(dict(model=model,calibrator=cal,columns=cols,fit_day=fit,training_frames=sorted(rows.frame_id.unique())),A/f'models/{family}_{cls}_{fit}.joblib',compress=3)
                ledger.append(dict(family=family,fit=fit,held=held,cls=cls,rows=len(rows),frames=rows.frame_id.nunique(),ap=AP(q.loc[vm,'error'],po[vm])))
                print('MODEL',ledger[-1],flush=True)
            pts.append(pp)
        pt=np.mean(pts,axis=0)
        for split,rows,p in [('oof',q,po),('test',te,pt)]:
            d=pd.DataFrame(dict(id=rows.id,probability=p))
            if split=='test':d['p25']=pts[0];d['p26']=pts[1]
            d.to_csv(A/f'features/{family}_{split}.csv',index=False)
        for cls in CL:
            for direction in ['rescue','penalty','both']:
                for weight in [.25,.5]:
                    scores=[]
                    for split,rows,base,p,g in [('oof',q,base_oof,po,geo_o),('test',te,current,pt,geo_t)]:
                        lo=(p<base)&(p<.4)&guard(split,rows,cls);hi=(p>base)&(p>.8)
                        if cls=='person':
                            lo&=g.head_count.eq(1).to_numpy()&g.body_outside_fraction.lt(.05).to_numpy()
                            hi&=g.head_count.ge(1).to_numpy()
                        if split=='test':lo&=np.maximum(*pts)<.5;hi&=np.minimum(*pts)>.7
                        gate=rows.class_name.eq(cls).to_numpy() & (lo if direction=='rescue' else hi if direction=='penalty' else lo|hi)
                        score=base.copy();score[gate]+=weight*(p[gate]-base[gate]);scores.append(score)
                    register(f'{family}_{cls}_{direction}_w{weight}',*scores,dict(type='image_boundary_quality',family=family,class_name=cls,direction=direction,weight=weight))
    pd.DataFrame(ledger).to_csv(A/'results/boundary_training_ledger.csv',index=False)

if __name__=='__main__':run()
