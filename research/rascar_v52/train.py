"""Cross-day feature-family ablations. All preprocessing is training-day only."""
from prepare import *
import argparse,joblib,time,warnings
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP
from catboost import CatBoostClassifier
sys.path.insert(0,str(ROOT/'rascar_v41'))
from common import baseline,metrics,DAYS
sys.path.pop(0)
EMB=['clip','vit_b','vit_l','dino_small','siglip2','vit_b448','vit_l448']
def configurations():
    conf={
      'base13':(['base13'],[]), 'base23':(['base23'],[]), 'frozen_all':(['frozen_all'],[]),
      'base23_owl':(['base23','owl'],[]), 'base23_text':(['base23','clip_text'],[]),
      'base23_pixels':(['base23','pixels'],[]), 'all_frozen':(['frozen_all','owl','clip_text'],[]),
      'all_frozen_pixels':(['frozen_all','owl','clip_text','pixels'],[]),
      'all_frozen_position':(['frozen_all','owl','clip_text','position'],[]),
      'nested_dense':(['nested'],[]), 'base23_nested':(['base23','nested'],[]),
      'all_frozen_nested':(['frozen_all','owl','clip_text','nested'],[]),
      'all_visual':(['frozen_all','owl','clip_text','pixels','nested'],[]),
      'pixels_only':(['pixels'],[]), 'semantic_prototypes':(['prototypes'],[]),
      'base23_prototypes':(['base23','prototypes'],[]),
      'all_frozen_prototypes':(['frozen_all','owl','clip_text','prototypes'],[]),
      'emb_all':([],EMB), 'base23_emb_all':(['base23'],EMB),
      'all_frozen_emb_all':(['frozen_all','owl','clip_text'],EMB),
      'all_visual_emb_all':(['frozen_all','owl','clip_text','pixels','nested','prototypes'],EMB),
      'clip_dino':(['base23'],['clip','vit_l448']),
      'siglip_dino':(['base23'],['siglip2','dino_small']),
    }
    for e in EMB:conf['base23_'+e]=(['base23'],[e]);conf['embedding_'+e]=([],[e])
    return conf
def load_bank(k=16,purged_prototypes=False):
    rows=pd.read_csv(A/'inputs/rows.csv');static=pd.read_csv(A/'features/static.csv').set_index('id').loc[rows.id].reset_index(drop=True);groups=json.loads((A/'inputs/groups.json').read_text());dense=pd.read_csv(ROOT/'rascar_v48/features/meta_features.csv');banks={};transforms={};proto=pd.read_csv(A/'inputs/prototypes.csv')
    for fit in DAYS:
        train=rows.split.eq('oof')&rows.day.eq(fit);bank=static.copy();dd=dense[dense.outer_fit_day.eq(fit)].set_index('id').loc[rows.id];cols=[c for c in dd if c.startswith(('siglip2_','small_control_','vit_b_','vit_l_'))];groups['nested']=['dense_'+c for c in cols]
        for c in cols:bank['dense_'+c]=dd[c].to_numpy()
        for suffix in ['iou','mean','coverage']:
            x=dd[[key+'_'+suffix for key in ['small_control','vit_b','vit_l','siglip2']]].to_numpy();bank['dense_'+suffix+'_spread']=np.ptp(x,axis=1);groups['nested'].append('dense_'+suffix+'_spread')
        groups['prototypes']=[]
        for e in ['clip','vit_b','vit_l']:
            x=np.load(A/f'features/{e}.npy');p=np.load(A/f'features/{e}_prototypes.npy');dim=x.shape[1]//3
            # ROI representation is the middle third, not a model output trained on errors.
            x=x[:,dim:2*dim];p=p[:,dim:2*dim];x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8);p=p/np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-8)
            own=[]
            for c in CL:
                use=proto.day.eq(fit)&proto.class_name.eq(c);sim=x@p[use].T
                valid=np.ones(sim.shape,bool)
                if purged_prototypes:
                    qt=pd.to_datetime(rows.frame_id.str.rsplit('_',n=2).str[-2:].str.join(''),format='%Y%m%d%H%M%S').astype('int64').to_numpy()/1e9
                    pt=pd.to_datetime(proto.loc[use,'frame_id'].str.rsplit('_',n=2).str[-2:].str.join(''),format='%Y%m%d%H%M%S').astype('int64').to_numpy()/1e9
                    valid[train]=abs(qt[train,None]-pt[None,:])>120
                vv=np.where(valid,sim,-10);top=np.sort(vv,axis=1)[:,-min(3,use.sum()):];v=np.maximum(top,0).sum(1)/np.maximum((top>-9).sum(1),1);mean=np.where(valid,sim,0).sum(1)/np.maximum(valid.sum(1),1)
                if not purged_prototypes:v=np.sort(sim,axis=1)[:,-min(3,use.sum()):].mean(1);mean=sim.mean(1)
                for tag,values in [('top3',v),('mean',mean)]:name=f'proto_{e}_{c}_{tag}';bank[name]=values;groups['prototypes'].append(name)
                own.append(v)
            own=np.stack(own,1);ci=rows.class_name.map(CL.index).to_numpy();name=f'proto_{e}_margin';bank[name]=own[np.arange(len(rows)),ci]-np.max(np.where(np.arange(4)[None,:]==ci[:,None],-10,own),axis=1);groups['prototypes'].append(name)
        transforms[fit]={}
        for e in EMB:
            x=np.load(A/f'features/{e}.npy');scale=StandardScaler().fit(x[train]);x=scale.transform(x);pca=PCA(n_components=min(k,int(train.sum())-2),svd_solver='randomized',random_state=23).fit(x[train]);z=pca.transform(x);groups['embed_'+e]=[f'emb_{e}_{j}' for j in range(z.shape[1])]
            for j,c in enumerate(groups['embed_'+e]):bank[c]=z[:,j]
            transforms[fit][e]=(scale,pca)
        banks[fit]=bank
    return rows,banks,groups,transforms
def main():
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning);p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=23);p.add_argument('--names',nargs='*');p.add_argument('--models',nargs='*',default=['cb2','cb4']);p.add_argument('--pca',type=int,default=16);p.add_argument('--purged-prototypes',action='store_true');p.add_argument('--tag',default='');args=p.parse_args();rows,banks,groups,transforms=load_bank(args.pca,args.purged_prototypes);q,te,h=inputs();bo,bt=baseline(q,te);conf=configurations();results=[];trainmanifest=[];imports=[];suffix='' if args.seed==23 and args.pca==16 else f'_s{args.seed}_p{args.pca}'
    if args.purged_prototypes:suffix+='_purgedproto'
    if args.tag:suffix+='_'+args.tag
    joblib.dump(transforms,A/f'models/transforms{suffix}.joblib');(A/f'inputs/configurations{suffix}.json').write_text(json.dumps(conf,indent=2))
    for name,(gr,emb) in conf.items():
        if args.names and name not in args.names:continue
        cols=list(dict.fromkeys(c for g in gr+['embed_'+e for e in emb] for c in groups[g]));cols+=['class_name'];assert not set(cols)&{'id','day','frame_id','error','score'}
        for kind in args.models:
            modelname=f'{name}_{kind}{suffix}';start=time.time();preds=[]
            for fit in DAYS:
                train=rows.split.eq('oof')&rows.day.eq(fit);evaluate=~train;X=banks[fit][cols];assert np.isfinite(X.drop(columns='class_name').to_numpy()).all();y=rows.loc[train,'error'];weight=np.ones(int(train.sum()))
                if kind=='lr':
                    numeric=[c for c in cols if c!='class_name'];scale=StandardScaler().fit(X.loc[train,numeric]);oh=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(X.loc[train,['class_name']]);xt=np.c_[scale.transform(X.loc[train,numeric]),oh.transform(X.loc[train,['class_name']])];xe=np.c_[scale.transform(X.loc[evaluate,numeric]),oh.transform(X.loc[evaluate,['class_name']])];model=LogisticRegression(C=.1,max_iter=3000,random_state=args.seed).fit(xt,y);prob=model.predict_proba(xe)[:,1];joblib.dump((numeric,scale,oh,model),A/f'models/{modelname}_fit{fit[-2:]}.joblib')
                else:
                    depth=int(kind[2:]);model=CatBoostClassifier(iterations=250 if depth==2 else 400,depth=depth,learning_rate=.03,l2_leaf_reg=10 if depth==2 else 20,loss_function='Logloss',random_seed=args.seed,thread_count=3,verbose=False,allow_writing_files=False);model.fit(X.loc[train],y,cat_features=['class_name'],sample_weight=weight);prob=model.predict_proba(X.loc[evaluate])[:,1];model.save_model(str(A/f'models/{modelname}_fit{fit[-2:]}.cbm'))
                    imports.extend(dict(model=modelname,fit_day=fit,feature=c,importance=float(v)) for c,v in zip(cols,model.feature_importances_))
                d=rows.loc[evaluate,['id','frame_id','day','class_name','split','error']].copy();d['p_error']=prob;d['fit_day']=fit;preds.append(d);trainmanifest.append(dict(model=modelname,fit_day=fit,train_ids=rows.loc[train,'id'].tolist(),train_frames=sorted(rows.loc[train,'frame_id'].unique()),features=cols,test_in_fit=False,held_day_in_fit=False,nested_dense_used='nested' in gr))
            pp=pd.concat(preds);pp.to_csv(A/f'features/{modelname}_predictions.csv',index=False);oo=pp[pp.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();test=np.stack([g.set_index('id').loc[te.id].p_error.to_numpy() for _,g in pp[pp.split.eq('test')].groupby('fit_day')]).mean(0)
            for alpha in [0.05,.1,.2,.5,1000000.]:
                tag='standalone' if alpha>100 else f'a{int(alpha*100):02}';s=oo if alpha>100 else (bo+alpha*oo)/(1+alpha);t=test if alpha>100 else (bt+alpha*test)/(1+alpha);exp=f'{modelname}_{tag}';q.assign(score=s).to_csv(A/f'results/oof/{exp}.csv',index=False);te[['id']].assign(score=t).to_csv(A/f'results/submissions/{exp}.csv',index=False);results.append(dict(name=exp,model=modelname,feature_set=name,kind=kind,seed=args.seed,pca=args.pca,alpha=alpha,features=len(cols),**metrics(q,s)))
            pd.DataFrame(results).to_csv(A/f'results/experiments{suffix}.csv',index=False);pd.DataFrame(imports).to_csv(A/f'audit/importances{suffix}.csv',index=False);(A/f'models/manifest{suffix}.json').write_text(json.dumps(trainmanifest,indent=2));last=pd.DataFrame(results[-5:]);print(modelname,'standalone',round(AP(q.error,oo),6),'blend',round(last[last.alpha.lt(100)].ap.max(),6),'sec',round(time.time()-start,1),flush=True)
    print(pd.DataFrame(results).sort_values('ap',ascending=False).head(20).to_string(index=False),flush=True)
if __name__=='__main__':main()
