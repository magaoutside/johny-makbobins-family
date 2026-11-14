"""Additional frozen OWLv2/YOLO embeddings combined with prior families."""
from train import *
def main():
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning);parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=23);parser.add_argument('--names',nargs='*');args=parser.parse_args();suffix='' if args.seed==23 else f'_s{args.seed}';rows,banks,groups,_=load_bank();q,t,h=inputs();bo,bt=baseline(q,t);transforms={}
    for fit in DAYS:
        tr=rows.split.eq('oof')&rows.day.eq(fit);transforms[fit]={}
        for e in ['owl','yolo']:
            x=np.load(A/f'features/{e}.npy');scale=StandardScaler().fit(x[tr]);xx=scale.transform(x);pca=PCA(n_components=16,svd_solver='randomized',random_state=23).fit(xx[tr]);z=pca.transform(xx);groups['embed_'+e]=[f'emb_{e}_{j}' for j in range(16)]
            for j,c in enumerate(groups['embed_'+e]):banks[fit][c]=z[:,j]
            transforms[fit][e]=(scale,pca)
    if args.seed==23:joblib.dump(transforms,A/'models/detector_transforms.joblib')
    out=[];manifest=[]
    configs=[('owl',['base23'],['owl']),('yolo',['base23'],['yolo']),('owl_yolo',['base23'],['owl','yolo']),('siglip_owl',['base23'],['siglip2','owl']),('clip_yolo',['base23'],['clip','yolo']),('all_backbones',['base23'],EMB+['owl','yolo']),('all_features',['frozen_all','owl','clip_text','pixels','nested'],EMB+['owl','yolo']),('detector_only',[],['owl','yolo'])]
    for name,gr,emb in configs:
        if args.names and name not in args.names:continue
        cols=list(dict.fromkeys(c for g in gr+['embed_'+e for e in emb] for c in groups[g]))+['class_name']
        for depth in [2,4]:
            key=f'detector_{name}_d{depth}'+suffix;pp=[]
            for fit in DAYS:
                tr=rows.split.eq('oof')&rows.day.eq(fit);ev=~tr;X=banks[fit][cols];model=CatBoostClassifier(iterations=250 if depth==2 else 400,depth=depth,learning_rate=.03,l2_leaf_reg=10 if depth==2 else 20,random_seed=args.seed,thread_count=3,verbose=False,allow_writing_files=False);model.fit(X.loc[tr],rows.loc[tr,'error'],cat_features=['class_name']);pp.append(rows.loc[ev,['id','frame_id','day','class_name','split']].assign(p_error=model.predict_proba(X.loc[ev])[:,1],fit_day=fit));model.save_model(str(A/f'models/{key}_fit{fit[-2:]}.cbm'));manifest.append(dict(model=key,fit_day=fit,training_ids=rows.loc[tr,'id'].tolist(),features=cols,backbones=emb))
            pp=pd.concat(pp);pp.to_csv(A/f'features/{key}_predictions.csv',index=False);oo=pp[pp.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();tt=pp[pp.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
            for a in [.05,.1,.2,.5,1000000.]:
                tag='standalone' if a>100 else f'a{int(a*100):02}';s=oo if a>100 else (bo+a*oo)/(1+a);st=tt if a>100 else (bt+a*tt)/(1+a);exp=key+'_'+tag;q.assign(score=s).to_csv(A/f'results/oof/{exp}.csv',index=False);t[['id']].assign(score=st).to_csv(A/f'results/submissions/{exp}.csv',index=False);out.append(dict(name=exp,model=key,alpha=a,**metrics(q,s)))
            pd.DataFrame(out).to_csv(A/f'results/detector_experiments{suffix}.csv',index=False);(A/f'models/detector_manifest{suffix}.json').write_text(json.dumps(manifest,indent=2));print(key,'standalone',AP(q.error,oo),'blend',pd.DataFrame(out[-5:]).iloc[:4].ap.max(),flush=True)
if __name__=='__main__':main()
