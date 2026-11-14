"""Separate classifier-only synthetic augmentation; no presence->bbox label shortcut."""
from train import *
def main():
    rec=pd.read_csv(ROOT/'rascar_v41/inputs/records.csv');q,te,h=inputs();bo,bt=baseline(q,te);simrec=pd.read_csv(ROOT/'rascar_v41/inputs/clip_text_records.csv');simidx=simrec.reset_index().set_index('id').loc[rec.id,'index'];sim=np.load(ROOT/'rascar_v41/features/clip_text_similarities.npy')[simidx];raw={e:np.load(ROOT/f'rascar_v41/features/{e}_frozen.npy',mmap_mode='r')[:,:-13] for e in ['clip','vit_b','vit_l']};banks={};manifests=[];transforms={}
    for fit in DAYS:
        tr=rec.day.eq(fit)&rec.split.isin(['query','synthetic']);bank=pd.DataFrame(index=rec.index);transforms[fit]={}
        for e,x in raw.items():
            scale=StandardScaler().fit(x[tr]);z=scale.transform(x);pca=PCA(n_components=16,svd_solver='randomized',random_state=23).fit(z[tr]);z=pca.transform(z)
            for j in range(16):bank[e+f'_{j}']=z[:,j]
            transforms[fit][e]=(scale,pca)
        for j in range(14):bank[f'clip_text_{j}']=sim[:,j]
        wh=rec[['x2','y2']].to_numpy()-rec[['x1','y1']].to_numpy();bank['width']=wh[:,0]/1280;bank['height']=wh[:,1]/720;bank['aspect']=np.log(np.maximum(wh[:,0],1)/np.maximum(wh[:,1],1));bank['class_name']=rec.class_name;banks[fit]=bank
    joblib.dump(transforms,A/'models/augmented_transforms.joblib');out=[]
    for depth in [2,4,6]:
        for budget in [0.,.25,1.]:
            name=f'aug_cb{depth}_budget{int(budget*100)}';pp=[]
            for fit in DAYS:
                tr=rec.day.eq(fit)&(rec.split.eq('query')|(rec.split.eq('synthetic')&(budget>0)));ev=rec.split.eq('test')|(rec.split.eq('query')&rec.day.ne(fit));assert not set(rec.loc[tr].frame_id)&set(rec.loc[ev].frame_id)
                # Equal total auxiliary/original loss budget, then balance synthetic frames.
                w=np.ones(int(tr.sum()));d=rec.loc[tr];syn=d.split.eq('synthetic').to_numpy()
                if syn.any():
                    sw=d.loc[syn,'weight'].to_numpy()/d.loc[syn].groupby('frame_id').id.transform('size').to_numpy();w[syn]=sw/sw.sum()*(~syn).sum()*budget
                model=CatBoostClassifier(iterations=400,depth=depth,learning_rate=.03,l2_leaf_reg=20,loss_function='Logloss',random_seed=23,thread_count=3,verbose=False,allow_writing_files=False);model.fit(banks[fit].loc[tr],rec.loc[tr,'error'],cat_features=['class_name'],sample_weight=w);pred=model.predict_proba(banks[fit].loc[ev])[:,1];model.save_model(str(A/f'models/{name}_fit{fit[-2:]}.cbm'));pp.append(rec.loc[ev,['id','frame_id','day','class_name','split']].assign(p_error=pred,fit_day=fit));manifests.append(dict(model=name,fit_day=fit,train_ids=d.id.tolist(),train_frames=sorted(d.frame_id.unique()),features=banks[fit].columns.tolist(),auxiliary_budget=budget,synthetic_rows=int(syn.sum()),original_rows=int((~syn).sum()),pool_presence_used_as_bbox_error=False,test_in_fit=False,held_day_in_fit=False,pca_fit_includes_same_day_synthetics_even_budget0=True))
            pp=pd.concat(pp);pp.to_csv(A/f'features/{name}_predictions.csv',index=False);oo=pp[pp.split.eq('query')].set_index('id').loc[q.id].p_error.to_numpy();tt=np.stack([g.set_index('id').loc[te.id].p_error.to_numpy() for _,g in pp[pp.split.eq('test')].groupby('fit_day')]).mean(0)
            for a in [.05,.1,.2,.5,1000000.]:
                tag='standalone' if a>100 else f'a{int(a*100):02}';s=oo if a>100 else (bo+a*oo)/(1+a);t=tt if a>100 else (bt+a*tt)/(1+a);exp=name+'_'+tag;q.assign(score=s).to_csv(A/f'results/oof/{exp}.csv',index=False);te[['id']].assign(score=t).to_csv(A/f'results/submissions/{exp}.csv',index=False);out.append(dict(name=exp,depth=depth,budget=budget,alpha=a,**metrics(q,s)))
            pd.DataFrame(out).to_csv(A/'results/augmented_experiments.csv',index=False);(A/'models/augmented_manifest.json').write_text(json.dumps(manifests,indent=2));print(name,'standalone',AP(q.error,oo),'blend',pd.DataFrame(out[-5:]).iloc[:4].ap.max(),flush=True)
    print(pd.DataFrame(out).sort_values('ap',ascending=False).head(10).to_string(index=False))
if __name__=='__main__':main()
