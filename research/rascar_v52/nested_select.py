"""Feature/config selection inside training day only. No held-day early stopping."""
from train import *
def main():
    q,te,h=inputs();bo,bt=baseline(q,te);rows=pd.read_csv(A/'inputs/rows.csv');static=pd.read_csv(A/'features/static.csv').set_index('id').loc[rows.id].reset_index(drop=True);groups=json.loads((A/'inputs/groups.json').read_text());conf={k:configurations()[k] for k in ['base13','base23','base23_text','base23_owl','all_frozen','base23_siglip2','base23_vit_l448','siglip_dino']};raw={e:np.load(A/f'features/{e}.npy') for e in EMB};seconds=pd.to_datetime(rows.frame_id.str.rsplit('_',n=2).str[-2:].str.join(''),format='%Y%m%d%H%M%S').astype('int64').to_numpy()/1e9;predictions=[];scores=[];manifest=[];selected=[]
    for fit in DAYS:
        outer=rows.split.eq('oof')&rows.day.eq(fit);frames=sorted(rows.loc[outer,'frame_id'].unique());blocks=np.array_split(frames,3);cache={}
        for k,held in enumerate(blocks):
            val=outer&rows.frame_id.isin(held);timestamps=seconds[val];distance=np.abs(seconds[:,None]-timestamps[None]).min(1);tr=outer&~val&(distance>120);assert rows.loc[tr,'error'].nunique()==2;assert not set(rows.loc[tr].frame_id)&set(rows.loc[val].frame_id)
            X=static.copy()
            for e in ['siglip2','vit_l448','dino_small']:
                scale=StandardScaler().fit(raw[e][tr]);z=scale.transform(raw[e]);pc=PCA(n_components=min(16,int(tr.sum())-2),svd_solver='randomized',random_state=23).fit(z[tr]);z=pc.transform(z);groups['embed_'+e]=[f'emb_{e}_{j}' for j in range(z.shape[1])]
                for j,c in enumerate(groups['embed_'+e]):X[c]=z[:,j]
            cache[k]=(tr,val,X)
            manifest.append(dict(fit_day=fit,block=k,train_ids=rows.loc[tr,'id'].tolist(),held_ids=rows.loc[val,'id'].tolist(),purge_seconds=120,preprocessing_inner_train_only=True))
        for name,(gr,emb) in conf.items():
            for depth in [2,4]:
                pred=np.zeros(int(outer.sum()));inner=[]
                for k,(tr,val,X) in cache.items():
                    cols=list(dict.fromkeys(c for g in gr+['embed_'+e for e in emb] for c in groups[g]))+['class_name'];model=CatBoostClassifier(iterations=250 if depth==2 else 400,depth=depth,learning_rate=.03,l2_leaf_reg=10 if depth==2 else 20,random_seed=23,thread_count=3,verbose=False,allow_writing_files=False);model.fit(X.loc[tr,cols],rows.loc[tr,'error'],cat_features=['class_name']);pp=model.predict_proba(X.loc[val,cols])[:,1];inner.append(rows.loc[val,['id','error']].assign(p_error=pp));model.save_model(str(A/f'models/inner_{name}_d{depth}_fit{fit[-2:]}_b{k}.cbm'))
                inner=pd.concat(inner);ap=AP(inner.error,inner.p_error);scores.append(dict(fit_day=fit,feature_set=name,depth=depth,inner_ap=ap));inner.to_csv(A/f'features/inner_{name}_d{depth}_fit{fit[-2:]}.csv',index=False)
        ranking=pd.DataFrame(scores);ranking=ranking[ranking.fit_day.eq(fit)].sort_values(['inner_ap','feature_set','depth'],ascending=[False,True,True]);winner=ranking.iloc[0];stem=f'{winner.feature_set}_cb{winner.depth}';path=A/f'features/{stem}_predictions.csv';pp=pd.read_csv(path);pp=pp[pp.fit_day.eq(fit)].copy();pp['selected_model']=stem;predictions.append(pp);selected.append(winner.to_dict());print('SELECT',fit,winner.to_dict(),flush=True)
    pp=pd.concat(predictions);pp.to_csv(A/'features/nested_selected_predictions.csv',index=False);pd.DataFrame(scores).to_csv(A/'results/nested_selection_scores.csv',index=False);(A/'models/nested_selection.json').write_text(json.dumps(dict(selected=selected,blocks=manifest,criterion='inner pooled AP on official error labels; alpha not selected, held-day labels unused; families are historically researched'),indent=2));oo=pp[pp.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();tt=pp[pp.split.eq('test')].groupby('id').p_error.mean().loc[te.id].to_numpy();out=[]
    for a in [.05,.1,.2,.5,1000000.]:
        tag='standalone' if a>100 else f'a{int(a*100):02}';s=oo if a>100 else (bo+a*oo)/(1+a);st=tt if a>100 else (bt+a*tt)/(1+a);name='nested_selected_'+tag;q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=st).to_csv(A/f'results/submissions/{name}.csv',index=False);out.append(dict(name=name,alpha=a,**metrics(q,s)))
    pd.DataFrame(out).to_csv(A/'results/nested_selected_experiments.csv',index=False);print(pd.DataFrame(out).to_string(index=False))
if __name__=='__main__':main()
