"""Presence-reviewed pool as semantic exemplars, never bbox-error targets."""
from train import *
def main():
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning);rows,banks,groups,_=load_bank();q,t,h=inputs();bo,bt=baseline(q,t);rec=pd.read_csv(ROOT/'rascar_v41/inputs/records.csv');ix=rec.reset_index().set_index('id').loc[rows.id,'index'].to_numpy();pool=rec[rec.split.eq('pool_presence')&rec.presence.eq(1)].copy();assert not set(pool.frame_id)&set(t.frame_id);qt=pd.to_datetime(rows.frame_id.str.rsplit('_',n=2).str[-2:].str.join(''),format='%Y%m%d%H%M%S').astype('int64').to_numpy()/1e9;cols=[];counts=[]
    for fit in DAYS:
        tr=rows.split.eq('oof')&rows.day.eq(fit);cols=[]
        for e in ['clip','vit_b','vit_l']:
            raw=np.load(ROOT/f'rascar_v41/features/{e}_frozen.npy',mmap_mode='r');dim=(raw.shape[1]-13)//3;x=np.asarray(raw[ix,dim:dim*2]);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8);values=[]
            for c in CL:
                p=pool[pool.day.eq(fit)&pool.class_name.eq(c)];z=np.asarray(raw[p.index,dim:dim*2]).copy();z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8);sim=x@z.T;pt=pd.to_datetime(p.frame_id.str.rsplit('_',n=2).str[-2:].str.join(''),format='%Y%m%d%H%M%S').astype('int64').to_numpy()/1e9;valid=np.ones(sim.shape,bool);valid[tr]=abs(qt[tr,None]-pt[None,:])>120
                count=valid.sum(1);sort=np.sort(np.where(valid,sim,-10),axis=1)[:,-min(3,len(p)):];top=np.where(sort>-9,sort,0).sum(1)/np.maximum((sort>-9).sum(1),1);mean=np.where(valid,sim,0).sum(1)/np.maximum(count,1);missing=(count==0).astype(float)
                for tag,v in [('top3',top),('mean',mean),('missing',missing)]:name=f'poolproto_{e}_{c}_{tag}';banks[fit][name]=v;cols.append(name)
                values.append(top);counts.append(dict(fit_day=fit,backbone=e,class_name=c,examples=len(p),frames=p.frame_id.nunique(),minimum_available_train=int(count[tr].min())))
            v=np.stack(values,1);ci=rows.class_name.map(CL.index).to_numpy();name=f'poolproto_{e}_margin';banks[fit][name]=v[np.arange(len(rows)),ci]-np.max(np.where(np.arange(4)[None,:]==ci[:,None],-10,v),axis=1);cols.append(name)
    # This curated pool contains no trays: explicit missing flags prevent claiming coverage.
    pd.DataFrame(counts).to_csv(A/'audit/pool_prototype_counts.csv',index=False);out=[];manifest=[]
    for key,gr,emb in [('poolproto',['base23'],[]),('poolproto_siglip',['base23'],['siglip2']),('poolproto_all',['frozen_all','owl','clip_text'],[])]:
        cc=list(dict.fromkeys(cols+[c for g in gr+['embed_'+e for e in emb] for c in groups[g]]))+['class_name']
        for depth in [2,4]:
            pp=[];name=f'{key}_d{depth}'
            for fit in DAYS:
                tr=rows.split.eq('oof')&rows.day.eq(fit);ev=~tr;X=banks[fit][cc];model=CatBoostClassifier(iterations=250 if depth==2 else 400,depth=depth,learning_rate=.03,l2_leaf_reg=10 if depth==2 else 20,random_seed=23,thread_count=3,verbose=False,allow_writing_files=False);model.fit(X.loc[tr],rows.loc[tr,'error'],cat_features=['class_name']);pp.append(rows.loc[ev,['id','frame_id','day','class_name','split']].assign(p_error=model.predict_proba(X.loc[ev])[:,1],fit_day=fit));model.save_model(str(A/f'models/{name}_fit{fit[-2:]}.cbm'));manifest.append(dict(name=name,fit_day=fit,features=cc,training_ids=rows.loc[tr,'id'].tolist(),prototype_ids=pool.loc[pool.day.eq(fit),'id'].tolist(),presence_is_not_bbox_error=True,training_neighbor_purge_seconds=120))
            pp=pd.concat(pp);pp.to_csv(A/f'features/{name}_predictions.csv',index=False);oo=pp[pp.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();tt=pp[pp.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
            for a in [.05,.1,.2,.5,1000000.]:
                tag='standalone' if a>100 else f'a{int(a*100):02}';s=oo if a>100 else (bo+a*oo)/(1+a);st=tt if a>100 else (bt+a*tt)/(1+a);exp=name+'_'+tag;q.assign(score=s).to_csv(A/f'results/oof/{exp}.csv',index=False);t[['id']].assign(score=st).to_csv(A/f'results/submissions/{exp}.csv',index=False);out.append(dict(name=exp,model=name,alpha=a,**metrics(q,s)))
            pd.DataFrame(out).to_csv(A/'results/pool_experiments.csv',index=False);(A/'models/pool_manifest.json').write_text(json.dumps(manifest,indent=2));print(name,AP(q.error,oo),flush=True)
    print(pd.DataFrame(out).sort_values('ap',ascending=False).head(10).to_string(index=False))
if __name__=='__main__':main()
