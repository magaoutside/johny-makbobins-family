"""Equal seed/family ensembles; never choose a winning seed."""
from train import *
def main():
    q,t,h=inputs();bo,bt=baseline(q,t);out=[]
    stems=['base23_siglip2_cb2','siglip_dino_cb2','base23_prototypes_cb4','base23_text_cb2','base23_nested_cb2','semantic_prototypes_cb2','detector_siglip_owl_d4','detector_detector_only_d2'];ensembles={}
    for stem in stems:
        suffixes=['','_s24','_s25'] if stem.startswith('detector_') else ['','_s24_p16','_s25_p16'];pp=pd.concat([pd.read_csv(A/f'features/{stem}{suffix}_predictions.csv').assign(rep=i) for i,suffix in enumerate(suffixes)]);key='ensemble_'+stem;ensembles[key]=pp.groupby(['id','fit_day','split'],as_index=False).p_error.mean();ensembles[key].to_csv(A/f'features/{key}_predictions.csv',index=False)
    families=['ensemble_base23_siglip2_cb2','ensemble_base23_text_cb2','ensemble_base23_nested_cb2','ensemble_detector_siglip_owl_d4'];ensembles['diverse_equal']=pd.concat([ensembles[k].assign(family=k) for k in families]).groupby(['id','fit_day','split'],as_index=False).p_error.mean();ensembles['diverse_equal'].to_csv(A/'features/diverse_equal_predictions.csv',index=False)
    for stem,pp in ensembles.items():
        oo=pp[pp.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();tt=pp[pp.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
        for scope in ['all']+CL:
            m=np.ones(len(q),bool) if scope=='all' else q.class_name.eq(scope).to_numpy();mt=np.ones(len(t),bool) if scope=='all' else t.class_name.eq(scope).to_numpy()
            for a in [.05,.1,.2]:
                s=bo.copy();st=bt.copy();s[m]=(bo[m]+a*oo[m])/(1+a);st[mt]=(bt[mt]+a*tt[mt])/(1+a);name=stem+'_'+scope+f'_a{int(a*100):02}';q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False);t[['id']].assign(score=st).to_csv(A/f'results/submissions/{name}.csv',index=False);out.append(dict(name=name,model=stem,scope=scope,alpha=a,**metrics(q,s)))
    pd.DataFrame(out).to_csv(A/'results/ensembles_complete.csv',index=False);(A/'models/ensemble_definitions.json').write_text(json.dumps(dict(seeds=[23,24,25],families=families,weights='equal; no best-seed selection',posthoc_class_scopes=True),indent=2));print(pd.DataFrame(out).sort_values('ap',ascending=False).head(12).to_string(index=False))
    purged=[]
    for depth in [2,4]:
        paths=[A/f'features/base23_prototypes_cb{depth}{suffix}_purgedproto_predictions.csv' for suffix in ['', '_s24_p16','_s25_p16']]
        if not all(p.exists() for p in paths):continue
        pp=pd.concat([pd.read_csv(p).assign(seed=k) for k,p in zip([23,24,25],paths)]);pp.to_csv(A/f'features/purged_ensemble_d{depth}_predictions.csv',index=False);oo=pp[pp.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy();tt=pp[pp.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
        for a in [.05,.1,.2]:
            m=q.class_name.eq('person');mt=t.class_name.eq('person');s=bo.copy();st=bt.copy();s[m]=(bo[m]+a*oo[m])/(1+a);st[mt]=(bt[mt]+a*tt[mt])/(1+a);name=f'purged_ensemble_d{depth}_person_a{int(a*100):02}';q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False);t[['id']].assign(score=st).to_csv(A/f'results/submissions/{name}.csv',index=False);purged.append(dict(name=name,**metrics(q,s)))
    if purged:pd.DataFrame(purged).to_csv(A/'results/purged_ensemble_experiments.csv',index=False)
if __name__=='__main__':main()
