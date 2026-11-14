"""Fixed V31 blend/cut; no new threshold search or test annotation inputs."""
from common38 import *

def main():
    pred=pd.read_csv(A/'features/multilevel_predictions.csv');manifest=json.loads((A/'models/multilevel_manifest.json').read_text());assert len(manifest)==18
    # Reuse V37 last-layer heads under EXACTLY the same new gate/fallback.
    # This isolates added features from multi-seed aggregation changes.
    control=pd.read_csv(ROOT/'rascar_v37/features/refit_predictions.csv').assign(kind='last1_control')
    pred=pd.concat([pred,control],ignore_index=True)
    results=[]
    prior={}
    v22=pd.read_csv(ROOT/'rascar_v22/results/oof/final_reviewed.csv').set_index('id').loc[q.id].score.to_numpy()
    inherited=pd.read_csv(ROOT/'rascar_v18/results/oof/release_main.csv').set_index('id').loc[q.id].score.to_numpy()-pd.read_csv(ROOT/'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv').set_index('id').loc[q.id].score.to_numpy()
    prior['oof']=np.clip(v22-inherited,0,1)
    prior['test']=pd.read_csv(ROOT/'rascar_v22/submission_v22_candidate.csv').set_index('id').loc[te.id].score.to_numpy()
    for kind in ['all4','early2','last1_control']:
        target={};gates={}
        for split,rows in [('oof',q),('test',te)]:
            for mode in (['daily'] if split=='oof' else ['daily','full']):
                d=pred[pred.kind.eq(kind)&pred.split.eq(split)]
                d=d[d.fit_day.eq('all')] if mode=='full' else d[d.fit_day.ne('all')]
                if split=='oof':
                    assert all(d.merge(q[['id','day']],on='id').fit_day!=d.merge(q[['id','day']],on='id').day)
                    pivot=d.pivot(index='id',columns='seed',values='overlap').reindex(rows.id)
                else:pivot=d.pivot(index='id',columns=['fit_day','seed'],values='overlap').reindex(rows.id)
                vals=np.clip((.85-pivot.to_numpy().T)/.5,0,1);base=prior[split];up=np.all(vals>=base,axis=0);down=np.all(vals<=base,axis=0)
                target[(split,mode)]=np.where(up,np.min(vals,axis=0),np.max(vals,axis=0))
                gates[(split,mode)]=np.isfinite(vals).all(0)&(up|down)
        for cls in ['all']+CL:
            for mode in ['daily','full']:
                scores={}
                for split,rows in [('oof',q),('test',te)]:
                    key=(split,'daily' if split=='oof' else mode);mask=gates[key].copy()
                    if cls!='all':mask &= rows.class_name.eq(cls).to_numpy()
                    score=BASE[split].copy();score[mask]=.75*prior[split][mask]+.25*target[key][mask]
                    scores[split]=score
                name=f'v38_{kind}_{cls}_{mode}_RESEARCH'
                result=save(name,scores,'Frozen V31 .25 blend and .5 mask cut; new multilevel references with all-seed direction agreement, V31 fallback. Full-fit test has the same cross-day proxy as daily, not a separate validation.')
                result.update(kind=kind,classes=cls,test_fit=mode,weight=.25,cut=.5,status='research_only')
                results.append(result)
    pd.DataFrame(results).to_csv(A/'results/multilevel_experiments.csv',index=False)
    print(pd.DataFrame(results)[['experiment','ap','2026-10-25','2026-10-26','changed_test']].to_string(index=False))

if __name__=='__main__':main()
