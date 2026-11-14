"""Reload all six saved estimators, check fold isolation, compose first candidate."""
from common56 import *
sys.path.insert(0,str(A/'vendor'));sys.path.insert(0,str(ROOT/'rascar_v53'))
from experiment import FeatureBank
from tabicl import TabICLClassifier
import torch,shutil

def main():
    torch.set_num_threads(2);bank=FeatureBank();q,t,h=inputs();bo,bt=incumbent(q,t);checks=[];oo=[];tt=[];start=time.time()
    for seed,n in [(5601,8),(5602,16),(5603,16)]:
        name=f'tabiclv1_anchor_n{n}_s{seed}';cached=pd.read_csv(A/f'features/{name}_predictions.csv');pred=[]
        for fit in DAYS:
            use=bank.rows.split.eq('oof')&bank.rows.day.eq(fit);ev=~use
            m=[m for m in json.loads((A/f'models/tabiclv1_manifest_s{seed}.json').read_text()) if m['name']==name and m['fit_day']==fit][0]
            assert m['train_ids']==bank.rows.loc[use,'id'].tolist()
            assert set(bank.rows.loc[use,'frame_id']).isdisjoint(bank.rows.loc[ev,'frame_id'])
            # Rebuild prototype features independently and compare to stored matrix.
            X,g=bank.build(use,fit);cols=m['features'];saved=pd.read_csv(A/f'features/anchor_fit{fit[-2:]}_bank.csv')
            numeric=[c for c in cols if c!='class_name'];delta=float(np.abs(X[numeric].to_numpy()-saved[numeric].to_numpy()).max());assert delta<1e-10
            model=TabICLClassifier.load(A/f'models/{name}_fit{fit[-2:]}.pkl',device='cpu')
            p=model.predict_proba(X.loc[ev,cols])[:,1]
            expected=cached[cached.fit_day.eq(fit)].set_index('id').loc[bank.rows.loc[ev,'id'],'p_error'].to_numpy()
            err=float(abs(p-expected).max());assert err<1e-6,err
            pred.append(bank.rows.loc[ev,['id','split']].assign(p_error=p));checks.append(dict(name=name,fit=fit,rows=int(ev.sum()),feature_max_error=delta,prediction_max_error=err,train_n=int(use.sum()),validation_frames_disjoint=True))
            print('REPLAY',checks[-1],flush=True)
        d=pd.concat(pred);oo.append(d[d.split.eq('oof')].set_index('id').loc[q.id,'p_error'].to_numpy());tt.append(d[d.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy())
    po,pt=np.mean(oo,axis=0),np.mean(tt,axis=0);s,z=bo.copy(),bt.copy();om=q.class_name.eq('nude_hands');tm=t.class_name.eq('nude_hands')
    s[om]=(s[om]+.4*po[om])/1.4;z[tm]=(z[tm]+.4*pt[tm])/1.4
    for split,df,score in [('oof',q,s),('submissions',t,z)]:
        old=pd.read_csv(A/f'results/{split}/v1_anchor_mean3_nude_hands_a40.csv').set_index('id').loc[df.id,'score'].to_numpy();assert abs(old-score).max()<1e-7
    release=A/'release';release.mkdir(exist_ok=True)
    source=A/'results/submissions/v1_anchor_mean3_nude_hands_a40.csv';target=release/'submission_v56_tabicl_hands_a40.csv';shutil.copy2(source,target)
    sent=pd.read_csv(target);assert sent.columns.tolist()==['id','score'];assert sent.id.tolist()==t.id.tolist();assert sent.id.is_unique and len(sent)==362
    assert np.isfinite(sent.score).all() and sent.score.between(0,1).all();assert np.array_equal(sent.score.to_numpy()[~tm],bt[~tm])
    allpred=pd.concat([q[['id','class_name']].assign(split='oof',baseline=bo,p_error=po),t[['id','class_name']].assign(split='test',baseline=bt,p_error=pt)])
    allpred.to_csv(release/'replay_inputs.csv',index=False)
    row=pd.read_csv(A/'results/ensemble_v1.csv');row=row[row.name.eq('v1_anchor_mean3_nude_hands_a40')].iloc[0].to_dict()
    report=dict(candidate=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),baseline_public=.96300,candidate_public=None,selection_adjusted=False,metrics=row,checks=checks,seconds=time.time()-start,scope='Only 131 nude_hands test scores; other 231 exact V52 values',formula='(V52 + 0.4 * mean_3_seeds_2_day_fits(TabICL_v1.1_error_probability)) / 1.4',notes='Cross-day development set repeatedly used for selection; bootstrap is descriptive, not independent evidence or a promised Public gain.')
    (release/'verification.json').write_text(json.dumps(report,indent=2));(A/'audit/first_candidate_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
