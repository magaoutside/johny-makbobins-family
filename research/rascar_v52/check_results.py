"""Data/score contract, exact reproduction and saved-model replay checks."""
from train import *
from scipy.optimize import linear_sum_assignment
import hashlib
def main():
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning);q,t,h=inputs();bo,bt=baseline(q,t);checks={};reproduced=0
    from common import iou
    for (fid,cls),g in q.groupby(['frame_id','class_name']):
        gt=h[h.frame_id.eq(fid)&h.class_name.eq(cls)][B].to_numpy();ov=iou(g[B],gt);err=np.ones(len(g),int)
        if ov.size:
            rr,cc=linear_sum_assignment(-((ov>=.6)*(max(ov.shape)+1)+ov*(ov>=.6)));err[rr[ov[rr,cc]>=.6]]=0
        assert np.array_equal(err,g.error.to_numpy());reproduced+=len(g)
    checks['official_error_labels_reproduced']=reproduced;checks['baseline_sha256']=hashlib.sha256((ROOT/'rascar_v31/submission_v31_dense_oof098784.csv').read_bytes()).hexdigest();n=0;count=0
    for f in sorted((A/'results/submissions').glob('*.csv')):
        d=pd.read_csv(f);assert list(d.columns)==['id','score'];assert d.id.is_unique and d.id.tolist()==t.id.tolist();assert np.isfinite(d.score).all() and d.score.between(0,1).all();oo=A/'results/oof'/f.name;assert oo.exists();d=pd.read_csv(oo);assert d.id.tolist()==q.id.tolist();assert np.array_equal(d.error,q.error);assert np.isfinite(d.score).all() and d.score.between(0,1).all();n+=1
    checks['valid_submission_oof_pairs']=n
    for path in (A/'models').glob('*manifest*.json'):
        data=json.loads(path.read_text())
        if not isinstance(data,list):continue
        for m in data:
            ids=m.get('train_ids',m.get('training_ids'));fit=m.get('fit_day')
            if ids is None or fit is None:continue
            # Augmentation has its own explicit source/day ledger.
            if path.name=='augmented_manifest.json':source=pd.read_csv(ROOT/'rascar_v41/inputs/records.csv').set_index('id')
            else:source=q.set_index('id')
            assert set(ids)<=set(source.index);assert source.loc[ids,'day'].eq(fit).all();assert not set(ids)&set(t.id);assert not set(m.get('features',[]))&{'id','frame_id','day','error','score'};count+=1
    checks['training_manifests_checked']=count
    source=pd.read_csv(A/'features/static.csv');assert source.id.is_unique and source.id.tolist()==pd.concat([q.id,t.id]).tolist();assert np.isfinite(source.drop(columns=['id','class_name']).to_numpy()).all();checks['static_features']=source.shape[1]-1
    rows,banks,groups,_=load_bank();maxerr=0.
    for stem in ['base13_cb2','base23_siglip2_cb2','base23_prototypes_cb4']:
        pred=pd.read_csv(A/f'features/{stem}_predictions.csv');manifest=json.loads((A/'models/manifest.json').read_text())
        for fit in DAYS:
            columns=next(m['features'] for m in manifest if m['model']==stem and m['fit_day']==fit);model=CatBoostClassifier();model.load_model(str(A/f'models/{stem}_fit{fit[-2:]}.cbm'));p=pred[pred.fit_day.eq(fit)].set_index('id');ix=rows.reset_index().set_index('id').loc[p.index,'index'];actual=model.predict_proba(banks[fit].loc[ix,columns])[:,1];err=float(np.max(abs(actual-p.p_error.to_numpy())));assert err<1e-10,err;maxerr=max(maxerr,err)
    checks['saved_model_replay_max_error']=maxerr
    p=pd.read_csv(A/'features/base13_cb2_predictions.csv');oo=p[p.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();m=metrics(q,oo);assert abs(m['ap']-.955963)<5e-7 and abs(m['2026-10-25']-.941732)<5e-7 and abs(m['2026-10-26']-.977209)<5e-7;checks['attached_MD_catboost13_reproduced']=m
    # Exact library contract for the frozen-prior correction, with saved model.
    from catboost import Pool
    from scipy.special import logit
    pr=[]
    for fit in DAYS:
        mm=next(m for m in json.loads((A/'models/prior_manifest.json').read_text()) if m['name']=='prior_base23_sam_i100' and m['fit_day']==fit);X=banks[fit][mm['features']];base=logit(np.clip(banks[fit].sam_score,.001,.999));model=CatBoostClassifier();model.load_model(str(A/f'models/prior_base23_sam_i100_fit{fit[-2:]}.cbm'));r=model.predict(X,prediction_type='RawFormulaVal');z=model.predict(Pool(X,cat_features=['class_name'],baseline=base),prediction_type='RawFormulaVal');err=float(np.max(abs(z-r-base)));assert err<1e-5;pr.append(err)
    checks['catboost_baseline_contract_max_error']=max(pr)
    checks['note']='Checks establish mechanics and split integrity, not improvement on hidden leaderboard labels.';(A/'results/integrity_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
