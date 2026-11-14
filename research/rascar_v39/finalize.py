"""Validate, inventory and report V39 without changing any previous release."""
from common39 import *
from set_detector import Data

FROZEN={
 'rascar_v22/submission_v22_candidate.csv':'44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c',
 'rascar_v31/submission_v31_dense_oof098784.csv':'8720eb732639532bab49eff75b956a6bd4db21b96ede309ed901889777f4ab2c',
 'rascar_v37/submission_v37_person_refit_RESEARCH.csv':'105e980e9465c00692e5a50bc8a5cf895f2b5e9a2b55f6c7e4682e6278f3afb1'}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    for f,sha in FROZEN.items():assert digest(ROOT/f)==sha,f
    sample=pd.read_csv(ROOT/'rascar_data_full/sample_submission.csv');assert set(sample.id)==set(te.id)
    artifacts=[];pairs=np.triu_indices(len(te),1);base_order=np.sign(BASE['test'][:,None]-BASE['test'][None,:])[pairs]
    for path in sorted((A/'results/submissions').glob('*.csv')):
        d=pd.read_csv(path);assert list(d.columns)==['id','score'] and d.id.is_unique and set(d.id)==set(te.id)
        s=d.set_index('id').loc[te.id].score.to_numpy();assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
        name=path.stem.removeprefix('submission_');oof=pd.read_csv(A/f'results/oof/{name}.csv');assert oof.id.is_unique and set(oof.id)==set(q.id)
        o=oof.set_index('id').loc[q.id];assert np.array_equal(o.error.to_numpy(),q.error.to_numpy());assert np.isfinite(o.score).all() and o.score.between(0,1).all()
        order=np.sign(s[:,None]-s[None,:])[pairs]
        artifacts.append(dict(experiment=name,sha256=digest(path),ap=float(AP(q.error,o.score)),test_changed=int((abs(s-BASE['test'])>1e-10).sum()),test_changed_order_pairs=int((order!=base_order).sum()),exact_values_equal_v31=bool(np.array_equal(s,BASE['test'])),within_1e12_v31=bool(np.max(abs(s-BASE['test']))<1e-12),numeric_test_fingerprint=hashlib.sha256(s.round(12).tobytes()).hexdigest(),numeric_oof_fingerprint=hashlib.sha256(o.score.to_numpy().round(12).tobytes()).hexdigest()))
    checks=pd.DataFrame(artifacts);checks.to_csv(A/'results/artifact_checks.csv',index=False)
    data=Data();rec=data.rec;models=[]
    for path in sorted((A/'models').glob('*_manifest.json')):
        entries=json.loads(path.read_text());assert len(entries)==(3 if '_full_' in path.name else 6),path
        for m in entries:
            fit=m['fit_day'];expected=rec[(rec.day.isin(DAYS) if fit=='all' else rec.day.eq(fit))&rec.split.isin(['manual','empty'])]
            assert set(m['fit_frames'])==set(expected.frame_id)
            assert not set(m['fit_frames'])&set(te.frame_id)
            if fit!='all':assert not set(m['fit_frames'])&set(q.loc[q.day.ne(fit),'frame_id'])
            pred=pd.read_csv(A/f'features/{m["name"]}_boxes.csv');frame_days=pred.frame_id.str.extract(r'_(2026\d{4})_',expand=False)
            if fit=='all':assert set(pred.split)=={'test'}
            else:assert not frame_days[pred.split.ne('test')].eq(fit.replace('-','')).any()
            weights=A/f'models/{m["name"]}.pt';assert weights.exists() and np.isfinite(m['last_loss'])
            models.append(dict(name=m['name'],fit_day=fit,steps=m['steps'],fit_frames=len(m['fit_frames']),last_loss=m['last_loss'],sha256=digest(weights)))
    for path in (A/'features').glob('*matching_evidence.csv'):
        d=pd.read_csv(path);o=d[d.split.eq('oof')].merge(q[['id','day']],on='id',validate='many_to_one');assert len(o)==len(q)*3 and o.fit_day.ne(o.day).all()
        t=d[d.split.eq('test')];assert set(t.id)==set(te.id);assert t.groupby('id').size().nunique()==1
        assert len(t)==len(te)*(3 if set(t.fit_day)=={'all'} else 6)
    pd.DataFrame(models).to_csv(A/'audit/model_manifest.csv',index=False)
    # Full-fit scores are judged only by their corresponding cross-day proxy.
    full_oof=A/'results/oof/v39_set_full_geometry_all_RESEARCH.csv'
    if full_oof.exists():assert np.allclose(pd.read_csv(full_oof).score,pd.read_csv(A/'results/oof/v39_set_geometry_all_RESEARCH.csv').score,rtol=0,atol=1e-14)
    long_full_oof=A/'results/oof/v39_set_long_full_identity_geometry_all_RESEARCH.csv'
    if long_full_oof.exists():assert np.allclose(pd.read_csv(long_full_oof).score,pd.read_csv(A/'results/oof/v39_set_long_identity_geometry_all_RESEARCH.csv').score,rtol=0,atol=1e-14)
    run_tables=[pd.read_csv(p) for p in sorted((A/'results').glob('*_experiments.csv')) if p.name!='all_experiments.csv']
    extra=[]
    for p in (A/'results').glob('v39_*guard.json'):extra.append(json.loads(p.read_text()))
    experiments=pd.concat(run_tables+[pd.DataFrame(extra)],ignore_index=True);experiments.to_csv(A/'results/all_experiments.csv',index=False)
    assert len(experiments)==len(checks)
    for _,r in experiments.iterrows():
        c=checks.set_index('experiment').loc[r.experiment];assert abs(r.ap-c.ap)<1e-12 and int(r.changed_test)==int(c.test_changed)
    state=dict(status='completed_research_not_promoted',best_confirmed='V31',best_public_ap=.96239,best_oof_ap=float(AP(q.error,BASE['oof'])),new_public_ap=None,submitted=False,recommended_submission=None,models=len(models),research_csv=len(checks),unique_numeric_test_outputs=int(checks.numeric_test_fingerprint.nunique()),unique_numeric_oof_outputs=int(checks.numeric_oof_fingerprint.nunique()),frozen_files=FROZEN,source_sha256={p.name:digest(p) for p in sorted(A.glob('*.py'))},manual_test_annotations_added=False,validation_warning='198 queries / 35 frames repeatedly used for development; full fits have cross-day proxy only.')
    (A/'release_manifest.json').write_text(json.dumps(state,indent=2));(A/'audit/pipeline_checks.json').write_text(json.dumps(dict(frozen_hashes_passed=True,all_csv_ids_and_values_passed=True,oof_metrics_recomputed=True,training_day_exclusion_passed=True,test_frames_never_fit=True,full_fit_predictions_test_only=True,full_fit_oof_proxy_identical=True,matching_behavior_checks=json.loads((A/'audit/matching_behavior_checks.json').read_text())),indent=2))
    print(json.dumps(state,indent=2))

if __name__=='__main__':main()
