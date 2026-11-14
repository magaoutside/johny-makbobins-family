"""Package research output, audit invariants, preserve prior releases.

This does not select a new best model or send a Kaggle submission.
"""
from common37 import *
import shutil, importlib.metadata
from stability import NAME

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    frozen={
      'rascar_v22/submission_v22_candidate.csv':'44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c',
      'rascar_v31/submission_v31_dense_oof098784.csv':'8720eb732639532bab49eff75b956a6bd4db21b96ede309ed901889777f4ab2c',
      'rascar_v35/submission_v35_temporal_guarded.csv':'2e2b18e2bb515d6e8fc0fe0209585c038b2dfc7c75cc00e2521bffbc09f44e21',
      'rascar_v36/submission_v36_candidate.csv':'9b3b01854e5eddb993bf7bc50a8aa38b813895d8fecc51336fb4b11ba138e7dd',
      'rascar_v36/submission_v36_narrow_control.csv':'a01f1e94cf9ff314a8814de91b9af79d48b358a0189482781bd1b0cfd90911e5',
    }
    for path,expected in frozen.items():assert sha(ROOT/path)==expected,path
    experiments=[]
    for path in sorted((A/'results').glob('*experiments.csv')):
        if path.name=='all_experiments.csv':continue
        d=pd.read_csv(path);d['family_file']=path.name;experiments.append(d)
    experiments=pd.concat(experiments,ignore_index=True);assert experiments.experiment.is_unique
    experiments.to_csv(A/'results/all_experiments.csv',index=False)
    checks=[]
    for name in experiments.experiment:
        op=A/f'results/oof/{name}.csv';tp=A/f'results/submissions/submission_{name}.csv'
        od=pd.read_csv(op);td=pd.read_csv(tp)
        assert od.id.is_unique and td.id.is_unique
        assert set(od.id)==set(q.id) and set(td.id)==set(te.id)
        assert list(td.columns)==['id','score'] and len(td)==362
        for d in [od,td]:assert np.isfinite(d.score).all() and d.score.between(0,1).all()
        observed=AP(q.error,od.set_index('id').loc[q.id].score)
        recorded=experiments.loc[experiments.experiment.eq(name),'pooled_ap'].iloc[0]
        assert abs(observed-recorded)<1e-12
        checks.append(dict(experiment=name,submission_sha256=sha(tp),oof_sha256=sha(op),rows=362))
    # Audit each new model's fit-frame provenance against OOF day and test.
    model_counts={}
    for path in sorted((A/'models').glob('*manifest.json')):
        models=json.loads(path.read_text());model_counts[path.stem]=len(models)
        for model in models:
            frames=set(model['fit_frames']);assert not frames&set(te.frame_id)
            fit=model['fit_day']
            if fit!='all':
                stamp=fit.replace('-','');assert all(f[-15:-7]==stamp for f in frames)
                assert not frames&set(q.loc[q.day.ne(fit),'frame_id'])
    # Final candidate intentionally isolates one person-training hypothesis.
    dest=A/'submission_v37_person_refit_RESEARCH.csv'
    shutil.copyfile(A/f'results/submissions/submission_{NAME}.csv',dest)
    td=pd.read_csv(dest).set_index('id').loc[te.id]
    unrelated=te.class_name.ne('person').to_numpy()
    assert np.allclose(td.score.to_numpy()[unrelated],BASE['test'][unrelated],rtol=0,atol=1e-15)
    pools=json.loads((A/'inputs/blind_review.json').read_text())
    assert sum(len(pools[k]) for k in ['present','absent','ambiguous'])==64
    poolsets=[set(pools[k]) for k in ['present','absent','ambiguous']]
    assert len(set.union(*poolsets))==64
    out=dict(status='RESEARCH_ONLY_NOT_A_NEW_BEST',recommendation='Hold: weak OOF delta, no demonstrated public gain. Do not spend remaining attempts on all variants.',
        best_public_version='V31',best_public_ap=.96239,public_ap=None,submitted=False,
        candidate=dest.name,sha256=sha(dest),experiment=NAME,
        validation_scope='Two opposite-day folds. Full-fit inference on both days is a distinct deployment policy, not separately validated.',
        diagnostics=json.loads((A/'results/person_stability.json').read_text()),
        variants_checked=len(checks),fit_model_manifests=model_counts,prior_release_hashes=frozen,
        pool_roles=dict(appearance='64 train-pool crops, blinded before predictions; never fitted',
                        hand_context='V36/V37 known presence labels promoted into own training day; only opposite-day diagnostics',
                        near_hand='New V36/V37 pool not fitted',
                        negative_occupancy='Reviewed negative presence regions promoted into own training day',
                        person_release='No pool labels fitted'),
        test_annotations_used_for_training_or_id_overrides=False,
        test_inspection='Visual diagnosis and branch rejection; not official labels or a test AP estimate.')
    (A/'release_manifest.json').write_text(json.dumps(out,indent=2));pd.DataFrame(checks).to_csv(A/'results/artifact_checks.csv',index=False)
    versions={}
    for package in ['numpy','pandas','scikit-learn','torch','transformers','opencv-python','Pillow','joblib']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not_found_under_this_name'
    (A/'runtime_versions.json').write_text(json.dumps(versions,indent=2))
    print(json.dumps(dict(candidate=dest.name,sha256=sha(dest),variants_checked=len(checks),model_manifests=model_counts,status=out['status']),indent=2))

if __name__=='__main__':main()
