"""Record completed V40 experiments without promoting a release."""
from common40 import *

def main():
    model_rows=[]
    for path in sorted((A/'models').glob('*_manifest.json')):
        for entry in json.loads(path.read_text()):
            fit=entry['fit_day'];frames=set(entry['fit_frames']);assert len(frames)==42
            assert all(f'_{fit.replace("-", "")}_' in f for f in frames)
            assert not frames&set(te.frame_id);assert not frames&set(q[q.day.ne(fit)].frame_id)
            model=A/'models'/f'{entry["name"]}.pt'
            model_rows.append(dict(name=entry['name'],fit_day=fit,seed=entry['seed'],steps=entry['steps'],fit_frames=len(frames),last_loss=entry['last_loss'],sha256=hashlib.sha256(model.read_bytes()).hexdigest()))
    assert len(model_rows)==18
    pd.DataFrame(model_rows).to_csv(A/'audit/model_manifest.csv',index=False)
    summaries=[pd.read_csv(A/f'results/{p}') for p in ['refine16_identity_experiments.csv','refine32_identity_experiments.csv','quality_experiments.csv']]
    summaries.append(pd.DataFrame([json.loads((A/'results/quality_verified.json').read_text())]))
    all_results=pd.concat(summaries,ignore_index=True);all_results.to_csv(A/'results/all_experiments.csv',index=False)
    artifacts=[]
    for split,rows in [('oof',q),('test',te)]:
        prefix='results/oof' if split=='oof' else 'results/submissions'
        for path in sorted((A/prefix).glob('*.csv')):
            d=pd.read_csv(path);assert d.id.is_unique and set(d.id)==set(rows.id)
            s=d.set_index('id').loc[rows.id].score.to_numpy();assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
            if split=='test':assert list(d.columns)==['id','score']
            ss=np.sign(s[:,None]-s[None]);bb=np.sign(BASE[split][:,None]-BASE[split][None]);upper=np.triu(np.ones(ss.shape,bool),1)
            artifacts.append(dict(path=str(path.relative_to(ROOT)),split=split,rows=len(s),changed_scores=int((abs(s-BASE[split])>1e-10).sum()),changed_pair_orders=int(((ss!=bb)&upper).sum()),ap=float(AP(q.error,s)) if split=='oof' else None,vector_sha256=hashlib.sha256(np.round(s,14).tobytes()).hexdigest(),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    art=pd.DataFrame(artifacts);art.to_csv(A/'audit/artifact_checks.csv',index=False)
    assert art.groupby('split').size().to_dict()=={'oof':51,'test':51}
    frozen={
        'rascar_v22/submission_v22_candidate.csv':'44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c',
        'rascar_v31/submission_v31_dense_oof098784.csv':'8720eb732639532bab49eff75b956a6bd4db21b96ede309ed901889777f4ab2c',
        'rascar_v37/submission_v37_person_refit_RESEARCH.csv':'105e980e9465c00692e5a50bc8a5cf895f2b5e9a2b55f6c7e4682e6278f3afb1'}
    for path,expected in frozen.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected
    manifest=dict(status='completed_research_not_promoted',submitted=False,baseline='V31',baseline_public_ap_user_reported=.96239,baseline_oof_ap=float(AP(q.error,BASE['oof'])),trained_models=len(model_rows),research_variants=51,unique_oof_vectors=int(art[art.split.eq('oof')].vector_sha256.nunique()),unique_test_vectors=int(art[art.split.eq('test')].vector_sha256.nunique()),frozen_files=frozen,scope='Finish already launched 16/32 grid and quality runs; audit frozen V31. No new training branches after user requested analysis.',new_test_labels=False)
    (A/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
