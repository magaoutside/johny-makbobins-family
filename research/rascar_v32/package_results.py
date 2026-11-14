"""Integrity checks, preserved research candidates, complete experiment ledger."""
from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,base_oof,current,metric,align
from sklearn.metrics import average_precision_score

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    baseline=ROOT/'rascar_v22/submission_v22_candidate.csv'
    assert sha(baseline)=='44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c'
    v31=ROOT/'rascar_v31/submission_v31_dense_oof098784.csv'
    assert sha(v31)=='8720eb732639532bab49eff75b956a6bd4db21b96ede309ed901889777f4ab2c'
    ledger=[]
    for path in sorted((A/'results').glob('*_experiments.csv')):
        if path.name=='all_experiments.csv':continue
        d=pd.read_csv(path);d['ledger_source']=str(path.relative_to(A));ledger.append(d)
    ledger=pd.concat(ledger,ignore_index=True);assert ledger.experiment.is_unique
    for _,r in ledger.iterrows():
        p=A/f'results/submissions/submission_{r.experiment}.csv';s=pd.read_csv(p)
        assert s.columns.tolist()==['id','score'] and s.id.tolist()==te.id.tolist() and len(s)==362 and s.id.is_unique
        assert np.isfinite(s.score).all() and s.score.between(0,1).all()
        oo=align(A/f'results/oof/{r.experiment}.csv',q)
        assert abs(average_precision_score(q.error,oo.score)-r.pooled_ap)<1e-12
        for day in ['25','26']:
            ix=q.day.eq('2026-10-'+day)
            assert abs(average_precision_score(q.loc[ix,'error'],oo.loc[ix.to_numpy(),'score'])-r['day_'+day])<1e-12
    ledger.sort_values('pooled_ap',ascending=False).to_csv(A/'results/all_experiments.csv',index=False)
    # All new trainable heads use their fitting day only. Do not treat test audit as labels.
    training=[]
    for file in ['quality_training_manifest.json','conditional_training_manifest.json']:
        entries=json.loads((A/'models'/file).read_text())
        for e in entries:
            fit=e['fit_day'];frames=e['fit_frames'];stamp=fit.replace('-','')
            assert all('_'+stamp+'_' in f for f in frames)
            assert not set(frames)&set(te.frame_id)
            assert not set(frames)&set(q.loc[q.day.ne(fit),'frame_id'])
            training.append(dict(source=file,fit_day=fit,frames=len(frames)))
    chosen='compose_watershed1_confirmed_hand_protection2'
    src=A/f'results/submissions/submission_{chosen}.csv';dst=A/'submission_v32_research_oof098800.csv';shutil.copyfile(src,dst)
    oof=align(A/f'results/oof/{chosen}.csv',q);test=align(dst,te).score.to_numpy()
    row=ledger[ledger.experiment.eq(chosen)].iloc[0]
    stability=pd.read_csv(A/'results/frame_stability.csv');st=stability[stability.experiment.eq('V32_composed')].iloc[0].to_dict()
    raw_oof=align(A/'results/oof/compose_watershed1_confirmed_hand_protection0.csv',q)
    assert np.allclose(oof.score,raw_oof.score,atol=1e-15,rtol=0)
    meta=dict(file=dst.name,sha256=sha(dst),experiment=chosen,metrics=metric(q,oof.score.to_numpy()),
        public_ap=None,submitted=False,release_status='research_only_not_promoted_over_confirmed_V22',
        best_confirmed_public_ap=.95610,best_confirmed_file=str(baseline.relative_to(ROOT)),
        previous_exploratory_oof=.9878440103167188,changed_test=int((abs(test-current)>1e-9).sum()),
        bootstrap=st,bootstrap_caveat='Post-selection exploratory interval. Not a probability of leaderboard improvement.',
        fresh_test_audit={'known_error_scores_decreased_gt002':int(row.known_error_score_down),'plausible_correct_scores_increased_gt002':int(row.plausible_correct_score_up),
                          'caveat':'Visual categories are nonofficial, uncertain and sometimes contradicted by later boundary review; counts are warnings, not errors or AP.'},
        rule_scope='All matching queries; no hard-coded detection IDs. Existing V22 rescue guard retained.',
        no_upload_reason='User reports exhausted leaderboard submissions')
    (A/'submission_v32_research_oof098800.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    control='compose_watershed1_prior_vetoes_preserved'
    control_src=A/f'results/submissions/submission_{control}.csv';control_dst=A/'submission_v32_control_oof098711.csv';shutil.copyfile(control_src,control_dst)
    control_oof=align(A/f'results/oof/{control}.csv',q)
    (A/'submission_v32_control_oof098711.json').write_text(json.dumps(dict(file=control_dst.name,sha256=sha(control_dst),experiment=control,
        metrics=metric(q,control_oof.score.to_numpy()),public_ap=None,submitted=False,release_status='research_control_not_promoted',
        distinction='Preserves prior semantic/duplicate/structure rescue vetoes for every class. Lower OOF, fewer visually suspicious rescues.'),indent=2))
    hard=pd.read_csv(A/'audit/hard_case_score_comparison.csv').drop(columns=['v32_guarded','v32_control'],errors='ignore').merge(pd.read_csv(dst).rename(columns={'score':'v32_guarded'}),on='id')
    hard=hard.merge(pd.read_csv(control_dst).rename(columns={'score':'v32_control'}),on='id')
    hard.to_csv(A/'audit/hard_case_score_comparison.csv',index=False)
    checks=dict(leaderboard_uploads=0,experiment_rows=len(ledger),submissions_validated=len(ledger),oof_metrics_recomputed=True,
        baseline_hash_unchanged=True,v31_hash_unchanged=True,models_newly_trained=len(training),fold_checks=training,
        winner_oof_equals_bootstrapped_composition=True,manual_pool_controls_not_official_gt=True)
    (A/'results/integrity_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(meta,ensure_ascii=False,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
