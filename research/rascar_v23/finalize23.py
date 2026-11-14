"""Freeze the candidate, research ablation, experiment registry and audit facts."""
from common23 import *
from collections import Counter
import shutil
import joblib

NAME = 'combined_catboost_cup_rescue_w0.4'
TABLE = 'table_edge_far_w0.25'
R = A / 'release'
(R / 'models').mkdir(parents=True, exist_ok=True)
candidate = json.loads((A / f'results/{NAME}.json').read_text())
table = json.loads((A / f'results/{TABLE}.json').read_text())
base = json.loads((A / 'results/baseline_v22.json').read_text())
checks = {'base_sha256_unchanged': hashlib.sha256(baseline.read_bytes()).hexdigest(), 'folds': []}

for fit, held in zip(DAYS, DAYS[::-1]):
    training = pd.read_csv(A / f'inputs/train_{fit}.csv')
    assert training.day.eq(fit).all()
    assert not set(training.frame_id) & set(q.loc[q.day.eq(held), 'frame_id'])
    model = joblib.load(A / f'models/combined_catboost_cup_{fit}.joblib')
    assert model['fit_day'] == fit
    assert set(model['training_frames']) <= set(training.frame_id)
    shutil.copyfile(A / f'models/combined_catboost_cup_{fit}.joblib', R / 'models' / f'cup_{fit}.joblib')
    for split, source, rows in [('oof', 'held', q[q.day.eq(held)]), ('test', 'test', te)]:
        features = align(A / f'features/{source}_{fit}.csv', rows)
        assert np.isfinite(features.to_numpy(float)).all()
        features.reset_index().to_csv(R / f'{split}_features_{fit}.csv', index=False)
    checks['folds'].append(dict(fit=fit, held=held, training_rows=len(training),
                               training_frames=int(training.frame_id.nunique()),
                               held_queries=int(q.day.eq(held).sum()), frame_overlap=0))

for split, rows, baseline_scores in [('oof', q, base_oof), ('test', te, current)]:
    structure = align(ROOT / f'rascar_v21/object_structure_{split}.csv', rows)
    semantic = align(ROOT / f'rascar_v16/semantic_both_{split}.csv', rows, 'p_cup')
    frozen = rows[['id', 'class_name', 'frame_id']].copy()
    frozen['day'] = rows.day.to_numpy() if 'day' in rows else pd.to_datetime(rows.frame_id.str[-15:-7], format='%Y%m%d').dt.strftime('%Y-%m-%d').to_numpy()
    frozen['baseline'] = baseline_scores
    frozen['semantic_probability'] = semantic
    frozen['distinct_instances'] = structure.distinct_instances.to_numpy()
    frozen['ambiguous_duplicate'] = structure.ambiguous_duplicate.astype(bool).to_numpy()
    frozen.to_csv(R / f'{split}_inputs.csv', index=False)
    source = A / (f'results/oof/{NAME}.csv' if split == 'oof' else f'results/submissions/submission_{NAME}.csv')
    expected = pd.read_csv(source)[['id', 'score']]
    assert expected.id.equals(rows.id.reset_index(drop=True)) and not expected.id.duplicated().any()
    assert np.isfinite(expected.score).all() and expected.score.between(0, 1).all()
    assert np.max(abs(expected.score.to_numpy()[rows.class_name.ne('cup')] - baseline_scores[rows.class_name.ne('cup')])) < 1e-12
    expected.to_csv(R / f'{split}_expected.csv', index=False)

shutil.copyfile(A / f'results/submissions/submission_{NAME}.csv', A / 'submission_v23_cv_cup_candidate.csv')
shutil.copyfile(A / f'results/submissions/submission_{TABLE}.csv', A / 'submission_v23_table_zone_experiment.csv')

# Keep the baseline's user-reported LB separate from unsubmitted experiments.
records = [json.loads(p.read_text()) for p in (A / 'results').glob('*.json')]
for record in records:
    if record['experiment'] == 'baseline_v22':
        record['lb'] = .95610
        record['submitted'] = True
        record['lb_source'] = 'User confirmed main V22 candidate'
pd.DataFrame(records).sort_values('pooled_ap', ascending=False).to_csv(A / 'results/experiment_registry.csv', index=False)

# Paired frame bootstrap within days; descriptive uncertainty after selection.
new = align(A / f'results/oof/{NAME}.csv', q, 'score')
rng = np.random.default_rng(2301)
groups = [[np.flatnonzero(q.frame_id.eq(fid).to_numpy()) for fid in q.loc[q.day.eq(day), 'frame_id'].unique()] for day in DAYS]
deltas = []
for _ in range(3000):
    indices = np.concatenate([np.concatenate([group[k] for k in rng.integers(0, len(group), len(group))]) for group in groups])
    y = q.error.to_numpy()[indices]
    deltas.append(AP(y, new[indices]) - AP(y, base_oof[indices]))
bootstrap = dict(draws=3000, seed=2301, unit='frame, stratified by day',
                 delta=float(AP(q.error, new) - AP(q.error, base_oof)),
                 percentile_95=list(map(float, np.quantile(deltas, [.025, .975]))),
                 fraction_positive=float(np.mean(np.asarray(deltas) > 0)),
                 caveat='Descriptive only: selected on these same 35 frames; does not correct experiment selection or predict leaderboard gain.')

review = pd.read_csv(A / f'results/reviews/{NAME}.csv')
review[review.delta.abs() > 1e-9].to_csv(A / 'audit/candidate_changed_queries.csv', index=False)
table_scores = align(A / f'results/oof/{TABLE}.csv', q, 'score')
penalized = q.assign(baseline_v22=base_oof, new_score=table_scores)
penalized = penalized[penalized.error.eq(0) & penalized.new_score.gt(penalized.baseline_v22 + .01)]
penalized.to_csv(A / 'audit/table_penalized_correct_oof.csv', index=False)
checks.update(candidate_rows=362, candidate_unique_ids=True, candidate_score_range_valid=True,
              candidate_only_cups_changed=True, known_test_audit_not_used_as_fitting_labels=True,
              new_models_outer_day_separation_verified=True,
              scope='Checks cover new V23 models and preserved ranking inputs; upstream V22 detector OOF provenance is inherited, not independently regenerated.',
              inner_calibration_note='GroupKFold splits training records by frame. Fit-day aggregate feature priors are shared within inner folds; the outer held day remains excluded.',
              invalid_initial_skin_experiments_excluded=True)
(A / 'validation_integrity.json').write_text(json.dumps(checks, indent=2))

summary = dict(baseline=base, candidate=candidate, table_experiment=table,
               experiment_count=len(records), experiment_families=dict(Counter(r['details'].get('type', 'baseline') for r in records)),
               fitted_heads=len(list((A / 'models').glob('*.joblib'))),
               bootstrap=bootstrap, extra_annotations=json.loads((A / 'inputs/extra_annotations_usage.json').read_text()),
               newly_closed_at_frozen_threshold=0, known_mismatches_remaining=26,
               priority_passes=18, priority_total=41, priority_pass_fraction=18/41,
               table_correct_oof_penalized_gt001=len(penalized),
               leaderboard_submissions_this_round=0, confirmed_best_lb=.95610)
(A / 'summary.json').write_text(json.dumps(summary, indent=2))
note = dict(frozen_audit_preserved=True,
            observation='Several manually correct person boxes end near the tabletop. A short-looking occluded extent is not sufficient evidence of an official error.',
            effect='Keep torso-only/lower-extent visual cases provisional; do not claim these are confirmed hidden-test errors or automatically stretch all seated people.',
            candidate_review='All 14 changes above 0.02 inspected. 13 plausible-correct cups improve ranking margin. The remaining one is an uncertain duplicate winner (#40); loser #39 stays 0.95. Its tighter box is visually reasonable, but hidden GT is unknown.',
            hand_rule_review='Visual review confirms #183 and #191 forearm regressions; wrist proximity does not certify box boundaries.',
            inspected_files=[p.name for p in (A / 'audit').glob('*.jpg')])
(A / 'audit/review_record.json').write_text(json.dumps(note, indent=2))
manifest = {str(p.relative_to(A)): hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file()}
(A / 'release_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(experiments=len(records), bootstrap=bootstrap, candidate=candidate['pooled_ap'], table_correct_penalized=len(penalized)), indent=2))
