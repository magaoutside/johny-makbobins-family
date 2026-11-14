"""Export a self-contained cached-feature replay and immutable release files."""
from common19 import *
import shutil
import importlib.util
spec = importlib.util.spec_from_file_location('v19_release_replay', A / 'replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
predict = replay.predict

INPUT = A / 'replay_inputs'
INPUT.mkdir(exist_ok=True)
(A / 'release_oof').mkdir(exist_ok=True)
sources = {}


def saved(path, rows):
    sources[str(path.relative_to(A.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return align(path, rows)


checks = {}
for rows, base, test in [(q, q.score.to_numpy(), False), (te, current, True)]:
    split = 'test' if test else 'oof'
    columns = ['id', 'frame_id', 'class_name'] + B + ([] if test else ['day', 'error'])
    d = rows[columns].reset_index(drop=True).copy()
    if test:
        d['day'] = audit.set_index('id').loc[rows.id, 'day'].to_numpy()
    d['baseline_v18'] = base
    d['baseline_v16'] = old.old.current if test else old.old.q.score.to_numpy()
    point = saved(V18 / f'person_body_ridge_{split}.csv', rows)
    for c in ['height_ratio', 'top_error', 'width_ratio']:
        d['body_' + c] = point[c].to_numpy()
    d['body_point_eligible'] = boolean(point.eligible).astype(int)
    seat = saved(V17 / ('seat_features_test_all.csv' if test else 'seat_features_oof.csv'), rows)
    for c in ['head_count', 'template_n', 'seat_distance']:
        d[c if c == 'seat_distance' else 'seat_' + c] = seat[c].to_numpy()
    d['seat_back'] = boolean(seat.back).astype(int)
    prob = saved(A / f'body_uncertainty_{split}.csv', rows)
    d['body_error_probability'] = prob.body_error_probability.to_numpy()
    d['body_residual_support'] = prob.residual_support.to_numpy()
    d['head_coverage'] = saved(A / f'body_head_coverage_{split}.csv', rows).head_coverage.to_numpy()
    mp = saved(A / f'landmark_joint_features_{split}.csv', rows)
    for c in ['iou', 'views', 'turns', 'coherence', 'point_fraction', 'palm_fraction', 'instances']:
        d['mp_' + c] = mp[c].to_numpy()
    # Test values are a fold mean (0, .5, 1); .5 must NOT be cast to True.
    d['mp_assigned'] = (pd.to_numeric(mp.assigned, errors='coerce').fillna(pd.Series(boolean(mp.assigned), index=mp.index).astype(float))).to_numpy()
    for col, file in [('yolo_old_instances', 'tta_v15'), ('yolo_refined_instances', 'tta_refined')]:
        d[col] = saved(V17 / f'{file}_{split}.csv', rows).instances.to_numpy()
    d['duplicate_veto'] = duplicate_veto(rows, test).astype(int)
    d['wevis_hand_instances'] = saved(V16 / f'wevis_features_{split}.csv', rows).hand_instances.to_numpy()
    if not test:
        without = align(V18 / 'results/oof/release_main_no_inherited_contradiction.csv', rows, 'score')
        d['inherited_oof_only_correction'] = base - without
    d.to_csv(INPUT / f'{split}.csv', index=False)
    roundtrip = pd.read_csv(INPUT / f'{split}.csv')
    for variant, weight in [('main', .75), ('conservative', .5)]:
        name = f'grouped_body{weight}_hands0.5'
        src = R / ('submissions' if test else 'oof') / (('submission_' if test else '') + name + '.csv')
        dst = A / f'submission_v19_{variant}.csv' if test else A / 'release_oof' / f'{variant}.csv'
        shutil.copy2(src, dst)
        expected = align(src, rows, 'score')
        actual = predict(roundtrip, body_weight=weight)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        untouched = rows.class_name.isin(['cup', 'tray']).to_numpy()
        np.testing.assert_array_equal(actual[untouched], base[untouched])
        checks[f'{split}_{variant}'] = dict(rows=len(rows), max_absolute_replay_error=float(abs(actual - expected).max()), ids_unique=bool(rows.id.is_unique), cup_tray_unchanged=True)

# Link the release to the exact features consumed; no test-audit column is used
# by replay.predict. The audit stays separate from the model input tables.
(A / 'replay_sources_sha256.json').write_text(json.dumps(sources, indent=2))
(A / 'replay_checks.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
