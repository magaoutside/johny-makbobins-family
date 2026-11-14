"""Freeze, validate and package V53's next submission without using test labels."""
from experiment import *
import shutil
import subprocess
import zipfile
from scipy.stats import spearmanr


def main():
    q, te, h = inputs()
    bo, bt = baseline(q, te)
    incumbent = pd.read_csv(V / 'results/oof/purged_ensemble_d2_person_a20.csv')
    incumbent_test = pd.read_csv(V / 'release/submission_v52_candidate_purged_a20.csv')
    p = pd.read_csv(A / 'features/anchor_d2_initial_predictions.csv')
    fold = p[p.split.eq('test') & p.fit_day.ne('full')].groupby('id').p_error.mean().loc[te.id].to_numpy()
    full = p[p.split.eq('test') & p.fit_day.eq('full')].groupby('id').p_error.mean().loc[te.id].to_numpy()
    mask = te.class_name.eq('person').to_numpy()
    release = A / 'release'
    release.mkdir(exist_ok=True)
    checks = {}
    for weight in [.25, .5, 1.]:
        score = bt.copy()
        probability = (1 - weight) * fold + weight * full
        score[mask] = (score[mask] + .2 * probability[mask]) / 1.2
        output = te[['id']].assign(score=score)
        assert output.id.tolist() == incumbent_test.id.tolist()
        assert np.array_equal(score[~mask], incumbent_test.score.to_numpy()[~mask])
        assert output.id.is_unique and np.isfinite(score).all() and output.score.between(0, 1).all()
        name = f'submission_v53_refit{round(weight * 100):02}'
        output.to_csv(A / f'results/submissions/{name}.csv', index=False)
        if weight == .5:
            output.to_csv(release / f'{name}.csv', index=False)
        old = incumbent_test.score.to_numpy()
        diff = abs(score - old)
        a = np.sign(old[:, None] - old[None, :])
        b = np.sign(score[:, None] - score[None, :])
        checks[name] = dict(rows=len(output), changed_scores=int((diff > 1e-10).sum()),
                           maximum_score_change=float(diff.max()), mean_person_score_change=float(diff[mask].mean()),
                           changed_rank_pairs=int(np.triu(a != b, 1).sum()),
                           spearman=float(spearmanr(old, score).statistic),
                           oof_claim='No independent OOF for the full-data refit; original recipe is unchanged.')
        te.assign(old_score=old, new_score=score, change=score-old).to_csv(A / f'audit/{name}_test_changes.csv', index=False)
    # Preserve the evaluated feature-only alternative separately; do not mix it into refit50.
    cal = pd.read_csv(A / 'features/calibrated_d2_confirm_predictions.csv')
    oldp = pd.read_csv(V / 'features/purged_ensemble_d2_predictions.csv')
    co = cal[cal.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy()
    ct = cal[cal.split.eq('test') & cal.fit_day.ne('full')].groupby('id').p_error.mean().loc[te.id].to_numpy()
    oo = oldp[oldp.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy()
    score, st = bo.copy(), bt.copy()
    m = q.class_name.eq('person').to_numpy()
    score[m] = (score[m] + .2 * (.75 * oo[m] + .25 * co[m])) / 1.2
    st[mask] = (st[mask] + .2 * (.75 * fold[mask] + .25 * ct[mask])) / 1.2
    name = 'calibrated25_person_a20'
    q.assign(score=score).to_csv(A / f'results/oof/{name}.csv', index=False)
    te[['id']].assign(score=st).to_csv(A / f'results/submissions/{name}.csv', index=False)
    calibration_metrics = metrics(q, score)
    pd.DataFrame([dict(name=name, **calibration_metrics)]).to_csv(A / 'results/calibrated_alternative.csv', index=False)
    # Inspect every initial and confirmation result, including weak/negative outcomes.
    experiments = pd.concat([pd.read_csv(A / f'results/experiments_{tag}.csv') for tag in ['initial', 'confirm']], ignore_index=True)
    for row in experiments.itertuples():
        oof = pd.read_csv(A / f'results/oof/{row.name}.csv')
        test = pd.read_csv(A / f'results/submissions/{row.name}.csv')
        assert oof.id.tolist() == q.id.tolist() and test.id.tolist() == te.id.tolist()
        assert np.array_equal(oof.error, q.error)
        for x in [oof, test]:
            assert x.id.is_unique and np.isfinite(x.score).all() and x.score.between(0, 1).all()
        actual = metrics(q, oof.score)
        recorded = experiments.set_index('name').loc[row.name]
        assert all(abs(value - recorded[key]) < 1e-12 for key, value in actual.items())
    experiments.sort_values('ap', ascending=False).to_csv(A / 'results/all_experiments.csv', index=False)
    # Verify split ledgers for both cross-day/full models and the mixed-day probe.
    rows = pd.read_csv(V / 'inputs/rows.csv').set_index('id')
    models_checked = 0
    for tag in ['initial', 'confirm']:
        for entry in json.loads((A / f'models/manifest_{tag}.json').read_text()):
            tr = rows.loc[entry['train_ids']]
            assert tr.split.eq('oof').all()
            if entry['fit_day'] != 'full':
                assert tr.day.eq(entry['fit_day']).all()
                assert all(entry['fit_day'].replace('-', '') in f for f in entry['prototype_frames'])
            assert not set(entry['features']) & {'id', 'day', 'frame_id', 'error', 'score'}
            models_checked += 1
    for entry in json.loads((A / 'models/refit_probe_manifest.json').read_text()):
        tr, val = rows.loc[entry['training_ids']], rows.loc[entry['held_ids']]
        assert tr.split.eq('oof').all() and not set(tr.frame_id) & set(val.frame_id)
        assert np.abs(timestamps(tr)[:, None] - timestamps(val)[None, :]).min() > 120
        assert not set(entry['prototype_frames']) & set(val.frame_id)
        models_checked += 1
    # Export the old six fold models and the three full-data refits on the SAME seeds.
    original = json.loads((V / 'release/manifest.json').read_text())
    shutil.copytree(V / 'release/inputs', release / 'inputs', dirs_exist_ok=True)
    (release / 'models').mkdir(exist_ok=True)
    entries = []
    for entry in original['models']:
        shutil.copy2(V / 'release' / entry['path'], release / entry['path'])
        entries.append(entry)
    for entry in json.loads((A / 'models/manifest_initial.json').read_text()):
        if entry['name'] == 'anchor_d2' and entry['fit_day'] == 'full':
            path = entry['model_path']
            shutil.copy2(A / path, release / path)
            entries.append(dict(entry, path=path))
    assert len(entries) == 9
    incumbent[['id', 'score']].to_csv(release / 'inputs/v52_oof_reference.csv', index=False)
    hashes = {str(path.relative_to(release)): hashlib.sha256(path.read_bytes()).hexdigest()
              for folder in ['models', 'inputs'] for path in sorted((release / folder).glob('*')) if path.is_file()}
    manifest = dict(models=entries, sha256=hashes, full_fit_weight=.5, alpha=.2, scope='person',
                    seeds=[23, 24, 25], baseline_user_reported_public_ap=.96300,
                    full_fit_training_queries=198, original_fit_training_queries=[68, 130],
                    validation='Original cross-day recipe unchanged. Mixed-day blocked probe is easier and diagnostic only.',
                    predicted_hidden_test_ap=None)
    (release / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(A / 'replay_release.py', release / 'replay_release.py')
    shutil.copy2(V / 'release/requirements.txt', release / 'requirements.txt')
    subprocess.run([sys.executable, str(release / 'replay_release.py')], check=True)
    checks['replay'] = json.loads((release / 'replayed/checks.json').read_text())
    checks['checked_named_experiments'] = len(experiments)
    checks['training_manifests_checked'] = models_checked
    checks['baseline_sha256'] = hashlib.sha256((V / 'release/submission_v52_candidate_purged_a20.csv').read_bytes()).hexdigest()
    (A / 'results/integrity_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    probe = pd.read_csv(A / 'results/refit_probe.csv')
    refit = probe[probe.recipe.eq('anchor_d2')]
    report = f'''V53 — продолжение после Public AP 0.96300, 14 ноября 2026

ОТПРАВЛЕННАЯ БАЗА
Пользователь сообщил 0.96300 после рекомендации V52 purged a20.
Это +0.00061 к ранее сообщённому V31 0.96239. Сайт Kaggle независимо не проверялся.
Путь и SHA256 отправленного кандидата: inputs/incumbent.json.

СЛЕДУЮЩИЙ КАНДИДАТ
release/submission_v53_refit50.csv
Тот же CatBoost base23 + purged CLIP/DINO GT-прототипы, тот же person-only alpha=0.2.
50% p_error от прежнего ансамбля из 6 моделей + 50% от трёх новых моделей,
обученных на всех 198 официальных query и GT-прототипах обоих обучающих дней.
Оставлены прежние seeds 23,24,25; выбор удачного seed не проводился.
Раньше каждая модель видела только 68 или 130 query. Классы cup/hands/tray не меняются.
При создании обучающих признаков эталоны того же кадра и в пределах 120 секунд
по-прежнему исключаются. Никакие тестовые метки или ручные статусы не использованы.

ОЦЕНКА
OOF исходного рецепта остаётся 0.988579935. Это НЕ новая независимая OOF-оценка
full-data refit: модель, обученная на всех данных, не имеет held-out строк среди них.
Проверка дополнительной гипотезы: исключались 3 временных блока в каждом дне,
остальные кадры обоих дней шли в обучение; соседние 120 секунд и GT-прототипы
всего проверочного временного блока исключались. Все 198 query проверены по одному разу.
Это диагностический, более лёгкий split; он не заменяет перенос на новый день.
{refit.to_string(index=False, float_format=lambda v: f'{v:.9f}')}

ПЕРЕБОР
{len(experiments)} комбинации моделей/коэффициентов в initial+confirm.
Проверены: эталоны из разных кадров, контекст и разность ROI/context;
сходство с размеченными правильными/ошибочными query (только train, purge 120 секунд);
геометрические соседи, дополнительные геометрические признаки, обучение только
на person, вес person, Ordered boosting, регуляризация и отсутствие bootstrap;
регрессия GT IoU вместо бинарной ошибки; калибровка границ OWLv2 по train GT.
Для шести семейств повторены эксперименты на пяти дополнительных seeds 26–30.
Выигрыш многих вариантов оказался меньше шума или сопровождался падением одного дня.
Исходный рецепт V52 воспроизведён ТОЧНО: максимальное расхождение p_error = 0.
Три GT-кадра без официальных query сохранены в банке обучения, как в V52.

КАЛИБРОВКА OWLv2 — ОТДЕЛЬНАЯ АЛЬТЕРНАТИВА
По train GT оценены медианные смещения каждой границы, shrinkage n/(n+10),
ограничение +/-0.4 размера рамки, reference confidence>=0.1 и train IoU>=0.35.
Для обучающих query калибратор исключает их кадр и соседние 120 секунд.
25% такого CatBoost в p_error V52: OOF {calibration_metrics['ap']:.9f}.
Дневные AP {calibration_metrics[DAYS[0]]:.9f} / {calibration_metrics[DAYS[1]]:.9f};
это небольшой post-hoc прирост, без подтверждения на Kaggle.
Файл results/submissions/calibrated25_person_a20.csv сохранён отдельно и не входит
в основной refit50: иначе следующий Public score смешал бы две разные гипотезы.

ПРОВЕРКИ
Все {len(experiments)} таблицы AP сверены с OOF CSV. Проверены ID, порядок, диапазон,
уникальность, NaN/inf, отсутствие train/test пересечений в {models_checked} записях обучения.
Пакет содержит 9 весов и все кэшированные признаки для автономного replay.
replay_release.py заново строит прототипные признаки, загружает веса, воспроизводит
новый test CSV и старый OOF reference с точностью <1e-10. Он не читает error-метки.
Пакет не является extractor для новых фотографий; он воспроизводит эти 362 query.
Изменения refit50: {json.dumps(checks['submission_v53_refit50'], ensure_ascii=False)}

ФАЙЛЫ
results/all_experiments.csv — все initial/confirm результаты, включая проигравшие.
results/refit_probe.csv — диагностика добавления второго дня в обучение.
results/integrity_checks.json — проверки.
models/ — все веса и протоколы разбиения.
release/submission_v53_refit50.csv — следующий рекомендуемый эксперимент.
release/replay_release.py — воспроизведение без сети; запуск python3 replay_release.py.

Public/Private AP V53 пока неизвестен. Улучшение относительно 0.96300 не заявляется.
Кандидат предназначен проверить пользу обучения на всех доступных данных.

Документация параметров CatBoost:
https://catboost.ai/docs/en/references/training-parameters/common
Описание штатных embedding features (в V53 сравниваются собственные purged признаки):
https://catboost.ai/docs/en/concepts/algorithm-main-stages_embedding-to-numeric
'''
    (A / 'REPORT_RU.txt').write_text(report)
    (release / 'REPORT_RU.txt').write_text(report)
    shutil.copy2(A / 'results/integrity_checks.json', release / 'integrity_checks.json')
    archive = ROOT / 'rascar_v53_refit50_replay.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(release.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                z.write(path, 'rascar_v53_refit50/' + str(path.relative_to(release)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps(checks, indent=2))
    print('ARCHIVE', archive, archive.stat().st_size)


if __name__ == '__main__':
    main()
