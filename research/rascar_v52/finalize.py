"""Reconcile V52 results, verify metrics, and export a replayable candidate."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
A = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'rascar_v41'))
from common import inputs, baseline, metrics, DAYS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    q, test, _ = inputs()
    bo, bt = baseline(q, test)
    source_tables = []
    exclude = {'all_experiments', 'robustness', 'family_summary', 'nested_selection_scores'}
    for path in sorted((A / 'results').glob('*.csv')):
        if path.stem in exclude:
            continue
        frame = pd.read_csv(path)
        if {'name', 'ap'} <= set(frame.columns):
            source_tables.append(frame.assign(source_table=path.name))
    raw = pd.concat(source_tables, ignore_index=True)
    for name, group in raw.groupby('name'):
        assert np.ptp(group.ap.to_numpy()) < 1e-12, name
    results = raw.drop_duplicates('name').copy()
    submission_names = {p.stem for p in (A / 'results/submissions').glob('*.csv')}
    assert set(results.name) == submission_names
    max_metric_error = 0.
    fingerprints = set()
    for row in results.itertuples():
        oof = pd.read_csv(A / f'results/oof/{row.name}.csv')
        submission = pd.read_csv(A / f'results/submissions/{row.name}.csv')
        assert oof.id.is_unique and oof.id.tolist() == q.id.tolist()
        assert np.array_equal(oof.error, q.error)
        assert list(submission.columns) == ['id', 'score']
        assert submission.id.is_unique and submission.id.tolist() == test.id.tolist()
        for frame in [oof, submission]:
            assert np.isfinite(frame.score).all() and frame.score.between(0, 1).all()
        actual = metrics(q, oof.score)
        source = results.loc[results.name.eq(row.name)].iloc[0]
        for key, value in actual.items():
            error = abs(value - source[key])
            assert error < 1e-12, (row.name, key, error)
            max_metric_error = max(max_metric_error, error)
        fingerprints.add(hashlib.sha256(np.round(np.r_[oof.score, submission.score], 12).tobytes()).hexdigest())
    results = results.sort_values(['ap', 'name'], ascending=[False, True])
    results.to_csv(A / 'results/all_experiments.csv', index=False)
    family = raw.sort_values('ap', ascending=False).groupby('source_table').head(1)
    family = family[['source_table', 'name', 'ap', *DAYS]].copy()
    family['named_variants'] = family.source_table.map(raw.groupby('source_table').size())
    family.to_csv(A / 'results/family_summary.csv', index=False)

    release = A / 'release'
    for name in ['inputs', 'models', 'expected']:
        (release / name).mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(A / 'inputs/rows.csv')
    rows[['id', 'frame_id', 'day', 'class_name', 'split']].to_csv(release / 'inputs/rows.csv', index=False)
    shutil.copy2(A / 'inputs/prototypes.csv', release / 'inputs/prototypes.csv')
    groups = json.loads((A / 'inputs/groups.json').read_text())
    static = pd.read_csv(A / 'features/static.csv')
    static[['id', *groups['base23'], 'class_name']].to_csv(release / 'inputs/static.csv', index=False)
    for backbone in ['clip', 'vit_b', 'vit_l']:
        for suffix, target in [('', '_roi'), ('_prototypes', '_prototype_roi')]:
            x = np.load(A / f'features/{backbone}{suffix}.npy')
            width = x.shape[1] // 3
            np.save(release / f'inputs/{backbone}{target}.npy', x[:, width:2 * width])
    pd.DataFrame({'id': q.id, 'score': bo}).to_csv(release / 'inputs/baseline_oof.csv', index=False)
    pd.DataFrame({'id': test.id, 'score': bt}).to_csv(release / 'inputs/baseline_test.csv', index=False)
    entries = []
    for seed, suffix in [(23, ''), (24, '_s24_p16'), (25, '_s25_p16')]:
        stem = f'base23_prototypes_cb2{suffix}_purgedproto'
        manifest = json.loads((A / f'models/manifest{suffix}_purgedproto.json').read_text())
        for fit in DAYS:
            source = next(m for m in manifest if m['model'] == stem and m['fit_day'] == fit)
            name = f'{stem}_fit{fit[-2:]}.cbm'
            shutil.copy2(A / 'models' / name, release / 'models' / name)
            entries.append(dict(source, seed=seed, path='models/' + name))
    for alpha in [.1, .2]:
        tag = f'a{round(alpha * 100)}'
        name = f'purged_ensemble_d2_person_{tag}'
        oof = pd.read_csv(A / f'results/oof/{name}.csv')
        oof[['id', 'score']].to_csv(release / f'expected/oof_{tag}.csv', index=False)
        for target in [f'expected/test_{tag}.csv', f'submission_v52_candidate_purged_{tag}.csv']:
            shutil.copy2(A / f'results/submissions/{name}.csv', release / target)
    shutil.copy2(A / 'replay_candidate.py', release / 'replay_candidate.py')
    versions = {name: importlib.metadata.version(name) for name in ['catboost', 'numpy', 'pandas']}
    (release / 'requirements.txt').write_text(''.join(f'{name}=={version}\n' for name, version in versions.items()))
    hashes = {str(p.relative_to(release)): digest(p) for folder in ['inputs', 'models', 'expected']
              for p in sorted((release / folder).glob('*')) if p.is_file()}
    definition = dict(
        status='Research candidate, not promoted over V31; no hidden-test score',
        fit_days=DAYS, seeds=[23, 24, 25], alpha=[.1, .2], scope='person',
        formula='(V31 + alpha * mean(p_error))/(1 + alpha) for person; V31 otherwise',
        oof='three equal-seed models trained on the opposite day only',
        test='six equal-weight models: two training days times three seeds; no full-data refit',
        prototypes='Training-day GT crops only. Same/nearby training frames excluded within 120 seconds.',
        selection='Depth, class scope and alpha compared on repeatedly used development OOF',
        feature_count=50, numeric_features=49, categorical_features=['class_name'],
        models=entries, sha256=hashes, versions=versions,
        baseline_sha256=digest(ROOT / 'rascar_v31/submission_v31_dense_oof098784.csv'),
    )
    (release / 'manifest.json').write_text(json.dumps(definition, indent=2) + '\n')
    subprocess.run([sys.executable, str(release / 'replay_candidate.py')], check=True)
    replay_checks = json.loads((release / 'replayed/checks.json').read_text())
    integrity = json.loads((A / 'results/integrity_checks.json').read_text())
    assert integrity['valid_submission_oof_pairs'] == len(results)
    verification = dict(named_variants=len(results), distinct_prediction_pairs_rounded_12_digits=len(fingerprints),
                        all_table_metrics_recomputed=True, maximum_metric_error=max_metric_error,
                        all_submission_ids_and_ranges_checked=True, replay=replay_checks,
                        baseline_sha256=definition['baseline_sha256'])
    (A / 'results/final_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    baseline_metrics = metrics(q, bo)
    names = ['base13_cb2_standalone', 'base23_siglip2_cb2_a10',
             'ensemble_base23_prototypes_cb4_person_a10', 'purged_ensemble_d2_person_a10',
             'purged_ensemble_d2_person_a20', 'nested_selected_a10']
    table = [{'name': 'V31 baseline', **baseline_metrics}]
    table += [results.set_index('name').loc[name].to_dict() | {'name': name} for name in names]
    comparison = pd.DataFrame(table)[['name', 'ap', *DAYS, 'person']]
    robust = pd.read_csv(A / 'results/robustness.csv').set_index('name')
    chosen = robust.loc['purged_ensemble_d2_person_a20']
    conservative = robust.loc['purged_ensemble_d2_person_a10']
    model_count = len(list((A / 'models').glob('*.cbm')))
    report = f'''V52 — CatBoost на признаках бокса и визуальных эмбеддингах
Итог восстановлен и проверен 14 ноября 2026 года.

РЕЗУЛЬТАТ
Перебор завершён: {len(results)} именованных вариантов score, {len(fingerprints)} различных
пар OOF/test предсказаний после округления до 12 знаков, {model_count} файлов CatBoost.
Варианты включают повторяющиеся предсказания, коэффициенты и ансамбли; это не
{len(results)} независимых экспериментов. Файлы моделей включают folds/seeds/inner splits.

Экспериментальный кандидат: purged_ensemble_d2_person_a20.
OOF AP {chosen.ap:.9f}; V31 {baseline_metrics['ap']:.9f}; разница {chosen.delta:+.9f}.
Более слабая поправка a10: {conservative.ap:.9f}.
Основным проверенным решением остаётся V31. Его Public AP 0.96239 ранее сообщил
пользователь. Новые CSV на Kaggle не отправлялись; их Public/Private AP неизвестен.

СРАВНЕНИЕ (все значения — повторно используемый development OOF)
{comparison.to_string(index=False, float_format=lambda v: f'{v:.9f}')}

ЧТО БЫЛО ПРОВЕРЕНО
Исходные 13/23 признака; полная замороженная геометрия SAM/WeVis/pose;
OWLv2 confidence, геометрия, semantic margin; CLIP text similarities;
240 пиксельных признаков рамки, контекста и их разности; позиции/размеры;
CLIP ViT-B/32, DINOv2 small 448, base 224/448, large 224/448, SigLIP2 base 512;
замороженные OWLv2 и YOLO26n embeddings. Использованы девять массивов представлений,
а не девять независимых архитектур. Полный статический банк: 320 числовых признаков
и class_name. Размерность эмбеддингов уменьшалась PCA по обучающему дню.

Сравнивались CatBoost depth 2/4, логрег, нативные embedding_features CatBoost,
ранжирование, поправка frozen prior, синтетические боксы, вес кадра,
признаки границ, типы ошибки, эталоны из размеченных кропов и машинного пула,
вложенные выходы обучаемых dense-голов, несколько seed и равновесные ансамбли.
Все источники и лучшие строки семейств: results/family_summary.csv.
Все варианты: results/all_experiments.csv. Весовые файлы: models/.

КАК УСТРОЕН ВЫДАВАЕМЫЙ КАНДИДАТ
22 числовых признака исходного base23 + 27 признаков сходства с эталонами
CLIP/DINOv2-B/DINOv2-L + категория class_name = 50 входных колонок.
CatBoost: depth=2, iterations=250, learning_rate=0.03, l2_leaf_reg=10;
seeds 23,24,25 без выбора лучшего seed. OOF: обучение на другом дне.
Test: среднее шести моделей (два обучающих дня, три seed), без full-data refit.
Эталонами служат GT-кропы только обучающего дня. Для обучающих примеров
исключены эталоны того же кадра и соседние в пределах 120 секунд.
В признаки не входят id, frame_id, day, error или исходный итоговый score V31.
Новый score для person: (V31 + 0.2 * p_error) / 1.2. Другие классы сохраняют V31.
Меняются {int(chosen.changed_test)} из {len(test)} тестовых строк. Вариант a10 использует 0.1 / 1.1.
Важнейшие признаки по сохранённым моделям: sam_score, best_iou, cover_ref,
left_offset, wevis_score. Эмбеддинги дополняют исходную геометрию.

ПОЧЕМУ ЭТО ПОКА НЕ ДОКАЗАННОЕ УЛУЧШЕНИЕ
Всего {len(q)} размеченных query на {q.frame_id.nunique()} кадрах и двух днях.
Глобальный максимум 0.988718805 использует эталоны без исключения соседних
обучающих кадров. Для выдаваемого кандидата используется версия с исключением.
Выбор depth, scope person и коэффициента делался после просмотра OOF, поэтому
даже аккуратный split не устраняет смещение от большого числа сравнений.
Выбор семейства внутри обучающего дня дал nested_selected_a10 = 0.987585454,
ниже V31. Эта вложенная проверка охватывала восемь семейств без прототипов;
она не является отдельной проверкой окончательного purged-кандидата.

У кандидата a20: {int(chosen.repaired_pairs)} исправленных и {int(chosen.broken_pairs)} испорченных пар ошибка/правильный бокс.
Это число пар ранжирования, а не число заново правильно классифицированных фото.
95% frame-bootstrap интервал разницы AP: [{chosen.frame_bootstrap_low:+.6f}, {chosen.frame_bootstrap_high:+.6f}].
Он включает ухудшение и условен относительно уже выбранного варианта.
Минимальная разница при исключении одного кадра: {chosen.leave_frame_min:+.9f}.
Ни bootstrap, ни leave-frame не заменяют новую независимую выборку.

ПРОВЕРКИ И ВОСПРОИЗВЕДЕНИЕ
Воспроизведены все 198 error-меток по GT и официальному правилу сопоставления.
CatBoost-13 воспроизводит приложенный MD: OOF 0.955962588.
Повторно посчитаны все метрики {len(results)} вариантов из фактических OOF CSV;
максимальное расхождение с таблицами {max_metric_error:.3g}.
Проверены ID, порядок, уникальность, отсутствие NaN/inf и диапазон [0,1]
во всех {len(results)} парах OOF/submission. Проверено {integrity['training_manifests_checked']} записей обучения.
Шесть конечных весов загружены заново; признаки прототипов пересчитаны из
сохранённых ROI-эмбеддингов; оба кандидата OOF/test воспроизводятся с ошибкой <1e-10.
Хеш исходного V31 не изменился: {definition['baseline_sha256']}.

ГОТОВЫЕ ФАЙЛЫ
release/submission_v52_candidate_purged_a20.csv — исследовательский кандидат.
release/submission_v52_candidate_purged_a10.csv — более слабая поправка.
release/replay_candidate.py — автономный replay из кэшированных признаков.
release/manifest.json — 6 моделей, признаки, seeds, train IDs и SHA256 входов.
release/requirements.txt — версии библиотек для replay.
results/final_verification.json — сверка всех таблиц и replay.
audit/purged_ensemble_d2_person_a20_* — пары, изменения, картинки и устойчивость.

Запуск из release: python3 replay_candidate.py
Архив содержит всё для replay на сохранённых query, без скачивания моделей.
Это не полный pipeline обработки новых фотографий: исходные SAM/WeVis/pose
и извлечение frozen embeddings остаются в прежних версиях проекта.

ВОССТАНОВЛЕНИЕ ЧАТА
В старом чате зафиксирован Error running remote compact task: Bad Request.
Это сообщение операции сжатия контекста Codex; причина отказа сервера не раскрыта.
Обучение и артефакты V52 сохранились. После восстановления пересчитаны проверки,
обновлён полный индекс вариантов и создан автономный пакет кандидата.

СЛЕДУЮЩИЙ СОДЕРЖАТЕЛЬНЫЙ ШАГ
Проверить зафиксированный кандидат на новых полностью размеченных кадрах другого
дня или отдельным сабмитом. Дальнейший перебор на этих 35 кадрах не даёт независимой
оценки. V51 с калибровкой границ остался отдельным незавершённым направлением;
он не включён в метрики и файлы этого CatBoost-эксперимента.
'''
    (A / 'REPORT_RU.txt').write_text(report)
    (release / 'REPORT_RU.txt').write_text(report)
    (release / 'README.txt').write_text(
        'V52: экспериментальный CatBoost-кандидат, не подтверждённое улучшение V31.\n'
        'Сабмит: submission_v52_candidate_purged_a20.csv (или более слабый a10).\n'
        'Воспроизведение из этой папки: python3 replay_candidate.py\n'
        'Зависимости перечислены в requirements.txt; модели не скачиваются.\n'
        'Скрипт проверяет SHA256, пересчитывает признаки, загружает 6 весов,\n'
        'пишет replayed/*.csv и сверяет их с expected/*.csv.\n'
        'Он не читает error-метки и не обучается; входы относятся к сохранённым query.\n'
        'Полный вывод и ограничения: REPORT_RU.txt.\n')
    for path in ['results/final_verification.json', 'results/integrity_checks.json',
                 'results/robustness.csv', 'results/all_experiments.csv', 'results/family_summary.csv']:
        shutil.copy2(A / path, release / Path(path).name)
    archive = ROOT / 'rascar_v52_candidate_replay.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(release.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                z.write(path, 'rascar_v52_candidate/' + str(path.relative_to(release)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps(dict(verification=verification, archive=str(archive),
                         archive_bytes=archive.stat().st_size), indent=2))


if __name__ == '__main__':
    main()
