"""Read-only audit of the user-confirmed V22 submission; writes new reports only."""
from pathlib import Path
from collections import Counter
from itertools import groupby
import csv
import hashlib
import json
import math

OUT = Path(__file__).resolve().parent
A = OUT.parents[1]
ROOT = A.parent
CLASSES = ['person', 'nude_hands', 'cup', 'tray']
FLAGS = {'clear_error_low_score', 'likely_error_low_score', 'plausible_correct_high_score'}


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def indexed(path):
    rows = read(path)
    assert len({r['id'] for r in rows}) == len(rows)
    return {r['id']: r for r in rows}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average_precision(rows, field):
    positives = sum(int(r['error']) for r in rows)
    assert positives
    tp = seen = result = 0
    for _, group in groupby(sorted(rows, key=lambda r: -r[field]), key=lambda r: r[field]):
        group = list(group)
        p = sum(int(r['error']) for r in group)
        tp += p
        seen += len(group)
        result += p * tp / seen
    return result / positives


def pass_visual(row, field):
    assert row['visual_status'] != 'uncertain'
    bad = row['visual_status'] in ['clear_error', 'likely_error']
    return (float(row[field]) >= threshold) == bad


submission = A / 'submission_v22_candidate.csv'
assert digest(submission) == '44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c'
test_scores = indexed(submission)
reviews = read(A / 'results/reviews/final_reviewed.csv')
threshold = json.loads((A / 'audit/protocol.json').read_text())['fixed_threshold']
assert len(reviews) == len(test_scores) == len({r['id'] for r in reviews}) == 362
for r in reviews:
    value = float(test_scores[r['id']]['score'])
    assert math.isfinite(value) and 0 <= value <= 1
    assert abs(value - float(r['new_score'])) < 1e-12

known = [r for r in reviews if r['visual_status'] != 'uncertain']
uncertain = [r for r in reviews if r['visual_status'] == 'uncertain']
remaining = [r for r in known if not pass_visual(r, 'new_score')]
newly_passed = [r for r in known if not pass_visual(r, 'baseline_v20') and pass_visual(r, 'new_score')]
priority = [r for r in reviews if r['diagnostic_flag'] in FLAGS]
class_audit = []
for c in CLASSES:
    rr = [r for r in reviews if r['class_name'] == c]
    kk = [r for r in known if r['class_name'] == c]
    pp = [r for r in priority if r['class_name'] == c]
    class_audit.append(dict(class_name=c, test_queries=len(rr), visually_non_uncertain=len(kk),
        uncertain=len(rr)-len(kk), unresolved_v20=sum(not pass_visual(r, 'baseline_v20') for r in kk),
        unresolved_v22=sum(not pass_visual(r, 'new_score') for r in kk),
        priority_total=len(pp), priority_passes=sum(pass_visual(r, 'new_score') for r in pp),
        changed_scores=sum(abs(float(r['new_score'])-float(r['baseline_v20'])) > 1e-9 for r in rr)))

oof = read(A / 'results/oof/final_reviewed.csv')
prior = indexed(ROOT / 'rascar_v18/results/oof/release_main.csv')
without = indexed(ROOT / 'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv')
assert len(oof) == len({r['id'] for r in oof}) == 198
correction_rows = []
for r in oof:
    r['score'] = float(r['score'])
    correction = float(prior[r['id']]['score'])-float(without[r['id']]['score'])
    r['score_without_inherited_correction'] = min(1, max(0, r['score']-correction))
    if abs(correction) > 1e-12:
        correction_rows.append(r.copy())


def metrics(rows):
    clean = 'score_without_inherited_correction'
    return dict(queries=len(rows), frames=len({r['frame_id'] for r in rows}),
        error_labels=sum(int(r['error']) for r in rows), legacy_ap=average_precision(rows, 'score'),
        ap_without_inherited_correction=average_precision(rows, clean),
        false_error_flags=sum(int(r['error']) == 0 and r[clean] >= threshold for r in rows),
        missed_error_flags=sum(int(r['error']) == 1 and r[clean] < threshold for r in rows))


oof_metrics = dict(overall=metrics(oof),
    by_day={day: metrics([r for r in oof if r['day'] == day]) for day in sorted({r['day'] for r in oof})},
    by_class={c: metrics([r for r in oof if r['class_name'] == c]) for c in CLASSES})
saved = json.loads((A / 'results/final_reviewed.json').read_text())
assert abs(oof_metrics['overall']['legacy_ap']-saved['pooled_ap']) < 1e-12
assert abs(oof_metrics['overall']['ap_without_inherited_correction']-saved['without_inherited_oof_only_correction']['pooled_ap']) < 1e-12
assert len(remaining) == 26 and len(newly_passed) == 7 and len(priority) == 41
assert sum(pass_visual(r, 'new_score') for r in priority) == 18
assert sum(not pass_visual(r, 'baseline_v20') for r in known) == 33

def compact(r):
    return {k:r[k] for k in ['id','review_no','frame_id','class_name','visual_status','issue','note','baseline_v20','new_score']}

sources = [submission, A/'results/reviews/final_reviewed.csv', A/'results/oof/final_reviewed.csv',
    A/'audit/protocol.json', A/'summary.json', ROOT/'rascar_v18/results/oof/release_main.csv',
    ROOT/'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv']
result = dict(audit_date='2026-11-13', submission=submission.name, sha256=digest(submission),
    leaderboard=dict(score=0.95610, prior_best=0.95417, delta=round(0.95610-0.95417,5),
        source='User reported score and explicitly confirmed submission_v22_candidate.csv in this conversation',
        leaderboard_split='not specified by user', submission_performed_during_this_audit=False),
    test=dict(queries=len(reviews), frames=len({r['frame_id'] for r in reviews}),
        visual_status_counts=dict(Counter(r['visual_status'] for r in reviews)),
        known_subset=len(known), unresolved_v20=33, unresolved_v22=26, newly_passing=7,
        uncertain_count=len(uncertain), uncertain_issue_counts=dict(Counter(r['issue'] for r in uncertain)),
        fixed_threshold=threshold, priority_passes=18, priority_total=41, class_audit=class_audit,
        remaining_cases=[compact(r) for r in sorted(remaining,key=lambda r:int(r['review_no']))],
        newly_passing_cases=[compact(r) for r in sorted(newly_passed,key=lambda r:int(r['review_no']))],
        uncertain_cases=[compact(r) for r in sorted(uncertain,key=lambda r:int(r['review_no']))]),
    oof=oof_metrics, inherited_correction_cases=correction_rows,
    limitations=['Visual annotations are assistant assessments, not hidden test GT.',
        'Diagnostic threshold is descriptive; official ranking AP is threshold-free.',
        'Test audit has informed model selection and is not an independent holdout.',
        'Repeated experiment selection on the same two OOF days can overfit.',
        'Cannot attribute the aggregate LB delta to individual classes or rows without hidden labels.',
        'This audit recalculates all saved rows; it is not a new independent visual labeling of all images.'],
    source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources})
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

report = '''RASCAR V22: аудит лучшей версии после LB 0,95610
Дата аудита: 13 ноября 2026. Источник результата LB — сообщение пользователя и явное подтверждение основного CSV.

Лучшая версия: submission_v22_candidate.csv. SHA256: 44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c.
LB: 0,95610 против 0,95417 у V20. Прирост +0,00193. Контрольный CSV имеет другой состав чашек; его LB этим сообщением не установлен.
Старые REPORT.txt/summary.json отражают состояние до загрузки на LB и сохранены без изменений. Код, веса и сабмиты в этом аудите не менялись.

Что работает в отправленной версии
Основа V20 сохранена. Люди и подносы дают в точности прежние score. Изменены 49 из 362 score: 30 кистей и 19 чашек. Score — оценка ошибочности исходного бокса, а не вероятность присутствия объекта. Координаты исходных запросов в CSV не исправляются.
Кисти: структурные штрафы V21, целевой YOLO26 на увеличенных фрагментах, согласование двух фолдов, проверки класса и геометрии, запрет ослаблять штраф при нескольких центрах кистей или признаках дубля. Новые DINOv2-классификаторы блокируют ошибочное подтверждение ткани как кисти. DINOv3 из прежней основы сохранён.
Чашки: объединены осторожные исправления V21 и дополнительного DINOv2 SVM. Поэтому результат основного CSV отличается от контрольного именно чашками.
Новые геометрические и совместные модели людей отвергнуты: они снижали штраф у реально коротких рамок. Более агрессивный детектор кистей у окна также отвергнут. 50 новых обученных моделей и 129 экспериментальных вариантов сохранены, но это не означает, что все они используются в финальном ансамбле.

Валидация
198 запросов на 35 кадрах. Для проверки 25 октября используем обучение на 26 октября, и наоборот. На 25-е приходится 68 запросов/12 кадров, на 26-е — 130/23. Для обучения детекторов имеется отдельный набор 84 ручных кадров, по 42 на день; это не размер OOF. Тест: 362 запроса на 86 кадрах 27–28 октября.
Историческая OOF AP: V20 0,98294694; V22 0,98603984. Без унаследованной поправки: V20 0,98006018; V22 0,98321983.
По дням V22: историческая AP 25-го 0,98408316, 26-го 0,99040568. Без поправки: 25-го 0,97316997, 26-го 0,99040568.
Уточнение прежнего отчёта: старая поправка основана на расхождении детекторов и семантики. Само правило применялось к OOF и тесту, но сработало на одном OOF-примере кисти и на нуле тестовых. Это не доказательство утечки меток и не физическая невозможность применить правило на тесте. Отдельная метрика без него показывает зависимость OOF от этого одиночного случая: score 0,22002 стал 0,74001, AP выросла на 0,00282.
AP без поправки по классам: person 0,95545147; nude_hands 0,99487716; cup 0,99068071; tray 1,00000. Чашки локально уступают V20 (0,99556792), хотя общий отправленный CSV дал больший LB. По единственному общему LB нельзя доказать, что именно чашки обеспечили прирост.
Разрыв между OOF без поправки и LB составляет 0,02712. Это разные выборки, разрыв не является оценкой утечки или доли неправильных боксов. Всего два дня, малая выборка и многократный подбор экспериментов ограничивают надёжность OOF. Визуальные случаи теста уже использованы в разработке и выборе правил, поэтому независимой проверкой они не являются.

Полный сохранённый визуальный аудит
Есть записи на все 362 запроса: 149 выглядят правильными, 61 явно ошибочный, 28 вероятно ошибочных, 124 неоднозначных. Это визуальные оценки ассистента, не официальная тестовая разметка. В этом проходе пересчитаны все записи и повторно просмотрены сохранённые примеры кистей, защит и регрессий людей; все фото заново независимо не размечались.
На 238 случаях без статуса uncertain при фиксированном старом диагностическом пороге 0,68899368 осталось 26 расхождений вместо 33 у V20. Убрано 7 из 33 прежних расхождений (21,2%). Новых направленных ухудшений score на этой визуально определённой части не найдено.
Из 26 остатков: 17 предполагаемо плохих боксов получают недостаточный штраф, 9 правдоподобно правильных получают завышенный штраф. Порог нужен только для описания аудита; AP на лидерборде оценивает ранжирование без этого порога.

Класс       Всего тестовых   Неоднозначных   Остаток V20 -> V22   Старый приоритетный список
person      153             67              15 -> 15            8/22 проходят
nude_hands  131             44              10 -> 7             7/14 проходят
cup          54             11               5 -> 1             3/4 проходят
tray         24              2               3 -> 3             0/1 проходят

В старом фиксированном списке 41 приоритетного случая проходят 18 (43,9%), ранее 13. Осталось 23. Это отдельный исторический список, а не все ошибки теста. Дополнительные три остатка полного аудита: person #308, tray #73 и #150. Нельзя говорить, что закрыто 43,9% всех ошибок, либо что LB 0,95610 означает 95,61% правильных боксов.

Где остаются проблемы
Люди: 13 предполагаемо коротких рамок за столом, одна рамка на двух людях (#67), одна обрезанная рамка вставшего человека (#96). У ряда сидящих людей полный нижний контур закрыт столом, поэтому визуальная оценка требует согласования с правилами GT. Присутствие лица/торса не доказывает полноту бокса. Шаблоны посадочных мест пока не дали пригодного улучшения ранжирования. Это главный незакрытый блок, 15 из 26 определённых расхождений.
Кисти: семь правдоподобно правильных боксов всё ещё штрафуются слишком сильно: #98, #151, #200, #248, #266, #333, #354. Есть мелкий масштаб, свет от окна, смаз, соседние предметы и частичные перекрытия. #116 улучшен с 0,873 до 0,635. Остальные не следует объявлять исправленными только потому, что общий AP кистей высокий.
Проверка детектора в фиксированном левом ROI на 84 ручных кадрах: 21 GT-кисть. При IoU >= 0,6, confidence >= 0,5 и входе 640 старая модель находит 0/21 с 4 ложными рамками, целевая 2/21 с 4 ложными, агрессивная 2/21 с 6 ложными. Это частная ROI-проверка, не общая полнота модели. Недостаток устойчивых обучающих примеров здесь сохраняется.
Чашки: остаётся #143, стакан в руке, score ошибки 0,90076. Нужна проверка границ и перекрытия кистью, а не общее уменьшение штрафа всем чашкам. Структурные проверки объединённых чашек должны сохраняться.
Подносы: #6 вероятно обрезан снизу (0,23566), #73 визуально похож на тарелку вместо подноса (0,66641), #150 выглядит правильным, но штраф высок (0,73678). AP 1,0 на 18 валидационных запросах не закрывает эти тестовые случаи.
Неоднозначные случаи: 124. Среди них 30 записей duplicate_candidate, 12 overlapping_hands, 43 occluded_extent, 14 person_lower_extent и другие. 30 записей дублей — не 30 независимых групп. Их надо разбирать совместно с соседними запросами и правилами назначения экземпляра. Здесь процент закрытия не установлен.

Что действительно улучшилось
Кисть #116: 0,873 -> 0,635; предплечье #190: 0,531 -> 0,723; объединённые кисти #193: 0,619 -> 0,768. Правильные чашки #1, #162, #247, #285 перешли на нужную сторону диагностического порога. Это семь новых проходящих случаев по всему сохранённому аудиту относительно V20.
Ткань #139 сохраняет высокий штраф 0,957, несмотря на ложное подтверждение YOLO. Защита нескольких центров не позволяет ошибочно снижать штраф перекрывающихся кистей #119. Сохранение этих защит не следует считать новыми исправлениями V22 относительно V20.

Приоритет следующей работы
1. Полные границы человека: отдельная качественная разметка для дальнего сидящего, ближнего сидящего и стоящего, с одним правилом для закрытого телa. Из 339 принятых ранее person-псевдометок подтверждено присутствие, но не полные границы; использовать их как эталон геометрии нельзя. Шаблон места должен быть признаком с неопределённостью, а не обязательным растяжением каждого бокса до стола/пола.
2. Кисти у окна: собирать разнообразные полностью размеченные фрагменты обоих дней, включая мелкие и соприкасающиеся кисти, ткань и предплечья. Из последних 43 кадров пула приняты только 13 кистей на 9 кадрах и 2 отрицательных фрагмента. При текущей слабости учителя массовое самообучение закрепит его ошибки.
3. Группы дублей, перекрытия и предметы в руках: проверять каждую группу целиком и отдельно контролировать #143, #6, #73, #150. Отсутствие референса само по себе не доказывает ошибку. Дополнить эксперименты разбором ошибок ранжирования, а не только количеством переходов через порог.
Во всех ветках сохранять V22 как контроль, проверять оба дня отдельно, общий AP и случаи явных регрессий. Новый независимый размеченный набор необходим для более надёжного выбора после многократных итераций. В этом аудите обучение и новые отправки не выполнялись.

Реестр оставшихся 26 расхождений
Номера ниже — review_no из визуального аудита; фактический submission id и кадр приведены рядом.
'''
for r in sorted(remaining,key=lambda r:(CLASSES.index(r['class_name']),int(r['review_no']))):
    report += f"#{r['review_no']} | {r['class_name']} | score={float(r['new_score']):.6f} | {r['visual_status']} | {r['issue']}\n"
    report += f"  id={r['id']}; кадр={r['frame_id']}\n  {r['note']}\n"
report += '\nДиагностика OOF без старой поправки при том же пороге\n'
for c in CLASSES:
    m=oof_metrics['by_class'][c]
    report += f"{c}: {m['queries']} запросов, AP={m['ap_without_inherited_correction']:.8f}; ложный штраф правильным={m['false_error_flags']}, недооценённые ошибки={m['missed_error_flags']}.\n"
report += '\nИсточники и воспроизводимость\n'
for p in sources:
    report += f'{p.relative_to(ROOT)}; SHA256 {digest(p)}\n'
report += 'Состав ранжирования проверен по replay.py, compose_guarded.py, compose_final_candidates.py, compose_reviewed.py. Происхождение старой поправки проверено по rascar_v16/calibrate_hand_boxes.py и guarded_structure.py. Подробные вычисления и все 124 неоднозначные записи сохранены в audit.json рядом с отчётом.\n'
(OUT/'AUDIT.txt').write_text(report)
assert digest(submission) == result['sha256']
print(json.dumps(dict(report=str(OUT/'AUDIT.txt'), leaderboard=result['leaderboard'],
    all_test_queries=len(reviews), remaining=len(remaining), uncertain=len(uncertain),
    oof=oof_metrics['overall'], class_audit=class_audit),ensure_ascii=False,indent=2))
