"""Package completed V38 research and pipeline audit without promoting a release."""
from common38 import *
import platform,importlib.metadata

def main():
    exp=pd.read_csv(A/'results/multilevel_experiments.csv');artifacts=[]
    for r in exp.itertuples():
        for split,rows in [('oof',q),('test',te)]:
            p=A/(f'results/oof/{r.experiment}.csv' if split=='oof' else f'results/submissions/submission_{r.experiment}.csv');d=pd.read_csv(p)
            assert d.id.is_unique and list(d.id)==list(rows.id) and len(d)==len(rows)
            assert np.isfinite(d.score).all() and d.score.between(0,1).all()
            if split=='test':assert list(d.columns)==['id','score']
            else:assert abs(AP(q.error,d.score)-r.ap)<1e-12
            artifacts.append(dict(file=str(p.relative_to(A)),rows=len(d),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pd.DataFrame(artifacts).to_csv(A/'results/artifact_checks.csv',index=False)
    name='v38_early2_tray_daily_RESEARCH';s=pd.read_csv(A/f'results/oof/{name}.csv').score.to_numpy();base=BASE['oof'];pairs=[]
    for i in np.flatnonzero(q.error.eq(1)):
        for j in np.flatnonzero(q.error.eq(0)):
            before=int(np.sign(base[i]-base[j]));after=int(np.sign(s[i]-s[j]))
            if before!=after:pairs.append(dict(error_id=q.iloc[i].id,correct_id=q.iloc[j].id,error_day=q.iloc[i].day,correct_day=q.iloc[j].day,before=before,after=after,improved=after>before))
    pairdf=pd.DataFrame(pairs);pairdf.to_csv(A/'audit/tray_pair_changes.csv',index=False)
    groups=[[np.flatnonzero(q.frame_id.eq(f)) for f in q[q.day.eq(day)].frame_id.unique()] for day in DAYS];rng=np.random.default_rng(3801);deltas=[]
    for _ in range(2000):
        idx=np.concatenate([np.concatenate([g[k] for k in rng.integers(0,len(g),len(g))]) for g in groups]);y=q.error.to_numpy()[idx]
        if len(np.unique(y))>1:deltas.append(AP(y,s[idx])-AP(y,base[idx]))
    leave=[]
    for f in q.frame_id.unique():
        m=q.frame_id.ne(f);leave.append(dict(frame_id=f,delta_ap=float(AP(q.loc[m,'error'],s[m])-AP(q.loc[m,'error'],base[m]))))
    pd.DataFrame(leave).to_csv(A/'audit/tray_leave_frame_out.csv',index=False)
    stability=dict(experiment=name,delta_ap=float(AP(q.error,s)-AP(q.error,base)),bootstrap_frame_stratified_by_day_n=len(deltas),conditional_ci95=np.quantile(deltas,[.025,.975]).tolist(),fraction_positive=float(np.mean(np.array(deltas)>1e-12)),
      improved_pairs=sum(p['improved'] for p in pairs),worsened_pairs=sum(not p['improved'] for p in pairs),changed_same_day_pairs=sum(p['error_day']==p['correct_day'] for p in pairs),unique_error_queries=len(set(p['error_id'] for p in pairs)),unique_correct_queries=len(set(p['correct_id'] for p in pairs)),min_leave_frame_out_delta=min(x['delta_ap'] for x in leave),warning='Conditional diagnostic after adaptive selection, not independent generalization evidence.')
    (A/'results/tray_stability.json').write_text(json.dumps(stability,indent=2))
    checks=json.loads((A/'audit/pipeline_checks.json').read_text());oracle=json.loads((A/'results/occupancy_oracle_summary.json').read_text());geometry=pd.read_csv(A/'results/geometry_pool_summary.csv')
    expected={'rascar_v22/submission_v22_candidate.csv':'44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c','rascar_v31/submission_v31_dense_oof098784.csv':'8720eb732639532bab49eff75b956a6bd4db21b96ede309ed901889777f4ab2c','rascar_v37/submission_v37_person_refit_RESEARCH.csv':'105e980e9465c00692e5a50bc8a5cf895f2b5e9a2b55f6c7e4682e6278f3afb1'}
    for path,digest in expected.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    manifest=dict(status='completed_research_rejected_for_promotion',public_best={'version':'V31','ap':.96239,'source':'user-confirmed result','file':'../rascar_v31/submission_v31_dense_oof098784.csv'},newly_trained_heads=18,reused_control_heads=9,scored_files=30,distinct_oof_configs=15,
       promoted_submission=None,automatic_submissions=0,experiment='Early DINOv2 layers for local geometry, with fixed V31 cut/blend and a same-aggregation last-layer control.',
       source_model='facebook/dinov2-small',weights_url='https://huggingface.co/facebook/dinov2-small',source_code='https://github.com/facebookresearch/dinov2',
       data_roles={'official_manual':'fit-day crops only','official_queries':'opposite-day development OOF only','train_pool_notes':'20 approximate local ROIs / 23 boxes, diagnostic only, never training','test':'feature extraction and qualitative model-output inspection; no new human test labels or ID overrides'},
       frozen_files=expected,limitations=checks['limits'],decision='No robust demonstrated improvement. Preserve V31; do not spend a submission on V38.')
    (A/'release_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    versions={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','pandas','scikit-learn','scipy','opencv-python']};versions['python']=platform.python_version();(A/'runtime_versions.json').write_text(json.dumps(versions,indent=2))
    table=exp[exp.test_fit.eq('daily')&exp.classes.eq('all')][['kind','ap','2026-10-25','2026-10-26','changed_test']].to_string(index=False)
    text=f'''RASCAR V38 — завершение эксперимента и повторный аудит действующего решения

РЕШЕНИЕ
V38 не повышен до основной версии. Новый сабмит на лидерборд не отправлен.
Лучший подтверждённый: V31, Public AP 0.96239, OOF AP 0.98784401.
Его CSV и исходные V22/V37 сохранены с прежними SHA256.
Это завершённый отрицательный эксперимент, а не обещание дальнейшего обучения.

ЧТО ДОДЕЛАНО
Проверено добавление слоёв 3/6/9/12 замороженного DINOv2-small и отдельно
ранних слоёв 3/6. Обучено 18 линейных голов: 2 варианта × 3 seed ×
fit 25 / fit 26 / оба дня. Сохранены веса, протокол, предсказания, 30
исследовательских CSV: 20 новых и 10 контрольных на ранее обученных головах.
Это 15 разных OOF-вариантов: test daily/full имеют ОДИН cross-day proxy,
а не две независимые оценки. Порог маски .5 и вес .25 не подбирались заново.
Одинаковый контроль на последнем слое из V37 отделяет эффект признаков от
изменения правил согласия seed. Полное обучение на двух днях оценено только
через cross-day proxy; его собственного независимого OOF нет.

Общий V31: 0.98784401; день 25: 0.97624060; день 26: 0.99250544.
{table}

Ветка early2 только для tray: OOF {AP(q.error,s):.8f}, delta {stability['delta_ap']:+.8f}.
AP обоих дней и AP класса tray не изменяются. Изменены {len(pairs)} error/correct-пар,
все между разными днями; улучшенных {stability['improved_pairs']}, ухудшенных {stability['worsened_pairs']}.
На test daily меняются только 3 подноса; full-fit — 20, но отдельной оценки
full-fit нет. Conditional bootstrap по кадрам внутри дня:
95% диапазон [{stability['conditional_ci95'][0]:+.8f}; {stability['conditional_ci95'][1]:+.8f}].
Это уже выбранный вариант на многократно использованном development-наборе.
Диапазон не учитывает выбор среди экспериментов и сдвиг будущих дней.
Причин расходовать на него одну из оставшихся попыток недостаточно.

ГЛАВНАЯ НАХОДКА — ПОТЕРЯ РАЗДЕЛЕНИЯ ОБЪЕКТОВ
В голову поданы идеальные официальные occupancy-targets вместо предсказаний
нейросети. После connected components, фильтра размера и max-IoU получено
34 расхождения с официальной бинарной меткой из 198: 24 правильных запроса
ошибочно отвергнуты по геометрии и 10 ошибок ошибочно подтверждены.
По классам: person 17, nude_hands 10, cup 4, tray 3.
В 25 расхождениях восстановленная рамка содержит центры нескольких GT-объектов.
Просмотрены все 34 случая: видны объединения соседних людей, кистей и подносов;
также есть чувствительность маленьких объектов к сетке 32×32 и порогу IoU .60.
Даже точный max-IoU с отдельными GT даёт одно расхождение с официальной меткой:
для дубликата нужно one-to-one matching, одного max-IoU недостаточно.

Это НЕ 34 ошибки финального V31 и НЕ потолок всей системы. Это диагностическое
ограничение конкретной геометрической ветки. Ранние признаки его не устраняют.
Осмысленный следующий приоритет — сохранять отдельные экземпляры объектов
при восстановлении рамок и учитывать конкуренцию запросов за один референс.
В этом цикле архитектура/decoder не заменялись, согласно просьбе закончить
текущий эксперимент и перепроверить сделанное.

ЧТО ПЕРЕПРОВЕРЕНО
1. Задача — ранжировать существующие ошибки, error=1, score выше означает
   более подозрительный бокс. Пропущенные объекты без запроса вне скоринга.
2. Из исходного human_boxes_train.csv заново восстановлены 1032 рамки;
   координаты совпали с используемыми с точностью 2.3e-13.
3. Для всех 670 train-запросов заново выполнен официальный matching:
   класс/кадр, IoU >= .60, сначала число пар, затем сумма IoU. Ошибок меток 0.
   В основном OOF 198 не-table запросов на 35 кадрах, 109 error=1.
4. Столы исключены из оцениваемого OOF, как в test; как контекст разрешены.
   Конвенция person действительно включает скрытые столом части тела.
5. AP независимо пересчитан по группам одинаковых score и совпал со sklearn.
6. V22 восстановлен из кэшированных модельных сигналов, V31 из его основания
   и dense-reference evidence. Максимальное расхождение V31: 2.3e-16.
7. Все 362 test ID совпадают с sample_submission, уникальны, координаты верны,
   train/test кадры и ID не пересекаются. Исследовательские CSV конечны в [0,1].
8. Все 170 имеющихся изображений целевых train/test-запросов — 1280×720.
   Машинный пул 640×360 масштабируется к системе координат 1280×720.
9. Все 1219 targets V31 заново восстановлены из официальных GT: разница 0.
   Последний слой новых features побитово совпал со старым float16 кэшем.
10. Проверены fit-кадры V31/V38: 659 кропов из 38 вручную размеченных кадров,
    только свой день для OOF; held/test не выбираются в обучающий batch.
    Есть targets held-day в общем массиве, но они не входят в fit-индексы.
11. Проверены day-gates и сохранённые данные V20/V22: pseudo 109/384 строки,
    hand-ledger 274/788 кропов и spatial train 2739/3730 записей. Чужих дней
    и пересечений spatial train с held/test в этих проверках не обнаружено.

ОБНАРУЖЕННЫЕ ОГРАНИЧЕНИЯ И ОШИБКИ ОТЧЁТНОСТИ
— 35 оценочных кадров многократно использовались для выбора моделей, порогов
  и правил. OOF теперь development-метрика, а не независимая оценка качества.
— Дневное обучение OOF и консенсус двух дней на test отличаются. Pooled AP
  чувствителен к взаимной калибровке дней и классов; отдельный общий прирост
  без улучшения внутри дней нельзя считать достаточным подтверждением.
— В старом V22 присутствовала OOF-only коррекция одной строки на ~.520.
  Она уже вычтена в V31. Проверено: используемые .98784401 её не включают.
  Исторический V22 .98603984 и исправленный .98321983 нельзя смешивать в
  таблице прогресса без этого пояснения.
— В отчёте V37 было написано «64 разных кадра»; фактически 64 запроса на
  63 кадрах. Текст исправлен, данные/модели/предсказания не менялись.
— В архиве есть старые таблицы ручных visual_status теста. Правила запрещают
  ручную разметку теста. Эти статусы нельзя считать GT, использовать как
  обучающие цели или отчёт о проценте закрытых ошибок. В V38 новые тестовые
  метки не создавались; score строится без этих таблиц и без ID-подстановок.
  В V22/V31 они используются в диагностической отчётности, но не в проверенной
  формуле расчёта score. Исторический выбор вариантов по визуальным тестовым
  оценкам всё равно является источником адаптации; этот риск не исчезает
  от отсутствия прямой подстановки ответов.
— Replay ранних ансамблей доказывает соответствие кэшам, а не полную
  воспроизводимость обучения всех моделей V1–V22 с нуля. Все старые GPU-модели
  заново не обучались. Утверждать «абсолютно вся история доказанно чиста» нельзя.

СВЕЖАЯ TRAIN-ГЕОМЕТРИЯ
Просмотрены 37 presence-positive запросов пула. Сохранены осторожные заметки
для 20 локальных областей / 23 объектов: 7 cup, 9 tray, 4 hand-области.
Для границ указана неопределённость ±3/4 пикселя в системе 1280×720.
Разметка частичная, выбранная по presence, с низким исходным разрешением.
Она НЕ добавлена в обучение и НЕ объявлена новым competition AP.
Проверка выполнена моделями противоположного дня.

{geometry.to_string(index=False)}

Число in_scope различается между моделями: агрегаты на разных подмножествах
нельзя напрямую трактовать как величину прироста качества.
Области, за которые выходит модельная рамка, не оцениваются как полностью
размеченные. Все четыре hand-области в этом смысле не дают пригодной оценки.
Консервативные границы IoU оставляют 19/20 запросов неоднозначными около .60;
у одного верхняя граница ниже .60. Эти заметки не заменили независимый bbox-
holdout. Включить их как уверенную точную разметку было бы ошибкой.

ВИЗУАЛЬНАЯ ПРОВЕРКА TEST
Просмотрены 22 модельных изменения с наибольшим |delta|, по классам и разным
кадрам. Это инспекция выходов моделей, не ручное присвоение test error.
Видны объединённые референсы для соседних чашек/рук/людей, большие person-
рамки с соседним посетителем и нестабильные рамки подносов. Оснований считать
этот вариант закрывающим прежние проблемы нет. Иллюстрации показывают один
фиксированный reference fit25/seed3101; score использует весь консенсус.
Точное число/процент исправленных test-ошибок без официальных меток неизвестно.
Равный округлённый Public AP не доказывает, что изменения только в Private:
AP может не измениться при изменении величин score и некоторых перестановках.

АРТЕФАКТЫ
audit/pipeline_checks.json — независимые проверки основной цепочки.
audit/inherited_training_checks.json — проверенные дневные данные V20/V22.
audit/occupancy_oracle.csv и official_oracle_failure_*.jpg — 34 случая.
results/multilevel_experiments.csv — все варианты и AP по дням/классам.
results/tray_stability.json — условная устойчивость слабого tray-прироста.
results/artifact_checks.csv — 60 CSV с проверенными ID/значениями/SHA256.
inputs/train_geometry_notes.json — частичные TRAIN-заметки, не training GT.
models/ — все 18 новых голов; features/ — кэши и предсказания.
release_manifest.json — статус: completed_research_rejected_for_promotion.

ВОСПРОИЗВЕДЕНИЕ ИЗ КОРНЯ ПРОЕКТА
python3 rascar_v38/multilevel_features.py
python3 rascar_v38/multilevel_train.py
python3 rascar_v38/evaluate_multilevel.py
python3 rascar_v38/audit_pipeline.py
python3 rascar_v38/diagnose_representation.py
python3 rascar_v38/geometry_annotations.py
python3 rascar_v38/geometry_check.py
python3 rascar_v38/visual_checks.py
python3 rascar_v38/finalize.py
Требуются сохранённые исходные данные и артефакты V18/V22/V31/V37.
Это локальный воспроизводимый пакет проекта, не автономный Kaggle-ноутбук.

Модель: facebook/dinov2-small (публичные замороженные веса).
https://huggingface.co/facebook/dinov2-small
https://github.com/facebookresearch/dinov2
'''
    (A/'REPORT.txt').write_text(text)
    print(json.dumps(stability,indent=2));print('VERIFIED',len(artifacts),'CSV files; no promotion; V31 unchanged.')

if __name__=='__main__':main()
