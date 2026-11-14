"""Package measured results, audit evidence, and replay inputs."""
from common20 import *
from evaluate_sahi import old_veto
import sys,platform,importlib.metadata,html,zipfile

(A/'replay_inputs').mkdir(exist_ok=True)
fo=pd.concat([pd.read_csv(A/f'person_dino_mode_held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id].reset_index()
tt=[align(A/f'person_dino_mode_test_{fit}.csv',te) for fit in DAYS];ft=(tt[0]+tt[1])/2
for split,rows,base,f,test in [('oof',q,q.score.to_numpy(),fo,False),('test',te,current,ft,True)]:
    d=f.reset_index(drop=True).copy();d['id']=rows.id.to_numpy();d['class_name']=rows.class_name.to_numpy();d['baseline']=base
    d['front_probability']=align(A/f'person_dino_modes_logreg_{split}.csv',rows,'probability')
    d['back_probability']=align(A/f'person_dino_modes_catboost_{split}.csv',rows,'probability')
    d['both_front_agree']=(tt[0].mode_front.ge(.5).to_numpy()&tt[1].mode_front.ge(.5).to_numpy()) if test else True
    hand=align(A/f'hand_expanded_sahi_{split}.csv',rows)
    for col in ['assigned','instances','overlap','confidence','duplicate_loser']:d['hand_'+col]=hand[col].to_numpy()
    d['hand_fold_agreement']=hand.folds.ge(2).to_numpy() if test else True
    d['semantic_hand']=align(ROOT/f'rascar_v16/semantic_tight_{split}.csv',rows,'p_nude_hands')
    d['old_duplicate_veto']=old_veto(rows,test)
    if not test:d['day']=rows.day.to_numpy();d['error']=rows.error.to_numpy()
    d.to_csv(A/f'replay_inputs/{split}.csv',index=False)

baseline=metric(q,q.score.to_numpy());main=json.loads((R/'release_main.json').read_text());cons=json.loads((R/'release_conservative.json').read_text())
no=metric(q,align(ROOT/'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv',q,'score'))
review=pd.read_csv(R/'reviews/release_main.csv')
prior=review.diagnostic_flag.isin(['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'])
bad=review.visual_status.isin(['clear_error','likely_error']);good=review.visual_status.eq('plausible_correct')
basepass=(bad&review.baseline_v18.ge(threshold))|(good&review.baseline_v18.lt(threshold))
classaudit=[]
for c,g in review[prior].groupby('class_name'):
    classaudit.append(dict(class_name=c,priority_cases=len(g),baseline_passes=int(basepass[g.index].sum()),v20_passes=int(g.diagnostic_pass.sum())))
pd.DataFrame(classaudit).to_csv(A/'audit/class_diagnostics.csv',index=False)
verified=pd.read_csv(A/'audit/final_review/reviewed_cases.csv')
verified['reviewed_in_v20']=True
verified['final_note']=np.where(verified.visual_status.eq('uncertain'),'Ambiguous extent/instance ownership; not counted as a resolved error.','Direction agrees with frozen visual audit, or score unchanged.')
verified.to_csv(A/'audit/final_review/visual_review_decisions.csv',index=False)
experiments=[]
for p in sorted(R.glob('*.json')):
    d=json.loads(p.read_text())
    if 'pooled_ap' in d:
        experiments.append({k:d.get(k) for k in ['experiment','pooled_ap','day_25','day_26','priority_improved_vs_v18','priority_passes','visual_regressions_gt002','changed_test_rows']})
pd.DataFrame(experiments).to_csv(A/'experiment_comparison.csv',index=False)
versions=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','catboost','joblib','Pillow']})
(A/'local_runtime_versions.json').write_text(json.dumps(versions,indent=2))
manifest=dict(baseline=baseline,baseline_without_inherited_correction=no,main=main,conservative=cons,experiment_count=len(experiments),visual_review_cases=len(verified),class_audit=classaudit,known_best_lb=.95259,new_lb=None,all_test_errors_closed=False,private_weights_dataset='shapodev/rascar-v20-trained-checkpoints')
(A/'release_manifest.json').write_text(json.dumps(manifest,indent=2))

text=f'''RASCAR V20 — отчёт, 13 ноября 2026

Итог: подготовлены submission_v20_main.csv и более слабый вариант
submission_v20_conservative.csv. Основа — V18 с подтверждённым LB 0.95259.
V19 не используется. Новый сабмит на соревнование автоматически не отправлялся.
LB V20 неизвестен; небольшой прирост OOF не доказывает улучшение на LB.

1. Что включено
Отдельные вероятности режимов человека: дальнее сидячее место, ближнее
сидячее место, стоящий. Классификатор использует замороженные DINOv3-признаки
и 77 визуально проверенных людей из ручного TRAIN. Геометрический эксперт
использует положение головы, позу, шаблоны людей и относительное положение
в кафе. Для ближних мест — логистическая регрессия с весом 0.3, для явно
коротких дальних рамок — CatBoost с весом 0.15. При неопределённости сохраняется
V18. Это изменение score ошибки существующей рамки, а не координат в сабмите.
Стоящие люди пока распознаются плохо: из пяти на каждом дне верно определён
только один. Отдельный режим реализован, но надёжного исправления стоящих нет.
Общая точность режимов: 46/50 = 92% на 26-м, 23/27 = 85.2% на 25-м.

YOLO26 для кистей дообучен на ручных данных и проверенном машинном пуле.
Независимый поиск SAHI по всему кадру: тайлы 256/384, перекрытие 25%.
Подтверждение требует согласия масштабов, confidence >= 0.7, IoU >= 0.6,
DINO-класса кисти, единственного сопоставленного объекта и согласия двух
дневных моделей на тесте. Дубли защищены взаимно-однозначным сопоставлением
и предыдущими запретами на понижение ошибки проигравшего дубля.
Сохранено прежнее линейное геометрическое ранжирование; отвергнутый expit
эксперимент не используется. В основной версии вес нового подтверждения 0.5,
в запасной 0.25. Изменены 27 тестовых боксов: 18 кистей и 9 людей.

2. Машинный пул и обучение
Пул: 5 388 кадров / 41 964 машинные рамки. После исключения кадров ближе
10 минут к ручным обучающим кадрам осталось 2 313 кадров. Teacher-проход
проверил 273/545 кадра двух дней с подходящими предложениями целевых классов.
Согласие моделей само по себе недостаточно: на 26-м дне 2 из 6 принятых
person оказались ошибочными по официальным query-меткам. Поэтому 339 person
используются только как сигнал наличия человека, не как правильная геометрия.
Визуально просмотрены все 192 кандидата кистей/чашек. Приняты 137 кистей и
17 чашек, 38 рамок отклонены из-за объединённых кистей, смаза или неясных
границ. Это дополнительная неофициальная разметка ассистента, а не истинный GT.
Несколько кадров могут содержать практически одинаковую позу; число боксов
нельзя считать числом независимых сцен. Исходные фото пула 640x360, координаты
1280x720 — при обучении масштаб согласован явно.

YOLO26: две дневные модели × clean/expanded, по 12 эпох, seed 201, AdamW,
lr=0.00005, размер 640, batch 16, без Mosaic. Использован last.pt, выбор эпохи
по held-day не делался. Ручных кропов 138/268; безопасных дополнительных 9/210,
всего 219. Кропы с обрезанной разметкой или пересекающейся непроверенной
машинной кистью исключены. Pseudo берутся только из дня обучения.
12 эпох дают разное число шагов clean/expanded: это сравнение конечных
вариантов, не эксперимент с одинаковым вычислительным бюджетом.
Ultralytics печатает mAP на train как на val из-за технического train=val
конфига. Этот mAP НЕ является нашей валидацией и в отчёт не включён.

Отдельная CNN качества: ImageNet ResNet18, RGB + маска query, четыре выхода
на класс (ошибка, IoU, несколько объектов, наличие объекта). Дообучаются
conv1/layer4/head, ранние слои и BatchNorm заморожены. По 400 шагов, batch 48,
одинаковый бюджет clean/expanded; доля pseudo в sampling 30%.
Для person pseudo geometry-loss выключен. После фильтров осталось 257 таких
примеров присутствия. CNN clean: 994/1996 записей; expanded добавляет 387/1549
синтетических и presence-записей (не независимых фотографий).
Standalone OOF AP CNN вырос с 0.54783 до 0.64295, но качество кистей снизилось,
а бленд портит визуально правильные боксы. CNN сохранена, в сабмит не включена.

3. Валидация
Обучение на 25 октября → оценка на 26; обучение на 26 → оценка на 25.
Новые обучаемые модели и шаблоны используют только свой день. Training-признаки
шаблонов/режимов вычисляются с исключением собственного кадра. Тестовые
визуальные метки не используются как таргеты обучения; они используются
для отбора вариантов и защитных условий, поэтому это development-аудит.
Ручной train: 84 кадра, по 42 в день. Официальные query: 198 боксов на 35
кадрах, 109 ошибок и 89 правильных; 68 query на 25-м и 130 на 26-м.
Ранжируем вероятность ошибки query. Таргет: сопоставление одного объекта того
же класса при IoU >= 0.6; столы не входят в оцениваемые классы.
Псевдоразметка не добавлена в основной validation GT: иначе получится оценка
согласия с собственным teacher вместо сопоставимой метрики.

                         V18              V20 main
OOF pooled AP            {baseline['pooled_ap']:.8f}       {main['pooled_ap']:.8f}
25 октября AP            {baseline['day_25']:.8f}       {main['day_25']:.8f}
26 октября AP            {baseline['day_26']:.8f}       {main['day_26']:.8f}
person AP                {baseline['ap_person']:.8f}       {main['ap_person']:.8f}
nude_hands AP            {baseline['ap_nude_hands']:.8f}       {main['ap_nude_hands']:.8f}
cup AP                   {baseline['ap_cup']:.8f}       {main['ap_cup']:.8f}
tray AP                  {baseline['ap_tray']:.8f}       {main['ap_tray']:.8f}
LB                       0.95259          ещё неизвестен

По отдельным дням и классам AP не изменился. Прирост pooled AP +0.00005139
объясняется перестановкой между группами, а не улучшением порядка внутри дня.
Базовый V18 наследует OOF-only коррекцию V16 одного бокса кисти (тестовых
изменений от неё нет). Без неё: V18 {no['pooled_ap']:.8f}, V20
{main['without_inherited_oof_only_correction']['pooled_ap']:.8f}. Оба варианта
подсчёта сохранены. OOF многократно использовался для выбора экспериментов;
его нельзя представлять как независимую оценку скрытого теста или «1 в 1 LB».

4. Детектор кистей: отдельная cross-day проверка
IoU >= 0.6, уверенные референсы >= 0.25 с согласованием двух масштабов,
ручной GT на кадрах официальных query: 26 кистей 25-го дня и 57 — 26-го.
Вариант                 recall 25      recall 26      AP06 25 / 26
V17 frozen + SAHI       14/26 53.8%    11/57 19.3%    0.51887 / 0.15321
YOLO clean + SAHI       11/26 42.3%    11/57 19.3%    0.40263 / 0.16419
YOLO expanded + SAHI    13/26 50.0%    13/57 22.8%    0.45984 / 0.18048
Без тайлов recall соответствующих моделей всего 0–11.5%.
Пул помогает относительно clean-дообучения на обоих днях, но относительно
frozen V17 улучшение есть только на 26-м. Универсальной победы новой модели нет.

5. Проверка проблемных тестовых фото
Повторно просмотрены 68 боксов: все 27 изменённых, все 41 приоритетный случай
и защищённые контрольные примеры (есть пересечения). Шесть листов сохранены
в audit/final_review; жёлтая рамка — query, голубая — новый референс кисти.
7/41 случаев получили полезный сдвиг score >0.02. Порог диагностики фиксирован
на {threshold:.8f} из старого V15. Его проходят 13/41 (31.7%), было
9/41 (22.0%): четыре новых перехода порога, +9.8 процентного пункта.
Это НЕ процент всех исправленных ошибок теста. У нас нет официальных test GT,
видимая обрезка не всегда означает IoU <0.6, неясные случаи не считаются решёнными.
В запасном варианте 12/41. Ухудшений >0.02 относительно замороженных уверенных
визуальных меток нет; это не гарантия отсутствия скрытых регрессий.

Новые переходы: правильные люди #89 0.830→0.638, #90 0.863→0.646,
#223 0.862→0.634; маленькая правильная кисть #204 0.865→0.616.
#100 и #309 тоже улучшились, но уже проходили порог в V18.
#128 короткий person: 0.276→0.301, остаётся нерешённым.
Ложная кисть #139 (ткань/предмет) остаётся 0.957: новый YOLO ошибочно её
подтверждал, но confidence-фильтр не позволил понизить score.
Предплечье #190 остаётся 0.531; слитые кисти #193 — 0.619.
Короткие люди #96/#308 не исправлены; #7/#9/#16/#141/#149 и другие дальние
сидящие требуют уточнения видимых/амодальных границ. Механическое удлинение
до низа стола понизит IoU у части корректных рамок.
Правильные маленькие кисти #98/#116/#151/#200/#248/#266/#333/#354 всё ещё
получают высокую ошибку. Их преимущественно не находит уверенно новый детектор.
Чашки #1/#143/#247/#285 и поднос #6 остаются проблемными. Новые cup/tray SAHI
варианты не дали улучшения приоритетных случаев, а rescue добавлял регрессию.
Неясные изменённые #117/#241/#355/#356 сохраняют статус uncertain; их
не включали в число исправленных. Контроли #66/#234/#272 сохранены.

6. Что отклонено и что дальше
Сохранено {len(experiments)} вариантов ранжирования, включая неудачные.
Глобальный person-logreg имел OOF 0.98418, но 19 визуальных регрессий;
массовое применение отвергнуто. CNN expanded-hands имел OOF 0.98330,
но 46 регрессий; отвергнут. Чашки/подносы в итоговом CSV оставлены как V18.
Следующий полезный сбор данных: новые разнообразные маленькие кисти у окна,
кисти на телефоне и пересечения двух рук; обязательная полная разметка
соседних кистей в выбранных кропах. Сейчас этот фильтр резко ограничивает
пул 25-го дня. Для людей важнее размечать точные нижние границы и стоящих,
чем добавлять много уверенных, но укороченных машинных рамок. Для чашек
нужна отдельная проверенная выборка наклонённых чашек у окна; 17 pseudo
примеров недостаточно, чтобы обещать устойчивое исправление.

7. Воспроизводимость
python3 replay.py: строит оба CSV и OOF из замороженных признаков, проверяет
198/362 уникальных id, диапазон score, соответствие релизу с точностью 1e-12
и контрольную сумму V18. Это воспроизведение ранжирования, не новое обучение.
v20_replay.ipynb запускает тот же код. Источники обучения: gpu_quality.py,
gpu_yolo_hands.py; псевдоразметки: gpu_pool_teacher*.py, gpu_pool_review.py;
режимы: person_modes.py/person_dino_modes.py; геометрия: evaluate_sahi.py;
сборка: compose_release.py. Полные исходные GPU журналы и 12 checkpoints
сохранены отдельно; пути и SHA256 — gpu_model_manifest.json.
Компактный приватный ноутбук Kaggle успешно выполнен за 33 секунды:
https://www.kaggle.com/code/shapodev/rascar-v20-reviewed-pool-and-person-modes
Он подтвердил обе сборки и сохранил CSV в Output. Большой рабочий draft
не удалось закоммитить из-за лимита исходника 1 МБ; компактная версия решает
сохранение воспроизводимого ранжирования без встроенных бинарных архивов.
Веса сохранены в приватном датасете:
https://www.kaggle.com/datasets/shapodev/rascar-v20-trained-checkpoints
Kaggle распаковывает загруженные ZIP в каталоги с соответствующими именами.
GPU-сессия обучения остановлена после сохранения весов.
Манифест CNN первоначально перечислял три головы, но фактически код и веса
имеют четыре (включая presence); описание в этом отчёте соответствует коду.
Первичные технические источники:
https://obss.github.io/sahi/predict/
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
'''
(A/'report_v20.txt').write_text(text)
sections=''.join(f'<a href="audit/final_review/page_{i}.jpg"><img src="audit/final_review/page_{i}.jpg" alt="Визуальная проверка, лист {i}"></a>' for i in range(1,7))
(A/'report_v20.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>RASCAR V20</title><style>body{max-width:1100px;margin:32px auto;padding:0 18px;font:16px/1.5 system-ui;background:#f7f8fa;color:#172335}pre{white-space:pre-wrap;font:inherit;background:white;padding:28px;border-radius:12px}img{width:48%;margin:1%;vertical-align:top}a{color:#1462be}</style><body><p><a href="submission_v20_main.csv">Основной CSV</a> · <a href="submission_v20_conservative.csv">Запасной CSV</a> · <a href="experiment_comparison.csv">Все эксперименты</a></p><pre>'+html.escape(text)+'</pre>'+sections+'</body></html>')
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=[dict(cell_type='markdown',metadata={},source=['# RASCAR V20 — воспроизведение ранжирования\n','Положите ноутбук рядом с replay.py после распаковки bundle. Используются сохранённые cross-day признаки; GPU не требуется. Это не повторное обучение.']),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=['from pathlib import Path\n','import runpy\n','root = Path.cwd()\n','if not (root / "replay.py").exists():\n','    root = root / "rascar_v20"\n','assert (root / "replay.py").exists(), "Распакуйте rascar_v20_reproducibility.zip"\n','runpy.run_path(str(root / "replay.py"), run_name="__main__")\n'])])
(A/'v20_replay.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))
print('REPORT_AND_REPLAY_READY',len(experiments))
