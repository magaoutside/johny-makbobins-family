# Воспроизведение: данные, веса, проверки

Основной маршрут запуска — в [README](README.md). Здесь перечислено происхождение каждого вида входов, чтобы кэш не был необъяснимой зависимостью.

## Три разных уровня воспроизведения

| Уровень | Что реально вычисляется | Статус |
|---|---|---|
| `predict.py` | Финальные CatBoost и выбранная формула смешивания | Проверен: максимальная разница каждого скора ≤ `5e-13`. |
| `train.py` и `rebuild_upstream.py` | Новые CatBoost; последовательная ранняя цепочка CPU-моделей и правил из сохранённых детекций / эмбеддингов | Проверены: ранний скор отличается на `3.33e-16`; прототипные модели — в пределах `1e-10`. |
| `from_images.py` | Все признаки из исходных JPG и CSV, затем вся выбранная цепочка; `--retrain` обучает модели | Код и веса приложены. Проверены метаданные, геометрия, контракты моделей и отдельные численные этапы. **Новый объединённый GPU-запуск целиком ещё не проверен.** |

Численное совпадение доказывает корректность переноса сохранённого решения, а не независимую способность обобщать. `--allow-drift` сохраняет измеренную разницу в отчёт и не превращает её в точное совпадение.

## Исходные данные

Источник — соревнование Kaggle `rascar-hack-case-1`. Архив должен содержать:

- `train.csv`, `test.csv`, `human_boxes_train.csv`, `scoring_policy.json`;
- `train_frames.csv`, `test_frames.csv`, `train_context.csv`, `test_context.csv`;
- `machine_frames.csv`, `machine_predictions.csv`;
- JPG целевых кадров, соседних кадров и машинного train-пула.

Скрипты ищут таблицы и JPG рекурсивно. Для каждой таблицы ожидается ровно один файл с соответствующим именем. Изображения идентифицируются по `frame_id`, размер исходного кадра — 1280×720.

`rascar/records.py` получает из исходных таблиц 560 запросов, 1 032 ручных аннотации всех классов, 240 GT-прототипов целевых классов и 1 219 плотных кропов. Координаты и порядок проверены против исходного запуска. 198 train-запросов — только четыре оцениваемых класса, а ручная разметка обучения содержит также другие классы.

### Дополнительные аннотации обучения

`upstream/annotations/` содержит сохранённые решения разметки машинного пула **только 25–26 октября**:

- `approved_pseudo_boxes_*.csv` — отобранные псевдобоксы для расширения train;
- `reviewed_pool_hands.csv` — проверенные вручную кисти;
- `reviewed_pool_crops.csv` — проверенные кропы, включая отрицательные;
- дополнительные таблицы выбранных тренировочных кропов, если они нужны соответствующему worker.

Это входная разметка, как и `human_boxes_train.csv`, а не признаки, которые нужно получить нейросетью. Код обучения строит по ней изображения-кропы, YOLO-labels, синтетические возмущения и список обучающих примеров. Тестовых `error`-меток в этих аннотациях нет. Для точного повторения нужны именно опубликованные аннотации; новый ручной пересмотр — уже другое обучение.

## Откуда берётся каждый кэш

| Артефакт | Генератор | Первичный вход |
|---|---|---|
| `queries.csv`, `train_labels.csv`, `manual_boxes.csv`, `prototypes.csv`, `dense_records.csv` | `prepare.py metadata` → `rascar/records.py` | Исходные CSV |
| SAM3-референсы и временное подтверждение | `prepare.py sam3` → `rascar/sam3.py` | JPG, соседние кадры, SAM3.1 |
| `wevis_full.csv`, `zoom_boxes.csv`, `heads.csv`, `hands.csv` | `prepare.py wevis-full/wevis-zoom/wevis-heads/wevis-hands` | JPG, WeVis, текстовые prompts в `rascar/detectors.py` |
| DINOv3 retrieval и frozen baseline | `prepare.py retrieval`, `rascar/retrieval.py:assemble` | Кропы, GT train, SAM3 и WeVis |
| Четырёхклассовый YOLO + ранний геометрический CatBoost | `rascar/foundation_models.py` | JPG, manual GT, независимые референсы |
| Локальные YOLO-детекции и начальные hand-детекции | `upstream/code/local_evidence/extract_local.py`, `train_hand_specialist.py` | JPG + опубликованные YOLO-checkpoints |
| `semantic_oof.csv`, `semantic_test.csv` | `upstream/code/local_evidence/extract_semantics.py` | DINOv3-кропы, ручная разметка, два линейных классификатора |
| `dino_records.csv`, `dino_features.npy` раннего этапа | `prepare.py semantic` → `rascar/synthetic.py` | JPG и детерминированные возмущения боксов |
| Pose-детекции | `extract_pose.py`, `prepare.py pose` | JPG + YOLO26m-pose |
| TTA-детекции рук | `upstream/code/seat_and_tta/gpu_worker.py` | Кропы, повороты, начальные и уточнённые hand-checkpoints |
| SAHI-детекции | `upstream/code/modes_and_sahi/gpu_yolo_hands.py` | JPG, расширенный hand-checkpoint |
| Ранние пространственные признаки / вероятности | `build_spatial_records.py`, `gpu_spatial.py` в `boundary_review` | DINOv2-B + train/GT/дополнительные аннотации |
| Targeted zoom-детекции | `upstream/code/boundary_review/gpu_hand_train.py` | JPG, целевой hand-checkpoint |
| `guarded_scores.csv` | `rebuild_upstream.py` | Все предыдущие виды свидетельств, последовательные CPU-модели и правила |
| `static_features.csv` | `prepare.py geometry` → `rascar/geometry.py` | Независимые детекции, координаты запросов |
| `embeddings/*_{queries,prototypes}.npy` | `prepare.py roi` → `rascar/vision.py` | JPG + CLIP/DINOv2-B/DINOv2-L |
| `dense_small.npy`, `upstream_scores.csv` | `prepare.py dense-small`, `rebuild_baseline.py` | JPG + DINOv2-S + две линейные головы |

`upstream/data/` — минимальные листовые входы CPU-пересчёта. Выходы промежуточных CPU-моделей не подставляются из старого архива: `rebuild_upstream.py` вычисляет их и передаёт следующему этапу. `from_images.py` начинает с пустой папки признаков и вызывает указанные генераторы.

## Порядок ранних этапов

1. `local_evidence`: независимые локальные детекции, семантика и pose.
2. `geometry_guards`: DINO-признаки, части тела, геометрические ограничения.
3. `seat_and_tta`: устойчивость положений и повороты кропов рук.
4. `body_and_hands`: полнота тела/кисти и объединение свидетельств.
5. `modes_and_sahi`: режимы объектов и детекция рук по тайлам.
6. `kernel_quality`: классификаторы качества и защитные правила.
7. `boundary_review`: пространственное качество границ и целевые zoom-детекции.

Точный список исполняемых CPU-скриптов — `upstream/plan.json`. Все скрипты запускаются общим launcher; вручную разбирать десятки старых версий не нужно. Исторические имена некоторых внутренних файлов оставлены для точного соединения этапов, но ветки и пользовательские команды используют смысловые имена.

## Какие веса включены

| Семейство | Количество | Код обучения |
|---|---:|---|
| Четырёхклассовый YOLO | 3: два дня + оба дня | `rascar/foundation_models.py:detector` |
| Начальный hand-YOLO | 3 | `local_evidence/train_hand_specialist.py` |
| Уточнённый hand-YOLO | 3 | `seat_and_tta/gpu_worker.py` |
| Расширенный hand-YOLO | 2 | `modes_and_sahi/gpu_yolo_hands.py` |
| Targeted hand-YOLO | 2 | `boundary_review/gpu_hand_train.py` |
| Ранний геометрический CatBoost | 2 | `rascar/foundation_models.py:geometry` |
| DINOv3 semantic / quality heads | 4 | `local_evidence/extract_semantics.py` |
| Ранние пространственные SVM / LogisticRegression | 6 | `boundary_review/gpu_spatial.py` |
| Линейные DINOv2-S головы | 2 | `rebuild_baseline.py --train` |
| Прототипные CatBoost | 6 | `train.py` |

Имена ранних весов, их источник, размер и SHA256 — `upstream/weights/manifest.json`. Их рабочие пути — `upstream/model_paths.json`. Для ранних spatial-моделей проверяется содержимое обучающих записей через `upstream/spatial_training.json`: удаление неиспользуемого эксперимента с людьми изменило абсолютные индексы строк, но выбранные записи рук/чашек и их порядок совпадают.

Остальные ранние CPU-классификаторы намеренно переобучаются при `rebuild_upstream.py`: код обучения включён, кэши достаточны, исходный результат проверен численно. Это не требует скрытых checkpoint-файлов.

## Окружения и источники pretrained-моделей

| Окружение | Этапы | Requirements |
|---|---|---|
| `.venv` | CPU, CatBoost, DINOv2, YOLO, SAHI, TabICL | `requirements.txt` / `requirements-detectors.txt` |
| `.venv-vlm` | WeVis, DINOv3 | `requirements-wevis.txt` |
| `.venv-sam3` | SAM3.1 | `requirements-sam3.txt` |

`setup_env.py --gpu` создаёт окружения; `from_images.py` запускает нужный Python для каждого этапа. Произвольные пути можно передать через `--python-cv`, `--python-vlm`, `--python-sam3`, `--python-sam2`.

Все foundation-модели перечислены в `weights.json`: официальный репозиторий, закреплённая ревизия, файлы и SHA256 либо snapshot-manifest. Для snapshot-моделей downloader сохраняет SHA256 фактически скачанных файлов в `download_manifest.json`. Источники: OpenAI CLIP, Meta DINOv2 / DINOv3 / SAM2 / SAM3.1, WeVis, TabICL, Ultralytics YOLO. Их использование подчиняется лицензиям авторов и условиям источников; веса foundation-моделей в Git не дублируются.

CLIP-зависимость SAM3 закреплена на ревизии `7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40` официального форка Ultralytics. В исходном эксперименте ставился тогдашний HEAD; отдельный исходный manifest этого пакета не восстановлен, поэтому идентичность этой зависимости старому окружению не подтверждена.

## Проверки переноса

- CPU-предсказания обеих веток: каждый из 560 скоров совпадает в допуске `1e-10`.
- Повторное обучение шести прототипных CatBoost: совпадение в допуске `1e-10`.
- Ранний CPU-ансамбль: максимальная разница `3.33e-16`.
- Два ранних геометрических CatBoost, обученные заново: максимальная разница `5e-16` на соответствующих train-запросах.
- Метаданные: исходные ID, порядок, 240 прототипов, 1 219 dense-кропов восстановлены.
- 22 статических геометрических признака: совпадение в допуске `1e-10`.
- 5 336 ранних синтетических записей: идентификаторы, типы, метки и веса совпадают; максимальное расхождение координат `6.8e-13`.
- Инференс двух приложенных линейных DINOv2-S голов на исходных полных картах: максимальная разница `0`.
- Исходный финальный generic-YOLO: SHA256 совпал с историческим manifest.
- Дополнительные аннотации + генератор spatial-записей: выбранные training records совпадают по содержимому и порядку.

GitHub Actions проверяет CPU-предсказания, SHA256 и повторное обучение прототипных моделей. Оно не запускает тяжёлые GPU-этапы.
