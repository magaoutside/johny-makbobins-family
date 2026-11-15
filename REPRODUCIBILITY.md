# Данные, веса и воспроизведение

Команды установки и запуска — в [README](README.md).

## Маршруты

| Команда | Что вычисляет | Входы |
|---|---|---|
| `python predict.py` | Финальные предсказания | Сохранённые признаки и обученные модели ветки |
| `python train.py` | Шесть прототипных CatBoost | Признаки и train-метки |
| `python rebuild_upstream.py` | Ранний CPU-ансамбль | Сохранённые детекции и эмбеддинги |
| `python from_images.py` | Признаки и всю цепочку предсказаний | Исходные JPG/CSV, веса, решения разметки |
| `python from_images.py --retrain --out outputs/full_training` | Признаки, машинный пул и обучение выбранных моделей | Исходные JPG/CSV, foundation-веса, решения разметки |

`verify.py` проверяет SHA256 артефактов, ID, порядок строк, диапазон скоров и OOF AP. Стандартный режим сравнивает скоры с исходным решением с допуском `1e-10`; `--allow-drift` сохраняет численное отличие без требования точного совпадения. Отчёт — `verification.json` в папке предсказаний.

## Исходные данные

Источник — Kaggle `rascar-hack-case-1`. Загрузчик: `download_data.py`.

- Запросы и разметка: `train.csv`, `test.csv`, `human_boxes_train.csv`, `scoring_policy.json`.
- Кадры и соседи: `train_frames.csv`, `test_frames.csv`, `train_context.csv`, `test_context.csv`.
- Машинный пул: `machine_frames.csv`, `machine_predictions.csv`.
- JPG всех нужных кадров; поиск рекурсивный, имя файла соответствует `frame_id`.

Для каждой таблицы нужен один файл с указанным именем. Координаты боксов заданы в системе 1280×720; изображения машинного пула приводятся к этому размеру перед обработкой кропов.

## Машинный пул

**Генератор — [generate_pool.py](generate_pool.py).** Его автоматически вызывает `from_images.py` между этапами `body` и `sahi`. Готовые псевдобоксы из репозитория в этот маршрут не подставляются.

1. Берутся машинные предложения 25–26 октября. Исключаются размеченные кадры и кадры ближе 600 секунд к ним.
2. Дневной четырёхклассовый YOLO проверяет класс и совпадение границ.
3. Дневной детектор рук проверяет два масштаба кропа; отдельный проход проверяет устойчивость на двух различных контекстах.
4. DINOv3 и обученная семантическая голова проверяют класс на точном и расширенном кропах.
5. Применяются пороги confidence, IoU, семантики и устойчивости; пересекающиеся предложения исключаются.
6. Машинные кандидаты соединяются с сохранёнными решениями визуального отбора **обучающего пула**. Новые кандидаты без решения записываются в `pending_review.csv` и не добавляются в геометрическую разметку автоматически. Для `person` автоматический отбор даёт только метку присутствия.

```bash
python generate_pool.py --raw cache/competition --out outputs/machine_pool --device cuda:0
```

В выходной папке: `pool_frame_ledger.csv`, журналы YOLO/DINOv3 по дням, `approved_pseudo_boxes_*.csv`, `manifest.json`. Отдельные шаги доступны через `--stage metadata|detectors|semantic|review|assemble`.

Разделение входов:

| Файл | Содержимое |
|---|---|
| `upstream/annotations/pool_review_decisions.csv` | Только ID, день и решения просмотра: без координат и предсказаний моделей. |
| `upstream/annotations/boundary_review/reviewed_pool_hands.csv` | 13 дополнительных аннотаций кистей, включая 4 добавленных при просмотре. |
| `upstream/annotations/boundary_review/reviewed_pool_crops.csv` | 11 проверенных обучающих кропов, включая 2 отрицательных. |
| `upstream/annotations/*/approved_pseudo_boxes_*.csv` | Сохранённый результат исходного отбора для сравнения; полный маршрут создаёт эти файлы заново. |

В машинных псевдобоксах координаты берутся из исходного `machine_predictions.csv`. Worker строит по отобранным боксам изображения-кропы, YOLO-labels и аугментации. Дневные модели используют пул своего дня.

Семантические головы пула включены как `upstream/weights/pool_semantic_*.joblib`. Их обучение на DINOv3-признаках:

```bash
python generate_pool.py --stage train-heads --out outputs/pool_models
```

Для применения переобученных голов поместите их вместе с `generic_*.pt` и `hand_initial_*.pt` в отдельную папку и передайте её через `--models`. В полном `--retrain` используются головы и детекторы текущего запуска через `--work`; команды копирования не нужны.

## Генераторы признаков

| Артефакт / этап | Код |
|---|---|
| Запросы, train-метки, ручные боксы, прототипы, dense-записи | `prepare.py metadata` → `rascar/records.py` |
| SAM3 и подтверждение соседними кадрами | `prepare.py sam3` → `rascar/sam3.py` |
| WeVis: кадр, zoom, головы, кисти | `prepare.py wevis-full/wevis-zoom/wevis-heads/wevis-hands` |
| DINOv3 retrieval | `prepare.py retrieval` → `rascar/retrieval.py` |
| Четырёхклассовый YOLO и геометрический CatBoost | `rascar/foundation_models.py` |
| Локальные детекции, pose, начальные руки | `upstream/code/local_evidence/`, `prepare.py pose` |
| DINOv3-признаки возмущённых боксов | `prepare.py semantic` → `rascar/synthetic.py` |
| TTA, SAHI, целевые zoom-детекции | `upstream/code/seat_and_tta/`, `modes_and_sahi/`, `boundary_review/` |
| Ранний ансамбль | `rebuild_upstream.py`; порядок скриптов — `upstream/plan.json` |
| Геометрические признаки | `prepare.py geometry` → `rascar/geometry.py` |
| ROI-эмбеддинги CLIP/DINOv2 | `prepare.py roi` → `rascar/vision.py` |
| DINOv2-S карты и линейная голова | `prepare.py dense-small`, `rebuild_baseline.py` |

## Веса и окружения

Наши обученные модели находятся в `models/` и `upstream/weights/`. Происхождение ранних весов — `upstream/weights/manifest.json`; протокол финального обучения — `models/training.json`.

Foundation-веса загружает `download_weights.py` по версиям из `weights.json`: CLIP, DINOv2/DINOv3, SAM3.1, WeVis, YOLO. Для SAM3.1 и DINOv3 нужен `HF_TOKEN` с доступом к соответствующим репозиториям.

`setup_env.py --gpu` создаёт Python 3.11 окружения: `.venv` — CPU/CV/YOLO, `.venv-vlm` — WeVis/DINOv3, `.venv-sam3` — SAM3. Скрипты выбирают нужный Python автоматически; пути переопределяются через `--python-cv`, `--python-vlm`, `--python-sam3`. Kaggle-ноутбук устанавливает Python 3.11 и использует `uv` для общей загрузки пакетов.
