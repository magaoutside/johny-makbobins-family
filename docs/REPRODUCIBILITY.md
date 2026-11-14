# Воспроизведение

[← Главная](../README.md) · [Артефакты](ARTIFACTS.md) · [Данные](DATA.md)

## Проверенный запуск из этого репозитория

Нужен Python 3.11. Команды выполняются из корня репозитория:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py --version all
```

На Windows активация: `.venv\Scripts\activate`.

Сначала проверяются SHA256 артефактов, затем запускаются оба replay. Проверяются 362 уникальных `id`, исходный порядок строк, диапазон score [0,1] и совпадение с оригинальными сабмитами. Итоги сохраняются в `outputs/verification.json`.

### V52

```bash
python scripts/reproduce.py --version v52
```

Восстанавливаются прототипные признаки по сохранённым эмбеддингам, загружаются шесть CatBoost, рассчитываются OOF/test и выполняется финальное смешивание. Новое обучение не запускается, error-метки replay не читает.

Выход: `outputs/v52/test_a20.csv`. Проверка чисел — с допуском 1e-10; в исходной локальной проверке максимальное расхождение составляло 1.39e-16 из-за сериализации float. Оригинальный отправленный файл: [`submissions/v52.csv`](../submissions/v52.csv).

### V58

```bash
python scripts/reproduce.py --version v58
```

Применяется финальная формула к замороженным прогнозам экспертов из `replay_inputs.npz`. Выход: `outputs/v58/submission.csv`. После записи 12 знаков после запятой требуется **побайтовое совпадение** с оригинальным CSV:

```text
1ccc86a345fed7590f0462a2216aa9f07f032df0c68a63decf6cdcde2d8087d6
```

Это воспроизведение сборки score, а не повторный RGB-инференс. Шесть сохранённых DINO-голов отдельно воспроизвели исходные карты на Kaggle с максимальной ошибкой 0; [протокол](../reports/v58/spatial_checkpoint_replay.json).

## Обучение и RGB-инференс

В [`research/`](../research/) сохранена полная доступная история авторских исходников. Они ожидают исходную структуру рабочего каталога: `rascar_v31`, `rascar_v41`, `rascar_v52`, `rascar_v56`, `rascar_v58` и данные соревнования рядом. Одних исходников недостаточно для запуска всей истории с нуля: нужны изображения, кэши признаков и промежуточные модели.

### V52

1. Подготовить исходные таблицы и изображения из соревнования.
2. Восстановить frozen-признаки и базовый V31 из предыдущих этапов.
3. Запустить `rascar_v52/prepare.py` в подготовленной структуре.
4. Обучить выбранное семейство:

```bash
python rascar_v52/train.py --seed 23 --names base23_prototypes --models cb2 --purged-prototypes
python rascar_v52/train.py --seed 24 --names base23_prototypes --models cb2 --purged-prototypes
python rascar_v52/train.py --seed 25 --names base23_prototypes --models cb2 --purged-prototypes
```

`finalize.py` собирает release на основе таблиц экспериментов. Он также ожидает результаты других сравнений из рабочего проекта. Для получения уже зафиксированного V52 используйте автономный replay выше.

### V58 на Kaggle

Исходная GPU-сессия: **2 × Tesla T4**. Входы и адаптированный общий код готовятся через [`prepare_kaggle.py`](../research/rascar_v56/prepare_kaggle.py). Для Kaggle используется `common56.py` из `kaggle_payload`, где зависимости от локальной истории заменены подготовленными таблицами.

[`kaggle_bootstrap.py`](../research/rascar_v56/kaggle_payload/rascar_v56/kaggle_bootstrap.py) загружает изображения соревнования, DINO и TabICL, извлекает признаки и запускает базовых экспертов. Затем выбранная новая ветка:

```bash
# Уже подготовленный /kaggle/working с V56 inputs, dense-сетками и DINO.
python /kaggle/working/rascar_v58/gpu_dense.py vit_b cuda:0
python /kaggle/working/rascar_v58/gpu_spatial_tta.py
```

После выгрузки прогнозов сравнение и аудит выполнялись через `study.py`, `stress_test.py`, `finalize.py`. Эти скрипты ожидают сохранённые V56/V57 результаты и таблицы визуального аудита в исходной структуре проекта.

| Компонент | Зафиксированная версия |
|:--|:--|
| PyTorch / CUDA | 2.11.0+cu130, образ Kaggle |
| NumPy GPU-сессии | 2.1.3 |
| transformers | 4.53.1 |
| tokenizers | 0.21.4 |
| huggingface-hub | 0.36.0 |
| tabicl | 2.2.0 |
| scikit-learn | 1.6.1 |
| einops | 0.8.1 |

`requirements-gpu.txt` фиксирует доустановленные пакеты, но не является полным lock-файлом образа Kaggle. Установка этой таблицы сама по себе не восстанавливает данные и кэши.

## Где большие артефакты

Основной [Kaggle notebook](https://www.kaggle.com/code/shapodev/rascar-v56-gpu-final-experiments), **Version 2 — V58 final verified**, сохранён с Output. Он принадлежит аккаунту `shapodev`; доступ определяется разрешениями Kaggle. В Output остались большие dense-сетки и checkpoints fine-tuning-экспериментов. В Git включены выбранные компактные DINO-головы и модели-предшественники, а не весь многогигабайтный рабочий каталог.

## Автоматическая проверка

[`Reproduce submissions`](../.github/workflows/replay.yml) запускает CPU replay V52 и V58 на push, pull request или вручную. Workflow не обучает модели, не скачивает данные соревнования и не отправляет сабмиты.

Проверка на чистом клоне подтверждает переносимость replay. Она не доказывает воспроизводимость полного обучения из RGB; такой чистый end-to-end запуск пока не заявляется.
