# Карта артефактов

[← Главная](../README.md) · [Запуск](REPRODUCIBILITY.md)

## Оригинальные сабмиты

| Версия | CSV | Public AP |
|:--|:--|--:|
| V58 | [`submissions/v58.csv`](../submissions/v58.csv) | 0.97564 |
| V52 | [`submissions/v52.csv`](../submissions/v52.csv) | 0.96300 |

```text
V58 SHA256
1ccc86a345fed7590f0462a2216aa9f07f032df0c68a63decf6cdcde2d8087d6

V52 SHA256
d30da6f6f938cba285c01a6e8ddb4fe8d095f38e0ea65ec38a2aa87091705ab5
```

## Включено в Git

- Полная доступная история авторского кода до V58 — `research/`, 600 файлов. Исходники сохранены, выводы ноутбуков удалены. Третьесторонние библиотеки устанавливаются отдельно.
- Шесть финальных V52 CatBoost и необходимые сохранённые признаки/прототипы.
- Шесть выбранных V58 DINO-B spatial голов, их fit-day/seed metadata.
- Компактные TabICL-модели-предшественники V58. Их `.pkl` содержат конфигурацию и обучающие данные; фундаментальный TabICL checkpoint устанавливается отдельно.
- Замороженные входы V58 для точной сборки score.
- OOF, таблицы экспериментов, seed-проверки, bootstrap и аудит ранжирования.

[`artifacts.sha256.json`](../artifacts.sha256.json) фиксирует содержимое решения, исходников и отчётов. `scripts/reproduce.py` проверяет его до запуска моделей.

## Вне Git

Исходные RGB-фото, виртуальные окружения, pretrained foundation checkpoints, большие dense-сетки и все fine-tuning checkpoints не включены. Это существенно больше исходного кода и компактного replay.

Большие рабочие артефакты сохранены в Output приватного [Kaggle notebook](https://www.kaggle.com/code/shapodev/rascar-v56-gpu-final-experiments), **Version 2 — V58 final verified**. Если ноутбук недоступен, потребуется доступ от его владельца; это не общедоступная download-ссылка.

Локальные полные архивы передачи `V52_FULL_CODE.zip` и `V58_FULL_CODE.zip` дополнительно содержат сохранённые промежуточные прогнозы и локальные экспериментальные веса. Их контрольные суммы сохранены в [`reports/delivery_archives.json`](../reports/delivery_archives.json). Сами ZIP в Git не дублируются.
