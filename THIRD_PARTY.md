# Модели, библиотеки и происхождение

Решение использует внешние модели и библиотеки. Их лицензии и ограничения действуют отдельно; наличие ссылок здесь не означает передачу прав на исходные данные соревнования или веса третьих лиц.

| Компонент | Источник |
|:--|:--|
| DINOv2 base / small / large | [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base), [small](https://huggingface.co/facebook/dinov2-small), [large](https://huggingface.co/facebook/dinov2-large) |
| SAM2.1 Tiny | [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) |
| TabICL v1.1 checkpoint | [jingang/TabICL](https://huggingface.co/jingang/TabICL), файл `tabicl-classifier-v1.1-20250506.ckpt` |
| TabICL runtime | [soda-inria/tabicl](https://github.com/soda-inria/tabicl), версия 2.2.0 |
| CatBoost | [catboost/catboost](https://github.com/catboost/catboost), версия 1.2.10 для replay |
| CLIP | [openai/CLIP](https://github.com/openai/CLIP), использован в семантических прототипах |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) |
| NumPy / pandas / scikit-learn | [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/) |

Исторические ветки также пробовали другие детекторы, pose/hand-модели и эмбеддинги. Конкретные источники загрузки сохранены в `research/rascar_v*/download*.py` и соответствующих скриптах извлечения признаков.

Pretrained foundation checkpoints и исходные RGB-фотографии в Git не включены. Компактные обученные головы, CatBoost-модели и saved-state TabICL относятся к артефактам этого решения.

Для проверки сабмитов не требуется скачивание pretrained-моделей. Для новых обучений и RGB-инференса необходимо отдельно получить модели и данные с соблюдением их условий.
