"""Package the frozen V27 release, its evidence, and an exact score replay."""
from pathlib import Path
import json, hashlib, html, zipfile, base64
import pandas as pd
from PIL import Image, ImageDraw

A=Path(__file__).resolve().parent
ROOT=A.parent
s=json.loads((A/'summary.json').read_text())
r=pd.read_csv(A/'audit/candidate_changes.csv').query('review_no == 266').iloc[0]
d=pd.read_csv(A/'gpu/temporal_references.csv');d=d[d.id.eq(r.id)]
w=pd.read_csv(A/'features/weak_temporal_test.csv').set_index('id').loc[r.id]
src=d[d.source_frame.eq(w.source_frame)].sort_values('ncc',ascending=False).iloc[0]
canvas=Image.new('RGB',(1100,570),'white');draw=ImageDraw.Draw(canvas)
for i,(fid,box) in enumerate([(r.frame_id,r[['x1','y1','x2','y2']].to_numpy(float)),(src.source_frame,src[['source_x1','source_y1','source_x2','source_y2']].to_numpy(float))]):
    path=ROOT/'rascar_v16/test_images'/f'{fid}.jpg'
    if not path.exists():path=A/'gpu/neighbor_images'/f'{fid}.jpg'
    im=Image.open(path).convert('RGB');dr=ImageDraw.Draw(im);dr.rectangle(tuple(box),outline='yellow' if i==0 else 'cyan',width=2)
    crop=(130,150,350,360);im=im.crop(crop).resize((550,525))
    canvas.paste(im,(i*550,45));draw.text((i*550+8,8),('Target #266 / query' if i==0 else 'Neighbor +30s / WiLoR detection'),fill='black')
    draw.text((i*550+8,25),fid,fill='black')
canvas.save(A/'audit/hand266_temporal_evidence.jpg')

report='''RASCAR V27 — результаты и ограничения
Дата фиксации: 13 ноября 2026. На лидерборд ничего не отправлено.

РЕЗУЛЬТАТ
Подтверждённый лучший Public AP остаётся V22: 0.95610.
Основной OOF: V22 0.983219826 → V27 0.984863494 (+0.001643668).
25 октября: 0.973169965 → 0.974727860.
26 октября: 0.990405676 → 0.991209968.
AP по классам V27: person 0.955451466; nude_hands 0.995226562;
cup 0.990680706; tray 1.0. Межклассовый порядок меняется, поэтому общий
AP может расти даже при неизменном AP внутри cup.

ВАЛИДАЦИЯ
198 официально размеченных запросов / 35 кадров: 25 октября 68/12,
26 октября 130/23; 109 ошибок, 89 правильных боксов. Обучение моделей
25→26 и 26→25; scaler/PCA/calibrator обучаются только внутри fit-day.
Положительный класс — ошибка; метрика sklearn.average_precision_score.
Дополнительная разметка: 84 кадра / 1032 объекта, включая 82 person,
83 nude_hands и 25 tray. Синтетические варианты не являются новыми фото.
Основной OOF исключает унаследованную поправку, активную только на одной
OOF-строке. Исторические показатели (0.986039840 → 0.987631755) сохранены
отдельно; эти две шкалы сравнивать между собой нельзя.
Bootstrap по кадрам, отдельно по дням, 2000 выборок: интервал delta
[0.000264222, 0.003706275]. Это описательная устойчивость после отбора
экспериментов, а не независимое подтверждение или прогноз Public AP.

ЧТО В КАНДИДАТЕ
1. Ранее проверенный ансамбль экспертов чашек V24: 22 тестовые строки.
2. Ранее проверенные V25/V26 pose-подтверждения: 8 кистей.
3. V26 геометрия подносов с перспективой: #142 и #150.
4. Новый групповой штраф за два человека внутри одного bbox: #67.
   Требуются два уверенных reference, почти целиком внутри запроса,
   с разнесёнными центрами и сопоставимой высотой; один bbox не должен
   достаточно хорошо совпадать с любым из этих людей по отдельности.
   Score #67: 0.680592 → 0.842237.
5. Новое согласованное слабое подтверждение кисти: #266.
   WiLoR на текущем и соседнем кадре, перенос по LK optical flow,
   высокая фотометрическая согласованность, независимый reference,
   boundary-expert, семантические и duplicate-veto должны согласоваться.
   Score #266: 0.841061 → 0.610098. Без этих условий слабый detector
   confidence не используется как основание для снижения штрафа.
Итого 34 изменённых тестовых score и 11 OOF-score относительно V22.
Контрольный файл отличается от основного только отсутствием нового
снижения score для #266. OOF у обоих одинаковый.

ОГРАНИЧЕНИЕ НОВЫХ ПРАВИЛ
Ни групповой person-rule, ни слабое temporal-hand правило не активируются
на основном OOF. Его прирост в основном дают сохранённые эксперты чашек.
Поэтому нельзя приписывать этот прирост двум новым правилам.
Групповой person-rule: 361 вспомогательный запрос, 12/24 ошибочных
объединений обнаружены, 0/82 целых GT-боксов оштрафованы; всего 13
срабатываний, все на ошибках (ещё один — увеличенный bbox).
Temporal-hand: после независимого reference-gate 21 подходящий
синтетический вариант, все правильные; они приходятся лишь на 3 кадра
26 октября. На 25 октября активаций нет. Это не 21 независимый объект.
Вспомогательные кропы центрировались по GT; это проверка геометрии,
а не end-to-end оценка детектора. Test audit участвовал в разработке,
поэтому он также не является независимой валидацией.

ЧТО ДЕЙСТВИТЕЛЬНО ЗАКРЫТО В АУДИТЕ
Фиксированный диагностический список V22: 26 расхождений, теперь 23.
Через прежнюю диагностическую границу перешли #150, #67, #266:
3/26 = 11.5% остатка V22; два из них новые относительно V26.
Из старого списка 41 приоритетного случая проходят 20/41 = 48.8%
против 18/41 = 43.9% у V22. Это статистика визуального аудита при
фиксированном диагностическом пороге, не доля всех ошибок hidden test.
Этот порог не является метрикой AP и не использовался как target модели.
Все 34 изменённых кропа просмотрены. Явных новых визуальных регрессий
не обнаружено; неоднозначные дубликаты #40/#145/#334 не объявлены верными.
Остаются 6 сложных кистей: #98, #151, #200, #248, #333, #354.
Нерешённые torso-only случаи, чашка #143 и semantic tray #73 сохранены.
Нельзя утверждать, что все недостатки устранены.

УТОЧНЕНИЯ ВИЗУАЛЬНОГО GT
#96: видимые ноги действительно выходят из bbox, но приближённый IoU
полного видимого тела около 0.649; при сдвиге ручной границы ±5px
диапазон 0.582–0.689 пересекает 0.6. Наличие ноги снаружи само по себе
не доказывает ошибку. Статус переоценён как uncertain; старый frozen
audit не переписан и этот случай не засчитан как исправленный.
#308: аналогичная оценка 0.511, диапазон 0.441–0.514 — вероятная ошибка
охвата остаётся. Hidden GT неизвестен.

НОВЫЕ ЭКСПЕРИМЕНТЫ, ВКЛЮЧАЯ ОТРИЦАТЕЛЬНЫЕ
Всего 206 сохранённых вариантов ранжирования, 36 финальных обученных
пакетов моделей. Это не 206 независимых архитектур.
• Калибровка границ 21 hand-keypoints: Ridge и ExtraTrees, 12 пакетов,
  48 вариантов. Ни одну из семи сложных кистей не закрыла — отклонено.
• DINOv3 same-object pairwise LogReg / PairLogitPairwise CatBoost:
  24 пакета, 72 варианта. Лучший hand-head OOF около 0.984667,
  но ухудшает 12 визуально правильных кистей — отклонено.
• Эмпирическое распределение ошибок reference-координат: 48 вариантов.
  Person ухудшает один день, cup не добавляет к старому ансамблю.
• Проверки согласия quality/geometry: 6 вариантов; самостоятельного
  подтверждённого улучшения не дали.
• WiLoR current TTA + temporal transfer: реально выполнено на Tesla T4,
  290 запросов, 2042 reference, 246.99 секунд. Использованы соседние
  кадры своего дня ±65 секунд, до четырёх; четыре поворота кропа.
  Условное совпадение IoU≥0.6 на 83 manual-hand: current 33, temporal 10,
  combined 38. На 13 отдельно проверенных сложных pool-hand — 0.
  Это результат конкретных query-centred кропов, не full-image AP.
  24 широких варианта не включены. Включено только описанное выше
  сильное согласование слабых сигналов для #266, с контрольным файлом.
• Геометрия четырёх поворотов и обратного optical-flow переноса проверена
  на прямоугольном изображении и синтетическом сдвиге; ошибка <0.002px.

ПОЛНЫЙ DEEP RESEARCH, ПОЛУЧЕННЫЙ ПОСЛЕ РАСЧЁТОВ
Исходный текст сохранён в inputs/deep_research_user.txt. Это внешний
исследовательский документ, не доказательство качества на RASCAR.
ReDC, SAM 3D Body и WeVisOne проверены по официальным репозиториям.
Новые перспективные направления: координатная неопределённость,
3D-body как дополнительный признак, semantic contrast tray/plate.
Повторять RTMW/RTMPose и WiLoR detector как будто они ещё не проверены
не нужно: V25/V27 уже содержат такие прогоны. Полный hand-mesh отличается
от использованного WiLoR detector и пока не запускался.
Exact max-cardinality matching стоит проверить отдельно: в V24
использовался Hungarian по сумме IoU, что не гарантирует максимум числа
пар с IoU≥0.6. Новую реализацию нужно сверять с исходной логикой задания,
а не считать любой Hungarian автоматически идентичным evaluator.
SAM 3D Body требует доступа к checkpoint; запуск и прирост не заявляются.
Оценки времени и величины пользы из research не считаются измерениями.

ИСТОЧНИКИ МЕТОДОВ
https://github.com/rolpotamias/WiLoR
https://arxiv.org/abs/2407.08634
https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html
https://catboost.ai/docs/en/concepts/loss-functions-ranking
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
https://github.com/JyLee-vilab/ReDC
https://github.com/facebookresearch/sam-3d-body
https://github.com/Tencent/WeVisOne-VL

ВОСПРОИЗВОДИМОСТЬ
replay27.py собирает оба CSV из замороженных компонентов replay_inputs.
Проверено: максимальное расхождение с выпущенными score <2e-16.
rascar_v27_replay.ipynb содержит эти входы внутри себя и работает
без предыдущих папок. Это воспроизведение финального ранжирования,
не переобучение всех моделей с исходных фото.
Полные скрипты обучения сохранены отдельно; им нужны frozen features
предыдущих версий V15–V26. Новый GPU-скрипт gpu_temporal.py сохранён.
Все 206 CSV проверены: 362 уникальных правильных ID, finite score [0,1].
SHA256 baseline V22 совпадает с исходным. Старые эксперименты сохранены.
'''
(A/'REPORT.txt').write_text(report)
links=''.join(f'<p><a href="{x}">{x}</a></p>' for x in ['submission_v27_candidate.csv','submission_v27_control.csv','rascar_v27_replay.ipynb'])
imgs=''.join(f'<figure><img loading="lazy" src="audit/candidate_changes_{i:02d}.jpg" style="max-width:100%"></figure>' for i in range(6))
(A/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>RASCAR V27</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.5 system-ui;color:#172234}pre{white-space:pre-wrap}a{color:#185ac8}</style><h1>RASCAR V27 — OOF 0.984863</h1>'+links+'<pre>'+html.escape(report)+'</pre><h2>Новые свидетельства для кисти #266</h2><img style="max-width:100%" src="audit/hand266_temporal_evidence.jpg"><h2>Все изменённые детекции</h2>'+imgs)
payload={}
for f in ['replay_inputs/oof.csv','replay_inputs/test.csv']:
    frame=pd.read_csv(A/f)
    if f.endswith('oof.csv'):
        labels=pd.read_csv(A/'results/oof/release_candidate.csv')[['id','error','day']]
        frame=frame.merge(labels,on='id',validate='one_to_one')
    payload[f]=base64.b64encode(frame.to_csv(index=False).encode()).decode()
code='''import base64, io\nfrom pathlib import Path\nimport pandas as pd, numpy as np\nfrom sklearn.metrics import average_precision_score\nPAYLOAD = '''+repr(payload)+'''\nfor split, encoded in PAYLOAD.items():\n    d = pd.read_csv(io.BytesIO(base64.b64decode(encoded)))\n    for mode in ['control', 'candidate']:\n        score = (d.assembled + d.merged_delta + (d.weak_hand_delta if mode == 'candidate' else 0)).clip(0, 1)\n        out = pd.DataFrame({'id': d.id, 'score': score})\n        if split.endswith('test.csv'):\n            assert len(out) == 362 and out.id.is_unique and np.isfinite(score).all()\n            out.to_csv(f'submission_v27_{mode}.csv', index=False)\n            print(mode, 'saved', len(out))\n        else:\n            print(mode, 'OOF AP', average_precision_score(d.error, score))\n'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'markdown','metadata':{},'source':['# RASCAR V27: exact replay\n','Frozen score components, not end-to-end training. Public AP unknown; best confirmed V22 = 0.95610.\n','Cross-day primary OOF = 0.9848634937162036. Candidate differs from control only for hand #266.']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
(A/'rascar_v27_replay.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
files=[p for p in A.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix not in ['.zip','.pyc']]
manifest={str(p.relative_to(A)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='MANIFEST.json'}
(A/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print('Wrote report, replay notebook, manifest. Full package assembled after final checks.')
