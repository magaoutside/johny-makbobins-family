from research_checks import *
import shutil,hashlib

name='v27_group_coordinates_catboost_shallow_w005'
source=A/f'results/submissions/submission_{name}.csv'
dest=A/'submission_v28_meta_ranker_research.csv';shutil.copyfile(source,dest)
d=pd.read_csv(A/f'results/reviews/{name}.csv')
d['v27_score']=d.id.map(pd.read_csv(ROOT/'rascar_v27/submission_v27_candidate.csv').set_index('id').score)
diag=[];pairs=[]
for label,states in [('clear',['plausible_correct','clear_error']),('inclusive',['plausible_correct','clear_error','likely_error'])]:
    z=d[d.visual_status.isin(states)];bad=z.visual_status.ne('plausible_correct');b=z.loc[bad];g=z.loc[~bad]
    old=b.v27_score.to_numpy()[:,None]-g.v27_score.to_numpy();new=b.new_score.to_numpy()[:,None]-g.new_score.to_numpy()
    diag.append(dict(subset=label,rows=len(z),v27_diagnostic_ap=float(AP(bad,z.v27_score)),v28_diagnostic_ap=float(AP(bad,z.new_score)),
        improved_pairs=int(((old<=0)&(new>0)).sum()),worsened_pairs=int(((old>0)&(new<=0)).sum())))
    if label=='clear':
        for i,j in np.argwhere((old>0)&(new<=0)):
            pairs.append(dict(error_review_no=int(b.iloc[i].review_no),correct_review_no=int(g.iloc[j].review_no),
                old_margin=float(old[i,j]),new_margin=float(new[i,j])))
pd.DataFrame(pairs).to_csv(A/'audit/clear_pair_regressions.csv',index=False)
(A/'audit/rank_diagnostics.json').write_text(json.dumps(diag,indent=2))
r=json.loads((A/f'results/{name}.json').read_text())
summary=dict(best_experiment=r,automatic_promotion=False,reason='Higher OOF on both days/classes, but 8 adverse ordering reversals among clear-error/plausible-correct visual cases and no beneficial reversals in that diagnostic subset.',
    baseline_rollback='V22 public AP 0.95610',working_candidate='V27',new_variants=36,final_model_bundles=18,
    matching_official_label_agreement=dict(compared=192,different=0,not_compared=6),visual_diagnostics=diag,
    submitted=False,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
(A/'summary.json').write_text(json.dumps(summary,indent=2))
text='''RASCAR — проверка полного Deep Research, V28
13 ноября 2026. Результаты сохранены; на LB не отправлено.

Итог: идеи из отчёта проверены частично, новым основным решением V28
не объявляется. V22 остаётся подтверждённым LB baseline 0.95610;
V27 — отдельно сохранённый кандидат, V28 — исследовательский CSV.

1. Cardinality-first matching реализован.
На синтетическом контрпримере max-sum IoU выбирает одну допустимую
пару вместо двух. Новая реализация совпала с полным перебором на
160 случайных маленьких матрицах. Приоритет числа пар гарантируется
бонусом min(n_queries,n_refs)+1, второй критерий — сумма IoU.
На 192 официальных запросах с доступным ручным GT новый matcher
воспроизвёл error без единого расхождения. Ещё 6 запросов без GT
нужного класса в этом сравнении пропущены; метки не перезаписывались.
На текущих frozen SAM-reference назначение assigned/unassigned
совпало с прежним max-sum для всех OOF/test запросов. Следовательно,
эта исправленная формулировка сама по себе не принесла новых очков.

2. Координатные и групповые признаки действительно обучены.
Три ablation: group / coordinates / group+coordinates.
LogReg C=0.03/0.1 и CatBoost depth=2, 100 iterations, L2=20;
18 финальных fold-пакетов, 36 сохранённых blends. Предсказания
строго 25→26, 26→25. Веса кадров нормированы, scaler fit-day-only.
В признаки входят confidence, reference IoU, offsets по четырём
координатам, разброс коррелированных SAM views, bbox-mask offsets,
геометрия, класс, assignment margin/competitors. Только frozen
pretrained SAM; finetuned YOLO, baseline score и test visual labels
не входят в обучение. Это MetaDetect/ReDC-inspired ablation,
а не воспроизведение всей опубликованной ReDC.

Лучший сохранённый малый blend: 95% V27 + 5% shallow CatBoost.
                           V22          V27          V28 research
OOF pooled                 0.983219826  0.984863494  0.985681679
25 октября                 0.973169965  0.974727860  0.976386896
26 октября                 0.990405676  0.991209968  0.991672998
person                     0.955451466  0.955451466  0.957306059
nude_hands                 0.994877156  0.995226562  0.995589406
cup                        0.990680706  0.990680706  0.991210687
tray                       1.000000000  1.000000000  1.000000000

Почему пока не заменяем V27:
Метрика зависит от порядка, поэтому проверены именно перестановки
error/correct, а не только изменение score или старый порог аудита.
На frozen visual clear_error+plausible_correct subset (210 строк)
получились 8 ухудшающих порядок перестановок и 0 улучшающих.
Диагностический AP 0.960597 → 0.958797. Если добавить likely_error,
39 ухудшающих пар, 0 улучшающих; AP 0.929592 → 0.925931.
Это не hidden test AP, labels визуальные и уже участвовали в разработке.
Но сигнал предупреждает: растущий OOF не гарантирует устранение остатка.
Просмотрены 12 наиболее неблагоприятных изменений: возвращается
снижение штрафа person-on-clothing #208 и коротких рамок; у нескольких
хороших кистей штраф возрастает. Даже после улучшения OOF автоматически
смешивать этот ranker в основной CSV преждевременно.

3. Что в полном ресёрче требует поправки по фактам проекта.
WeVisOne-VL-4B уже использован: rascar_v16/extract_wevis.py загружает
именно tencent/WeVisOne-VL-4B, есть кэши heads/hands и более ранний
wevis_full. Повторный полный прогон не будет новым независимым экспертом.
RTMW/WholeBody и WiLoR detector также уже проверены в V25/V27.
WiLoR detector — не HaMeR/WiLoR mesh, полноценный mesh пока не запускался.
#96 не является почти детерминированной ошибкой только из-за ног
за рамкой: приблизительный IoU около 0.649 и неопределённость пересекает
0.6. #308 остаётся более убедительным случаем недостаточного охвата.
Старые seat-templates нельзя превращать в GT. Согласен с подходом
mesh как uncertainty-feature, а не автоматическим расширением запроса.

4. Приоритет дальнейшего coding-этапа с учётом фактически пройденного.
P0: SAM 3D Body / HMR2 feasibility на небольшом заранее фиксированном
manual subset. Проверить доступ к официальным весам, зависимости,
время/память; затем projected_bbox и стабильность y2. На fit-day
калибровать признаки только по manual visible/full-body convention.
Принимать по error separation на противоположном дне, включая целые
GT и намеренно укороченные/расширенные варианты. Mesh не служит target.
P0: добавить независимую семантику к coordinate ranker прежде чем
переносить его на все строки: отдельно clothing/person, tray/plate,
hand/forearm. Уже существующие DINO/WeVis признаки сначала использовать
в fold-safe ablation, не повторять получение тех же предсказаний.
P1: полноценный hand-mesh verifier с body-wrist identity и видимостью
суставов. Начать с manual hands и известных FP, лишь потом hard test.
P1: DINOv3 patch retrieval с настоящими hard negatives, без выдачи
ближайшего бокса за GT. Background residual — слабый вспомогательный
признак, проверить смену освещения; не применять к невидимым ногам.
P2: новые внешние API/большие VLM и fine-tuning SAM до появления
положительного независимого сигнала. Не тратить весь срок на setup.

Для каждого следующего опыта сохранять features, fold-models, AP
pooled/day/class, изменённые пары ранжирования, adverse-case images,
тайминг и источник весов. Не выбирать коэффициенты по LB. Новый
исследовательский CSV не заменяет подтверждённый baseline автоматически.

Проверенные первичные источники новых направлений:
ReDC paper: https://arxiv.org/abs/2607.29040
ReDC code: https://github.com/JyLee-vilab/ReDC
SAM 3D Body: https://github.com/facebookresearch/sam-3d-body
WeVisOne: https://github.com/Tencent/WeVisOne-VL
Веса SAM 3D Body требуют запроса доступа по официальной инструкции;
здесь не заявляется, что они уже скачаны или модель уже запускалась.
'''
(A/'REPORT.txt').write_text(text)
print(json.dumps(summary,indent=2))
