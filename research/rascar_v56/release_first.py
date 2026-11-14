"""Build a fixed candidate from completed predictions without neural computation."""
from pathlib import Path
import csv,json,hashlib,shutil
A=Path(__file__).resolve().parent;ROOT=A.parent;R=A/'release';R.mkdir(exist_ok=True)
base=ROOT/'rascar_v52/release/submission_v52_candidate_purged_a20.csv'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='d30da6f6f938cba285c01a6e8ddb4fe8d095f38e0ea65ec38a2aa87091705ab5'
def read(p):return list(csv.DictReader(p.open()))
b=read(base);d=read(A/'results/submissions/v1_anchor_mean3_nude_hands_a40.csv');m={r['id']:r for r in read(A/'features/v1_anchor_mean3.csv') if r['split']=='test'}
new={r['id']:r['score'] for r in d};classes={r['id']:r['class_name'] for r in read(ROOT/'rascar_v52/inputs/rows.csv') if r['split']=='test'}
rows=[];replay=[];changed=0
for r in b:
    hand=classes[r['id']]=='nude_hands';p=float(m[r['id']]['p_error']);v=(float(r['score'])+.4*p)/1.4 if hand else float(r['score'])
    assert abs(v-float(new[r['id']]))<1e-12
    row={'id':r['id'],'score':new[r['id']] if hand else r['score']};rows.append(row);replay.append(dict(id=r['id'],baseline=r['score'],p_error=m[r['id']]['p_error'],is_hand=int(hand)))
    changed+=abs(float(row['score'])-float(r['score']))>1e-9
assert changed==131 and len(rows)==362 and len({r['id'] for r in rows})==362
for name,items in [('submission_v56_tabicl_hands_a40.csv',rows),('replay_inputs.csv',replay)]:
    with (R/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
summary=[r for r in read(A/'results/ensemble_v1.csv') if r['name']=='v1_anchor_mean3_nude_hands_a40'][0]
manifest=dict(file='submission_v56_tabicl_hands_a40.csv',sha256=hashlib.sha256((R/'submission_v56_tabicl_hands_a40.csv').read_bytes()).hexdigest(),public=None,baseline_public=.963,development=summary,changed_test=131,other_231_score_strings_identical=True,selection_adjusted=False,formula='hands: (V52 + 0.4 * mean of three TabICL v1.1 seed probabilities) / 1.4; two fit days averaged for test')
(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
(R/'README_RU.txt').write_text('V56 — первый кандидат для проверки на Public\n\nФайл: submission_v56_tabicl_hands_a40.csv\n198 исходных OOF-боксов, строго противоположные дни обучения/оценки.\nV52 OOF AP 0.988579935 -> V56 0.989831038 (точное значение см. manifest.json).\nИспользован TabICL v1.1 на 50 признаках V52: геометрия, исходные замороженные модели, GT-прототипы только обучающего дня с исключением соседних кадров.\nТри seed 5601/5602/5603; 8/16/16 внутренних ансамблей. Выбран общий средний прогноз, не лучший seed.\nИзменяются 131 оценка рук. Остальные 231 строки score сохранены текстово в точности как V52.\nЛокально исправляет 18 неверно ранжированных пар, ухудшает 2. Выигрыш сохраняется при удалении любого из 35 кадров.\nBootstrap здесь описательный: development-выборка многократно использована для выбора моделей. Это не независимая проверка и не гарантия роста Public.\nПодтверждённый пользователем Public лучшей версии V52: 0.96300. V56 ещё не отправлен.\n\nКод и локальные модели: ../tabular.py, ../ensemble_audit.py, ../models/tabiclv1_anchor*.pkl. Проверка сохранённых моделей продолжается на Kaggle; 5 из 6 локально воспроизведены с максимальной ошибкой <3e-8 до остановки локальных вычислений.\nИсточник модели: https://github.com/soda-inria/tabicl\n')
print(json.dumps(manifest,indent=2))

