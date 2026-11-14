"""Package the fresh audit and unpromoted experiments; keep V22 as best."""
from pathlib import Path
import json,hashlib,shutil,base64,html,sys
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v29'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,current,base_oof,metric,align
from audit_geometry import interval_iou,iou

def encoded(path):
    typ='image/webp' if path.suffix=='.webp' else 'image/jpeg'
    return 'data:'+typ+';base64,'+base64.b64encode(path.read_bytes()).decode()

def run():
    d=pd.read_csv(A/'audit/all_362_fresh_audit.csv').fillna('')
    summary=json.loads((A/'audit/summary.json').read_text())
    metrics=pd.read_csv(A/'results/geometry_metrics.csv')
    copies={'submission_v29_rollback_v22.csv':ROOT/'rascar_v22/submission_v22_candidate.csv',
            'submission_v29_group_geometry_probe.csv':A/'results/submissions/submission_cup_uncertainty_one_to_one_penalty_w0.5.csv',
            'submission_v29_geometry_research.csv':A/'results/submissions/submission_cup_uncertainty_one_to_one_both_w0.5.csv'}
    for name,path in copies.items():shutil.copyfile(path,A/name)
    hashes={name:hashlib.sha256((A/name).read_bytes()).hexdigest() for name in copies}
    assert hashes['submission_v29_rollback_v22.csv']=='44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c'
    decisions=dict(best='V22',best_public_ap=.95610,v27_public_ap=.95159,source='user-reported leaderboard scores',
        recommended_submission='No new promoted submission. Preserve V22.',
        files={k:dict(sha256=v,promoted=(k=='submission_v29_rollback_v22.csv'),submitted=False) for k,v in hashes.items()},
        probe_note='Two cup-score increases (#136 and #160); primary OOF AP unchanged. Duplicate winner #136/#137 is not robust to all plausible manual boundaries.',
        research_note='20 test changes, +0.000717 primary OOF; no within-class AP gain, unsafe rescues of disputed cup boxes, not promoted.',
        no_test_visual_labels_used_for_fitting=True,test_visual_audit_used_for_qualitative_rejection=True)
    (A/'DECISION.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2))
    # Render explicit evidence with uncertainty, not a claim of hidden ground truth.
    est=pd.concat([pd.read_csv(A/'annotations/cup_tight_boxes.csv'),pd.read_csv(A/'annotations/other_tight_boxes.csv')])
    paths=[A/'audit/measurement/transforms.json',A/'audit/measurement/changed_hands_transforms.json']
    transforms={r['review_no']:r for path in paths for r in json.loads(path.read_text())}
    font=ImageFont.truetype('/System/Library/Fonts/Monaco.ttf',13)
    tiles=[]
    for no in [1,21,145,204,266,136]:
        t=Image.new('RGB',(480,545),'white');t.paste(Image.open(A/f'audit/measurement/{no:03d}.png'),(0,0));dr=ImageDraw.Draw(t)
        r=est[est.review_no.eq(no)].iloc[0];b=r[['x1','y1','x2','y2']].to_numpy(float);tr=transforms[no];u=float(r.uncertainty_px)
        def xy(box):return [55+(box[0]-tr['x0'])*tr['scale'],50+(box[1]-tr['y0'])*tr['scale'],55+(box[2]-tr['x0'])*tr['scale'],50+(box[3]-tr['y0'])*tr['scale']]
        dr.rectangle(xy(b+np.array([-u,-u,u,u])),outline='#ffbf00',width=2)
        dr.rectangle(xy(b),outline='#00c286',width=2)
        row=d[d.review_no.eq(no)].iloc[0]
        dr.text((10,487),f'Visual IoU {row.manual_max_iou:.3f}; range {row.iou_min:.3f}-{row.iou_max:.3f}',font=font,fill='black')
        dr.text((10,507),'Green: visual estimate, NOT official GT',font=font,fill='black')
        dr.text((10,526),f'Yellow: +/- {u:g} native pixels per boundary',font=font,fill='black');tiles.append(t)
    sheet=Image.new('RGB',(1440,1090),'white')
    for i,t in enumerate(tiles):sheet.paste(t,((i%3)*480,(i//3)*545))
    sheet.save(A/'audit/geometry_conflicts.jpg',quality=95)

    report='''RASCAR V29 — повторный аудит после регрессии V27
Дата: 14 ноября 2026. Скрытые метки теста недоступны.

РЕШЕНИЕ
Лучшая подтверждённая версия остаётся V22: Public AP 0.95610.
V27: Public AP 0.95159 по сообщению пользователя, ухудшение 0.00451.
База возвращена к V22 в отдельной папке V29; старые версии и результаты сохранены.
Новых подтверждённых улучшений для лидерборда в этом проходе нет. На LB ничего не отправлялось.

ЧТО ПЕРЕПРОВЕРЕНО
Повторно просмотрены все 362 исходные рамки в 86 кадрах: 153 person, 131 nude_hands,
54 cup, 24 tray. Контактные листы не показывали score и старые визуальные метки;
это повторный просмотр тем же ассистентом, а не независимая слепая экспертиза.
Для каждой рамки сохранены причина, уровень неопределённости, сравнение V22/V27.
Первичный визуальный разбор: 52 явные ошибки класса/фона, 16 вероятных ошибок
локализации, 95 визуально правдоподобных рамок, 199 неопределённых.
Эти числа НЕ означают число ошибок ранжирования модели и НЕ являются скрытыми GT.
Из неопределённых: 94 — скрытый контур человека, 41 — назначение дублей,
30 — свободная рамка, 34 — недостаточно визуальных данных.

Дополнительно измерены плотные рамки для 44 чашек и 15 других запросов (59 всего).
Для маленьких объектов разрешена погрешность каждой границы ±2 пикселя;
для пересвета и части других случаев ±3. Рассчитаны минимальный/максимальный IoU
в этом диапазоне, включая точки совпадения границ с исходной рамкой.
13 рамок остаются ниже 0.6 во всём заданном диапазоне, 45 пересекают порог,
одна уверенно проходит геометрию. Это условные выводы при принятых визуальных
границах, не статистические доверительные интервалы и не официальные метки.
10 чашек с ошибочным классом не требовали построения рамки чашки.

ЧТО БЫЛО НЕ ТАК С ПРЕЖНИМ АУДИТОМ
В некоторых случаях «предмет присутствует внутри рамки» принималось за
«рамка корректна». Для IoU >= 0.6 этого недостаточно: рамка может включать
слишком много фона, часть телефона/блюдца или обрезать пальцы.
Поэтому прежние проценты «закрытых ошибок» не следует трактовать как процент
исправленных скрытых меток. Здесь новый процент закрытия не заявляется.

НАИБОЛЕЕ ВАЖНЫЕ СЛУЧАИ
• #266, кисть на ноутбуке: V27 снизил score ошибки 0.841061 → 0.610098.
  Кисть есть, однако рамка содержит фон слева и обрезает пальцы снизу.
  Визуальный IoU 0.384, диапазон 0.201–0.595 при ±3 пикселях.
  Это существенный конфликт с прежним объявлением случая исправленным.
  Границы на пересвете условны; V22 уже выставлял здесь высокий score ошибки.
• #145, кисть у телефона: 0.704437 → 0.401855. Визуальный IoU 0.476;
  диапазон пересекает 0.6. Вместе с #146/#147 это одна кисть и три рамки.
  Нельзя понизить штраф только потому, что pose-модель нашла руку.
• #204, маленькая кисть ребёнка: 0.616231 → 0.437414. Визуальный IoU 0.407,
  диапазон 0.238–0.621: наличие кисти подтверждено, правильность рамки — нет.
• #1, чашка: V22 и V27 оставляют 0.629589. Визуальный IoU 0.395,
  верхняя граница диапазона 0.553. Большой запас фона — конкретная цель
  для алгоритма оценки границ. #73 — тарелка вместо подноса, score 0.666412.
  Это два случая с явной/геометрически устойчивой ошибкой ниже старого
  диагностического порога 0.688994. Порог не определяет AP и не говорит,
  что остальные рамки ранжированы оптимально.
• #67, люди: количество голов в рамке само по себе не доказывает ошибку.
  Нужен IoU с отдельным человеком по принятой политике скрытых частей тела.
  Прежний штраф за объединение не подтверждён настоящей разметкой теста.

ЧАШКИ И ДУБЛИ
V27 снизил score у 22 чашек. У 14 точечная визуальная оценка IoU < 0.6;
у всех 22 интервал неопределённости пересекает порог. Поэтому это обоснованный
список подозреваемых, но назвать конкретную чашку доказанной причиной Public
регрессии нельзя. Есть чашки вместе с блюдцем (#109), две чашки в одной
рамке (#157/#160/#161), чашка плюс соседняя посуда (#288).
Сохранены 20 групп повторных/перекрывающихся запросов, а также проверка
устойчивости назначения для семи групп с измеренными границами.
Нельзя принудительно оставлять одну «правильную» рамку: иногда ни одна
не достигает IoU 0.6. Например, в группе #145–147 центральная оценка
не пропускает ни одну; #145 только иногда проходит при расширении границ.

ПОЧЕМУ OOF НЕ ПРЕДУПРЕДИЛ
Первичный OOF: V22 0.98321983, V27 0.98486349. Улучшение OOF не гарантировало LB.
Официальная валидация мала: 198 запросов / 35 кадров, в том числе 33 чашки
(12 корректных, 21 ошибочная). 25 октября: 9 чашек; 26 октября: 24.
Используется cross-day: обучение/калибровка на 25 → оценка 26 и наоборот.
Для AP важен глобальный порядок всех классов. V27 изменил только 34 значения,
но поменял относительный порядок 921 пары на полном тесте. Не каждая такая
пара влияет на AP; состав public/private неизвестен.
Если бы все 33 пониженных запроса были правильными, а единственный повышенный
ошибочным, AP не мог бы ухудшиться. Значит, прежняя трактовка хотя бы части
изменений не соответствует public-разметке. Совпадающие LB-скоры сами по себе
не доказывают, что изменённые фотографии лежат в private.

НОВЫЙ ПРОВЕРЕННЫЙ ЭКСПЕРИМЕНТ
Оценка неопределённости границ чашки: замороженные SAM-референсы, эмпирические
совместные отклонения четырёх координат от ручной разметки, вес каждого кадра
нормирован, one-to-one matching в каждом сценарии. Дубли видов SAM объединены.
На 25-м дне для калибровки нашлось 18 соответствий в 10 кадрах,
на 26-м — 21 в 14. Тестовая визуальная разметка в обучение не передавалась.
Проверены 12 заранее заданных сочетаний: независимое/групповое назначение,
только повышение штрафа/двусторонняя коррекция, веса 0.1/0.25/0.5.

Лучший OOF: 0.98393693 против V22 0.98321983, прирост 0.00071711.
По дням: 25 — 0.97369126, 26 — 0.99083505 (точные значения в metrics CSV).
AP внутри класса cup остался 0.99068071. Выигрыш связан с относительным
положением чашек среди других классов; улучшение распознавания чашек не доказано.
Парный bootstrap с повторной выборкой кадров отдельно внутри дней
(2000 повторов) дал диапазон
прироста [0; 0.00249927]. Это описательная проверка после выбора эксперимента,
не независимый тест значимости. На тесте остаются сомнительные понижения
штрафа свободным рамкам, поэтому вариант отклонён как новая основная версия.

Сохранены два исследовательских CSV:
1. submission_v29_geometry_research.csv — 20 изменений, лучший OOF,
   НЕ рекомендован как доказанное улучшение.
2. submission_v29_group_geometry_probe.csv — только повышение score чашек
   #136 и #160, 2 изменения; OOF полностью совпадает с V22.
   Выбор победителя #136/#137 чувствителен к границам, поэтому это лишь проба.
submission_v29_rollback_v22.csv побайтово совпадает с лучшим V22.

ЧТО ДЕЛАТЬ ДАЛЬШЕ
1. Приоритет — локализация: раздельно оценивать наличие объекта и достаточность
   IoU исходной рамки. Не делать rescue по одному подтверждению класса/pose.
2. Для чашек и кистей проверять, сколько площади рамки занято фоном/телефоном,
   и какие пальцы/части предмета остались снаружи; учить на качественных
   дополнительных кадрах 25/26 и реальных пограничных ошибках геометрии.
3. Дубликаты проверять совместно с возможностью «все рамки ошибочны»;
   требовать устойчивого победителя при изменении границ референса.
4. Для людей со скрытыми ногами сначала согласовать политику разметки с
   размеченными кадрами, затем проверять шаблон места/амодальную геометрию.
   Уверенность в наличии человека не подтверждает правильность короткого бокса.
5. Новые исправления проверять одновременно по обоим дням, классу,
   глобальному AP и сохранённому списку проблемных кадров. Эта подборка
   уже используется в разработке и не является независимым test holdout.
'''
    # Insert exact day values rather than rounded hand transcription.
    best=metrics[metrics.experiment.eq('cup_uncertainty_one_to_one_both_w0.5')].iloc[0]
    report=report.replace('0.97369126',f'{best.day_25:.8f}').replace('0.99083505',f'{best.day_26:.8f}')
    (A/'REPORT.txt').write_text(report)
    labels={'clear_error':'Явная ошибка класса/фона','likely_error':'Вероятная ошибка границ','plausible_correct':'Визуально правдоподобно','uncertain':'Неопределённо'}
    items=[]
    for r in d.to_dict('records'):
        no=int(r['review_no']);r['thumbnail']=encoded(ROOT/f'rascar_test_visual_audit/crops/{no:03d}.webp');r['status_text']=labels[r['visual_status_v29']];items.append(r)
    data=json.dumps(items,ensure_ascii=False).replace('</','<\\/')
    doc='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RASCAR — повторный аудит 362 рамок</title><style>
*{box-sizing:border-box}body{margin:0;background:#10151e;color:#e8edf4;font:15px system-ui,sans-serif}header,main{max-width:1400px;margin:auto;padding:28px}h1{font-size:28px;margin:0 0 12px}p{line-height:1.55}.muted{color:#aab9c9}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{background:#1c2736;padding:14px 20px;border-radius:10px}.stat b{font-size:24px;display:block}.notice{background:#30291c;border-left:4px solid #edb64b;padding:14px 18px;border-radius:4px}details{background:#182230;border-radius:10px;padding:16px;margin:20px 0}pre{white-space:pre-wrap;font:14px/1.6 system-ui}.tools{position:sticky;top:0;background:#10151ef5;padding:14px 0;display:flex;gap:10px;flex-wrap:wrap;z-index:3}select,input,button{background:#243348;color:inherit;border:1px solid #42556c;border-radius:7px;padding:10px}input{flex:1;min-width:220px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}.card{background:#1b2635;border:1px solid #314156;border-radius:12px;overflow:hidden}.card img{width:100%;height:270px;object-fit:contain;background:#edf0f3;cursor:zoom-in}.body{padding:14px}.title{font-weight:700;font-size:18px}.badge{font-size:12px;padding:5px 7px;background:#364b65;border-radius:5px;display:inline-block;margin:8px 0}.clear_error{background:#743737}.likely_error{background:#755923}.uncertain{background:#414a59}.plausible_correct{background:#275b4b}.scores{font-variant-numeric:tabular-nums;padding:8px 0;color:#bbcadd}.note{line-height:1.5}.small{font-size:12px;color:#9daec2;word-break:break-word}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #3b4a5c;text-align:left}dialog{background:#182230;color:white;max-width:96vw;border:1px solid #62738a;border-radius:12px}dialog img{max-width:90vw;max-height:80vh}a{color:#80bfff}.evidence{max-width:100%;border-radius:8px}
</style><header><h1>RASCAR: повторный аудит после V27</h1>
<p>Основа: <b>V22 · Public AP 0.95610</b>. V27: <b>0.95159</b> · Δ −0.00451.</p>
<div class="stats"><div class="stat"><b>362 / 362</b>рамок просмотрено</div><div class="stat"><b>86</b>кадров</div><div class="stat"><b>59</b>геометрических измерений</div><div class="stat"><b>20</b>групп запросов</div></div>
<p class="notice">Это визуальный аудит, а не скрытая разметка Kaggle. «Предмет есть» не означает IoU ≥ 0.6. Процент исправленных скрытых ошибок неизвестен. Новые эксперименты сохранены, основная версия остаётся V22.</p>
<details open><summary>Главные расхождения геометрии</summary><p>Красная рамка — исходный запрос. Зелёная — визуальная оценка границ. Жёлтая — допустимое расширение при указанной погрешности. Интервалы условны, это не официальные GT.</p><img class="evidence" src="EVIDENCE"></details>
<details><summary>Полный отчёт, результаты и ограничения</summary><pre>REPORT</pre></details>
</header><main><div class="tools"><input id="search" placeholder="Номер, ID, причина, группа…"><select id="class"><option value="">Все классы</option><option>person</option><option>nude_hands</option><option>cup</option><option>tray</option></select><select id="status"><option value="">Все статусы</option><option value="clear_error">Явные ошибки</option><option value="likely_error">Вероятные ошибки</option><option value="uncertain">Неопределённые</option><option value="plausible_correct">Правдоподобные</option></select><select id="mode"><option value="all">Все рамки</option><option value="changed">Изменённые V27</option><option value="groups">Группы дублей</option><option value="measured">Измеренная геометрия</option><option value="robust">Устойчивый геометрический конфликт</option></select><select id="sort"><option value="number">По номеру</option><option value="change">По величине изменения V27</option><option value="score">По score V22</option></select></div><p id="count" class="muted"></p><div id="grid" class="grid"></div></main>
<dialog id="dialog"><button id="close">Закрыть</button><div id="detail"></div></dialog>
<script>const rows=DATA;const $=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const f=x=>typeof x==='number'?x.toFixed(3):'—';
function render(){const term=$('search').value.toLowerCase().trim(),cl=$('class').value,st=$('status').value,mo=$('mode').value;let a=rows.filter(r=>(!cl||r.class_name===cl)&&(!st||r.visual_status_v29===st)&&(!term||(/^#?[0-9]+$/.test(term)?String(r.review_no)===term.replace('#',''):[r.id,r.note_v29,r.group_id,r.frame_id].join(' ').toLowerCase().includes(term)))&&(mo==='all'||mo==='changed'&&Math.abs(r.v27_delta)>1e-9||mo==='groups'&&r.group_id||mo==='measured'&&r.geometry_status||mo==='robust'&&r.geometry_status==='robust_error_under_assumptions'));a.sort($('sort').value==='change'?(a,b)=>Math.abs(b.v27_delta)-Math.abs(a.v27_delta):$('sort').value==='score'?(a,b)=>b.v22_score-a.v22_score:(a,b)=>a.review_no-b.review_no);$('count').textContent=`Показано ${a.length} из 362. Score выше = сильнее подозрение на ошибку.`;$('grid').innerHTML=a.map(r=>`<article class="card"><img loading="lazy" src="${r.thumbnail}" alt="Рамка ${r.review_no}" data-no="${r.review_no}"><div class="body"><div class="title">#${r.review_no} · ${esc(r.class_name)}</div><span class="badge ${r.visual_status_v29}">${esc(r.status_text)}</span><div class="scores">V22 ${f(r.v22_score)} → V27 ${f(r.v27_score)}<br>Δ ${f(r.v27_delta)} · ранг ${r.v22_rank} → ${r.v27_rank}</div><div class="note">${esc(r.note_v29)}</div>${r.geometry_status?`<p>Визуальный IoU <b>${f(r.manual_max_iou)}</b><br>Диапазон ${f(r.iou_min)}–${f(r.iou_max)}<br><span class="small">Неофициальная оценка; отдельно учитывать дубли.</span></p>`:''}${r.group_id?`<p>${esc(r.group_id)} · видимых объектов: ${r.group_visible_objects}</p>`:''}<p class="small">${esc(r.id)}<br>${esc(r.frame_id)}</p></div></article>`).join('')}
for(const id of ['search','class','status','mode','sort'])$(id).addEventListener('input',render);$('grid').addEventListener('click',e=>{const n=e.target.dataset.no;if(!n)return;const r=rows.find(x=>x.review_no===+n);$('detail').innerHTML=`<h2>#${n} · ${esc(r.class_name)}</h2><img src="${r.thumbnail}"><p>${esc(r.note_v29)}</p>`;$('dialog').showModal()});$('close').onclick=()=>$('dialog').close();render();</script></html>'''
    doc=doc.replace('EVIDENCE',encoded(A/'audit/geometry_conflicts.jpg')).replace('REPORT',html.escape(report)).replace('DATA',data)
    (A/'REPORT.html').write_text(doc)
    # Meaningful integrity and bound checks.
    checks={'reviewed':len(d),'unique_ids':d.id.nunique(),'all_review_nos':set(d.review_no)==set(range(1,363)),
        'official_test_labels_used':False,'baseline_byte_identical':True,'saved_variants':len(metrics)}
    assert len(d)==d.id.nunique()==362 and checks['all_review_nos']
    for path in (A/'results/submissions').glob('*.csv'):
        x=pd.read_csv(path);assert x.id.equals(te.id) and x.id.nunique()==362 and np.isfinite(x.score).all() and x.score.between(0,1).all()
    rng=np.random.default_rng(29)
    for _ in range(20):
        b=np.r_[rng.uniform(10,30,2),rng.uniform(50,70,2)];ref=b+rng.uniform(-10,10,4);lo,hi=interval_iou(b,ref,3)
        vals=iou(b,ref+rng.uniform(-3,3,(1000,4)))[0];assert vals.min()>=lo-1e-10 and vals.max()<=hi+1e-10
    checks['iou_intervals_random_samples_inside_bounds']=20000
    (A/'checks.json').write_text(json.dumps(checks,indent=2))
    (A/'MANIFEST.json').write_text(json.dumps({str(p.relative_to(A)):hashlib.sha256(p.read_bytes()).hexdigest() for p in A.rglob('*') if p.is_file() and p.name!='MANIFEST.json'},indent=2))
    print(json.dumps(checks,indent=2));print('HTML MB',round((A/'REPORT.html').stat().st_size/1e6,2));print('SHA',hashes)

if __name__=='__main__':run()
