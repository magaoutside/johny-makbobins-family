"""Package the selected automatic correction and every flagged test change."""
from pathlib import Path
import json,html,base64,hashlib,zipfile
import pandas as pd
import numpy as np
A=Path(__file__).resolve().parent;O=A/'correction_audit'
sel=json.loads((O/'selection.json').read_text());rob=json.loads((O/'robustness.json').read_text())
q=pd.read_csv(O/'selected_oof.csv');sub=pd.read_csv(O/'submission_v57_auto_correction.csv')
t=pd.read_csv(A/'kaggle_payload/rascar_v56/inputs/t.csv')
assert len(q)==198 and len(sub)==362 and sub.id.is_unique and sub.id.tolist()==t.id.tolist()
assert np.isfinite(sub.score).all() and sub.score.between(0,1).all()
flags=pd.read_csv(O/'all_43_suspected_changes.csv').sort_values('review_no')
assert len(flags)==43 and flags.id.is_unique
seed_rows=[]
for seed in [5601,5602,5603]:
    r=pd.read_csv(O/f'seed_{seed}/validation_all.csv').set_index('name').loc[sel['winner']]
    seed_rows.append(dict(seed=seed,name=sel['winner'],ap=r.ap,day25=r['2026-10-25'],day26=r['2026-10-26'],min_day_delta=r.min_day_delta))
pd.DataFrame(seed_rows).to_csv(O/'seed_confirmation.csv',index=False)
methods=pd.read_csv(O/'validation_all.csv');all_scores=np.load(O/'all_scores.npz')
audit=pd.read_csv(A/'visual_audit/test_all_362.csv').set_index('id').loc[t.id]
sign=audit.flag.map({'up':1,'down':-1}).fillna(0).to_numpy();mask=sign!=0
comparisons=[]
for _,r in methods.iterrows():
    delta=(all_scores[r['name']+'_test']-t.v52.to_numpy())*sign
    comparisons.append(dict(name=r['name'],ap=r.ap,eligible=r.eligible,selected=r['name']==sel['winner'],favorable=int(((delta>=.01)&mask).sum()),unfavorable=int(((delta<=-.01)&mask).sum()),under_0_01=int(((abs(delta)<.01)&mask).sum())))
pd.DataFrame(comparisons).to_csv(O/'method_visual_comparison.csv',index=False)
notes=pd.read_csv(A/'visual_audit/train_all_198.csv')[['id','train_no','note']]
for name in ['train_all_changes','train_original_49_changes']:
    d=pd.read_csv(O/f'{name}.csv').drop(columns=['train_no','note'],errors='ignore').merge(notes,on='id',how='left',validate='one_to_one')
    d.to_csv(O/f'{name}.csv',index=False)
best=sel['selected'];base=sel['base'];v52=rob['visual_counts']['v52']['0.01'];a40=rob['visual_counts']['a40']['0.01']
txt=f'''V57 — автоматическая коррекция скоров, аудит 14.11.2026

Выбранный алгоритм: {sel['winner']}
На тесте ничего не размечалось для обучения корректора. Номера и визуальные
пометки тестовых боксов не используются в формуле и выборе параметров.
Параметры выбраны по OOF AP с требованием отсутствия ухудшения обоих дней;
для SAM2 дополнительно проверены отдельные seed 5601/5602/5603.

Валидация (AP; 198 боксов, 35 кадров; обучение на противоположном дне):
V52              {base['ap']:.9f}
V57              {best['ap']:.9f}
25 октября       {base['2026-10-25']:.9f} -> {best['2026-10-25']:.9f}
26 октября       {base['2026-10-26']:.9f} -> {best['2026-10-26']:.9f}
Неверные пары    {rob['train_pairs']['before']} -> {rob['train_pairs']['after']}
Исправлено пар: {rob['train_pairs']['fixed']}; появилось новых: {rob['train_pairs']['new']}.
Из 49 прежних проблемных train-боксов: {rob['original_49_train']['improved']} улучшились,
{rob['original_49_train']['worsened']} ухудшились, {rob['original_49_train']['unchanged']} без изменения числа неверных пар.
Полностью исчезли инверсии у {rob['original_49_train']['fully_resolved']} из этих 49 боксов.

Все 43 визуально подозрительных тестовых бокса (38 кадров), порог изменения 0.01:
Относительно V52: {v52['favorable']} в ожидаемую сторону ({v52['favorable_frames']} кадров),
{v52['unfavorable']} в противоположную, {v52['small_or_unchanged']} изменились менее чем на 0.01.
Относительно A40: {a40['favorable']} в ожидаемую сторону ({a40['favorable_frames']} кадров),
{a40['unfavorable']} в противоположную, {a40['small_or_unchanged']} изменились менее чем на 0.01.
Это НЕ число подтверждённо исправленных test-ошибок: test GT отсутствует.
Визуальная гипотеза может расходиться с официальной разметкой и matching.
Абсолютный скор — ранжирующий балл, не откалиброванная вероятность ошибки.

Формула базового двухстороннего корректора:
u = s_V52 + clip(0.2*(p_full-s_V52),0,0.1)
          + clip(0.2*(p_crop-s_V52),-0.1,0)
p_full/p_crop: средние предсказания 3 seed TabICL v1 с SAM2-признаками,
полный кадр / кроп; для test также среднее двух моделей по обучающим дням.
При конфликте направлений поправки частично компенсируются.
Вариант dual_plus_tray_m0.2 дополнительно применяет к tray:
sigmoid(logit(u)-0.2). Другие варианты и параметры — selection.json и validation_all.csv.

Промежуточный dual_emhands_k10_cap0.1_tray0.2 использует эту формулу для person/cup/tray,
а для nude_hands заменяет поправку следующим алгоритмом:
1) Берём средние вероятности трёх TabICL v1 anchor seed отдельно для каждого fit-day.
2) Для каждой группы целевой день × класс оцениваем новую долю ошибок EM по одним
   предсказаниям, без целевых меток. Исходная доля — (ошибки+1)/(число+2) fit-day.
3) posterior = sigmoid(logit(p)+logit(new_prior)-logit(source_prior));
   new_prior = (sum(posterior)+10*source_prior)/(N+10); до 100 итераций.
4) Усредняем test-предсказания двух fit-day моделей; score = s_V52 +
   clip(0.2*(posterior-s_V52),-0.1,0.1). OOF использует только противоположный день.
Точно та же адаптация применяется к неразмеченному validation-батчу, не только test.
Оценённые доли сохранены в prior_em_k10_priors.csv. Предположение об изменении
доли ошибок при сходном распределении признаков внутри классов является эвристикой.

ФИНАЛЬНЫЙ dual_emhands_monotone_tray_m0.4:
person/cup — двухсторонняя поправка SAM2; nude_hands — EM, описанный выше;
tray — только sigmoid(logit(s_V52)-0.4), без изменения порядка внутри класса.
Так устранена регрессия tray: его AP внутри класса снова 1.0.

Ограничения проверки:
Это OOF, использованный также для выбора алгоритма: не новый независимый holdout.
Бутстрэп по кадрам: 95% интервал прироста [{rob['bootstrap']['delta_95pct'][0]:.6f}, {rob['bootstrap']['delta_95pct'][1]:.6f}],
доля положительных приростов {rob['bootstrap']['p_delta_positive']:.1%}; он не учитывает подбор вариантов.
Исключение одного кадра: прирост положительный в {rob['leave_one_frame_out']['positive_delta']}/35 случаях.
Три seed проверяют случайность модели, а не независимость данных.
Public V52 = 0.96300; A40 = 0.96285. Для V57 Public неизвестен.
Последняя попытка отправки НЕ использована.

Файлы:
report.html — все 43 бокса с изображениями и старыми/новыми скорами;
all_43_suspected_changes.csv — полная таблица этих 43 боксов;
all_test_changes.csv — все 362 тестовых бокса;
train_original_49_changes.csv — все 49 прежних train-проблем;
train_pair_changes.csv — исправленные, оставшиеся и новые неверные пары;
validation_all.csv — все протестированные варианты;
robustness.json — бутстрэп, устойчивость и точные счётчики;
submission_v57_auto_correction.csv — экспериментальный кандидат.
replay_candidate.py + replay_inputs.npz — точное воспроизведение всех 560 скоров;
проверяются побитовое совпадение массивов и SHA256 submission.
SHA256 candidate: {rob['submission_sha256']}
'''
(O/'README_RU.txt').write_text(txt)
e=html.escape
cards=[]
for _,r in flags.iterrows():
    image=A/'visual_audit/crops'/f'{int(r.review_no):03d}.jpg'
    b64=base64.b64encode(image.read_bytes()).decode()
    outcome={'favorable':'В ожидаемую сторону','unfavorable':'Против ожидания','under_0.01':'Изменение < 0.01'}[r.outcome_vs_v52]
    cards.append(f'''<article class="{e(r.outcome_vs_v52)}"><img loading="lazy" src="data:image/jpeg;base64,{b64}" alt="Бокс {int(r.review_no)}"><div><h2>#{int(r.review_no)} · {e(r.class_name)}</h2><p>{e(r.id)}<br>{e(r.frame_id)}</p><p>Визуальная гипотеза: <b>{'снизить' if r.flag=='down' else 'повысить'}</b></p><table><tr><th>V52</th><th>A40</th><th>V57</th><th>Δ V52</th></tr><tr><td>{r.score_v52:.5f}</td><td>{r.score_a40:.5f}</td><td>{r.score_v57:.5f}</td><td>{r.delta_vs_v52:+.5f}</td></tr></table><p><b>{outcome}</b></p><p>{e(str(r.note))}</p></div></article>''')
doc=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V57: коррекция всех 43 подозрительных боксов</title><style>body{{font:16px system-ui;margin:0;background:#10151f;color:#e9eef7}}main{{max-width:1100px;margin:auto;padding:28px}}h1{{font-size:30px}}p{{line-height:1.5}}article{{display:grid;grid-template-columns:minmax(240px,40%) 1fr;background:#1b2433;padding:20px;margin:18px 0;border-left:5px solid #718096;border-radius:8px;gap:24px}}article img{{width:100%;max-height:450px;object-fit:contain}}.favorable{{border-color:#42bd86}}.unfavorable{{border-color:#ee827c}}h2{{margin-top:0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #445066}}pre{{white-space:pre-wrap;background:#1b2433;padding:20px;font:14px/1.5 ui-monospace}}@media(max-width:650px){{article{{grid-template-columns:1fr}}}}</style><main><h1>V57: автоматическая коррекция</h1><p><b>OOF AP: {base['ap']:.5f} → {best['ap']:.5f}.</b> Все 43 подозрительных бокса показаны ниже.</p><p>Относительно V52: {v52['favorable']} сдвинулись в ожидаемую сторону, {v52['unfavorable']} — в противоположную, {v52['small_or_unchanged']} — менее чем на 0,01. Цвет обозначает совпадение с визуальной гипотезой, не подтверждённую правильность.</p><details><summary>Метод, валидация и ограничения</summary><pre>{e(txt)}</pre></details>{''.join(cards)}</main></html>'''
(O/'report.html').write_text(doc)
integrity={'rows':len(sub),'unique_ids':int(sub.id.nunique()),'ids_equal_test':True,'scores_finite_in_unit_interval':True,'flagged_cards':len(cards),'submission_sha256':rob['submission_sha256'],'test_flags_used_in_fit_or_selection':False,'local_neural_training':False,'submitted':False}
(O/'integrity.json').write_text(json.dumps(integrity,indent=2))
archive=A/'v57_correction_audit.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(O.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(A))
    for p in [A/'correct_outliers.py',A/'validate_correction.py',A/'package_correction.py',A/'gpu_predictions.zip',A/'gpu_all_results.zip',A/'gpu_trained_models.zip',A/'rascar_v56_gpu_replay.ipynb',A/'audit/kaggle_gpu_completed.jpg']:
        z.write(p,p.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print('PACKAGE',archive.stat().st_size,integrity)
