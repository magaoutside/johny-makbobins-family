from common17 import *
import shutil,html,platform,importlib.metadata

main=sys.argv[1];alternate=sys.argv[2] if len(sys.argv)>2 else None
selected=[main]+([alternate] if alternate else [])
metrics=pd.DataFrame([json.loads(p.read_text()) for p in sorted(R.glob('*.json'))])
metrics.drop(columns=['details']).to_csv(A/'all_experiment_metrics.csv',index=False)
manifest={'main':main,'alternate':alternate,'known_best_public_lb':.92822,'known_best_lb_version':'V15 (user reported)',
    'new_public_lb':None,'validation':'train25 -> predict26; train26 -> predict25; concatenate OOF; official supplied-query error labels unchanged',
    'release_experiments':{},'baseline_sha256':hashlib.sha256((V16/'baseline_submission_0.92822.csv').read_bytes()).hexdigest(),
    'audit_protocol':json.loads((A/'audit_protocol.json').read_text())}
for i,name in enumerate(selected):
    src=R/'submissions'/f'submission_{name}.csv';df=pd.read_csv(src)
    assert df.columns.tolist()==['id','score'] and len(df)==362 and df.id.is_unique and df.id.tolist()==te.id.tolist()
    assert np.isfinite(df.score).all() and df.score.between(0,1).all()
    dest=A/('submission_v17_main.csv' if i==0 else 'submission_v17_alternative.csv');shutil.copyfile(src,dest)
    manifest['release_experiments'][name]=json.loads((R/f'{name}.json').read_text())
assert manifest['baseline_sha256']=='7aa06bec3809dedd2e60969c5c38672d5dbd82fe8ee9479960583dd6956dba54'
(A/'release_manifest.json').write_text(json.dumps(manifest,indent=2))

review=pd.read_csv(R/'reviews'/f'{main}.csv');threshold=manifest['audit_protocol']['fixed_threshold']
prior=review.diagnostic_flag.isin(['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'])
bad=review.visual_status.isin(['clear_error','likely_error']);good=review.visual_status.eq('plausible_correct')
sets={'Все приоритетные':prior,'Короткие рамки людей (весь аудит)':review.issue.eq('person_torso_only'),
    'Правильные кисти с высокой оценкой ошибки':good&review.class_name.eq('nude_hands')&prior,
    'Явные ошибки людей с низкой оценкой':review.class_name.eq('person')&review.diagnostic_flag.eq('clear_error_low_score')}
stats=[]
for label,m in sets.items():
    signed=np.where(bad,review.new_score-review.v15_score,np.where(good,review.v15_score-review.new_score,0))
    signed16=np.where(bad,review.delta,np.where(good,-review.delta,0))
    passed=(bad&(review.new_score>=threshold))|(good&(review.new_score<threshold))
    stats.append(dict(category=label,total=int(m.sum()),improved_vs_v15=int((m&(signed>.02)).sum()),improved_vs_v16=int((m&(signed16>.02)).sum()),diagnostic_passes=int((m&passed).sum())))
stats=pd.DataFrame(stats);stats.to_csv(A/'audit_closure_counts.csv',index=False)
review.to_csv(A/'audit_all_362_boxes.csv',index=False)

# Paired, day-stratified frame bootstrap: uncertainty description, no tuning.
pred=align(R/'oof'/f'{main}.csv',q,'score');rng=np.random.default_rng(171);boot=[]
for _ in range(1500):
    ix=[]
    for day,g in q.groupby('day'):
        frames=g.frame_id.unique();sample=rng.choice(frames,len(frames),replace=True)
        ix.extend(j for fid in sample for j in g.index[g.frame_id.eq(fid)])
    ix=np.array(ix);boot.append(AP(q.error.iloc[ix],pred[ix])-AP(q.error.iloc[ix],baseline15[ix]))
ci=np.quantile(boot,[.025,.5,.975]).tolist();(A/'paired_frame_bootstrap.json').write_text(json.dumps(dict(iterations=1500,seed=171,unit='frame, stratified by held day',baseline='V15',delta_ap_quantiles_025_50_975=ci,warning='development set reused for experiment selection; not a fresh holdout'),indent=2))
rows=[]
for name in ['baseline_v15','baseline_v16']+selected+['release_balanced_without_v16_contradiction']:
    m=json.loads((R/f'{name}.json').read_text());rows.append({k:m[k] for k in ['experiment','pooled_ap','day_25','day_26','ap_person','ap_nude_hands','ap_cup','ap_tray']})
table=pd.DataFrame(rows);table.to_csv(A/'release_metric_comparison.csv',index=False)
local=pd.read_csv(A/'tta_held_manual_diagnostic.csv');localization=local.groupby(['checkpoint','day']).agg(n=('best_iou','size'),mean_best_iou=('best_iou','mean'),fraction_iou06=('found_iou06','mean')).reset_index();localization.to_csv(A/'held_manual_localization_summary.csv',index=False)
main_metrics=manifest['release_experiments'][main]
text=f'''RASCAR V17 — результаты
Основной сабмит: submission_v17_main.csv ({main})
Альтернатива: {alternate}
OOF: {main_metrics['pooled_ap']:.8f}; 25 октября: {main_metrics['day_25']:.8f}; 26 октября: {main_metrics['day_26']:.8f}.
Лучший подтверждённый пользователем LB: V15 0.92822. Новый сабмит на LB не отправлялся.

Что изменилось
1. Шесть столов и 12 кандидатов мест; распределение ручных person-рамок привязано к голове. Карта и шаблоны берутся только с обучающего дня. Признаки обучающих примеров исключают собственный кадр. Поправка ограничена задней скамьёй, одной найденной головой, >=3 шаблонами и допустимым отношением сторон. Все места не покрыты одинаково: часть мест отсутствует в одном из дней.
2. YOLO26s для кистей дополнительно обучен 20 эпох на полных ручных боксах. Обучены 25, 26 и оба дня. Отсечены кропы с обрезанной кистью; добавлены настоящие фоновые кропы. Машинные боксы не использовались как новые истинные метки.
3. Старые и новые веса проверены одинаково: два размера кропа × четыре поворота. Предсказания объединяются по объекту; для изменения оценки нужны несколько согласованных просмотров. Сопоставление с запросами однозначное, ранее выявленные проигравшие дубли не спасаются.
4. Добавлен осторожный штраф объединённым кистям: оба YOLO видят несколько рук внутри запроса, а новый YOLO и WeVis расходятся с границами запроса. На OOF он меняет два уже ошибочных бокса, AP не меняется; на тесте улучшает проблемный #193, но недостаточно для прохождения порога.
5. Исходные рамки в submission не меняются: задача требует id,score, где высокий score означает ошибку рамки.

Валидация
198 запросов, {q.frame_id.nunique()} кадров, 109 ошибочных и 89 правильных боксов. Обучение 25 -> проверка 26; обучение 26 -> проверка 25. AP на объединённых OOF, отдельно по дням и классам. Метки не удалялись и не исправлялись под визуальные ожидания. Данные тестового аудита не используются как метки обучения.
Основной OOF включает унаследованное от V16 правило противоречия локализаторов. Оно меняет один OOF-бокс и ни одного тестового. Без этого правила тот же CSV имеет OOF 0.978837 (день 25: 0.959137; день 26: 0.988585). Этот отдельный результат сохранён для прозрачности.
Это прежний небольшой development-набор, на котором сравнивалось много вариантов; 1:1 соответствие скрытому LB не доказано. Bootstrap разницы с V15 по кадрам: 95% интервал [{ci[0]:.5f}, {ci[2]:.5f}]; он не учитывает смещение от выбора эксперимента.
В логе YOLO есть проверка на обучающих кропах: её mAP не считается OOF и не приводится как качество обобщения.

Процент закрытия
Фиксированный список: 41 бокс. Улучшение — изменение >0.02 в ожидаемую сторону относительно V15. Диагностическое прохождение — ошибочный бокс >= {threshold:.6f}, визуально правильный ниже порога; порог заранее выбран по V15 OOF. Это субъективная проверка, а не известная истинная доля исправленных ошибок теста. Все ошибки закрытыми считать нельзя. Основной вариант: 24/41 = 58.5% заметно улучшены относительно V15; 17/41 = 41.5% дополнительно улучшены относительно V16; 5/41 = 12.2% проходят строгий диагностический порог. Короткие люди: улучшены 13/15, порог проходит 1/15. Приоритетные кисти: улучшены 6/13 относительно V15 (3 новых относительно V16), порог проходят 3/13. Альтернатива с более сильной поправкой мест: 6/41 = 14.6% проходят порог, но OOF ниже.
{stats.to_string(index=False)}

Неудачные и спорные эксперименты сохранены
Широкий CatBoost по местам и принудительные полные шаблоны тела дают регрессии правильных людей. DINO-признаки качества + CatBoost/ExtraTrees не дали устойчивого выигрыша. Дообучение кистей помогает одному дню и ухудшает другой; сравнение весов и вариантов TTA сохранено. Высокий train-mAP не был основанием выбора.
Визуально повторно просмотрены все 41 приоритетный случай, все изменённые тестовые боксы, контрольные группы дублей и ложные руки #183/#190/#298. Новых ухудшений >0.02 по фиксированным уверенным/вероятным визуальным статусам относительно V16: 0. Неопределённые боксы #103/#213/#320 остаются неопределёнными. Метки не переписывались после просмотра.
Не решены все короткие рамки, перекрытые/смазанные кисти и неоднозначные группы. Аудит теста используется для разработки, поэтому его метрики нельзя считать независимой валидацией.

Файлы
all_experiment_metrics.csv — все эксперименты, включая неудачные.
audit_all_362_boxes.csv — оценки и статусы каждого бокса.
galleries/{main}/index.html — галерея всех 362 боксов с фильтрами; контактные листы приоритетных и изменённых случаев.
gpu_results/ — сырые предсказания, журналы, ведомости обучения и манифесты.
replay.py — воспроизведение сборки из сохранённых признаков/предсказаний; повторное GPU-обучение описано в reproduce_v17.ipynb.
'''
(A/'SUMMARY.txt').write_text(text)
report='<!doctype html><meta charset="utf-8"><title>RASCAR V17</title><style>body{max-width:1150px;margin:32px auto;font:16px/1.5 system-ui;color:#172536}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cbd2de;padding:7px;text-align:left}pre{white-space:pre-wrap}a{color:#2363ad}</style><h1>RASCAR V17</h1><p><a href="submission_v17_main.csv">Основной CSV</a> · <a href="submission_v17_alternative.csv">Альтернативный CSV</a> · <a href="galleries/'+html.escape(main)+'/index.html">Все 362 бокса</a></p>'+table.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'<h2>Контроль ошибок</h2>'+stats.to_html(index=False)+'<h2>Локализация ручных кистей на другом дне</h2><p>Условная диагностика на кропах вокруг GT; не full-frame detector AP. Эти кропы исключены из признаков запросов.</p>'+localization.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'<h2>Описание и ограничения</h2><pre>'+html.escape(text)+'</pre>'
(A/'REPORT.html').write_text(report)
versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','catboost','Pillow','joblib']};versions['python']=platform.python_version();(A/'local_runtime_versions.json').write_text(json.dumps(versions,indent=2))
print(table.to_string(index=False));print(stats.to_string(index=False));print('BOOTSTRAP',ci)
