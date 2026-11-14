from common18 import *
import shutil,html,platform,importlib.metadata

main='release_main';alternate='release_conservative'
metrics=pd.DataFrame([json.loads(p.read_text()) for p in sorted(R.glob('*.json'))])
metrics.drop(columns=['details']).to_csv(A/'all_experiment_metrics.csv',index=False)
for name,dest in [(main,'submission_v18_main.csv'),(alternate,'submission_v18_conservative.csv')]:
    src=R/'submissions'/f'submission_{name}.csv';s=pd.read_csv(src)
    assert s.columns.tolist()==['id','score'] and s.id.tolist()==te.id.tolist() and s.id.is_unique
    assert len(s)==362 and s.score.between(0,1).all() and np.isfinite(s.score).all()
    shutil.copyfile(src,A/dest)
d=pd.read_csv(R/'reviews'/f'{main}.csv');bad=d.visual_status.isin(['clear_error','likely_error']);good=d.visual_status.eq('plausible_correct')
prior=d.diagnostic_flag.isin(['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'])
signed17=np.where(bad,d.delta,np.where(good,-d.delta,0));signed15=np.where(bad,d.delta_vs_v15,np.where(good,-d.delta_vs_v15,0))
sets={'Все приоритетные':prior,'Короткие рамки людей':d.issue.eq('person_torso_only'),'Приоритетные правильные кисти':prior&good&d.class_name.eq('nude_hands'),'Приоритетные явные ошибки людей':d.class_name.eq('person')&d.diagnostic_flag.eq('clear_error_low_score')}
counts=pd.DataFrame([dict(category=n,total=int(m.sum()),improved_vs_v15=int((m&(signed15>.02)).sum()),improved_vs_v17=int((m&(signed17>.02)).sum()),diagnostic_passes=int((m&d.diagnostic_pass).sum())) for n,m in sets.items()])
counts.to_csv(A/'audit_closure_counts.csv',index=False);d.to_csv(A/'audit_all_362_boxes.csv',index=False)

# The stronger hand variant was visually inspected first; its changed set covers main.
reviewed=pd.read_csv(R/'reviews/combined_seat0.55_geometry0.5.csv')
controls=[66,91,92,120,214,234,272,85,44,55,186,183,190,194,195,193,192,110,111,145,146,147,259,320,325,298,253,169,25,308]
viewmask=prior|(reviewed.delta.abs()>1e-9)|d.review_no.isin(controls)
assert not ((d.delta.abs()>1e-9)&~viewmask).any()
ledger=d[viewmask][['id','review_no','class_name','issue','visual_status','score','new_score','delta']].copy()
ledger['reviewed']=True;ledger['review_method']='contact sheets with query and detector references; structural/remaining controls separately'
ledger['assessment']='Visual status retained; no newly observed confident regression'
notes={190:'Unresolved forearm: pose, YOLO and palm classifier still confuse it with a hand',193:'Two hands in one query; higher error score, still below fixed threshold',320:'Uncertain overlapping boundaries; increased penalty may be a risk',138:'Uncertain overlapping hands; rescue is a risk, status not changed',220:'Motion blur; exact boundary remains uncertain',66:'Front-seat child; new body guard preserves original score',234:'Image-edge truncation; preserved',272:'Front-seat body; preserved',308:'Short person remains unchanged: body extent evidence is insufficient',128:'Oblique short person remains unchanged',96:'Standing bent person undercoverage unresolved',253:'Coat on chair; absence evidence raises score',242:'Coat on chair; score now passes diagnostic threshold',208:'Coat on chair; score now passes diagnostic threshold',296:'Blurred hand: distal wrist direction and two YOLO models support rescue'}
for n,note in notes.items():ledger.loc[ledger.review_no.eq(n),'assessment']=note
ledger.to_csv(A/'visual_review_ledger.csv',index=False)

names=['baseline_v17_lb095180',main,alternate,'release_main_no_inherited_contradiction','release_conservative_no_inherited_contradiction']
table=metrics.set_index('experiment').loc[names,['pooled_ap','day_25','day_26','ap_person','ap_nude_hands','ap_cup','ap_tray']].reset_index()
table.to_csv(A/'release_metric_comparison.csv',index=False)
pred=align(R/'oof/release_main.csv',q,'score');rng=np.random.default_rng(181);boot=[]
for _ in range(1500):
    ix=[]
    for day,g in q.groupby('day'):
        sample=rng.choice(g.frame_id.unique(),g.frame_id.nunique(),replace=True)
        ix.extend(j for fid in sample for j in g.index[g.frame_id.eq(fid)])
    ix=np.array(ix);boot.append(AP(q.error.iloc[ix],pred[ix])-AP(q.error.iloc[ix],q.score.iloc[ix]))
ci=np.quantile(boot,[.025,.5,.975]).tolist()
(A/'paired_frame_bootstrap.json').write_text(json.dumps(dict(iterations=1500,seed=181,unit='frame stratified by day',baseline='V17 LB .95180',delta_ap_quantiles=ci,warning='reused development set; selection bias not included'),indent=2))
manifest=dict(version='V18',date='2026-11-13',baseline=json.loads((A/'baseline_provenance.json').read_text()),main=json.loads((R/'release_main.json').read_text()),conservative=json.loads((R/'release_conservative.json').read_text()),new_public_lb=None,visually_reviewed_rows=len(ledger),all_changed_rows_reviewed=True,priorities_reviewed=int((viewmask&prior).sum()),audit_protocol=json.loads((A/'audit_protocol.json').read_text()))
(A/'release_manifest.json').write_text(json.dumps(manifest,indent=2))

summary=f'''RASCAR V18 — продолжение после LB 0.95180
Основной файл: submission_v18_main.csv. На LB не отправлен; новый LB неизвестен.
База — сохранённый V17 main, к которому отнесён сообщённый пользователем результат 0.95180. Предыдущие версии сохранены.
OOF основного: 0.98289555 против 0.98181574. День 25: 0.97604117 против 0.96740451; день 26: 0.98817684 против 0.98858451.
Консервативный файл: submission_v18_conservative.csv, OOF 0.984246; затрагивает меньше известных проблем. Основной выбран по сочетанию междневной проверки и визуального аудита, не только по максимуму OOF.

Изменения
1. Ridge предсказывает границы полного тела относительно головы по координатам головы и ключевым точкам позы. Обучение на ручных person GT только своего дня; проверка на другом. Поправка применяется к задним местам кафе, одной голове, >=3 шаблонам, вдали от границ изображения. Со старым сигналом мест берётся максимум, чтобы не суммировать одну причину дважды. Вес прежнего распределения мест увеличен с 0.4 до 0.55.
2. CatBoost оценивает ошибку границ кисти по расхождению с фиксированными WeVis-референсами и распределению отклонений от ручной разметки. Для признаков обучающего примера исключается его собственный кадр. На тесте усредняются две дневные модели. Сила поправки 0.25; требуются семантическое подтверждение, один центр кисти, назначенный референс и отсутствие запрета для проигравшего дубля.
3. Для смазанных кистей добавлено подтверждение по направлению локоть -> запястье: центр бокса должен лежать дальше запястья, согласованно в >=3 видах, и оба YOLO должны совпадать с боксом (IoU >=0.6). Так исправляется #296: 0.87675 -> 0.62123. Обратное направление даёт осторожный штраф предплечьям.
4. Ложный person получает дополнительный штраф только при отсутствии головы и позы и p(person)<0.5 на tight/context DINO. #242: 0.68352 -> 0.84176; #208: 0.67925 -> 0.83963. Частично перекрытые настоящие люди с семантическим подтверждением не проходят этот фильтр.
5. Усилен существующий групповой штраф: оба YOLO находят >=2 кистей, а новый YOLO и WeVis не подтверждают границы общего бокса. Вес заменён с 0.25 на 0.5. #193: 0.42807 -> 0.61871; прежние проигравшие дубли не спасаются.
6. В новой ветке явно разбираются CSV-флаги True/False/0: строка False не должна превращаться в True. Проверка не изменила выбранный итоговый CSV, но убрала риск ошибки в воспроизведении.
Координаты исходных боксов в сабмите не меняются: формат задачи id,score, высокий score означает ошибку. В этой итерации обучены новые небольшие модели поверх сохранённых признаков; повторного GPU-обучения YOLO и добавления машинных фотографий не было.

Валидация и ограничения
198 исходных запросов, 35 кадров; обучение 25 -> проверка 26 и наоборот. AP считается на объединённых OOF, а также по дням/классам. Метки не удалялись и не исправлялись. Новые модели не обучаются на визуальных статусах теста; конкретные test id не участвуют в правилах.
В OOF остаётся унаследованное V16-правило, влияющее на один OOF-бокс и ноль тестовых. При вычитании его поправки основной имеет OOF 0.980009 (25: 0.964940; 26: 0.988177), базовый V17 — 0.978837. CSV теста идентичен при этой абляции. Это не новый независимый holdout.
Один небольшой development-набор использовался для многих экспериментов. Междневное разделение защищает от прямого обучения на другом дне, но не устраняет переобучение при выборе вариантов. 95% парный bootstrap по кадрам для разницы с V17: [{ci[0]:.6f}, {ci[2]:.6f}]; он не учитывает выбор эксперимента. Визуальный аудит участвует в разработке и не является независимой оценкой скрытого LB.

Аудит и процент закрытия
Проверено визуально {len(ledger)} бокса: все 82 изменённых, все 41 приоритетный, контрольные случаи правильных людей, ложных кистей и дублей. Галерея сохранена для всех 362 боксов. По прежним уверенным/вероятным визуальным статусам новых ухудшений >0.02 нет; для неопределённых случаев это утверждение не действует.
20/41 = 48.8% приоритетных случаев дополнительно улучшены относительно V17 (изменение >0.02 в ожидаемую сторону). Относительно V15 улучшены 26/41 = 63.4%. Через фиксированный диагностический порог {threshold:.6f} проходят 9/41 = 22.0%, раньше 5/41 = 12.2%. Это доля прохождения нашей субъективной диагностики, а не истинный процент закрытых ошибок на скрытом тесте. Все недостатки не закрыты.
{counts.to_string(index=False)}

Осталось
Предплечье #190 (0.531) ошибочно подтверждается несколькими моделями; новый palm-classifier его не разделил. Короткие #128/#308 и стоящий #96 не получили достаточного геометрического подтверждения. Мелкие/перекрытые кисти #98/#151/#200/#204/#266/#333/#354 всё ещё имеют высокие оценки ошибки; #116/#248 улучшены лишь частично. У общего бокса кистей #193 оценка повысилась, но порог ещё не пройден. Неопределённые #138/#220/#320 и границы одежды/окклюзий остаются риском; статусы не переписаны ради результата. В этой итерации cup/tray не изменены.

Неудачные варианты сохранены
Общий регрессор тела без ограничения места портил правильные #66/#234/#272. LogisticRegression и CatBoost геометрии кисти отдельно слегка снижали OOF, поэтому в основном используется мягкий вес 0.25. Классификатор полноты ладони на DINO не отличил #190 и при широком спасении ухудшал OOF до ~0.9777. Сильные поправки мест/кистей лежат среди экспериментов, но основной их не использует. Всего сохранено {len(metrics)} строк результатов, включая базу и абляции; это не {len(metrics)} независимых обучений.

Воспроизведение и файлы
python3 replay.py — автономно собирает оба CSV из replay_inputs, проверяет все 198 OOF/362 test строки против сохранённых результатов (допуск 1e-12).
python3 retrain_saved_features.py — заново обучает две выпущенные добавки (CatBoost кистей и Ridge тела) на сохранённых признаках и сверяет с моделями; проверяет разделение кадров по дням.
reproduce_v18.ipynb — запуск этих проверок. Новые joblib-модели, признаки, журналы и результаты сохранены. Полное извлечение признаков из фотографий использует прежние rascar_v16/rascar_v17, остающиеся в проекте; архив V18 является дополнением к ним, не новой автономной GPU-обучалкой всего решения.
all_experiment_metrics.csv — все сравнения; audit_all_362_boxes.csv — аудит; visual_review_ledger.csv — повторно просмотренные случаи; galleries/release_main/index.html — интерактивные фильтры галереи.
'''
(A/'SUMMARY.txt').write_text(summary)
report='<!doctype html><html lang="ru"><meta charset="utf-8"><title>RASCAR V18 — LB baseline 0.95180</title><style>body{max-width:1150px;margin:32px auto;padding:0 18px;font:16px/1.5 system-ui;color:#172536}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd3dd;padding:7px;text-align:left}pre{white-space:pre-wrap}a{color:#2363ad}</style><h1>RASCAR V18</h1><p><a href="submission_v18_main.csv">Основной CSV</a> · <a href="submission_v18_conservative.csv">Консервативный CSV</a> · <a href="galleries/release_main/index.html">Все 362 бокса</a> · <a href="all_experiment_metrics.csv">Все эксперименты</a></p>'+table.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'<h2>Фиксированный аудит</h2>'+counts.to_html(index=False)+'<h2>Изменения и ограничения</h2><pre>'+html.escape(summary)+'</pre></html>'
(A/'REPORT.html').write_text(report)
versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','catboost','Pillow','joblib']};versions['python']=platform.python_version();(A/'local_runtime_versions.json').write_text(json.dumps(versions,indent=2))
cells=[dict(cell_type='markdown',metadata={},source=['# RASCAR V18\nCPU replay of the saved cross-day evidence. Run from the extracted rascar_v18 directory. Full original feature extraction requires preserved V16/V17 inputs.']),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=['%run replay.py']),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=['%run retrain_saved_features.py'])]
(A/'reproduce_v18.ipynb').write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5),indent=2))
print(table.to_string(index=False));print(counts.to_string(index=False));print('REVIEWED',len(ledger),'CI',ci)
