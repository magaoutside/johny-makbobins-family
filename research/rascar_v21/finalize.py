"""Package preserved experiments, evidence and a portable inference replay."""
from common21 import *
import html
import platform
import sys
import zipfile
import sklearn

base_result = json.loads((ROOT/'rascar_v20/results/release_main.json').read_text())
result = json.loads((R/'release_main.json').read_text())
variants = ['main', 'conservative', 'seat_experiment']
release = {n: json.loads((R/f'release_{n}.json').read_text()) for n in variants}
all_results = [json.loads(p.read_text()) for p in sorted(R.glob('*.json'))]
comparison = pd.json_normalize(all_results).sort_values('pooled_ap', ascending=False)
comparison.to_csv(A/'experiment_comparison.csv', index=False)

# Saved-model inventory: the original 24 model files remain in models/.
models = [{'path':str(p.relative_to(A)), 'bytes':p.stat().st_size,
           'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((A/'models').glob('*.joblib'))]
(A/'model_manifest.json').write_text(json.dumps(models,indent=2))
(A/'runtime.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,
    pandas=pd.__version__,sklearn=sklearn.__version__,platform=platform.platform()),indent=2))

dest=A/'replay_inputs';dest.mkdir(exist_ok=True)
inherited=align(ROOT/'rascar_v18/results/oof/release_main.csv',q,'score')-align(ROOT/'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv',q,'score')
for split,rows,baseline in [('oof',q,q.score.to_numpy()),('test',te,current)]:
    keep=['id','frame_id','class_name']+(['day','error'] if split=='oof' else [])
    out=rows[keep].copy();out['baseline']=baseline
    for component,source in [('cup','cup_instance_quality_w0.5'),('hand','hand_quality_both_w0.75'),('person','person_distribution_w0.25')]:
        f=align(A/f'{source}_{split}.csv',rows)
        out[component+'_gate']=boolean(f.gate)
        out[component+'_target']=f.target.to_numpy()
    for name in variants:
        source=R/'oof'/f'release_{name}.csv' if split=='oof' else A/f'submission_v21_{name}.csv'
        out['expected_'+name]=align(source,rows,'score')
    if split=='oof':out['inherited_oof_only_delta']=inherited
    out.to_csv(dest/f'{split}.csv',index=False)
(dest/'manifest.json').write_text(json.dumps(dict(
    description='Cached quality probabilities and fixed geometric eligibility gates; not raw-image inference or retraining',
    input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.csv')},
    source_baseline_sha256=hashlib.sha256((A/'baseline_lb095417.csv').read_bytes()).hexdigest(),
    note='Expected columns are independent saved outputs used only to check reproduction, not to calculate scores'),indent=2))

# Descriptive paired frame bootstrap. This does not account for repeated
# experiment selection or visual feedback from test images.
rng=np.random.default_rng(210013);main_score=align(R/'oof/release_main.csv',q,'score')
groups=[[g.index.to_numpy() for _,g in q[q.day.eq(day)].groupby('frame_id')] for day in DAYS]
deltas=[]
for _ in range(2000):
    ix=np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g),size=len(g))]) for g in groups])
    y=q.error.to_numpy()[ix]
    if len(np.unique(y))<2:continue
    deltas.append(AP(y,main_score[ix])-AP(y,q.score.to_numpy()[ix]))
bootstrap=dict(samples=len(deltas),unit='frame, stratified by day, paired V20/V21',
    mean_delta=float(np.mean(deltas)),percentile_95=np.quantile(deltas,[.025,.975]).tolist(),
    positive_fraction=float(np.mean(np.array(deltas)>0)),
    caveat='Descriptive only; 35 frames and repeated selection, no model-selection uncertainty or LB guarantee')
(A/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2))

review=pd.read_csv(R/'reviews/release_main.csv')
review[review.delta.abs().gt(1e-9)].to_csv(A/'changed_test_cases.csv',index=False)
reviewed=pd.read_csv(A/'audit/final_review/cases.csv')
reviewed['reviewed_v21']=True
reviewed['review_scope']='context crop with candidate box; subjective visual check, no official test GT'
reviewed['decision']=np.where(reviewed.visual_status.isin(['clear_error','likely_error']),
    np.where(reviewed.delta.gt(.02),'error score increased','remaining error or preserved'),
    np.where(reviewed.visual_status.eq('plausible_correct'),
        np.where(reviewed.delta.lt(-.02),'false alarm reduced','unchanged or below delta threshold'),
        'uncertain; not counted as resolved'))
reviewed.to_csv(A/'audit/visual_review_decisions.csv',index=False)

def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
def number(v):return f'{v:.5f}'
metrics_table=table(['Вариант','OOF AP','25 октября','26 октября','LB','Диагностика'],[
    ['V20',number(base_result['pooled_ap']),number(base_result['day_25']),number(base_result['day_26']),'0.95417 — сообщил пользователь','13/41'],
    *[[f'V21 {name}',number(x['pooled_ap']),number(x['day_25']),number(x['day_26']),'не отправлен',str(x['priority_passes'])+'/41'] for name,x in release.items()]])
class_table=table(['Класс','V20 AP','V21 main AP','Изменение'],[
    [c,number(base_result['ap_'+c]),number(result['ap_'+c]),f"{result['ap_'+c]-base_result['ap_'+c]:+.5f}"] for c in CL])
ablation_table=table(['Без унаследованной OOF-only поправки','OOF AP','25 октября','26 октября'],[
    [name,*[number(d[k]) for k in ['pooled_ap','day_25','day_26']]] for name,d in [('V20',base_result['without_inherited_oof_only_correction']),('V21 main',result['without_inherited_oof_only_correction'])]])
changes=review[review.review_no.isin([1,247,285,190,193,143,139,157,160,161,288])].sort_values('review_no')
case_table=table(['№ аудита','Класс','Визуальная оценка','V20','V21','Примечание'],[
    [r.review_no,r.class_name,r.visual_status,number(r.baseline_v20),number(r.new_score),r.issue] for r in changes.itertuples()])
body=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>RASCAR V21 — результаты и следующие шаги</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172435}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}td,th{{border:1px solid #d4dce5;padding:9px;text-align:left}}th{{background:#edf3f8}}img{{max-width:100%;height:auto}}.note{{padding:16px;background:#fff3d9;border-left:4px solid #c98a12}}a{{color:#075ab7}}code{{background:#edf3f8;padding:2px 4px}}</style>
<h1>RASCAR V21: чашки, слитые кисти и проверка посадочных мест</h1>
<p>13 ноября 2026. База — неизмененный V20 с LB <b>0.95417</b>, результат сообщил пользователь.
Новый основной кандидат — <a href="submission_v21_main.csv">submission_v21_main.csv</a>. Он еще не проверен на LB.</p>
<p>Проведено {len(all_results)} сохраненных вариантов ранжирования, обучено {len(models)} моделей качества на замороженных DINOv3-признаках.
В этом цикле YOLO заново не обучалась: использованы сохраненные детекторы и SAHI-предсказания V20.</p>
<h2>Что вошло в основной кандидат</h2>
<ul><li><b>Чашки:</b> уменьшаем ошибку только при уверенном классе, подходящей оценке качества и отсутствии признаков нескольких чашек или неоднозначного дубля. Чистый DINO-классификатор ошибочно спасал рамки с двумя чашками; проверка отдельных объектов устранила эти регрессии на известном аудите.</li>
<li><b>Кисти:</b> повышаем ошибку только при согласии двух новых моделей качества и независимом геометрическом подтверждении: несколько кистей внутри рамки либо конфликт с локальными детекторами. Обычное отсутствие детекции не является достаточным основанием.</li>
<li>На тесте изменены 23 строки: 18 чашек и 5 кистей. Координаты входных боксов не изменяются: задача сабмита — ранжирование их ошибок. Ранжирование людей и подносов сохранено из V20.</li></ul>
<h2>Метрики и ограничения</h2>{metrics_table}{class_table}
<p class="note"><b>Компромисс:</b> общий AP растет, но AP чашек в main снижается с 0.99557 до 0.99068. Консервативная версия сохраняет AP чашек 0.99557, но слабее исправляет высокий score правильных чашек на тестовом аудите. Рост LB не гарантирован.</p>
<p>Валидация: 198 запросов, 35 кадров; 68 запросов за 25 октября и 130 за 26 октября. Обучение на 25-м дне → оценка на 26-м, и наоборот. Тест — 362 запроса на 86 кадрах 27–28 октября. Машинная разметка и визуальные оценки теста не используются как истинные метки в расчете AP.</p>
<p>Новые классификаторы обучены строго по дням. Калибровка каждого — внутренние 3 фолда с группировкой по кадру только на обучающем дне; синтетические варианты одного кадра не разделяются. Test-предсказание усредняет два дневных эксперта. Настройки ранжирования выбирались многократно по этой же небольшой валидации и известным тестовым случаям, поэтому это метрика разработки, а не независимая финальная оценка.</p>
<p>Для сопоставимости отдельно показана абляция унаследованного из V16 правила, сработавшего только на OOF. На тестовый CSV его удаление не влияет:</p>{ablation_table}
<p>В основном кандидате изменились 11 OOF-запросов: 9 чашек и 2 кисти. Обе измененные кисти — официальные ошибки, по одной на каждом дне. Этого мало для уверенного вывода о всех типах рук. Парный bootstrap по кадрам, с сохранением дней: средняя ΔAP {bootstrap['mean_delta']:+.5f}, описательный 95% интервал [{bootstrap['percentile_95'][0]:+.5f}; {bootstrap['percentile_95'][1]:+.5f}]. Он не учитывает выбор лучшего из многих экспериментов.</p>
<h2>Повторный визуальный аудит</h2>
<p>Просмотрены 69 тестовых случаев: все измененные боксы, замороженный список 41 проблемного случая и контрольные примеры. По старому фиксированному диагностическому порогу 0.688993678 проходят <b>17/41 (41.5%)</b> против 13/41 (31.7%) у V20. Четыре новых прохождения: чашки №1, №247, №285 и слитые кисти №193. Это доля прохождения субъективной диагностики, <b>не процент доказанно закрытых ошибок всего теста</b>.</p>
<p>По известным уверенным визуальным оценкам нет изменений в неверную сторону больше 0.02. Неоднозначные кисти №64, №344, №348 остаются неопределенными; их улучшением не объявляем. Предплечье №190 стало оцениваться как ошибка выше, но его нет в первоначальном списке 41, поэтому в числитель оно не добавлено.</p>
{case_table}<p><a href="changed_test_cases.csv">Все измененные строки</a> · <a href="audit/visual_review_decisions.csv">Решения визуального аудита</a></p>
<figure><img src="audit/hand_penalty_cases.jpg"><figcaption>Пять измененных кистей. Не все визуально однозначны; официальной тестовой разметки нет.</figcaption></figure>
<figure><img src="audit/cup_cases.jpg"><figcaption>Проверка чашек: снижение ложных тревог и сохранение штрафа за несколько объектов.</figcaption></figure>
<h2>Почему новая поправка коротких людей отдельно</h2>
<p>Проверена оценка по распределению эталонных рамок людей на дальнем месте: вероятность, что IoU с похожей посадкой ниже 0.6. На тесте она сдвигает шесть подозрительно коротких рамок в нужную по визуальному аудиту сторону. Однако все четыре затронутых OOF-человека официально правильные: IoU 0.768, 0.821, 0.763, 0.821. Разная поза и видимая часть тела делают средний шаблон ненадежным.</p>
<p>В результате AP людей снижается до 0.95199. В main эта поправка не включена. <a href="submission_v21_seat_experiment.csv">Seat experiment</a> сохранен отдельно; это рискованный изолированный вариант, не основной сабмит.</p>
<figure><img src="audit/person_template_validation_failures.jpg"><figcaption>Желтым — запрос, голубым — ручная разметка. Все четыре случая имеют IoU выше 0.6 и не должны получать штраф только за отличие от шаблона посадки.</figcaption></figure>
<h2>Оставшиеся проблемы и следующие действия</h2>
<ol><li><b>Мелкие кисти у окна: №98, 116, 151, 200, 248, 266, 333, 354.</b> Добрать из машинного пула похожие по масштабу, освещению и позиции кадры. Отбирать разнообразные кадры, а не соседние почти дубли; вручную проверить полные кропы и разметить все кисти, включая соседние и частично закрытые. Примеры низкой уверенности нельзя автоматически превращать в фон. Дообучить текущий hand-detector на этих данных с сохранением старых качественных примеров; сравнить SAHI при неизменном режиме валидации.</li>
<li><b>Предплечья, ткань и слитые объекты.</b> Добавить проверенные трудные отрицательные примеры: рукава, предплечья, салфетки; отдельно синтезировать рамки, объединяющие две разные кисти или чашки. В текущем DINO-наборе всего один синтетический merged-пример чашки — это конкретный пробел. Сам факт присутствия чашки внутри бокса не означает правильную рамку.</li>
<li><b>Короткие люди.</b> Сохранить режимы «дальнее место / ближнее место / стоит», но использовать положение головы, плеч и видимые границы тела, перекрытие столом и неопределенность, а не один средний прямоугольник. Обучить оценку вероятности IoU &lt; 0.6 на качественных рамочных парах, с повышенным весом корректных пограничных примеров IoU 0.6–0.85. Если нижняя граница не наблюдается и эксперты расходятся — не применять новую поправку.</li>
<li><b>Чашка №143 и поднос №6.</b> Отдельная проверка границ и маски видимой части объекта; обучение на сдвиге границ и неполном покрытии. На подносах текущий AP=1 на малом числе запросов не доказывает отсутствие ошибок на тесте.</li>
<li><b>Устойчивость валидации.</b> Для расширенной ручной проверки держать отдельный набор по времени и месту, исключать соседние кадры между обучением и проверкой. Псевдоразметку использовать в обучении с меньшим весом; не считать ее независимой истинной валидацией. Процент закрытых ошибок измерять только на фиксированном проверенном списке, сохраняя новые регрессии.</li></ol>
<p>Повороты 0/90/180/270 и SAHI уже использовались раньше: повторять их как новый метод не предлагается. Следующий полезный шаг — качество и разнообразие конкретных трудных примеров.</p>
<h2>Воспроизведение и сохраненные результаты</h2>
<p><a href="experiment_comparison.csv">Все {len(all_results)} результатов</a> · <a href="replay.ipynb">Ноутбук воспроизведения</a> · <a href="replay.py">Replay script</a> · <a href="model_manifest.json">Манифест 24 моделей</a></p>
<p><code>python3 replay.py</code> использует только сохраненные вероятности и геометрические условия из <code>replay_inputs</code>, воспроизводит три CSV и проверяет их против сохраненных выходов. Это воспроизведение ранжирования по кешу, не повторное обучение или инференс по новым фотографиям.</p>
<p>Полные обученные модели сохранены локально в <code>rascar_v21/models</code>. Для переобучения <code>kernel_quality.py</code> нужны прежние <code>rascar_v16/dino_features.npy</code>, <code>dino_records.csv</code>; для пересчета геометрии — сохраненные источники V15/V16/V18/V20, перечисленные в исходных скриптах. Компактный ZIP содержит код, исходные выходы моделей, метрики и аудит, но не тяжелые модели и датасеты. Ничего из предыдущих версий не удалено.</p>
<p>Методы: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">SVC</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html">KernelRidge</a>. Версии библиотек и контрольные суммы сохранены отдельно.</p>
</html>'''
(A/'report.html').write_text(body,encoding='utf-8')

notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
    {'cell_type':'markdown','metadata':{},'source':['# RASCAR V21 — cached inference replay\n','Unpack the compact bundle and run this notebook in its folder. Requires numpy and pandas. No GPU or network. This replays saved model outputs and geometric gates; it does not retrain models.\n','V20 reported LB: 0.95417. V21 LB: not yet known.']},
    {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':['from pathlib import Path\nimport subprocess, sys\n','root = Path.cwd()\nassert (root / "replay.py").exists()\n','subprocess.run([sys.executable, str(root / "replay.py")], check=True)\n']},
    {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':['import pandas as pd\n','pd.read_csv(root / "experiment_comparison.csv").query("experiment.str.startswith(\'release_\')", engine="python")[["experiment", "pooled_ap", "day_25", "day_26", "priority_passes"]]\n']}
]}
for i,c in enumerate(notebook['cells']):c['id']=f'v21-replay-{i}'
(A/'replay.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2))

assert hashlib.sha256((ROOT/'rascar_v20/submission_v20_main.csv').read_bytes()).hexdigest()==base_result['sha256']
for name in variants:
    frame=pd.read_csv(A/f'submission_v21_{name}.csv')
    assert frame.columns.tolist()==['id','score'] and frame.id.tolist()==te.id.tolist() and frame.id.is_unique
    assert np.isfinite(frame.score).all() and frame.score.between(0,1).all()
    assert hashlib.sha256((A/f'submission_v21_{name}.csv').read_bytes()).hexdigest()==release[name]['sha256']

print(json.dumps(dict(experiments=len(all_results),models=len(models),reviewed_cases=len(reviewed),bootstrap=bootstrap),indent=2))
