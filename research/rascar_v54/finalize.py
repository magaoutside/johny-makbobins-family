"""Freeze the augmentation experiment, controls, uncertainty, and candidate replay."""
from common54 import *
from rank_blend import reorder
from sklearn.metrics import average_precision_score as AP
import shutil, zipfile, importlib.metadata

SELECTED='rank_visual_d4_all_w100_actual_confirm_nude_hands_a05'


def predictions(model,q,t):
    p=pd.read_csv(A/f'features/{model}_predictions.csv')
    oo=p[p.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy()
    tt=p[p.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
    return oo,tt


def bootstrap(q,base,candidate,rng,n=3000):
    frames=list(q.frame_id.unique());lookup={f:i for i,f in enumerate(frames)}
    frame_index=q.frame_id.map(lookup).to_numpy();samples=[]
    day_frames=[np.array([lookup[f] for f in q.loc[q.day.eq(day),'frame_id'].unique()]) for day in DAYS]
    for _ in range(n):
        drawn=np.concatenate([rng.choice(v,len(v),replace=True) for v in day_frames])
        weights=np.bincount(drawn,minlength=len(frames))[frame_index]
        samples.append(AP(q.error,candidate,sample_weight=weights)-AP(q.error,base,sample_weight=weights))
    return dict(resamples=n,delta=float(AP(q.error,candidate)-AP(q.error,base)),
                frame_bootstrap_ci95=np.quantile(samples,[.025,.975]).tolist(),fraction_positive=float(np.mean(np.array(samples)>0)),
                selection_adjusted=False)


def main():
    q,t,h=inputs();bo,bt=incumbent(q,t);rng=np.random.default_rng(5403)
    candidate=pd.read_csv(A/f'results/oof/{SELECTED}.csv').set_index('id').loc[q.id,'score'].to_numpy()
    test=pd.read_csv(A/f'results/submissions/{SELECTED}.csv').set_index('id').loc[t.id,'score'].to_numpy()
    noaug,_=predictions('visual_d4_all_w000_actual_confirm',q,t)
    aug,_=predictions('visual_d4_all_w100_actual_confirm',q,t)
    repeated,_=predictions('visual_d4_all_w100_repeat_parent_controls',q,t)
    uncertainty=dict(candidate_vs_v52=bootstrap(q,bo,candidate,rng),aux_aug_vs_noaug=bootstrap(q,noaug,aug,rng),
                     aux_aug_vs_repeated=bootstrap(q,repeated,aug,rng))
    (A/'audit/bootstrap.json').write_text(json.dumps(uncertainty,indent=2))
    loo=[]
    for frame in q.frame_id.unique():
        ix=q.frame_id.ne(frame)
        loo.append(dict(omitted_frame=frame,delta=AP(q.loc[ix,'error'],candidate[ix])-AP(q.loc[ix,'error'],bo[ix])))
    pd.DataFrame(loo).to_csv(A/'audit/leave_one_frame_out.csv',index=False)
    sensitivity=[]
    for model in ['visual_d4_all_w000','visual_d4_all_w100']:
        sources=[pd.read_csv(A/f'features/{model}_actual_{tag}_predictions.csv') for tag in ['confirm','extra']]
        for label,p in [('seeds23_25',sources[0]),('seeds26_30',sources[1]),('seeds23_30',pd.concat(sources))]:
            po=p[p.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy()
            s=reorder(bo,po,q.class_name.eq('nude_hands').to_numpy(),.05)
            sensitivity.append(dict(model=model,seeds=label,standalone_ap=AP(q.error,po),**metrics(q,s)))
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(A/'audit/additional_seeds.csv',index=False)
    # Changes are constrained to the hand class and preserve its score multiset.
    mask=t.class_name.eq('nude_hands').to_numpy();assert np.array_equal(test[~mask],bt[~mask])
    assert np.allclose(np.sort(test[mask]),np.sort(bt[mask]),atol=1e-14,rtol=0)
    assert len(test)==362 and np.isfinite(test).all() and ((test>=0)&(test<=1)).all()
    changes=t[['id','frame_id','class_name']].assign(v52=bt,v54=test,delta=test-bt)
    changes[abs(changes.delta)>1e-10].to_csv(A/'audit/candidate_changes.csv',index=False)
    release=A/'release';release.mkdir(exist_ok=True);(release/'models').mkdir(exist_ok=True)
    submission='submission_v54_boxaug_hands.csv';t[['id']].assign(score=test).to_csv(release/submission,index=False)
    t[['id']].assign(score=bt).to_csv(release/'baseline.csv',index=False)
    t[['id','class_name']].to_csv(release/'test_metadata.csv',index=False)
    shutil.copy(A/'replay.py',release/'replay.py')
    manifest=json.loads((A/'models/manifest_confirm_actual.json').read_text());selected=[]
    rows=pd.read_csv(A/'inputs/records.csv')
    for m in manifest:
        if not m['name'].startswith('visual_d4_all_w100_actual_'):continue
        fit=m['fit_day'];cols=m['features'];features_file=f'features_fit{fit[-2:]}.csv'
        bank=pd.read_pickle(A/f'features/bank_fit{fit[-2:]}.pkl')
        bank.loc[rows.split.eq('test'),cols].assign(id=rows.loc[rows.split.eq('test'),'id']).to_csv(release/features_file,index=False)
        weights_file='models/'+m['name']+'.cbm';shutil.copy(A/'models'/Path(weights_file).name,release/weights_file)
        selected.append(dict(seed=m['seed'],fit_day=fit,columns=cols,features_file=features_file,weights_file=weights_file))
    assert len(selected)==6
    release_manifest=dict(submission_file=submission,scope='nude_hands',alpha=.05,models=selected,
            baseline='V52 public score 0.96300 reported by user; CSV SHA-256 d30da6f6f938cba285c01a6e8ddb4fe8d095f38e0ea65ec38a2aa87091705ab5',
            validation=metrics(q,candidate),v52_validation=metrics(q,bo),public_score=None,changed_test_rows=int((abs(test-bt)>1e-10).sum()),
            status='Experimental development candidate. Small selected OOF improvement; bootstrap CI includes zero. No new Kaggle result.',
            packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','catboost']})
    (release/'manifest.json').write_text(json.dumps(release_manifest,indent=2))
    (release/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in release_manifest['packages'].items())+'\n')
    records=json.loads((A/'inputs/protocol.json').read_text()); integrity=json.loads((A/'audit/integrity.json').read_text())
    expfiles=sorted((A/'results').glob('*experiments*.csv'))
    count=sum(len(pd.read_csv(p)) for p in expfiles)
    hashes={str(p.relative_to(release)):hashlib.sha256(p.read_bytes()).hexdigest() for p in release.rglob('*') if p.is_file()}
    (release/'sha256.json').write_text(json.dumps(hashes,indent=2))
    stats=dict(protocol=records,integrity=integrity,baseline=metrics(q,bo),candidate=metrics(q,candidate),
               auxiliary_noaug=metrics(q,noaug),auxiliary_aug=metrics(q,aug),auxiliary_repeat=metrics(q,repeated),
               uncertainty=uncertainty,changed_test_rows=int((abs(test-bt)>1e-10).sum()),max_abs_test_change=float(abs(test-bt).max()),
               saved_score_variants=count,saved_model_files=len(list((A/'models').glob('*.cbm'))),
               submission_sha256=hashlib.sha256((release/submission).read_bytes()).hexdigest())
    (A/'results/summary.json').write_text(json.dumps(stats,indent=2))
    baseline_ap=AP(q.error,bo);candidate_ap=AP(q.error,candidate);ci=uncertainty['candidate_vs_v52']['frame_bootstrap_ci95']
    lines=[
        'V54 — аугментации обучающих боксов. Отчёт об эксперименте',
        '',
        'Результат',
        f'Подтверждённый пользователем public Kaggle: V52 = 0.96300. Нового public-результата нет.',
        f'На исходных 198 запросах, с обучением на противоположном дне: V52 AP {baseline_ap:.9f}; кандидат V54 AP {candidate_ap:.9f}; разница {candidate_ap-baseline_ap:+.9f}.',
        f'95% интервал разницы при bootstrap по кадрам: [{ci[0]:+.9f}, {ci[1]:+.9f}]. Прирост мал; интервал включает ноль.',
        'Это многократно используемая development-выборка, а не новый независимый holdout; выбор лучшего рецепта повышает оптимизм оценки. Гарантии роста Kaggle нет.',
        '',
        'Аугментации и признаки',
        '9 579 дополнительных боксов от 198 исходных запросов / 35 кадров. Дополнительные рамки не являются независимыми наблюдениями.',
        'Сдвиги по X/Y; масштабирование; изменение отдельной стороны; случайный jitter четырёх сторон; движение к GT. Примеры: audit/augmentation_examples.jpg.',
        'Это аугментация проверяемой рамки при фиксированном изображении. Пиксельные аугментации не выполнялись; тестовые рамки не изменялись.',
        'Для каждой рамки заново решается исходное one-to-one сопоставление кадра/класса при IoU>=0.6: заменяется ровно один запрос.',
        '2 009 меток поменялись относительно родителя. В 99 синтетических случаях независимое правило max IoU<0.6 дало бы неверную метку из-за конкурирующих детекций.',
        'Пересчитаны геометрия относительно YOLO/OWLv2, pooled ROI из замороженных CLIP/DINOv2-B/L, а также признаки четырёх карт заполнения объектов.',
        'SAM/WeVis и уверенность исходного запроса не копируются на сдвинутые боксы. Замороженные crop-контексты фиксированы; все новые рамки внутри видимой области.',
        'GT-прототипы только обучающего дня, собственный кадр и соседи <=120 секунд исключены. Карты заполнения для обучения получены внутренними моделями, не видевшими родительский кадр.',
        'Валидация только на исходных запросах другого дня. Синтетические примеры и тестовые метки в валидацию/обучение соответственно не попадают.',
        '',
        'Парные проверки',
        f'Отдельная visual depth=4 модель, ансамбль seeds23/24/25: без аугментации AP {AP(q.error,noaug):.9f}; с ней {AP(q.error,aug):.9f}; повторение исходных строк с теми же весами {AP(q.error,repeated):.9f}.',
        'Это локальный AP отдельного вспомогательного классификатора; его значение нельзя путать с public Kaggle 0.96300.',
        'Вес каждой исходной строки = 1; суммарный вес синтетики каждого родителя = 1 в выбранном варианте. Число вариантов родителя не увеличивает его относительный вес.',
        'Проверены четыре семейства признаков, depth2/depth4, бюджеты аугментации 0/0.25/1, mild и no-GT политики, регрессия IoU, nested stacking, три основных и пять дополнительных random seeds.',
        'Прямое добавление вероятностей аугментированной модели не улучшило базовый ансамбль. Nested stacking тоже не дал прироста. Лучшие варианты сохранены вместе с отрицательными результатами.',
        '',
        'Кандидат для отдельной проверки',
        'Шесть CatBoost: два обучающих дня × seeds23/24/25; geometry+purged CLIP/DINO prototypes; все семейства аугментации; depth4, 400 итераций.',
        'Сортировка боксов nude_hands по V52_score + 0.05 * mean(aux_probability), затем присвоение отсортированных исходных V52-оценок этому порядку.',
        'Это сохраняет набор оценок класса и меняет только порядок. Остальные классы совпадают с известной V52. Нормализация использует только предсказания, без меток.',
        f'Изменено {stats["changed_test_rows"]} из 362 тестовых строк; максимальное абсолютное изменение {stats["max_abs_test_change"]:.9f}.',
        f'По дням: 25 октября {stats["baseline"]["2026-10-25"]:.9f} -> {stats["candidate"]["2026-10-25"]:.9f}; 26 октября {stats["baseline"]["2026-10-26"]:.9f} -> {stats["candidate"]["2026-10-26"]:.9f}.',
        'Рецепт выбран по development-результатам; статус экспериментальный. V52 и предыдущий V53 refit50 не перезаписаны.',
        '',
        'Дополнительные случайные инициализации (та же фиксированная формула кандидата)',
        sensitivity.to_string(index=False),
        '',
        'Проверки и воспроизведение',
        'Исходные 198 меток полностью воспроизведены. Независимо перепроверены 60 синтетических примеров; все геометрии/признаки конечны; у всех 9 579 рамок меняются пересчитанные признаки.',
        'Исходный feature-cache V41 соответствует сохранённому SHA-256 d518b44d24eb45709c4f483189248368c18f714dbb8a4b04357dabcf75163bec.',
        f'Сохранено {stats["saved_model_files"]} файлов моделей и {count} именованных вариантов оценки; общий поиск также включает предыдущие V31–V53.',
        f'CSV: release/{submission}',
        f'SHA-256: {stats["submission_sha256"]}',
        'Автономный replay использует готовые замороженные признаки и веса: python3 replay.py --output submission_replayed.csv. Обучающие метки и сеть ему не нужны.',
        'Полный эксперимент в рабочем проекте: prepare.py -> features.py -> train.py; парные проверки, stacked-модели и rank postprocessing находятся в отдельных скриптах.',
        'Архив с replay воспроизводит финальное ранжирование, но не заменяет исходные изображения/тяжёлые pretrained-backbone caches для извлечения признаков с нуля.',
    ]
    report='\n'.join(lines)+'\n';(A/'REPORT_RU.txt').write_text(report);(release/'REPORT_RU.txt').write_text(report)
    archive=ROOT/'rascar_v54_boxaug_candidate_replay.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(release.rglob('*')):
            if p.is_file():z.write(p,Path('rascar_v54_boxaug_candidate')/p.relative_to(release))
    print(json.dumps(stats,indent=2));print('ARCHIVE',archive)


if __name__=='__main__':main()
