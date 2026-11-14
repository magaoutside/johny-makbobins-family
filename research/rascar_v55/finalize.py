"""Record a negative/weak experiment without promoting it to a new submission."""
from common55 import *
from train import Head,ResidualHead
from sklearn.metrics import average_precision_score as AP
import torch,zipfile

def main():
    torch.set_num_threads(3);q,t,h=inputs();bo,bt=incumbent(q,t)
    initial=pd.read_csv(A/'results/experiments_initial.csv');confirm=pd.read_csv(A/'results/experiments_confirm.csv')
    selected='flip450_confirm_all_replace100'
    score=pd.read_csv(A/f'results/oof/{selected}.csv').set_index('id').loc[q.id,'score'].to_numpy()
    test=pd.read_csv(A/f'results/submissions/{selected}.csv').set_index('id').loc[t.id,'score'].to_numpy()
    rng=np.random.default_rng(5510);unique=list(q.frame_id.unique());indices=q.frame_id.map({f:i for i,f in enumerate(unique)}).to_numpy();deltas=[]
    byday=[np.array([unique.index(f) for f in q.loc[q.day.eq(day),'frame_id'].unique()]) for day in DAYS]
    for _ in range(3000):
        samples=np.concatenate([rng.choice(a,len(a),replace=True) for a in byday]);weight=np.bincount(samples,minlength=len(unique))[indices]
        deltas.append(AP(q.error,score,sample_weight=weight)-AP(q.error,bo,sample_weight=weight))
    pos=np.flatnonzero(q.error.eq(1));neg=np.flatnonzero(q.error.eq(0))
    old=bo[pos,None]>bo[None,neg];new=score[pos,None]>score[None,neg]
    pairchanges=dict(improved=int((new&~old).sum()),worsened=int((old&~new).sum()))
    loo=[]
    for frame in unique:
        keep=q.frame_id.ne(frame);loo.append(AP(q.loc[keep,'error'],score[keep])-AP(q.loc[keep,'error'],bo[keep]))
    manifests=[]
    for p in sorted((A/'models').glob('*.json')):
        m=json.loads(p.read_text());assert set(m['train_frames']).isdisjoint(m['validation_frames'])
        assert all('_'+m['fit_day'].replace('-','')+'_' in frame for frame in m['train_frames'])
        manifests.append(m)
    # Independently reload one augmented checkpoint and reproduce every saved map.
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');lookup=rec.reset_index().set_index('id')['index'];features=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r');replay=[]
    for fit in DAYS:
        name=f'flip450_s3101_fit{fit[-2:]}';model=Head();model.load_state_dict(torch.load(A/f'models/{name}.pt',weights_only=True));model.eval()
        saved=np.load(A/f'features/{name}_maps.npz');ids=saved['ids'];ix=lookup.loc[ids].to_numpy();pred=[]
        for off in range(0,len(ix),16):
            with torch.inference_mode():pred.append(model(torch.from_numpy(np.array(features[ix[off:off+16]],np.float32))).sigmoid().numpy())
        error=float(abs(np.concatenate(pred)-saved['maps']).max());assert error<1e-7
        replay.append(dict(model=name,rows=len(ids),max_abs_error=error))
    (A/'audit/checkpoint_replay.json').write_text(json.dumps(replay,indent=2))
    summaries=initial.groupby('recipe').apply(lambda g:g.loc[g.ap.idxmax()],include_groups=False).reset_index()
    summaries.to_csv(A/'results/recipe_summary.csv',index=False)
    ci=np.quantile(deltas,[.025,.975]).tolist()
    summary=dict(status='NOT_PROMOTED: no material validated improvement; retain V52 as confirmed public baseline.',
        baseline_public=.96300,new_public=None,baseline=metrics(q,bo),best_confirmed_development=metrics(q,score),
        selected_development_name=selected,delta=float(AP(q.error,score)-AP(q.error,bo)),
        frame_bootstrap_ci95=ci,bootstrap_resamples=3000,selection_adjusted=False,
        loo_frame_delta_min=float(min(loo)),pair_changes=pairchanges,changed_test=int((abs(test-bt)>1e-9).sum()),
        trained_models=len(manifests),recipes=initial.recipe.nunique(),augmented_rgb_crops=1977,manual_frames=38,
        checkpoint_replay=replay,original_baseline_unchanged=True)
    (A/'results/summary.json').write_text(json.dumps(summary,indent=2))
    report=[
      'V55 — проверка более существенного изменения визуального модуля',
      '',
      'РЕШЕНИЕ: не назначать новый основной сабмит. Существенного подтверждённого прироста не получено.',
      'Лучший подтверждённый пользователем Public AP остаётся 0.96300 (V52). Новых отправок не было.',
      f'Лучший результат после усреднения трёх seed: OOF AP {AP(q.error,score):.9f}, база {AP(q.error,bo):.9f}, delta {AP(q.error,score)-AP(q.error,bo):+.9f}.',
      f'95% frame-bootstrap интервал delta: [{ci[0]:+.9f}, {ci[1]:+.9f}]; минимум при исключении одного кадра {min(loo):+.9f}.',
      'Это малая, многократно используемая development-выборка; выбор рецепта/классов по ней не учтён в интервале. Новая независимая оценка не получена.',
      'Численно результат лучше V54, но всё ещё слишком мал, чтобы считать запрос на заметный рост выполненным.',
      '',
      'Что сделано',
      '1. Проанализированы оставшиеся ошибки V52, их GT-IoU и неверно упорядоченные пары. Галерея: audit/hard_examples.jpg.',
      '2. Созданы 1 977 реальных RGB-аугментаций 659 ручных кропов: отражение; gamma/яркость/контраст/насыщенность/лёгкое размытие; новые размеры и центры кропа.',
      '3. Для каждого изображения заново извлечены замороженные DINOv2-small признаки при 448px, сетка 32x32. Это не копирование прежнего embedding и не только jitter координат.',
      '4. Все GT-цели преобразуются вместе с изображением. Для отражений целевые карты в точности совпали с отражёнными исходными картами.',
      '5. Обучены линейные карты занятости GT-прямоугольниками (не истинная пиксельная сегментация). Дни строго раздельные, обучение только на ручных кадрах fit-day.',
      '6. Сравнены control450/control900; отражение; фотометрия; комбинированные аугментации; усиление loss у границ; hard-example mining; отдельная residual CNN 384->32->4 с фиксированной базой.',
      '7. Изменяется исходный dense-модуль ансамбля, а не только маленькая добавка к финальному score. Сильная V52 person-модель сохраняется.',
      '8. Проверена непрерывная интерполяция границ на сетке вместо округления к ячейкам. Существенного выигрыша не дала.',
      f'Всего обучено {len(manifests)} нейронных моделей. Для контрольного и наиболее перспективного рецепта проверены seeds3101/3102/3103.',
      '',
      'Контроль и данные',
      'Повтор исходного V31 seed3101 воспроизвёл OOF/test scores с ошибкой 1.11e-16.',
      'Новые кропы происходят всего из 38 ручных кадров. Официальные error-метки: 198 запросов из 35 кадров. 1 977 аугментаций не являются новыми независимыми наблюдениями.',
      'Ни один тестовый или противоположный дневной кадр не обучает соответствующую модель. Тестовые GT/ручные статусы не использованы для изменения scores.',
      'Классы и число шагов фиксируются на уровне рецепта, не выбираются по эпохам оценочного дня. Модели, карты и все отрицательные результаты сохранены.',
      'Визуальная проверка аугментаций: audit/rgb_augmentations.jpg. Загрузка весов и повтор карт проверены отдельно: audit/checkpoint_replay.json.',
      '',
      'Лучшие результаты каждого рецепта в первом seed (выбор максимумов post-hoc)',
      summaries[['recipe','scope','fraction','ap','delta']].to_string(index=False),
      '',
      'Сравнение трёх seed',
      confirm[['recipe','scope','fraction','ap','2026-10-25','2026-10-26','cup','person']].to_string(index=False),
      '',
      f'Лучший общий вариант исправляет {pairchanges["improved"]} пар error/correct и ухудшает {pairchanges["worsened"]}. Это не число исправленных фотографий.',
      f'Он изменяет {summary["changed_test"]} тестовых оценок; одновременно ухудшается AP чашек. Такая цена изменения при малом общем выигрыше не оправдывает новую рекомендацию отправки.',
      'Ветка сохранена для исследований, без папки release и без замены V52/V53/V54. CSV в results/submissions не объявляются улучшенным решением.',
      '',
      'Вывод',
      'В этой серии ни фотометрические аугментации, ни более гибкая сеть, ни hard-example mining не дали нужного заметного шага.',
      'Дополнительное размножение тех же кадров не решило проблему переноса на другие дни. Для следующей гипотезы важны новые независимые кадры с проверяемой геометрией, особенно person и неоднозначные руки; это направление, а не обещание прироста.',
      '',
      'Воспроизведение в текущем рабочем проекте',
      'python3 rascar_v55/extract.py',
      'python3 rascar_v55/train.py --tag initial',
      'python3 rascar_v55/train.py --names control450 flip450 --seeds 3102 3103 --tag extra',
      'python3 rascar_v55/train.py --names control450 flip450 --seeds 3101 3102 3103 --evaluate-only --tag confirm',
      'python3 rascar_v55/subpixel.py --names control450 flip450 --tag control',
      'python3 rascar_v55/finalize.py',
      'Нужны исходные изображения и V31/V52 кэши проекта; compact-архив содержит код, веса, протоколы и результаты, но не дублирует многогигабайтные признаки.',
    ]
    (A/'REPORT_RU.txt').write_text('\n'.join(report)+'\n')
    archive=ROOT/'rascar_v55_research.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(A.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if p.suffix in ['.npy','.npz']:continue
            if '/results/submissions/' in str(p) or '/results/oof/' in str(p):continue
            z.write(p,Path('rascar_v55')/p.relative_to(A))
    print(json.dumps(summary,indent=2));print('ARCHIVE',archive)

if __name__=='__main__':main()
