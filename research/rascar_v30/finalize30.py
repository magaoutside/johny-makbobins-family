from pathlib import Path
import json,sys,hashlib,shutil,zipfile,importlib.metadata
import numpy as np,pandas as pd
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v30'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,h,B,DAYS,iou,align,base_oof,current,metric
from features23 import image
from bounded_geometry import upper_iou

def run():
    tables=[pd.read_csv(A/f'results/{name}_metrics.csv') for name in ['bounded','plate_tray','topology','elbow','dominance']]
    metrics=pd.concat(tables,ignore_index=True);metrics.to_csv(A/'results/all_56_experiments.csv',index=False)
    assert len(metrics)==56
    # Reference localization benchmark for the rejected exemplar algorithm.
    bench=[]
    for fit in DAYS:
        f=pd.read_csv(A/f'features/exemplars_oof_fit{fit}.csv')
        for r in f[f['rank'].eq(0)].itertuples():
            gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)]
            if not len(gt):continue
            bb=np.array([getattr(r,c) for c in B]);ref=np.array([getattr(r,'ref_'+c) for c in B])
            bench.append(dict(fit_day=fit,frame_id=r.frame_id,class_name=r.class_name,similarity=r.similarity,
                original_iou=float(iou(ref,gt[B]).max()),exemplar_iou=float(iou(bb,gt[B]).max())))
    bench=pd.DataFrame(bench);bench.to_csv(A/'results/exemplar_boundary_benchmark.csv',index=False)
    pool=pd.read_csv(ROOT/'rascar_v25/inputs/reviewed_pool_cups.csv')
    pool['definitely_not_flagged']=pool.teacher_iou.ge(.6)
    pool['note']='Admission already selected teacher agreement; positive-only, biased control, not AP validation.'
    pool[['id','frame_id','day','teacher_iou','definitely_not_flagged','note']].to_csv(A/'audit/pool_cup_controls.csv',index=False)
    sub=align(A/'submission_v30_precision_postcorrection.csv',te,'score')
    review=pd.read_csv(ROOT/'rascar_v29/audit/all_362_fresh_audit.csv').set_index('id').loc[te.id].reset_index()
    review['v30_score']=sub;review['v30_delta']=sub-current
    review['v30_action']=np.where(abs(sub-current)>1e-9,'bounded_cup_penalty','keep_v22')
    review.to_csv(A/'audit/all_362_current_scores.csv',index=False)
    shutil.copy2(A/'results/submissions/submission_bounded_cup_u2_c0.65_w0.25.csv',A/'submission_v30_low_strength_control.csv')
    shutil.copy2(A/'inputs/baseline_v22.csv',A/'submission_v30_rollback_v22.csv')
    # Review the delivered edits and examples that vetoed a higher-OOF variant.
    rows=[(1,'Accepted: loose cup'),(157,'Accepted: two cups'),(185,'Rejected hand penalty'),(222,'Rejected hand penalty')]
    canvas=Image.new('RGB',(1000,760),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    idx=review.set_index('review_no')
    for j,(no,title) in enumerate(rows):
        r=idx.loc[no];b=r[B].to_numpy(float);wh=b[2:]-b[:2]
        lo=np.maximum(0,b[:2]-.6*wh);hi=np.minimum([1280,720],b[2:]+.6*wh)
        im=Image.fromarray(image(r.frame_id)[:,:,::-1]).crop(tuple(np.r_[lo,hi]))
        scale=min(470/im.width,285/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)))
        d=ImageDraw.Draw(im);d.rectangle(np.r_[(b[:2]-lo)*scale,(b[2:]-lo)*scale].tolist(),outline='red',width=3)
        x=(j%2)*500;y=(j//2)*380
        draw.text((x+10,y+8),f'#{no} {title}',fill='black',font=font)
        draw.text((x+10,y+36),f'V22 {r.v22_score:.6f}  ->  V30 {r.v30_score:.6f}',fill='black',font=font)
        canvas.paste(im,(x+10,y+75))
    canvas.save(A/'audit/release_review.jpg',quality=94)
    decision=dict(baseline_version='V22',baseline_public_ap=.95610,rejected_v27_public_ap=.95159,
        candidate='submission_v30_precision_postcorrection.csv',candidate_public_ap=None,
        primary_oof=metric(q,base_oof),changed_test=2,changed_oof=1,leaderboard_submitted=False,
        promoted_to_confirmed_best=False,selected='bounded_cup_u2_c0.65_w0.5',
        rationale='No cross-day AP loss, no manual positive-control flags, two visually supported geometry errors.',
        limitation='OOF AP unchanged. 2px envelope is a sensitivity assumption, not a guarantee about hidden GT.',
        excluded_higher_oof='Hand topology penalized plausible true hands; elbow gain is one OOF frame and no test score changes.')
    (A/'DECISION.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    report='''RASCAR V30 — отдельные посткоррекции сложных случаев
14 ноября 2026. Ничего не отправлено на лидерборд.

РЕЗУЛЬТАТ
Сохранён submission_v30_precision_postcorrection.csv: 362 строки, база V22,
изменены только 2 score чашек. Подтверждённый лучший Public AP остаётся 0.95610.
Public AP нового файла неизвестен. V27 с 0.95159 не используется.

Версия                          OOF AP        25 октября    26 октября
V22, основной протокол           0.983219826   0.973169965   0.990405676
V30, выбранные посткоррекции     0.983219826   0.973169965   0.990405676
Кисти, топология, отвергнут      0.983416181*  0.974088706*  0.990405676
Локоть вместо кисти, отдельно    0.984996990*  0.977599372*  0.990405676
* Точные вычисленные значения — results/all_56_experiments.csv; строки выше
  округлены для чтения. Эти варианты не включены в выбранный сабмит.

ЧТО ДЕЛАЕТ ВЫБРАННЫЙ АЛГОРИТМ
Для каждой чашки берутся все доступные SAM-референсы после NMS, а не только
удобный ближайший. Алгоритм перебирает допустимое отклонение каждой границы
референса на ±2 исходных пикселя и находит максимально возможный IoU с запросом.
Поправка включается, если даже максимум меньше 0.6, исходный IoU >=0.1,
confidence ближайшего SAM >=0.65, а DINO-score класса cup >=0.8.
Новый score = старый + 0.5 * max(0.9 - старый, 0).
Исходные боксы и число строк не меняются. Высокий score означает ошибку.
Это условная проверка чувствительности к границам, не доверительный интервал GT.
При допуске ±3 пикселя обе новые тестовые поправки исчезают: вывод зависит от
точности референса. Поэтому дополнительно просмотрены сами изображения.

ДВА ИЗМЕНЕНИЯ
#1: чашка присутствует, но рамка включает много фона сверху и справа.
0.629589 -> 0.764795; место в ранжировании ошибок 138 -> 108.
Максимальный IoU с машинными референсами при ±2px: 0.574703.
#157: рамка охватывает две чашки, а должна соответствовать одному объекту.
0.806787 -> 0.853394; место 96 -> 84; максимальный IoU 0.579134.
Идентификаторы этих случаев не используются в условии алгоритма.
Визуальные оценки V29 поддерживают обе поправки, но скрытой разметки теста нет.

ПРОВЕРКИ
• Основной OOF: 198 запросов / 35 кадров. Отдельно 25 и 26 октября.
• Убрана историческая единичная OOF-поправка, отсутствующая на тесте.
• В выбранной версии один OOF-score меняется: 9 перестановок между ошибочными
  рамками, 0 перестановок «ошибка/правильная рамка». AP сохраняется.
• Положительный контроль: все 50 ручных чашек и 83 ручные кисти, 0 ложных
  штрафов геометрического условия; у 109 есть референсы, у 24 — воздержание.
  Для этого контроля правильность класса задана ручной разметкой, без DINO-гейта.
• 17 ранее принятых чашек машинного пула также сохраняются. Это слабый контроль:
  их отбор уже учитывал согласие с учителем, отрицательных примеров здесь нет.
• В тесте проверены оба принятых изменения и 5 кистей, затронутых отвергнутой
  моделью. Полный предыдущий аудит 362 рамок сохранён с обновлёнными score.
• Повторный запуск отдельного postcorrect.py воспроизводит результат.
  Проверены 362 уникальных ID, диапазон score, неизменность V22, отсутствие
  обучения на тестовой разметке, разбиение по дням и 180000 случайных боксов
  для проверки верхней границы IoU. Подробности — checks.json.

ЧТО ЕЩЁ РЕАЛИЗОВАНО И ПОЧЕМУ НЕ ВКЛЮЧЕНО
1. Сопоставление с размеченными изображениями чашек/кистей: яркость и градиенты,
   пять масштабов, отдельный банк противоположного дня, пять различных кадров.
   Алгоритм цепляется за часть объекта. На сопоставляемых референсах доля IoU>=0.6
   у чашек падает с 27/39 до 2/39, у кистей с 31/43 до 1/43. Отвергнут.
2. Границы с допуском: 20 вариантов для чашек, кистей и людей. Варианты для людей
   снижают общий OOF, вплоть до 0.982177, и штрафуют визуально нормальных людей.
   Для кистей осторожные настройки не дают новых изменений. Выбраны только чашки.
3. Тарелка/поднос: 6 обученных моделей, HOG + радиальные профили края, 12 вариантов
   посткоррекции. Один проблемный поднос исправляется, но истинные подносы на
   валидации штрафуются. Лучший активный OOF 0.983051 < V22. В день 25 всего
   4 ручных подноса; модель плохо переносится. Не включено.
4. Топология кисти: 4 CatBoost-модели, 8 вариантов. Различаются покрытие ладони,
   кончиков пальцев, пересечения цепочек пальцев с границей и лишняя площадь.
   RTMW — уже сохранённые предсказания, новый GPU-инференс не заявляется.
   SimCC-score не трактуется как вероятность. Варианты RTMW — зависимые признаки.
   Лучший OOF слегка выше, но штрафуются нормальные кисти #185 и #222; отвергнуто.
5. Локоть вместо кисти: 8 вариантов правила по скелету, без обучения на тесте.
   Три и более вида pose подтверждают локоть внутри рамки и запястье дальше.
   Исправляется один OOF-пример: человек сидит, модель приняла локоть за ладонь.
   В тесте отмечены #122/#123, но V22 уже даёт им 0.977689/0.972392, поэтому
   их score не меняется. Нулевой новый охват теста; прирост OOF НЕ выдаётся за
   доказательство прироста сабмита. В выбранный файл этот эксперимент не включён.
   Геометрическое условие с >=3 видами не срабатывает на 83 истинных ручных кистях.
6. Дубли: 8 вариантов строгого геометрического доминирования. Победитель должен
   оставаться корректным при сдвиге всех границ и превосходить лучший возможный
   IoU проигравшего; вторые возможные объекты запрещают штраф. Сомнительные группы
   остаются без решения. Новых тестовых изменений нет.

Всего: 6 направлений, 56 сохранённых вариантов посткоррекции, 10 обученных
небольших классификаторов плюс банки визуальных образцов. Модели и таблицы сохранены.
Размер подноса с учётом перспективы дополнительно проверен на #73: его площадь
не является физически невозможной, поэтому запрет «слишком маленький поднос»
не обоснован. Не добавлен как искусственное правило под один ID.

ЧТО НЕ ЗАКРЫТО
Тарелка вместо подноса #73; неопределённые границы кистей #145/#204/#266;
неустойчивое назначение дублей #136/#137 и других групп; скрытые части людей.
У части из них V22 уже выставляет высокий score ошибки, но точной проверки
правильности рамки всё ещё нет. Нельзя честно заявить, что «все ошибки удалены»
или назвать процент исправлений относительно неизвестного GT. В этой итерации
есть 2 обоснованные перестановки score, а не доказанные 2 исправленные метки.

ОГРАНИЧЕНИЯ ВАЛИДАЦИИ
Многократный выбор на тех же двух днях делает OOF метрикой разработки.
Новый AP на лидерборде не известен. Фотографии теста использовались для проверки
и отбраковки кандидатов, поэтому их визуальный аудит не является независимым holdout.
Ни тестовые оценки аудитора, ни нарисованные тестовые границы не участвовали в
обучении новых моделей. Для обучаемых экспериментов весь отложенный день исключён
из fit; исходные V22 и CV-референсы заморожены. Широкий новый ансамбль не выпускался.

ВОСПРОИЗВЕДЕНИЕ
python3 rascar_v30/postcorrect.py
Для автономного пакета достаточно Python, numpy, pandas. Это посткоррекция
сохранённых выходов SAM/DINO/V22, а не повторный запуск исходных CV-моделей.
Исследовательские .py запускаются из исходного проекта и используют сохранённые
разметку, картинки и предсказания прошлых версий. Ничего из старых версий не удалено.

Источники: OpenCV template matching
https://docs.opencv.org/4.13.0/d4/dc6/tutorial_py_template_matching.html
Исходная работа с фиксированной камерой и анатомическими ограничениями —
проектные гипотезы, проверенные здесь; чужой опубликованный прирост не заявляется.
'''
    # Exact measured lines replace illustrative rounded placeholders.
    for name,label in [('hand_topology_d3_p0.7_w0.5','Кисти, топология, отвергнут'),('elbow_v3_sem0.7_w0.5','Локоть вместо кисти, отдельно')]:
        r=metrics.set_index('experiment').loc[name]
        old=next(line for line in report.splitlines() if line.startswith(label))
        report=report.replace(old,f'{label:32s}{r.pooled_ap:.9f}   {r.day_25:.9f}   {r.day_26:.9f}')
    (A/'REPORT.txt').write_text(report)
    (A/'README.txt').write_text('''V30: start with REPORT.txt and DECISION.json.
Candidate: submission_v30_precision_postcorrection.csv (public AP unknown).
Confirmed baseline: V22, public AP 0.95610.
Run postcorrect.py for standalone replay from frozen CV outputs.
Run verify30.py in the original workspace for metric/fold/numerical checks.
Research scripts use assets from earlier project versions; they are not a raw-image turnkey pipeline.
all_56_experiments.csv contains every evaluated postcorrection. No leaderboard submission was made.
''')
    versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','opencv-python','scikit-learn','scikit-image','catboost','joblib']}
    (A/'environment.json').write_text(json.dumps(versions,indent=2))
    (A/'requirements-release.txt').write_text('\n'.join(f'{k}=={versions[k]}' for k in ['numpy','pandas'])+'\n')
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[dict(cell_type='markdown',metadata={},source=['# V30 — replay of two geometry postcorrections\n',
          'Frozen V22/SAM/DINO inputs. No training or leaderboard upload. Public AP unknown.\n']),
          dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=[
            'from pathlib import Path\n','import runpy\n',
            'root = Path.cwd() if (Path.cwd() / "postcorrect.py").exists() else Path.cwd() / "rascar_v30"\n',
            'module = runpy.run_path(str(root / "postcorrect.py"))\n','submission, evidence = module["run"](root)\n',
            'evidence[evidence["delta"].abs() > 1e-9]\n'])])
    (A/'rascar_v30_postcorrection.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))
    manifest={str(p.relative_to(A)):hashlib.sha256(p.read_bytes()).hexdigest() for p in A.rglob('*')
              if p.is_file() and '__pycache__' not in str(p) and p.name!='MANIFEST.json'}
    (A/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    release=['postcorrect.py','rascar_v30_postcorrection.ipynb','requirements-release.txt','README.txt','REPORT.txt',
             'DECISION.json','checks.json','submission_v30_precision_postcorrection.csv','submission_v30_low_strength_control.csv',
             'submission_v30_rollback_v22.csv','release_evidence.csv']
    release += [str(p.relative_to(A)) for p in (A/'inputs').glob('*') if p.name in ['baseline_v22.csv','test_queries.csv',
        'semantic_test.csv','frozen_cup_references.csv','release_rule.json']]
    with zipfile.ZipFile(ROOT/'rascar_v30_postcorrection_release.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in release:z.write(A/name,'rascar_v30/'+name)
    with zipfile.ZipFile(ROOT/'rascar_v30_research_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in A.rglob('*'):
            if p.is_file() and '__pycache__' not in str(p):z.write(p,str(p.relative_to(ROOT)))
    print('FINALIZED',len(metrics),'variants',len(manifest),'files')

if __name__=='__main__':run()
