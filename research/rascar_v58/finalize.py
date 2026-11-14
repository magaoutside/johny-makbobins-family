"""Freeze the train-OOF-selected V58, then generate diagnostic audit and replay."""
from pathlib import Path
import json,hashlib,zipfile,shutil,base64,html
import pandas as pd,numpy as np
from sklearn.metrics import average_precision_score as AP
A=Path(__file__).resolve().parent;OLD=A.parent/'rascar_v56';R=A/'release';R.mkdir(exist_ok=True)
name='guarded_tta_cup_tray0.6_cap0.15'
Q=pd.read_csv(OLD/'kaggle_payload/rascar_v56/inputs/q.csv');T=pd.read_csv(OLD/'kaggle_payload/rascar_v56/inputs/t.csv');n=len(Q);Y=Q.error.to_numpy();S=np.load(A/'results/scores.npz');z=S[name];s=z[:n];b=S['V52'];v57=S['V57']
res=pd.read_csv(A/'results/candidates.csv').set_index('name');seeds=[]
for seed in [3101,3102,3103]:
 rr=pd.read_csv(A/f'results/seed_{seed}/candidates.csv').set_index('name');v=rr.loc[name];v0=rr.loc['V52'];v1=rr.loc['V57']
 assert v['ap']>v1['ap'] and min(v[d]-v0[d] for d in sorted(Q.day.unique()))>0
 seeds.append(dict(seed=seed,ap=v['ap'],day25=v['2026-10-25'],day26=v['2026-10-26'],delta57=v['ap']-v1['ap'],min_day_delta52=min(v[d]-v0[d] for d in sorted(Q.day.unique()))))
boot=pd.read_csv(A/'results/frame_bootstrap.csv').set_index('name').loc[name].to_dict();stress=pd.read_csv(A/'results/prior_shift_stress.csv').set_index('name').loc[name].to_dict()
assert res.loc[name,'ap']>=res.ap.max()-1e-12
assert min(res.loc[name,d]-res.loc['V57',d] for d in sorted(Q.day.unique()))>=-1e-12
assert abs(z-b).max()<=.15000001
sel=dict(winner=name,metrics=res.loc[name].to_dict(),v52=res.loc['V52'].to_dict(),v57=res.loc['V57'].to_dict(),candidate_count=len(res),unique_prediction_pairs=len({hashlib.sha256(np.round(S[k],12).tobytes()).hexdigest() for k in S.files}),seed_confirmation=seeds,bootstrap=boot,prior_shift_stress=stress,selection='Highest pooled development OOF, no day regression versus V57 in mean ensemble, all 3 seeds improve pooled AP over V57 and both days over V52. Tied OOF variants: prefer preserving previous dense component when new geometry is absent or contradictory. Absolute correction cap .15. Selection completed before reading visual test flags.',limitations=['These same 198 boxes / 35 frames / 2 days have been used repeatedly for adaptive model selection; not an independent holdout.','Bootstrap and prior stress use fixed OOF predictions, not selection-adjusted confidence.','Test averages two fit-day experts; OOF has one opposite-day expert.','No hidden test labels available; visual flags are hypotheses only.'],submission_status='not_yet_submitted')
(R/'selection.json').write_text(json.dumps(sel,ensure_ascii=False,indent=2))
# Above this point no visual test notes or flags have been loaded.
out=R/'submission_v58_final_guarded_tta.csv';T[['id']].assign(score=z[n:]).to_csv(out,index=False,float_format='%.12f');assert AP(Y,s)==AP(Y,np.round(s,12));sha=hashlib.sha256(out.read_bytes()).hexdigest()
assert list(pd.read_csv(out).id)==list(T.id);assert T.id.nunique()==362 and len(T)==362;assert np.isfinite(z).all() and z.min()>=0 and z.max()<=1
Q.assign(score_v58=s,score_v57=v57[:n],delta_vs_v52=s-b[:n]).to_csv(R/'selected_oof.csv',index=False)
pd.DataFrame(seeds).to_csv(R/'seed_confirmation.csv',index=False)
# Rank-error accounting on official labels.
def pairs(x):return x[Y==1,None]<=x[None,Y==0]
old,new=pairs(b[:n]),pairs(s);cnt0=np.zeros(n,int);cnt1=cnt0.copy();cnt0[Y==1]=old.sum(1);cnt0[Y==0]=old.sum(0);cnt1[Y==1]=new.sum(1);cnt1[Y==0]=new.sum(0)
train=Q.assign(score_v58=s,score_v57=v57[:n],pairs_before=cnt0,pairs_after=cnt1,pair_change=cnt1-cnt0)
train.to_csv(R/'all_train_198.csv',index=False);train[cnt0>0].to_csv(R/'original_49_train.csv',index=False)
rr=[]
for i,j in zip(*np.where(old|new)):
 rr.append(dict(error_id=Q.loc[Y==1,'id'].iloc[i],correct_id=Q.loc[Y==0,'id'].iloc[j],before=bool(old[i,j]),after=bool(new[i,j]),outcome='fixed' if old[i,j] and not new[i,j] else 'new' if new[i,j] and not old[i,j] else 'remaining'))
pd.DataFrame(rr).to_csv(R/'train_pair_changes.csv',index=False)
# Diagnostic-only visual hypotheses, after final method freeze.
audit=pd.read_csv(OLD/'visual_audit/test_all_362.csv').set_index('id').loc[T.id].reset_index();audit['score_v58']=z[n:];audit['score_v57']=v57[n:]
for ref,refscore in [('v52',b[n:]),('v57',v57[n:]),('a40',audit.score_a40.to_numpy())]:audit['delta_vs_'+ref]=z[n:]-refscore
flag=audit[audit.flag.isin(['up','down'])].copy();sign=flag.flag.map({'up':1,'down':-1});counts={}
for ref in ['v52','v57','a40']:
 d=sign*flag['delta_vs_'+ref];flag['direction_vs_'+ref]=d
 counts[ref]=dict(favorable_boxes=int((d>=.01).sum()),favorable_frames=int(flag.loc[d>=.01,'frame_id'].nunique()),unfavorable_boxes=int((d<=-.01).sum()),small=int((abs(d)<.01).sum()))
audit.to_csv(R/'all_test_362.csv',index=False);flag.to_csv(R/'all_43_suspected_changes.csv',index=False)
summary=dict(submission=out.name,sha256=sha,train_pairs=dict(before=int(old.sum()),after=int(new.sum()),fixed=int((old&~new).sum()),new=int((~old&new).sum()),remaining=int((old&new).sum())),original_49=dict(improved=int(((cnt0>0)&(cnt1<cnt0)).sum()),worsened=int(((cnt0>0)&(cnt1>cnt0)).sum()),unchanged=int(((cnt0>0)&(cnt1==cnt0)).sum()),fully_resolved=int(((cnt0>0)&(cnt1==0)).sum()),improved_frames=int(Q.loc[(cnt0>0)&(cnt1<cnt0),'frame_id'].nunique())),visual_hypotheses_only=counts,rows=362,changed_vs_v52_over01=int((abs(z[n:]-b[n:])>=.01).sum()),max_delta_vs_v52=float(abs(z[n:]-b[n:]).max()))
(R/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
# Cached-expert replay inputs include raw new dense overlap, not final test scores.
basein=dict(np.load(OLD/'correction_audit/replay_inputs.npz',allow_pickle=False));basein.pop('expected')
e=pd.concat([pd.read_csv(p) for p in sorted((A/'features').glob('tta_spatial_w0.5_mean3_fit*_evidence.csv'))]);oo=e[e.split.eq('oof')].set_index('id').loc[Q.id,'overlap'].to_numpy();tt=np.stack([g.set_index('id').loc[T.id,'overlap'].to_numpy() for _,g in e[e.split.eq('test')].groupby('fit_day')])
np.savez_compressed(R/'replay_inputs.npz',**basein,v22=np.r_[Q.v22,T.v22],v31=np.r_[Q.v31,T.v31],new_dense_oof_iou=oo,new_dense_test_iou=tt,expected=z)
replay='''"""Exact final score assembly from cached, day-held-out expert outputs.
GPU trained-checkpoint replay is separately recorded in spatial_checkpoint_replay.json.
"""
from pathlib import Path
import numpy as np,pandas as pd,hashlib
A=Path(__file__).resolve().parent;d=np.load(A/'replay_inputs.npz',allow_pickle=False)
b=d['baseline'];c=d['class_name'];n=len(d['new_dense_oof_iou'])
s=b+np.clip(.2*(d['p_full']-b),0,.1)+np.clip(.2*(d['p_crop']-b),-.1,0)
m=c=='nude_hands';s[m]=b[m]+np.clip(.2*(d['p_anchor_adapted'][m]-b[m]),-.1,.1)
s=np.clip(s,0,1)
o,t=d['new_dense_oof_iou'],d['new_dense_test_iou'];v22=d['v22'];dense=v22.copy()
valid=np.isfinite(o);target=np.clip((.85-o)/.5,0,1);dense[:n][valid]+=.25*(target-v22[:n])[valid]
target=np.clip((.85-t)/.5,0,1);delta=target-v22[n:][None,:]
support=np.isfinite(target).all(0)&((delta>=0).all(0)|(delta<=0).all(0))
safe=np.where(np.isfinite(target),target,v22[n:][None,:]);cons=np.where((delta>=0).all(0),safe.min(0),safe.max(0));dense[n:][support]+=.25*(cons-v22[n:])[support]
m=(c=='cup')&np.r_[valid,support];s[m]+=(dense-d['v31'])[m]
m=c=='tray';x=np.clip(b[m],.0001,.9999);s[m]=1/(1+np.exp(-(np.log(x/(1-x))-.6)))
s=np.clip(s,0,1);s=b+np.clip(s-b,-.15,.15)
assert np.allclose(s,d['expected'],rtol=0,atol=1e-14),float(abs(s-d['expected']).max())
assert np.array_equal(np.round(s,12),np.round(d['expected'],12))
out=A/'replayed_submission.csv';pd.DataFrame({'id':d['id'][n:],'score':s[n:]}).to_csv(out,index=False,float_format='%.12f')
sha=hashlib.sha256(out.read_bytes()).hexdigest();assert sha=='SHA_PLACEHOLDER'
print('EXACT_REPLAY_OK',len(s),'scores; CSV SHA256',sha)
'''.replace('SHA_PLACEHOLDER',sha)
(R/'replay_candidate.py').write_text(replay)
shutil.copy(A/'audit/spatial_checkpoint_replay.json',R/'spatial_checkpoint_replay.json')
for p in ['candidates.csv','frame_bootstrap.csv','prior_shift_stress.csv','slices.csv']:shutil.copy(A/'results'/p,R/p)
# Self-contained visual audit of all 43 flagged test hypotheses.
cards=[]
for _,r in flag.iterrows():
 im=OLD/'visual_audit/crops'/f'{int(r.review_no):03d}.jpg'
 if not im.exists():im=OLD/'visual_audit/crops'/f'{int(r.review_no):03d}.jpg'
 picture=f'<img loading="lazy" src="data:image/jpeg;base64,{base64.b64encode(im.read_bytes()).decode()}">' if im.exists() else ''
 cards.append(f'<article><h3>#{int(r.review_no)} · {html.escape(r.class_name)} · {html.escape(r.flag)}</h3>{picture}<p>V52 {r.score_v52:.4f} → V57 {r.score_v57:.4f} → <b>V58 {r.score_v58:.4f}</b></p><p>{html.escape(str(r.note))}</p><small>{r.id} · {r.frame_id}</small></article>')
htmlout=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>V58 — финальная проверка</title><style>body{{font:16px system-ui;background:#10131a;color:#e9edf5;margin:32px auto;max-width:1200px;padding:20px}}h1{{font-size:36px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}article{{background:#1b2130;border-radius:12px;padding:16px}}img{{width:100%;height:250px;object-fit:contain}}small{{overflow-wrap:anywhere;color:#abb6cc}}a{{color:#84b9ff}}table{{border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #49536b}}</style><h1>V58 — финальный кандидат</h1><p>OOF AP: V52 {res.loc['V52','ap']:.8f} → V57 {res.loc['V57','ap']:.8f} → <b>V58 {res.loc[name,'ap']:.8f}</b>. Неверные пары: {old.sum()} → {new.sum()}.</p><p>Выбран по train OOF до чтения test-заметок. Эти 43 test-случая — визуальные гипотезы, не подтверждённые ошибки. Изменение score в ожидаемую сторону не доказывает исправление. OOF использовался для выбора многократно.</p><p>Рецепт: V57 + DINOv2-B / пространственная голова / 3 seed / RGB flip TTA для cup; сохранение старого вклада при отсутствии нового референса; монотонная tray-поправка −0.6 в logit; общий лимит |Δ к V52| ≤ 0.15.</p><p><a href="submission_v58_final_guarded_tta.csv">CSV</a> · <a href="all_test_362.csv">Все 362 бокса</a> · <a href="all_43_suspected_changes.csv">Все 43 гипотезы</a> · <a href="selection.json">Метрики и ограничения</a></p><div class="grid">{''.join(cards)}</div></html>'''
(R/'report.html').write_text(htmlout)
assert htmlout.count('<article>')==43
sources={'DINOv2-B':'https://huggingface.co/facebook/dinov2-base','DINOv2-S fine-tuning experiment':'https://huggingface.co/facebook/dinov2-small','DINOv2-L experiment':'https://huggingface.co/facebook/dinov2-large','SAM2.1 Tiny':'https://huggingface.co/facebook/sam2.1-hiera-tiny','TabICL v1.1':'https://huggingface.co/jingang/TabICL'}
(R/'pretrained_sources.json').write_text(json.dumps(sources,indent=2))
text=f'''V58: финальный кандидат

Файл: {out.name}
SHA256: {sha}
362 строки; id в исходном порядке; score конечные, [0,1], CSV канонизирован до 12 знаков после запятой. AP при таком округлении не меняется. Численные расхождения порядка 1e-16 от чтения CSV не влияют на итоговый CSV.

V52 Public 0.96300 и V56 A40 Public 0.96285 подтверждены в интерфейсе Kaggle.
V58 ещё не отправлен на момент сборки этого пакета.

OOF V52: {res.loc['V52','ap']:.12f}
OOF V57: {res.loc['V57','ap']:.12f}
OOF V58: {res.loc[name,'ap']:.12f}
День25: {res.loc[name,'2026-10-25']:.12f}; день26: {res.loc[name,'2026-10-26']:.12f}.
Пары: {old.sum()} -> {new.sum()}; исправлено {(old&~new).sum()}, новых {(~old&new).sum()}.
Исходные49: {json.dumps(summary['original_49'],ensure_ascii=False)}
43 test-гипотезы: {json.dumps(counts,ensure_ascii=False)}

Выбрано после {len(res)} записей сравнения ({sel['unique_prediction_pairs']} уникальных векторов OOF+test).
Все новые нейросетевые эксперименты выполнены на 2 Tesla T4 в Kaggle; локально — сборка и метрики.
Проверены: TabICL совместные SAM2-признаки и модели по классам (60 fits), исключение train-кадра (105 fits), нелинейные DINO-B/L головы (24), fine-tuning последнего блока DINO-S/B с RGB-аугментациями (2 pilot +12 трёх-seed обучений), 350/750 шагов, RGB-flip TTA, регрессия IoU, ограничения ранга/поправок/калибровки.
Отдельные кандидаты, веса и классы не являются независимыми экспериментами.
Jackknife исключал query-метки кадра, но сохранял фиксированный GT-банк обучающего дня; это проверка чувствительности, не новый независимый split.

Суть выбранного изменения: V57 сохраняется для person/hands. Для cup старая dense-компонента V31 заменяется DINOv2-B с обученной пространственной головой 768->64->64->4 (1x1/GELU/3x3/GELU/dropout.1/1x1), 450 шагов, BCE, реальные RGB flips. Среднее 3 seed и двух видов (original/flip) берётся до выделения компонент. OOF — противоположный день; test требует согласия двух дневных экспертов. Если нового надёжного референса нет, сохраняется предыдущий вклад. Tray сохраняет порядок V52, sigmoid(logit(V52)-.6). Общая поправка ограничена .15.

Ограничения: 198 размеченных query,35 кадров,2дня. Данные многократно использовались при выборе. Bootstrap условен на фиксированных прогнозах, не учитывает адаптивный выбор. Визуальные test-заметки никогда не участвуют в study.py и не являются разметкой. Public не известен до отправки.

Воспроизведение финального score из сохранённых экспертных прогнозов:
python replay_candidate.py
Ожидается EXACT_REPLAY_OK и SHA256 выше. Это сборка score, а не повторное извлечение признаков из RGB.
Загруженные шесть DINO-голов отдельно воспроизвели сохранённые карты с max abs error =0 (spatial_checkpoint_replay.json).
Полные источники обучения: ../gpu_dense.py, ../gpu_spatial_tta.py, ../gpu_experts.py, ../gpu_jackknife.py, ../gpu_finetune.py, ../study.py.
Предыдущая часть рецепта и веса: ../../rascar_v56/v57_correction_audit.zip и ../../rascar_v56/gpu_trained_models.zip.
Новые выбранные веса: ../models/vit_b_spatial_s*_fit*.pt. Претренированные веса — pretrained_sources.json.
Kaggle: https://www.kaggle.com/code/shapodev/rascar-v56-gpu-final-experiments
'''
(R/'README_RU.txt').write_text(text)
print(json.dumps({'selection':sel,'audit':summary},ensure_ascii=False,indent=2))
