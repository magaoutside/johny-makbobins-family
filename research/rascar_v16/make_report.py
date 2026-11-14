from pathlib import Path
import json,html,base64,sys
import pandas as pd,numpy as np
from PIL import Image,ImageDraw
from io import BytesIO
A=Path(__file__).resolve().parent;R=A/'results';name=sys.argv[1]
metrics=pd.DataFrame([json.loads(p.read_text()) for p in R.glob('*.json')])
metrics.sort_values('pooled_ap',ascending=False).to_csv(A/'all_experiment_metrics.csv',index=False)
d=pd.read_csv(R/'reviews'/f'{name}.csv');d['rank_before']=d.score.rank(method='min',ascending=False).astype(int);d['rank_after']=d.new_score.rank(method='min',ascending=False).astype(int)
d.to_csv(A/'selected_test_audit.csv',index=False)
reg=(d.visual_status.isin(['clear_error','likely_error'])&(d.delta<-.02))|(d.visual_status.eq('plausible_correct')&(d.delta>.02))
d[reg].to_csv(A/'regression_review.csv',index=False)
priority=d.diagnostic_flag.fillna('').ne('')|d.issue.eq('person_torso_only')|d.issue.str.contains('duplicate')|(d.delta.abs()>.07)|reg
cards=[]
for r in d[priority].sort_values('review_no').itertuples():
    im=Image.open(A/'test_images'/f'{r.frame_id}.jpg').convert('RGB');b=np.array([r.x1,r.y1,r.x2,r.y2]);pad=np.maximum((b[2:]-b[:2])*.5,50)
    roi=b+np.r_[-pad,pad];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720)
    crop=im.crop(tuple(roi));scale=min(440/crop.width,320/crop.height);crop=crop.resize((round(crop.width*scale),round(crop.height*scale)))
    ImageDraw.Draw(crop).rectangle(tuple((b-np.tile(roi[:2],2))*scale),outline='red',width=2);buf=BytesIO();crop.save(buf,format='JPEG',quality=90)
    uri='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
    cards.append(f'<article data-class="{r.class_name}" data-status="{r.visual_status}"><h3>#{r.review_no} {r.class_name}</h3><img src="{uri}" alt="Query {r.review_no}"><p><strong>{r.score:.3f} → {r.new_score:.3f}</strong> · rank {r.rank_before} → {r.rank_after}</p><p>{html.escape(r.issue)} · {r.visual_status}</p><p>{html.escape(r.note)}</p><small>{r.id}</small></article>')
table=metrics[['experiment','pooled_ap','day_25','day_26','ap_person','ap_nude_hands','test_visual_diagnostic_ap_NOT_validation','test_regressions_gt002']].sort_values('pooled_ap',ascending=False).to_html(index=False,float_format=lambda v:f'{v:.6f}',border=0)
selected_names=['baseline_lb092822','final_targeted_short','final_targeted_short_with_contradiction','final_targeted_with_contradiction']
selected=metrics.set_index('experiment').loc[selected_names,['pooled_ap','day_25','day_26','ap_person','ap_nude_hands','ap_cup','ap_tray']].to_html(float_format=lambda v:f'{v:.6f}',border=0)
narrative=html.escape((A/'SUMMARY.txt').read_text()) if (A/'SUMMARY.txt').exists() else ''
report=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>RASCAR V16</title><style>body{{font:16px/1.5 system-ui;margin:28px;background:#f4f6fa;color:#182230}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}}article{{background:white;border-radius:12px;padding:16px}}article img{{max-width:100%;height:320px;object-fit:contain}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #dde}}.table{{overflow:auto}}small{{color:#567}}select{{padding:10px;margin:15px 12px 15px 0}}.notice{{background:#fff3cf;padding:18px;border-radius:10px}}pre{{white-space:pre-wrap;font:15px/1.55 system-ui}}</style>
<h1>RASCAR V16: {html.escape(name)}</h1><p class="notice">База — подтверждённый пользователем LB 0.92822 / OOF 0.9752788. Новые варианты ещё не отправлены на LB. Все изменения на фото показаны относительно этой базы. Тестовый визуальный аудит субъективен и используется при разработке; его AP не равна скрытой метрике. Метки тестового аудита не подавались в обучение.</p>
<h2>Сравнение с базой</h2><div class="table">{selected}</div><details open><summary>Результаты и ограничения</summary><pre>{narrative}</pre></details><details><summary>Все {len(metrics)} сохранённых экспериментов, включая отклонённые варианты</summary><div class="table">{table}</div></details><h2>Проблемные фото и регрессии</h2><p>{len(cards)} примеров; полный аудит всех 362 рамок — selected_test_audit.csv. Выше score — более вероятная ошибка входной рамки.</p>
<select id="cls"><option value="">Все классы</option><option>person</option><option>nude_hands</option><option>cup</option><option>tray</option></select><select id="status"><option value="">Все статусы</option><option>clear_error</option><option>likely_error</option><option>plausible_correct</option><option>uncertain</option></select><div class="cards">{''.join(cards)}</div><script>const cs=document.getElementById('cls'),ss=document.getElementById('status');for(const s of document.querySelectorAll('select'))s.onchange=()=>{{for(const c of document.querySelectorAll('article'))c.hidden=(cs.value&&cs.value!==c.dataset.class)||(ss.value&&ss.value!==c.dataset.status);}};</script></html>'''
(A/'REPORT.html').write_text(report)
print('REPORT',len(metrics),'experiments',len(cards),'cases',int(reg.sum()),'regressions')
