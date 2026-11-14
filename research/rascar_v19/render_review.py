from pathlib import Path
import sys,math,html
import pandas as pd,numpy as np
from PIL import Image,ImageDraw,ImageFont
A=Path(__file__).resolve().parent;B=['x1','y1','x2','y2'];name=sys.argv[1]
d=pd.read_csv(A/'results/reviews'/f'{name}.csv');folder=A/'galleries'/name;folder.mkdir(parents=True,exist_ok=True)
heads=pd.read_csv(A.parent/'rascar_v16/wevis_heads_references.csv');refs=pd.read_csv(A/'landmark_joint_refs_query_test.csv')
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',16);tiles=[];entries=[]
controls=[66,87,91,92,120,214,234,272,85,44,55,186,183,190,194,195,193,192,110,111,145,146,147,259,320,325,298,308]
connections=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
for r in d.itertuples():
    b=np.array([getattr(r,c) for c in B]);im=Image.open(A.parent/'rascar_v16/test_images'/f'{r.frame_id}.jpg').convert('RGB')
    pad=np.maximum((b[2:]-b[:2])*.65,70);roi=b+np.r_[-pad,pad];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720)
    crop=im.crop(tuple(roi));sf=min(455/crop.width,320/crop.height);crop=crop.resize((round(crop.width*sf),round(crop.height*sf)));dr=ImageDraw.Draw(crop)
    if r.class_name=='person':
        for bb in heads[heads.frame_id.eq(r.frame_id)][B].to_numpy():dr.rectangle(tuple((bb-np.tile(roi[:2],2))*sf),outline='#35aeff',width=2)
    if r.class_name=='nude_hands':
        rr=refs[refs.frame_id.eq(r.frame_id)&refs.views.ge(4)&refs.turns.ge(3)&refs.coherence.ge(.65)]
        for pp in rr.itertuples():
            xy=(np.array([[getattr(pp,f'kp{k}_x'),getattr(pp,f'kp{k}_y')] for k in range(21)])-roi[:2])*sf
            for u,v in connections:dr.line([tuple(xy[u]),tuple(xy[v])],fill='#00c943',width=2)
            for xx,yy in xy:dr.ellipse((xx-2,yy-2,xx+2,yy+2),fill='#00dafa')
    dr.rectangle(tuple((b-np.tile(roi[:2],2))*sf),outline='#ff2424',width=3)
    tile=Image.new('RGB',(470,390),'#f2f4f7');tile.paste(crop,((470-crop.width)//2,65));dr=ImageDraw.Draw(tile)
    dr.text((8,5),f'#{r.review_no} {r.class_name} | {r.score:.3f} -> {r.new_score:.3f}',font=font,fill='black');dr.text((8,29),str(r.issue)[:46],font=font,fill='black')
    fn=f'{int(r.review_no):03}.jpg';tile.save(folder/fn,quality=92)
    prior=r.diagnostic_flag in ['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'];changed=abs(r.delta)>1e-9
    if prior or changed or r.review_no in controls:tiles.append((r.review_no,tile))
    entries.append(f'<article class="case" data-priority="{int(prior)}" data-changed="{int(changed)}"><img loading="lazy" src="{fn}"><p>{html.escape(str(r.id))}<br>{html.escape(str(r.visual_status))}; {html.escape(str(r.note))}<br>V18 {r.score:.4f}; V19 {r.new_score:.4f}; Δ {r.delta:+.4f}</p></article>')
for start in range(0,len(tiles),12):
    sub=tiles[start:start+12];sheet=Image.new('RGB',(1880,390*math.ceil(len(sub)/4)),'white')
    for j,(_,tile) in enumerate(sub):sheet.paste(tile,((j%4)*470,(j//4)*390))
    sheet.save(folder/f'contact_{start//12+1:02}.jpg',quality=93)
pd.DataFrame(dict(review_no=[n for n,_ in tiles])).to_csv(folder/'contact_cases.csv',index=False)
(folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>RASCAR V19 audit</title><style>body{font:15px system-ui;background:#e6eaf0;margin:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}.case{background:white;padding:8px}.case img{width:100%}button{padding:8px;margin:8px}</style><h1>RASCAR V19 — '+html.escape(name)+'</h1><p>Красный: исходный бокс. Голубой: голова. Зелёные линии и бирюзовые точки: устойчивые 21 точки кисти MediaPipe. Высокий score означает ошибку бокса. Визуальные статусы субъективны, скрытой разметки теста нет.</p><button onclick="f(\'all\')">Все 362</button><button onclick="f(\'priority\')">41 приоритетный</button><button onclick="f(\'changed\')">Изменённые</button><div class="grid">'+''.join(entries)+'</div><script>function f(k){document.querySelectorAll(".case").forEach(e=>e.hidden=k!=="all"&&e.dataset[k]!=="1")}</script>')
print('GALLERY',len(d),'REVIEW',len(tiles),'SHEETS',math.ceil(len(tiles)/12))
