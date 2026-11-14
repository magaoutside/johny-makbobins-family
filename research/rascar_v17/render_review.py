"""Persist a full 362-row diagnostic gallery, plus contact sheets for changed cases."""
from pathlib import Path
import sys,math,html
import pandas as pd,numpy as np
from PIL import Image,ImageDraw,ImageFont
A=Path(__file__).resolve().parent;B=['x1','y1','x2','y2']
name=sys.argv[1];d=pd.read_csv(A/'results/reviews'/f'{name}.csv');folder=A/'galleries'/name;folder.mkdir(parents=True,exist_ok=True)
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',16) if Path('/System/Library/Fonts/Helvetica.ttc').exists() else ImageFont.load_default()
refs={c:pd.read_csv(A/f'tta_{c}_refs_test.csv') for c in ['v15','refined'] if (A/f'tta_{c}_refs_test.csv').exists()}
heads=pd.read_csv(A.parent/'rascar_v16/wevis_heads_references.csv')
tiles=[];entries=[]
for r in d.itertuples():
    b=np.array([getattr(r,c) for c in B]);im=Image.open(A.parent/'rascar_v16/test_images'/f'{r.frame_id}.jpg').convert('RGB')
    pad=np.maximum((b[2:]-b[:2])*.6,70);roi=b+np.r_[-pad,pad];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720)
    crop=im.crop(tuple(roi));scale=min(450/crop.width,320/crop.height);crop=crop.resize((round(crop.width*scale),round(crop.height*scale)))
    draw=ImageDraw.Draw(crop)
    display=[]
    if r.class_name=='person':display=[(heads[heads.frame_id.eq(r.frame_id)],'#35aeff')]
    if r.class_name=='nude_hands':display=[(rr[rr.frame_id.eq(r.frame_id)&rr.views.ge(3)&rr.turns.ge(2)&rr.coherence.ge(.65)],color) for c,color in [('v15','#f4b42c'),('refined','#00e15c')] for rr in [refs.get(c,pd.DataFrame(columns=['frame_id','views','turns','coherence']+B))]]
    for rr,color in display:
        for bb in rr[B].to_numpy():draw.rectangle(tuple((bb-np.tile(roi[:2],2))*scale),outline=color,width=2)
    draw.rectangle(tuple((b-np.tile(roi[:2],2))*scale),outline='#ff2424',width=3)
    tile=Image.new('RGB',(470,390),'#f2f4f7');tile.paste(crop,((470-crop.width)//2,65));dr=ImageDraw.Draw(tile)
    dr.text((8,5),f'#{r.review_no} {r.class_name} | {r.score:.3f} -> {r.new_score:.3f}',fill='black',font=font)
    dr.text((8,28),str(r.issue)[:45],fill='black',font=font)
    cropname=f'{int(r.review_no):03}.jpg';tile.save(folder/cropname,quality=91)
    prior=r.diagnostic_flag in ['clear_error_low_score','likely_error_low_score','plausible_correct_high_score']
    changed=abs(r.delta)>.02
    if prior or changed:tiles.append((r.review_no,tile))
    entries.append(f'<article class="case" data-priority="{int(prior)}" data-changed="{int(changed)}"><img loading="lazy" src="{cropname}"><p>{html.escape(str(r.id))}<br>{html.escape(str(r.visual_status))}; {html.escape(str(r.diagnostic_flag))}<br>V15 {r.v15_score:.4f}; V16 {r.score:.4f}; V17 {r.new_score:.4f}; Δ {r.delta:+.4f}</p></article>')
for start in range(0,len(tiles),12):
    subset=tiles[start:start+12];sheet=Image.new('RGB',(470*4,390*math.ceil(len(subset)/4)),'white')
    for i,(_,tile) in enumerate(subset):sheet.paste(tile,((i%4)*470,(i//4)*390))
    sheet.save(folder/f'contact_{start//12+1:02}.jpg',quality=93)
(folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>RASCAR V17 audit</title><style>body{font:15px system-ui;background:#e6eaf0;margin:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}.case{background:white;padding:8px;break-inside:avoid}.case img{width:100%}button{padding:8px;margin:8px}</style><h1>RASCAR V17 — '+html.escape(name)+'</h1><p>Красный: исходный бокс. Голубой: голова. Жёлтый: прежний YOLO с TTA. Зелёный: дообученный YOLO с TTA. Чем выше score, тем вероятнее ошибка бокса. Визуальные статусы субъективны и не являются скрытой разметкой теста.</p><button onclick="filterCases(\'all\')">Все 362</button><button onclick="filterCases(\'priority\')">41 приоритетный</button><button onclick="filterCases(\'changed\')">Изменение &gt;0.02</button><div class="grid">'+''.join(entries)+'</div><script>function filterCases(k){document.querySelectorAll(".case").forEach(e=>e.hidden=k!=="all"&&e.dataset[k]!=="1")}</script>')
print('GALLERY',len(d),'CONTACT_CASES',len(tiles),'SHEETS',math.ceil(len(tiles)/12),folder)
