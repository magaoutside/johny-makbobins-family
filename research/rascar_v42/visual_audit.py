from base42 import *
from PIL import Image,ImageDraw,ImageFont
q,te,h=inputs();bo,bt=baseline(q,te);rec=pd.read_csv(A/'inputs/records.csv')
preds={}
for name in ['tight','dual','dual_pool']:
 d=pd.read_csv(A/f'results/{name}_predictions_s4201.csv');preds[name]=d.groupby('id').score.mean()
viewer=json.loads((ROOT/'rascar_test_viewer/data.js').read_text().removeprefix('window.RASCAR_DATA = ').rstrip(';\n'));rank={x['id']:x['rank'] for x in viewer['detections']}
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',17);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
def render(rows,name):
 for off in range(0,len(rows),6):
  sheet=Image.new('RGB',(1280,1200),'#171b23');dr=ImageDraw.Draw(sheet)
  for k,(_,r) in enumerate(rows.iloc[off:off+6].iterrows()):
   x=k%2*640;y=k//2*400;b=r[B].to_numpy(float);rr=rec[rec.id.eq(r.id)].iloc[0];cb=rr[CB].to_numpy(float);gt=None
   if 'error' in r.index:
    gg=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy();gt=gg[np.argmax(iou(b,gg)[0])] if len(gg) else None
   lo=cb[:2];hi=cb[2:]
   if gt is not None:lo=np.minimum(lo,np.maximum(0,gt[:2]-10));hi=np.maximum(hi,np.minimum([1280,720],gt[2:]+10))
   im=Image.open(D/f'{r.frame_id}.jpg').convert('RGB').resize((1280,720)).crop(tuple(np.r_[lo,hi].astype(int)));s=min(620/im.width,305/im.height);im=im.resize((round(im.width*s),round(im.height*s)),Image.Resampling.BICUBIC);sheet.paste(im,(x+10,y+90));scale=np.tile([im.width/(hi[0]-lo[0]),im.height/(hi[1]-lo[1])],2);offset=np.tile([x+10,y+90],2)
   dr.rectangle(tuple((b-np.tile(lo,2))*scale+offset),outline='#ffe36b',width=3)
   if gt is not None:dr.rectangle(tuple((gt-np.tile(lo,2))*scale+offset),outline='#55f295',width=2)
   label=(f"GT error={int(r.error)}" if 'error' in r.index else f"test #{rank.get(r.id,'?')}")
   dr.text((x+10,y+4),f'{label} {r.class_name} {r.id}',font=font,fill='white')
   dr.text((x+10,y+29),' | '.join(f'{n}={preds[n].loc[r.id]:.3f}' for n in preds),font=font,fill='#dddddd')
   dr.text((x+10,y+54),'Yellow=query; green=official GT (validation only)',font=small,fill='#b3c2cf')
  sheet.save(A/f'audit/{name}_{off//6:02d}.jpg',quality=94)
t=te[te.id.map(rank).isin([41,53,63,64,67,77,78,82,87,117])].copy();t['rank']=t.id.map(rank);render(t.sort_values('rank'),'test_discussed')
v=q.copy();v['classifier_score']=v.id.map(preds['dual_pool']);bad=pd.concat([v[v.error.eq(0)].nlargest(6,'classifier_score'),v[v.error.eq(1)].nsmallest(6,'classifier_score')]);bad.to_csv(A/'audit/worst_official_cases.csv',index=False);render(bad,'official_failures')
print('AUDIT IMAGES DONE')
