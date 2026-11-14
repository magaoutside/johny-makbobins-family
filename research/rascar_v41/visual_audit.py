"""Render model predictions for inspection; creates no test training labels."""
from common import *
from PIL import Image,ImageDraw,ImageFont
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();prefix='final_' if args.final else ''
q,te,h=inputs();bo,bt=baseline(q,te)
viewer=json.loads((ROOT/'rascar_test_viewer/data.js').read_text().removeprefix('window.RASCAR_DATA = ').rstrip(';\n'))
rank={d['id']:d['rank'] for d in viewer['detections']}
d=pd.read_csv(A/'features/vit_l_expanded_predictions.csv');d=d[d.variant.eq('expanded_pool')]
score=pd.read_csv(A/'results/submissions/submission_vit_l_expanded_pool_person_blend0.2.csv').set_index('id').score
if args.final:
 d=pd.concat([pd.read_csv(A/f'features/vit_l_expanded_s{seed}_old_predictions.csv') for seed in [4142,4143,4144]])
 d=d.groupby(['id','fit_day'],as_index=False)[['reference_iou']+['ref_'+b for b in B]].mean()
 score=pd.read_csv(A/'submission_v41_pool_person_3seed.csv').set_index('id').score
t=te.copy();t['rank_v31']=t.id.map(rank);t['v31']=bt;t['candidate']=t.id.map(score);t['delta']=t.candidate-t.v31
t.to_csv(A/f'audit/{prefix}expanded_person_changes.csv',index=False)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
def sheet(rows,name):
 for start in range(0,len(rows),6):
  canvas=Image.new('RGB',(1280,1200),'#151920');draw=ImageDraw.Draw(canvas)
  for k,(_,r) in enumerate(rows.iloc[start:start+6].iterrows()):
   x=k%2*640;y=k//2*400;im=Image.open(D/f'{r.frame_id}.jpg').convert('RGB').resize((1280,720))
   b=r[B].to_numpy(float);refs=d[d.id.eq(r.id)];boxes=[b]+[rr[['ref_'+b for b in B]].to_numpy(float) for _,rr in refs.iterrows()];gtbox=None
   if 'error' in r.index:
    gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy()
    if len(gt):gtbox=gt[int(np.argmax(iou(b,gt)[0]))]
   valid=[b for b in boxes+([gtbox] if gtbox is not None else []) if np.isfinite(b).all()]
   lo=np.maximum(0,np.min([b[:2] for b in valid],axis=0)-35);hi=np.minimum([1280,720],np.max([b[2:] for b in valid],axis=0)+45)
   crop=im.crop(tuple(np.r_[lo,hi].astype(int)));scale=min(620/crop.width,320/crop.height);crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.BICUBIC);canvas.paste(crop,(x+10,y+75));sx=crop.width/(hi[0]-lo[0]);sy=crop.height/(hi[1]-lo[1])
   for j,box in enumerate(boxes):
    if np.isfinite(box).all():draw.rectangle(tuple((box-np.tile(lo,2))*[sx,sy,sx,sy]+[x+10,y+75,x+10,y+75]),outline=['#ffe066','#32d6ff','#ee77ff'][min(j,2)],width=3 if j==0 else 2)
   if gtbox is not None:draw.rectangle(tuple((gtbox-np.tile(lo,2))*[sx,sy,sx,sy]+[x+10,y+75,x+10,y+75]),outline='#5cff88',width=2)
   label=f"#{int(r.get('rank_v31',0))} {r.class_name} {r.id}"
   draw.text((x+10,y+3),label,font=font,fill='white')
   draw.text((x+10,y+26),f"V31 {r.v31:.4f} -> person-only {r.candidate:.4f} | delta {r.delta:+.4f}",font=font,fill='#dddddd')
   vals=' / '.join(f"fit{str(rr.fit_day)[-2:]} IoU={rr.reference_iou:.3f}" for _,rr in refs.iterrows())
   draw.text((x+10,y+51),('GREEN GT, error='+str(int(r.error))+'; ' if gtbox is not None else '')+('mean refs; ' if args.final else '')+vals,font=small,fill='#b0bbcc')
  canvas.save(A/f'audit/{prefix}{name}_{start//6:02d}.jpg',quality=94)
sheet(t[t.rank_v31.isin([41,53,63,64,67,77,78,82,87,117])].sort_values('rank_v31'),'discussed')
sheet(t.reindex(t.delta.abs().sort_values(ascending=False).index).head(12),'largest_test_changes')
o=pd.read_csv(A/('results/oof/v41_old_person_3seed.csv' if args.final else 'results/oof/vit_l_expanded_pool_person_blend0.2.csv')).set_index('id').loc[q.id].reset_index();o['v31']=bo;o['candidate']=o.score;o['delta']=o.score-bo
# Show all changed people with the biggest score changes, not cherry-picked successes.
sheet(o[o.class_name.eq('person')].reindex(o[o.class_name.eq('person')].delta.abs().sort_values(ascending=False).index).head(12),'validation_people')
o.to_csv(A/f'audit/{prefix}expanded_oof_changes.csv',index=False)
print('AUDIT RENDERED')
