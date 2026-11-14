"""AP loss decomposition; manual targets only, no use of test annotations for fit."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent; A=ROOT/'rascar_v31'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,h,B,base_oof,current,iou,metric,align
from features23 import image
from PIL import Image,ImageDraw

d=q.copy();d['score']=base_oof
d=d.sort_values('score',ascending=False,kind='stable').reset_index(drop=True)
d['rank']=np.arange(1,len(d)+1);d['precision']=d.error.cumsum()/d['rank']
d['ap_loss']=d.error*(1-d.precision)/d.error.sum()
d['max_gt_iou']=[float(iou(r[B],h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B]).max(initial=0)) for _,r in d.iterrows()]
d['geometry_bucket']=np.select([d.max_gt_iou.ge(.6),d.max_gt_iou.ge(.3),d.max_gt_iou.gt(.01)],['eligible_reference','iou_0.3_to_0.6','iou_0.01_to_0.3'],default='no_overlap')
# For each correct box, exact AP gain if this score were moved below every error.
from sklearn.metrics import average_precision_score
ap=average_precision_score(d.error,d.score)
d['oracle_single_demote_gain']=0.
for j in d.index[d.error.eq(0)]:
 s=d.score.copy();s.iloc[j]=-1
 d.loc[j,'oracle_single_demote_gain']=average_precision_score(d.error,s)-ap
d.to_csv(A/'audit/oof_ap_loss.csv',index=False)
summary=d.groupby('class_name').agg(n=('id','size'),errors=('error','sum'),ap_loss=('ap_loss','sum'))
summary['share_of_total_ap_loss']=summary.ap_loss/summary.ap_loss.sum();summary.to_csv(A/'audit/ap_loss_by_class.csv')
out=dict(primary_metrics=metric(q,base_oof),loss_sum=float(d.ap_loss.sum()),ap_loss_identity_residual=float(abs(d.ap_loss.sum()-(1-ap))),by_class=summary.reset_index().to_dict('records'),top_error_queries=d[d.error.eq(1)].nlargest(8,'ap_loss')[['id','class_name','score','ap_loss','max_gt_iou']].to_dict('records'),top_correct_blockers=d.nlargest(8,'oracle_single_demote_gain')[['id','class_name','score','oracle_single_demote_gain','max_gt_iou']].to_dict('records'))
(A/'audit/baseline_diagnosis.json').write_text(json.dumps(out,indent=2))
chosen=pd.concat([d[d.error.eq(1)].nlargest(8,'ap_loss'),d.nlargest(8,'oracle_single_demote_gain')])
canvas=Image.new('RGB',(4*440,4*340),'#191919');draw=ImageDraw.Draw(canvas)
for k,(_,r) in enumerate(chosen.iterrows()):
 im=Image.fromarray(image(r.frame_id)[:,:,::-1]);b=r[B].to_numpy(float);wh=b[2:]-b[:2];lo=np.maximum(0,np.floor(b[:2]-np.maximum(wh*.5,35))).astype(int);hi=np.minimum([1280,720],np.ceil(b[2:]+np.maximum(wh*.5,35))).astype(int)
 im=im.crop((*lo,*hi));scale=min(430/im.width,270/im.height);im=im.resize((int(im.width*scale),int(im.height*scale)))
 x=(k%4)*440;y=(k//4)*340;canvas.paste(im,(x,y+55))
 def box(bb,color):
  c=(np.asarray(bb)-np.r_[lo,lo])*scale+np.array([x,y+55,x,y+55]);draw.rectangle(tuple(c),outline=color,width=2)
 for g in h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy():
  if iou(b,g)[0,0]>.01:box(g,'#38ff4c')
 box(b,'#ff4747');draw.text((x+4,y+4),f'{r.id} {r.class_name}',fill='white');draw.text((x+4,y+20),f'error={r.error} score={r.score:.3f} IoU={r.max_gt_iou:.3f}',fill='white');draw.text((x+4,y+36),f'loss={r.ap_loss:.5f} demote_gain={r.oracle_single_demote_gain:.5f}',fill='white')
canvas.save(A/'audit/oof_priority_gallery.jpg')
print(json.dumps(out,indent=2))
