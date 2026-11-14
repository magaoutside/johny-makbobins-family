"""New train-pool semantic audit chosen before SigLIP predictions.
No geometry/error labels are claimed. Review file is supplied by visual audit.
"""
from common36 import *
from PIL import Image,ImageDraw
import cv2

def prepare():
 p=pd.read_csv(ROOT/'rascar_data_full/machine_predictions.csv');p['day']=pd.to_datetime(p.frame_id.str[-15:-7],format='%Y%m%d').dt.strftime('%Y-%m-%d')
 old=pd.read_csv(ROOT/'rascar_v34/inputs/semantic_records.csv');excluded=set(old.frame_id)|set(h.frame_id)
 for f in ['rascar_v33/inputs/rare_hand_review_candidates.csv','rascar_v32/inputs/pool_control_records.csv']:
  excluded|=set(pd.read_csv(ROOT/f).frame_id)
 stamps=np.array([pd.to_datetime(f[-15:],format='%Y%m%d_%H%M%S').value//10**9 for f in excluded]);z=p[p.day.isin(DAYS)&p.class_name.eq('nude_hands')&~p.frame_id.isin(excluded)].copy()
 z['seconds']=pd.to_datetime(z.frame_id.str[-15:],format='%Y%m%d_%H%M%S').astype('int64')//10**9
 distance=abs(z.seconds.to_numpy()[:,None]-stamps).min(1);z=z.loc[distance>120]
 z['cx']=(z.x1+z.x2)/2;z['cy']=(z.y1+z.y2)/2;z['min_side']=np.minimum(z.x2-z.x1,z.y2-z.y1);z=z[z.min_side.between(14,85)]
 z['stratum']=np.where(z.confidence<.65,'lower_conf',np.where((z.cy>400)|(z.cx>1000),'periphery','standard'))
 z['bucket']=z.day+'_'+(z.seconds//600).astype(str)+'_'+(z.cx//120).astype(int).astype(str)+'_'+(z.cy//100).astype(int).astype(str)
 z['draw']=np.random.default_rng(3617).random(len(z));z=z.sort_values('draw').drop_duplicates('bucket').drop_duplicates('frame_id')
 sel=pd.concat([z[z.day.eq(d)&z.stratum.eq(s)].head(6) for d in DAYS for s in ['lower_conf','periphery','standard']],ignore_index=True).rename(columns={'detection_id':'id'})
 sel['review_index']=np.arange(len(sel));sel['split']='blind_pool';assert len(sel)==36 and not set(sel.frame_id)&excluded
 sel.to_csv(A/'inputs/blind_pool_candidates.csv',index=False)
 (A/'inputs/blind_pool_selection.json').write_text(json.dumps(dict(seed=3617,n=36,excluded_frames=len(excluded),minimum_seconds_from_previously_used_frames=120,selection_uses_model_predictions=False,test_days_excluded=True,label_scope='Hand presence only; not competition error or exact IoU.'),indent=2))
 for off in range(0,len(sel),12):
  rr=sel.iloc[off:off+12];canvas=Image.new('RGB',(1500,1140),'#202020');dr=ImageDraw.Draw(canvas)
  for j,(_,r) in enumerate(rr.iterrows()):
   b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2;lo=np.maximum(0,np.floor(cc-1.8*wh)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+1.8*wh)).astype(int)
   im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));roi=Image.fromarray(im[lo[1]:hi[1],lo[0]:hi[0],::-1]);scale=min(370/roi.width,325/roi.height);roi=roi.resize((int(roi.width*scale),int(roi.height*scale)))
   x=j%4*375;y=j//4*380;canvas.paste(roi,(x,y+52));box=(b-np.r_[lo,lo])*scale+[x,y+52,x,y+52];dr.rectangle(tuple(box),outline='#ff4444',width=2)
   dr.text((x+4,y+4),f'{r.review_index} {r.day[-2:]} {r.stratum} {r.frame_id[-6:]}',fill='white');dr.text((x+4,y+22),f'Native box minimum side {r.min_side/2:.0f}px',fill='white')
  canvas.save(A/f'audit/blind_pool_{off//12+1}.jpg')
 print('Prepared blind pool:',len(sel),flush=True)

if __name__=='__main__':prepare()
