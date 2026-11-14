from pathlib import Path
import os,json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score as AP
A=Path(os.environ.get('RASCAR_V15_ROOT',Path(__file__).resolve().parent));P=A/'previous';E=P/'rascar_v14';O=P/'oof_validation_v11'
B=['x1','y1','x2','y2'];CL=['person','nude_hands','cup','tray'];DAYS=['2026-10-25','2026-10-26']
q=pd.read_csv(E/'stage2/catboost_person_hands0.5_yolo_cup_tray20_oof.csv').merge(pd.read_csv(O/'train.csv')[['id','confidence']+B],on='id',validate='one_to_one')
te=pd.read_csv(O/'test.csv');current=pd.read_csv(P/'submission_catboost_person_hands0.5_yolo_cup_tray20.csv').set_index('id').loc[te.id,'score'].to_numpy()
v8=pd.read_csv(E/'stage1/oof/v8_baseline.csv').set_index('id').loc[q.id,'score'].to_numpy();v8t=pd.read_csv(P/'submission.csv').set_index('id').loc[te.id,'score'].to_numpy()
fr=pd.read_csv(O/'train_frames.csv');fr['day']=pd.to_datetime(fr.timestamp,utc=True,format='mixed').dt.strftime('%Y-%m-%d')
h=pd.read_csv(O/'human_boxes_train.csv');h.class_name=h.class_name.replace(json.loads((O/'scoring_policy.json').read_text())['aliases']);h=h.merge(fr[['frame_id','day']],on='frame_id',validate='many_to_one')
h[B]=np.c_[(h.cx-h.w/2)*1280,(h.cy-h.h/2)*720,(h.cx+h.w/2)*1280,(h.cy+h.h/2)*720]
def iou(a,b):
 a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
 inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
 return inter/np.maximum(np.maximum(a[:,2:]-a[:,:2],0).prod(1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(1)[None]-inter,1e-9)
def metric(rows,s):
 out={'pooled_ap':float(AP(rows.error,s))}
 for d,g in rows.groupby('day'):out['day_'+d[-2:]]=float(AP(g.error,np.asarray(s)[g.index]))
 for c,g in rows.groupby('class_name'):out['ap_'+c]=float(AP(g.error,np.asarray(s)[g.index]))
 return out
def register(name,s,t,details=None):
 s=np.asarray(s,float);t=np.asarray(t,float)
 assert len(s)==198 and len(t)==362 and np.isfinite(s).all() and np.isfinite(t).all()
 assert ((s>=0)&(s<=1)).all() and ((t>=0)&(t<=1)).all()
 R=A/'results';R.mkdir(exist_ok=True);(R/'oof').mkdir(exist_ok=True);(R/'submissions').mkdir(exist_ok=True)
 q.assign(score=s).to_csv(R/'oof'/f'{name}.csv',index=False)
 fn=R/'submissions'/f'submission_{name}.csv';te[['id']].assign(score=t).to_csv(fn,index=False)
 row=dict(experiment=name,**metric(q,s),details=details or {},sha256=hashlib.sha256(fn.read_bytes()).hexdigest(),public_lb=None)
 if (A/'test_diagnostic_frozen.csv').exists():
  d=pd.read_csv(A/'test_diagnostic_frozen.csv').set_index('id').loc[te.id].reset_index();d['new_score']=t;d['delta']=t-current
  m=d.visual_status.isin(['clear_error','plausible_correct'])&~d.revised_after_scores
  row['test_visual_diagnostic_ap_NOT_validation']=float(AP(d.loc[m,'visual_status'].eq('clear_error'),t[m]))
  for key,m in [('clear_false_person',d.class_name.eq('person')&d.visual_status.eq('clear_error')&(d.score<.5)),('short_person',d.issue.eq('person_torso_only')),('plausible_hand_high',d.class_name.eq('nude_hands')&d.visual_status.eq('plausible_correct')&(d.score>=.8))]:
   row[key+'_n']=int(m.sum());row[key+'_mean_delta']=float(d.loc[m,'delta'].mean()) if m.any() else None
  d.to_csv(R/f'{name}_test_review.csv',index=False)
 (R/f'{name}.json').write_text(json.dumps(row,indent=2));print('RESULT',name,round(row['pooled_ap'],6),round(row['day_25'],6),round(row['day_26'],6),flush=True)
 return row
