from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'model_cache/mpl'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parent/'model_cache/yolo_config'))
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
A=Path(__file__).resolve().parent;ROOT=A.parent;D=ROOT/'rascar_data_full'
B=['x1','y1','x2','y2'];CB=['crop_x1','crop_y1','crop_x2','crop_y2'];CL=['person','nude_hands','cup','tray'];DAYS=['2026-10-25','2026-10-26']
def day(fid):return pd.to_datetime(fid.split('_')[-2],format='%Y%m%d').strftime('%Y-%m-%d')
def iou(a,b):
 a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
 z=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
 return z/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-z,1e-9)
def inputs():
 q=pd.read_csv(D/'train.csv');q['class_name']=q.class_name.replace({'nude_hand':'nude_hands'});q=q[q.class_name.isin(CL)].copy().reset_index(drop=True)
 te=pd.read_csv(D/'test.csv');h=pd.read_csv(D/'human_boxes_train.csv');h['class_name']=h.class_name.replace({'nude_hand':'nude_hands'})
 for col,expr in [('x1',(h.cx-h.w/2)*1280),('y1',(h.cy-h.h/2)*720),('x2',(h.cx+h.w/2)*1280),('y2',(h.cy+h.h/2)*720)]:h[col]=expr
 for d in [q,te,h]:d['day']=d.frame_id.map(day)
 return q,te,h
def baseline(q,te):
 p=ROOT/'rascar_v31/submission_v31_dense_oof098784.csv'
 assert hashlib.sha256(p.read_bytes()).hexdigest()=='8720eb732639532bab49eff75b956a6bd4db21b96ede309ed901889777f4ab2c'
 v=pd.read_csv(ROOT/'rascar_v31/results/oof/linear_mask_cut0.5_all_two_sided_w0.25.csv').set_index('id').loc[q.id]
 assert np.array_equal(v.error,q.error)
 return v.score.to_numpy(),pd.read_csv(p).set_index('id').loc[te.id,'score'].to_numpy()
def metrics(d,s):
 out={'ap':float(average_precision_score(d.error,s))}
 for col in ['day','class_name']:
  for k in sorted(d[col].unique()):
   m=d[col].eq(k);out[k]=float(average_precision_score(d.loc[m,'error'],np.asarray(s)[m]))
 return out
