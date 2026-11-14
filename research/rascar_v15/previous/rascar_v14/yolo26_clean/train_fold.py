from pathlib import Path
import os,sys,json,hashlib,time
os.environ['OMP_NUM_THREADS']='2';os.environ['WANDB_DISABLED']='true'
import numpy as np,pandas as pd,torch,ultralytics
from ultralytics import YOLO,settings
from sklearn.metrics import average_precision_score
from frozen_scoring import score_refs_v4
settings.update({k:False for k in ['wandb','mlflow','comet','clearml','dvc','neptune'] if k in settings})
torch.set_num_threads(2)
Y=Path(__file__).parent;D=Y/'official';ds=Y/'dataset'
fit_day=sys.argv[1];held_day='2026-10-26' if fit_day=='2026-10-25' else '2026-10-25';gpu=int(sys.argv[2])
plan=json.loads((Y/'experiment_plan.json').read_text());names=plan['classes']
fit_paths=(ds/f'day_{fit_day}.txt').read_text().splitlines();held_paths=(ds/f'day_{held_day}.txt').read_text().splitlines()
assert len(fit_paths)==len(held_paths)==42 and set(fit_paths).isdisjoint(held_paths)
assert all(Path(p).is_file() for p in fit_paths+held_paths)
for p in fit_paths:
 label=ds/'labels'/f'{Path(p).stem}.txt'
 for line in label.read_text().splitlines():assert len(line.split())==5
out=Y/f'fit_{fit_day}';out.mkdir(exist_ok=True)
(out/'fit_manifest.json').write_text(json.dumps({'fit_day':fit_day,'held_day':held_day,'fit_frames':[Path(p).stem for p in fit_paths],'held_frames':[Path(p).stem for p in held_paths],'epochs':plan['epochs'],'checkpoint_selection':'fixed last epoch; held labels not used for training, early stopping or checkpoint choice'},indent=2))
model=YOLO(str(Y/'yolo26s.pt'))
started=time.time()
model.train(data=str(ds/f'day_{fit_day}.yaml'),epochs=plan['epochs'],imgsz=plan['imgsz'],batch=4,device=gpu,workers=2,seed=17,deterministic=True,optimizer='AdamW',lr0=.0003,lrf=.1,weight_decay=.0005,warmup_epochs=3,patience=0,val=False,amp=True,cache=False,mosaic=.25,close_mosaic=10,mixup=0.0,scale=.25,translate=.05,fliplr=.5,flipud=0.0,save=True,save_period=20,plots=False,project=str(out),name='train',exist_ok=False,verbose=False)
checkpoint=out/'train/weights/last.pt';assert checkpoint.is_file()
history=pd.read_csv(out/'train/results.csv');history.columns=history.columns.str.strip()
assert len(history)==plan['epochs'],len(history)
trained=YOLO(str(checkpoint));assert list(trained.names.values())==names
# Held-day labels are accessed only after fixed-epoch training has completed.
metrics=trained.val(data=str(ds/f'day_{held_day}.yaml'),imgsz=plan['imgsz'],batch=4,device=gpu,workers=2,conf=.001,iou=.7,plots=False,project=str(out),name='heldout_detection',verbose=False)
refs=[]
for result in trained.predict(source=held_paths,imgsz=plan['imgsz'],batch=4,device=gpu,conf=.01,iou=.7,max_det=300,stream=True,verbose=False):
 fid=Path(result.path).stem
 for i,(box,confidence,cls) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy())):
  refs.append(dict(frame_id=fid,class_name=trained.names[int(cls)],confidence=float(confidence),reference_id=f'yolo26_{fit_day}_{fid}_{i}',x1=float(box[0]),y1=float(box[1]),x2=float(box[2]),y2=float(box[3])))
refs=pd.DataFrame(refs,columns=['frame_id','class_name','confidence','reference_id','x1','y1','x2','y2']);refs.to_csv(out/'heldout_references.csv',index=False)
train=pd.read_csv(D/'train.csv');held_ids={Path(p).stem for p in held_paths};queries=train[train.frame_id.isin(held_ids)].drop(columns='error').copy()
rank=score_refs_v4(refs[refs.confidence.ge(plan['reference_confidence'])],queries)
rank['fit_day']=fit_day;rank['day']=held_day
rank.to_csv(out/'heldout_scores.csv',index=False)
labels=train.set_index('id').loc[rank.id,'error'].to_numpy();ap=float(average_precision_score(labels,rank.score))
summary={'fit_day':fit_day,'held_day':held_day,'epochs_completed':len(history),'elapsed_seconds':time.time()-started,'checkpoint':'last.pt','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'ranking_ap':ap,'ranking_rows':len(rank),'detection_map50':float(metrics.box.map50),'detection_map50_95':float(metrics.box.map),'detection_precision':float(metrics.box.mp),'detection_recall':float(metrics.box.mr),'class_map50_95':{names[int(i)]:float(metrics.box.maps[int(i)]) for i in metrics.box.ap_class_index},'ultralytics':ultralytics.__version__}
(out/'fold_result.json').write_text(json.dumps(summary,indent=2));print('FOLD_FINISHED',json.dumps(summary),flush=True)
