
from pathlib import Path
import sys,os,json,time,hashlib
os.environ['OMP_NUM_THREADS']='2';os.environ['WANDB_DISABLED']='true'
import numpy as np,pandas as pd,torch,yaml
from ultralytics import YOLO,settings
E=Path('/kaggle/working/rascar_v14');Y=E/'yolo26_clean';S=E/'stage5';ds=S/'dataset';mode,fit=sys.argv[1:3];gpu=int(sys.argv[3]);out=S/f'{mode}_{fit}';out.mkdir(exist_ok=True);torch.set_num_threads(2);settings.update({k:False for k in ['wandb','mlflow','comet','clearml','dvc','neptune'] if k in settings})
fitdays=['2026-10-25','2026-10-26'] if fit=='all' else [fit];fitpaths=[]
for day in fitdays:
 fitpaths+=(ds/f'clean_{day}.txt').read_text().splitlines()
 if mode=='augmented':fitpaths+=(ds/f'pool_{day}.txt').read_text().splitlines()
assert len(fitpaths)==len(set(fitpaths));lst=out/'train.txt';lst.write_text('\n'.join(fitpaths)+'\n');config={'path':str(ds),'train':str(lst),'val':str(lst),'names':{0:'cup',1:'tray'}};cfg=out/'data.yaml';cfg.write_text(yaml.safe_dump(config));(out/'training_manifest.json').write_text(json.dumps(dict(fitting_days=fitdays,training_images=fitpaths,epochs=80,seed=17,checkpoint='last epoch',pool_mode=mode,heldout_used_for_training_or_early_stopping=False),indent=2))
model=YOLO(str(Y/'yolo26s.pt'));model.train(data=str(cfg),epochs=80,imgsz=1280,batch=4,device=gpu,workers=2,seed=17,deterministic=True,optimizer='AdamW',lr0=.0003,lrf=.1,weight_decay=.0005,warmup_epochs=3,patience=0,val=False,amp=True,cache=False,mosaic=.25,close_mosaic=10,mixup=0.0,scale=.25,translate=.05,fliplr=.5,flipud=0.,save=True,save_period=40,plots=False,project=str(out),name='train',exist_ok=False,verbose=False)
checkpoint=out/'train/weights/last.pt';hist=pd.read_csv(out/'train/results.csv');assert len(hist)==80;model=YOLO(str(checkpoint));assert list(model.names.values())==['cup','tray']
if fit=='all':
 fr=pd.read_csv(E/'stage2/test_frames.csv');te=pd.read_csv('/kaggle/working/oof_validation_v11/test.csv');paths=[str(Y/'test_images'/r.image_path) for r in fr[fr.frame_id.isin(te.frame_id)].itertuples()]
else:
 held='2026-10-26' if fit.endswith('25') else '2026-10-25';paths=(ds/f'clean_{held}.txt').read_text().splitlines();assert set(paths).isdisjoint(fitpaths)
refs=[]
for result in model.predict(paths,imgsz=1280,batch=4,device=gpu,conf=.01,iou=.7,max_det=300,stream=True,verbose=False):
 fid=Path(result.path).stem;scale=np.array([1280/result.orig_shape[1],720/result.orig_shape[0]]*2)
 for j,(b,c,k) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy())):refs.append(dict(frame_id=fid,class_name=model.names[int(k)],confidence=float(c),reference_id=f'{mode}_{fit}_{fid}_{j}',**dict(zip(['x1','y1','x2','y2'],b*scale))))
pd.DataFrame(refs,columns=['frame_id','class_name','confidence','reference_id','x1','y1','x2','y2']).to_csv(out/'references.csv',index=False);print('POOL_TRAIN_DONE',mode,fit,len(fitpaths),hashlib.sha256(checkpoint.read_bytes()).hexdigest(),flush=True)
