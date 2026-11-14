from pathlib import Path
import os,json,hashlib,shutil
os.environ['DISABLE_KAGGLE_CACHE']='true'
import numpy as np,pandas as pd,kagglehub,yaml
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
Y=Path('/kaggle/working/yolo26_clean');D=Y/'official';ds=Y/'dataset'
(ds/'images').mkdir(parents=True,exist_ok=True);(ds/'labels').mkdir(exist_ok=True)
policy=json.loads((D/'scoring_policy.json').read_text());names=['person','cup','nude_hands','tray']
train=pd.read_csv(D/'train.csv');frames=pd.read_csv(D/'train_frames.csv');human=pd.read_csv(D/'human_boxes_train.csv')
frames['utc_day']=pd.to_datetime(frames.timestamp,utc=True,format='mixed').dt.strftime('%Y-%m-%d')
target=frames[frames.frame_id.isin(train.frame_id)].copy();assert len(target)==84
human['class_name']=human.class_name.replace(policy['aliases'])
assert set(human.frame_id)==set(target.frame_id)
clean=human[human.class_name.isin(names)].copy();coords=['cx','cy','w','h']
assert np.isfinite(clean[coords]).all().all() and clean.w.gt(0).all() and clean.h.gt(0).all()
xyxy=np.column_stack([clean.cx-clean.w/2,clean.cy-clean.h/2,clean.cx+clean.w/2,clean.cy+clean.h/2])
assert (xyxy>=-1e-6).all() and (xyxy<=1+1e-6).all(), 'Out-of-frame boxes need review'
dups=clean.duplicated(['frame_id','class_name']+coords);assert not dups.any(),'Duplicate annotations need review'
clean.to_csv(Y/'clean_training_boxes.csv',index=False)
def fetch(row):
 dest=ds/'images'/row.image_path
 if not dest.exists():
  path=Path(kagglehub.competition_download('rascar-hack-case-1',path=row.image_path,output_dir=str(D/'images')))
  shutil.copy2(path,dest)
 with Image.open(dest) as im:assert im.size==(int(row.image_width),int(row.image_height))
 return {'frame_id':row.frame_id,'day':row.utc_day,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
with ThreadPoolExecutor(max_workers=6) as pool:ledger=list(pool.map(fetch,target.itertuples(index=False)))
for row in target.itertuples(index=False):
 b=clean[clean.frame_id.eq(row.frame_id)]
 lines=[str(names.index(r.class_name))+' '+' '.join(f'{getattr(r,c):.9f}' for c in coords) for r in b.itertuples()]
 (ds/'labels'/f'{row.frame_id}.txt').write_text('\n'.join(lines)+'\n' if lines else '')
for day,g in target.groupby('utc_day'):
 image_list=ds/f'day_{day}.txt';image_list.write_text('\n'.join(str(ds/'images'/r.image_path) for r in g.itertuples())+'\n')
 # Training configuration only contains the training day, even in val field.
 config={'path':str(ds),'train':str(image_list),'val':str(image_list),'names':dict(enumerate(names))}
 (ds/f'day_{day}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
ledger=pd.DataFrame(ledger);assert set(ledger[ledger.day.eq('2026-10-25')].sha256).isdisjoint(ledger[ledger.day.eq('2026-10-26')].sha256)
ledger.to_csv(Y/'training_frame_ledger.csv',index=False)
joined=clean.merge(target[['frame_id','utc_day']],on='frame_id',validate='many_to_one')
summary={'frames':len(target),'boxes':len(clean),'revision_boxes':int(clean.source.eq('revision').sum()),'negative_frames':int((~target.frame_id.isin(clean.frame_id)).sum()),'classes':names,'by_day_class':joined.groupby(['utc_day','class_name']).size().to_dict(),'by_day_frames':target.utc_day.value_counts().to_dict(),'invalid_boxes':0,'duplicate_boxes':0,'machine_pool_used':False,'heldout_image_hash_overlap':0}
summary['by_day_class']={day+'/'+cls:int(n) for (day,cls),n in summary['by_day_class'].items()}
(Y/'data_quality.json').write_text(json.dumps(summary,indent=2));print('DATASET_READY',json.dumps(summary),flush=True)
