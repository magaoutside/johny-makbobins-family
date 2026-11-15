"""Selected four-class YOLO and shallow geometry ranker, before hand specialists."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .core import queries, labels, BBOX
from .geometry import reference_features
from .foundation_scoring import score_refs_v4
DAYS=['2026-10-25','2026-10-26']
NAMES=['person','cup','nude_hands','tray']


def detector(data,raw,weights,models,out,device='cuda:0',train=False):
    import torch,yaml
    from ultralytics import YOLO,settings
    torch.set_num_threads(2)
    settings.update({k:False for k in ['wandb','mlflow','comet','clearml','dvc','neptune'] if k in settings})
    data,raw,weights,models,out=map(Path,(data,raw,weights,models,out));out.mkdir(parents=True,exist_ok=True)
    rows=queries(data);h=pd.read_csv(data/'manual_boxes.csv');files={p.stem:p for p in raw.rglob('*.jpg')}
    for fit in DAYS+['all']:
        checkpoint=models/f'generic_{fit}.pt'
        fit_frames=pd.read_csv(next(raw.rglob('train_frames.csv')))
        fit_frames=fit_frames[fit_frames.frame_id.isin(h.frame_id)].copy()
        fit_frames['day']=pd.to_datetime(fit_frames.timestamp,utc=True,format='mixed').dt.strftime('%Y-%m-%d')
        if fit!='all':fit_frames=fit_frames[fit_frames.day.eq(fit)]
        if train:
            ds=out/'training'/fit; (ds/'images').mkdir(parents=True,exist_ok=True);(ds/'labels').mkdir(exist_ok=True)
            for r in fit_frames.itertuples():
                shutil.copy2(files[r.frame_id],ds/'images'/f'{r.frame_id}.jpg')
                gt=h[h.frame_id.eq(r.frame_id)&h.class_name.isin(NAMES)]
                lines=[str(NAMES.index(b.class_name))+' '+' '.join(f'{getattr(b,c):.9f}' for c in ['cx','cy','w','h']) for b in gt.itertuples()]
                (ds/'labels'/f'{r.frame_id}.txt').write_text('\n'.join(lines)+'\n' if lines else '')
            listing=ds/'train.txt';listing.write_text('\n'.join(str((ds/'images'/f'{fid}.jpg').resolve()) for fid in fit_frames.frame_id)+'\n')
            config=ds/'data.yaml';config.write_text(yaml.safe_dump(dict(path=str(ds.resolve()),train=str(listing.resolve()),val=str(listing.resolve()),names=dict(enumerate(NAMES)))))
            model=YOLO(str(weights/'detector/yolo26s.pt'))
            model.train(data=str(config),epochs=80,imgsz=1280,batch=4,device=device,workers=2,seed=17,deterministic=True,optimizer='AdamW',lr0=.0003,lrf=.1,weight_decay=.0005,warmup_epochs=3,patience=0,val=False,amp=True,cache=False,mosaic=.25,close_mosaic=10,mixup=0.,scale=.25,translate=.05,fliplr=.5,flipud=0.,save=True,save_period=20,plots=False,project=str(ds),name='fit',exist_ok=False,verbose=False)
            models.mkdir(parents=True,exist_ok=True);shutil.copy2(ds/'fit/weights/last.pt',checkpoint)
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        model=YOLO(str(checkpoint))
        target=rows[rows.split.eq('test')] if fit=='all' else rows[rows.split.eq('oof')&rows.day.ne(fit)]
        # Frame order follows the official frame table, as in the original run.
        frame_table=pd.read_csv(next(raw.rglob('test_frames.csv' if fit=='all' else 'train_frames.csv')))
        frame_table=frame_table[frame_table.frame_id.isin(target.frame_id)]
        refs=[]
        for r in model.predict(source=[str(files[fid]) for fid in frame_table.frame_id],imgsz=1280,batch=4,device=device,conf=.01,iou=.7,max_det=300,stream=True,verbose=False):
            fid=Path(r.path).stem
            for j,(b,c,cls) in enumerate(zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy())):
                refs.append(dict(frame_id=fid,class_name=model.names[int(cls)],confidence=float(c),reference_id=f'generic_{fit}_{fid}_{j}',**dict(zip(BBOX,b))))
        pd.DataFrame(refs,columns=['frame_id','class_name','confidence','reference_id']+BBOX).to_csv(out/f'generic_{fit}.csv',index=False)


def geometry(data,foundation,detections,models,out,train=False):
    from catboost import CatBoostClassifier
    data,foundation,detections,models,out=map(Path,(data,foundation,detections,models,out));out.mkdir(parents=True,exist_ok=True)
    rows=queries(data);rows['error']=labels(data,rows)
    q=rows[rows.split.eq('oof')].reset_index(drop=True);t=rows[rows.split.eq('test')].reset_index(drop=True)
    base={split:pd.read_csv(foundation/f'foundation_{split}.csv').set_index('id').loc[r.id].reset_index() for split,r in [('oof',q),('test',t)]}
    features={}
    for split,name,r in [('oof','train',q),('test','test',t)]:
        x=reference_features(r,base[split],pd.read_csv(foundation/f'fused_{name}_references.csv'),pd.read_csv(detections/'zoom_boxes.csv'))
        # Preserve the historical class-column order for the fitted ranker.
        cols=list(x.columns);i=cols.index('class_person');cols[i:i+4]=['class_'+c for c in NAMES];x=x[cols]
        features[split]=x
        pd.concat([r[['id']],x],axis=1).to_csv(out/f'raw_frozen_{name}_features.csv',index=False)
    oof=np.zeros(len(q));test=[]
    for fit in DAYS:
        tr=q.day.eq(fit);va=~tr;path=models/f'geometry_{fit}.cbm'
        m=CatBoostClassifier(iterations=250,depth=2,learning_rate=.03,l2_leaf_reg=10,random_seed=17,verbose=False,allow_writing_files=False,thread_count=2)
        if train:
            m.fit(features['oof'].loc[tr],q.loc[tr,'error'].astype(int));models.mkdir(parents=True,exist_ok=True);m.save_model(str(path))
        else:m.load_model(str(path))
        oof[va]=m.predict_proba(features['oof'].loc[va])[:,1];test.append(m.predict_proba(features['test'])[:,1])
    for split,r,pred in [('oof',q,oof),('test',t,np.mean(test,axis=0))]:
        score=.5*base[split].score.to_numpy()+.5*pred
        detector_scores=np.zeros(len(r))
        for fit in (DAYS if split=='oof' else ['all']):
            mask=r.day.ne(fit) if split=='oof' else np.ones(len(r),bool)
            refs=pd.read_csv(detections/f'generic_{fit}.csv')
            detector_scores[mask]=score_refs_v4(refs[refs.confidence.ge(.25)],r.loc[mask]).set_index('id').loc[r.loc[mask,'id'],'score']
        cup=r.class_name.isin(['cup','tray'])
        score[cup]=.8*base[split].score.to_numpy()[cup]+.2*detector_scores[cup]
        r.assign(score=score).drop(columns='split').to_csv(out/f'geometry_{split}.csv',index=False)
