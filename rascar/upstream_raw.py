"""Connect image-derived artifacts to each selected inherited-model stage."""
from pathlib import Path
import json,os,shutil,subprocess,sys
import pandas as pd
from .core import ROOT,queries,labels
from .records import prepare
from .sam3 import table


def metadata(raw,data,work):
    prepare(raw,data)
    rows=queries(data);rows['error']=labels(data,rows)
    train=rows[rows.split.eq('oof')].copy();test=rows[rows.split.eq('test')].drop(columns=['split','error'])
    manual=pd.read_csv(data/'manual_boxes.csv')
    fr=table(raw,'train_frames.csv');fr=fr[fr.frame_id.isin(manual.frame_id)].copy();fr['split']='train'
    ft=table(raw,'test_frames.csv');ft['split']='test'
    frames=pd.concat([fr,ft],ignore_index=True);frames['day']=pd.to_datetime(frames.timestamp,utc=True,format='mixed').dt.strftime('%Y-%m-%d')
    images={p.stem:p for p in raw.rglob('*.jpg')}
    paths=pd.DataFrame([dict(frame_id=fid,path=str(images[fid].resolve())) for fid in frames.frame_id if fid in images])
    needed=set(manual.frame_id)|set(rows.frame_id)
    if not needed<=set(paths.frame_id):raise FileNotFoundError(f'Missing {len(needed-set(paths.frame_id))} target images below {raw}')
    old=work/'local_evidence/previous/oof_validation_v11';old.mkdir(parents=True,exist_ok=True)
    for name in ['train.csv','test.csv','human_boxes_train.csv','train_frames.csv','scoring_policy.json']:
        files=list(raw.rglob(name))
        if len(files)!=1:raise ValueError(f'Expected one {name}')
        shutil.copy2(files[0],old/name)
    for stage in ['local_evidence','seat_and_tta','modes_and_sahi','kernel_quality','boundary_review']:
        dest=work/stage;dest.mkdir(parents=True,exist_ok=True);paths.to_csv(dest/'image_paths.csv',index=False)
        inp=dest/'inputs' if stage in ['modes_and_sahi','kernel_quality','boundary_review'] else dest
        inp.mkdir(exist_ok=True);train.drop(columns='split').to_csv(inp/'train_queries.csv',index=False)
        test.to_csv(inp/'test_queries.csv',index=False);manual.to_csv(inp/'manual_boxes.csv',index=False);frames.to_csv(inp/'frames.csv',index=False)
        if stage in ['modes_and_sahi','boundary_review']:
            machine=table(raw,'machine_predictions.csv');machine.to_csv(inp/'machine_predictions.csv',index=False)
    for stage in ['modes_and_sahi','boundary_review']:
        src=ROOT/'upstream/annotations'/stage
        if src.exists():
            # Only the separate reviewed annotations are fixed inputs. Pool
            # model scores and approved pseudo boxes are regenerated later.
            for p in src.glob('reviewed_pool_*.csv'):shutil.copy2(p,work/stage/'inputs'/p.name)
    (work/'boundary_review/roots.json').write_text(json.dumps(dict(images=str(raw),checkpoints=str(work/'modes_and_sahi/gpu_results'))))


def install_models(work,models,retrain=False):
    manifest=json.loads((ROOT/'upstream/model_paths.json').read_text())
    for name,relative in manifest.items():
        if retrain:continue
        source=Path(models)/name.split('::')[0]
        if not source.is_file():raise FileNotFoundError(f'Missing fitted checkpoint {source}')
        dest=work/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)


def foundation_bridge(data,work,foundation):
    target=work/'local_evidence';p=target/'previous';e=p/'rascar_v14'
    for folder in [target/'v8',e/'stage1/oof',e/'stage2']:folder.mkdir(parents=True,exist_ok=True)
    for split,name in [('oof','train'),('test','test')]:
        f=pd.read_csv(foundation/f'foundation_{split}.csv')
        if split=='oof':f[['id','frame_id','class_name','day','error','score']].to_csv(e/'stage1/oof/v8_baseline.csv',index=False)
        else:f[['id','score']].to_csv(p/'submission.csv',index=False)
        g=pd.read_csv(foundation/f'geometry_{split}.csv')
        if split=='oof':g[['id','frame_id','class_name','day','error','score']].to_csv(e/'stage2/catboost_person_hands0.5_yolo_cup_tray20_oof.csv',index=False)
        else:g[['id','score']].to_csv(p/'submission_catboost_person_hands0.5_yolo_cup_tray20.csv',index=False)
        shutil.copy2(foundation/f'raw_frozen_{name}_features.csv',e/'stage1'/f'raw_frozen_{name}_features.csv')
        shutil.copy2(foundation/f'fused_{name}_references.csv',target/'v8'/f'fused_{name}_references.csv')


def worker(work,stage,script,args,python,raw,weights,device):
    env=dict(os.environ,RASCAR_RAW=str(raw),RASCAR_WEIGHTS=str(weights),RASCAR_DEVICE=device,
             PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([str(python),str(work/stage/script),*map(str,args)],cwd=work/stage,env=env,check=True)
