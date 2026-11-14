from pathlib import Path
import os,sys,subprocess,json,time,hashlib,zipfile,importlib.metadata
os.environ['DISABLE_KAGGLE_CACHE']='true';os.environ['TOKENIZERS_PARALLELISM']='false'
A=Path('/kaggle/working/rascar_v17')
def status(phase,**kw):
    (A/'gpu_status.json').write_text(json.dumps(dict(phase=phase,**kw),indent=2));print('V17_STATUS',phase,kw,flush=True)
try:
    status('install')
    subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.211'],check=True)
    import pandas as pd,kagglehub,torch
    from concurrent.futures import ThreadPoolExecutor
    frames=pd.read_csv(A/'frames.csv');status('download',frames=len(frames))
    def fetch(r):
        path=A/'images'/r.image_path
        if not path.is_file():path=Path(kagglehub.competition_download('rascar-hack-case-1',path=r.image_path,output_dir=str(A/'images')))
        return dict(frame_id=r.frame_id,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(fetch,frames.itertuples()))
    pd.DataFrame(paths).to_csv(A/'image_paths.csv',index=False)
    jobs=[];status('folds_training_and_tta')
    for fit,device in [('2026-10-25',0),('2026-10-26',1)]:
        log=open(A/f'gpu_{fit}.log','w');p=subprocess.Popen([sys.executable,str(A/'gpu_worker.py'),fit,str(device)],stdout=log,stderr=subprocess.STDOUT);jobs.append((fit,p,log))
    errors=[]
    for fit,p,log in jobs:
        code=p.wait();log.close()
        if code:errors.append(dict(fit=fit,returncode=code))
    if errors:raise RuntimeError(str(errors))
    status('all_training_and_tta')
    with open(A/'gpu_all.log','w') as log:subprocess.run([sys.executable,str(A/'gpu_worker.py'),'all','0'],stdout=log,stderr=subprocess.STDOUT,check=True)
    versions={name:importlib.metadata.version(name) for name in ['torch','ultralytics','numpy','pandas','Pillow','kagglehub']};(A/'gpu_runtime_versions.json').write_text(json.dumps(versions,indent=2))
    with zipfile.ZipFile(A/'gpu_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in A.rglob('*'):
            if p.is_file() and (p.name.startswith('tta_') or p.name in ['training_ledger.csv','manifest.json','gpu_runtime_versions.json','image_paths.csv'] or (p.suffix=='.log')):z.write(p,str(p.relative_to(A)))
    with zipfile.ZipFile(A/'refined_weights.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (A/'hand_refinement').glob('*/train/weights/last.pt'):z.write(p,str(p.relative_to(A)))
    status('done',results_bytes=(A/'gpu_results.zip').stat().st_size,weights_bytes=(A/'refined_weights.zip').stat().st_size)
except Exception as e:
    status('failed',error=type(e).__name__+': '+str(e));raise
