from pathlib import Path
import os,sys,subprocess,json,shutil,zipfile
os.environ['DISABLE_KAGGLE_CACHE']='true'
os.environ['TOKENIZERS_PARALLELISM']='false'
import kagglehub
A=Path('/kaggle/working/rascar_v22');A.mkdir(exist_ok=True)
def unpack(src,dest):
    with zipfile.ZipFile(src) as z:z.extractall(dest)
try:
    print('RESTORING_V20_CHECKPOINTS',flush=True)
    checkpoints=Path(kagglehub.dataset_download('shapodev/rascar-v20-trained-checkpoints'))
    old_inputs=Path(kagglehub.dataset_download('shapodev/rascar-v20-private-training-inputs'))
    roots={}
    for name,path in [('checkpoints',checkpoints),('old_inputs',old_inputs)]:
        zips=list(path.rglob('*.zip'))
        dest=A/name
        if zips:
            dest.mkdir(exist_ok=True)
            for p in zips:unpack(p,dest)
        else:dest=path
        roots[name]=str(dest)
        print('RESTORED',name,str(dest),[str(p.relative_to(dest)) for p in dest.rglob('last.pt')],flush=True)
    print('DOWNLOADING_COMPETITION_DATA',flush=True)
    raw=Path(kagglehub.competition_download('rascar-hack-case-1'))
    dest=A/'competition_data';dest.mkdir(exist_ok=True)
    if raw.is_file() and zipfile.is_zipfile(raw):unpack(raw,dest)
    elif raw.is_dir():
        jpg=list(raw.rglob('*.jpg'))
        if jpg:dest=raw
        for p in raw.rglob('*.zip'):unpack(p,dest)
    roots['images']=str(dest)
    (A/'roots.json').write_text(json.dumps(roots,indent=2))
    subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.211','sahi','transformers','scikit-learn','joblib'],check=True)
    import torch
    print('BOOTSTRAP_DONE',len(list(dest.rglob('*.jpg'))),'images','CUDA',torch.cuda.device_count(),flush=True)
except Exception as e:
    print('BOOTSTRAP_FAILED',type(e).__name__,str(e)[:1000],flush=True)
    raise
