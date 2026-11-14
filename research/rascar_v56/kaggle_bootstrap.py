from pathlib import Path
import os,sys,json,time,subprocess,zipfile
A=Path(__file__).resolve().parent;ROOT=A.parent
(A/'logs').mkdir(exist_ok=True)
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HOME']='/kaggle/temp/huggingface';os.environ['DISABLE_KAGGLE_CACHE']='true'
subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','transformers==4.53.1','tokenizers==0.21.4','huggingface-hub==0.36.0','tabicl==2.2.0','scikit-learn==1.6.1','einops==0.8.1'],check=True)
import torch,pandas as pd,kagglehub
from huggingface_hub import snapshot_download,hf_hub_download
assert torch.cuda.device_count()>=2,torch.cuda.device_count()
print('GPU_READY',[(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())],flush=True)
raw=Path(kagglehub.competition_download('rascar-hack-case-1'));paths={p.stem:p for p in raw.rglob('*.jpg')}
if not paths:
    dest=ROOT/'raw_images';dest.mkdir(exist_ok=True)
    for p in ([raw] if raw.is_file() else raw.rglob('*.zip')):
        with zipfile.ZipFile(p) as z:z.extractall(dest)
    paths={p.stem:p for p in dest.rglob('*.jpg')}
frames=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv').frame_id.unique();D=ROOT/'rascar_data_full';D.mkdir(exist_ok=True)
for f in frames:
    assert f in paths,f
    target=D/f'{f}.jpg'
    if not target.exists():target.symlink_to(paths[f])
print('IMAGES_READY',len(frames),flush=True)
hf_hub_download('jingang/TabICL','tabicl-classifier-v1.1-20250506.ckpt',local_dir=A/'models',token=False)
jobs=[]
for name,repo,device in [('vit_b','facebook/dinov2-base','cuda:0'),('vit_l','facebook/dinov2-large','cuda:1')]:
    snapshot_download(repo,local_dir=ROOT/f'rascar_v41/model_cache/{name}',allow_patterns=['config.json','model.safetensors'],token=False)
    log=open(A/f'logs/dense_{name}.log','w');p=subprocess.Popen([sys.executable,'-u',str(A/'dense_backbone.py'),'--backbone',name,'--device',device],stdout=log,stderr=subprocess.STDOUT)
    jobs.append((name,p,log));print('STARTED',name,'pid',p.pid,'device',device,flush=True)
log=open(A/'logs/tabular_gpu.log','w');p=subprocess.Popen([sys.executable,'-u',str(A/'kaggle_tabular.py')],stdout=log,stderr=subprocess.STDOUT);jobs.append(('tabular',p,log))
(A/'job_pids.json').write_text(json.dumps({name:p.pid for name,p,_ in jobs}))
while any(p.poll() is None for _,p,_ in jobs):
    state={name:p.poll() for name,p,_ in jobs};(A/'gpu_status.json').write_text(json.dumps(dict(state=state,utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())),indent=2));print('STATUS',state,flush=True)
    time.sleep(30)
codes={name:p.returncode for name,p,_ in jobs};print('FINISHED',codes,flush=True)
for _,_,log in jobs:log.close()
subprocess.run([sys.executable,str(A/'kaggle_finish.py')],check=True)
if any(codes.values()):raise RuntimeError(codes)
