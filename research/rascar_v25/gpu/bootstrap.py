from pathlib import Path
import os,sys,subprocess,json,zipfile
os.environ['DISABLE_KAGGLE_CACHE']='true'
os.environ['TOKENIZERS_PARALLELISM']='false'
A=Path('/kaggle/working/rascar_v25');A.mkdir(exist_ok=True)
import kagglehub
raw=Path(kagglehub.dataset_download('shapodev/rascar-v20-private-training-inputs'))
dest=A/'private_inputs';dest.mkdir(exist_ok=True)
for p in raw.rglob('*.zip'):
 with zipfile.ZipFile(p) as z:z.extractall(dest)
paths={p.stem:str(p) for root in [raw,dest] for p in root.rglob('*.jpg')}
print('PRIVATE_IMAGES',len(paths),flush=True)
if len(paths)<170:
 raw=Path(kagglehub.competition_download('rascar-hack-case-1'))
 if raw.is_file():
  with zipfile.ZipFile(raw) as z:z.extractall(dest)
 else:
  for p in raw.rglob('*.zip'):
   with zipfile.ZipFile(p) as z:z.extractall(dest)
 paths.update({p.stem:str(p) for root in [raw,dest] if root.is_dir() for p in root.rglob('*.jpg')})
(A/'image_paths.json').write_text(json.dumps(paths))
subprocess.run([sys.executable,'-m','pip','install','-q','rtmlib','onnxruntime-gpu','ultralytics==8.4.211'],check=True)
import torch,onnxruntime
print('BOOTSTRAP_DONE',len(paths),'CUDA',torch.cuda.device_count(),onnxruntime.get_available_providers(),flush=True)
