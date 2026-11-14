from pathlib import Path
import os,sys,json,subprocess
os.environ['DISABLE_KAGGLE_CACHE']='true'
import kagglehub
A=Path('/kaggle/working/rascar_v27')
raw=Path(kagglehub.competition_download('rascar-hack-case-1'))
image_paths={p.stem:str(p) for p in raw.rglob('*.jpg')}
print('AVAILABLE_IMAGES',len(image_paths),flush=True)
assert len(image_paths)>1000,len(image_paths)
(A/'image_paths.json').write_text(json.dumps(image_paths))
subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.211'],check=True)
subprocess.run([sys.executable,'-u',str(A/'gpu_temporal.py')],check=True)
