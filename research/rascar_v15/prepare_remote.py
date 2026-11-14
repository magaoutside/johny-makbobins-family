from pathlib import Path
import os,sys,json,zipfile,subprocess
os.environ['DISABLE_KAGGLE_CACHE']='true'
from kagglehub.clients import build_kaggle_client
from kagglehub.http_resolver import _build_notebook_download_request
from kagglehub.handle import parse_notebook_handle
import kagglehub,pandas as pd
from concurrent.futures import ThreadPoolExecutor
A=Path('/kaggle/working/rascar_v15');A.mkdir(exist_ok=True)
def download_output(handle,remote,dest):
 dest=Path(dest)
 if dest.is_file():return str(dest)
 req=_build_notebook_download_request(parse_notebook_handle(handle),remote)
 with build_kaggle_client() as client:r=client.kernels.kernels_api_client.download_kernel_output(req)
 r.raise_for_status();dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(r.content);return str(dest)
download_output('shapodev/rascar-v14-geometry-quality-and-learned-ranking/versions/9','rascar_v14_code_metrics_submissions.zip',A/'v14_bundle.zip')
with zipfile.ZipFile(A/'v14_bundle.zip') as z:
 assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(A/'previous')
v7=download_output('shapodev/rascar-open-vocabulary-rule-based-audit/versions/8','sam31_v7/candidate_reproduction_bundle.zip',A/'v7.zip')
with zipfile.ZipFile(v7) as z:
 assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(A/'v7')
print('DINO_MODEL', (A/'v7/sam31_v7/dino_b_manifest.json').read_text(),flush=True)
for fit in ['2026-10-25','2026-10-26']:
 download_output('shapodev/rascar-v14-geometry-quality-and-learned-ranking/versions/9',f'rascar_v14/yolo26_clean/fit_{fit}/train/weights/last.pt',A/f'weights/{fit}.pt')
download_output('shapodev/rascar-v14-geometry-quality-and-learned-ranking/versions/3','rascar_v14/yolo26_clean/final_train/weights/last.pt',A/'weights/all.pt')
O=A/'previous/oof_validation_v11';S=A/'previous/rascar_v14/stage2'
frames=pd.concat([pd.read_csv(O/'train_frames.csv'),pd.read_csv(S/'test_frames.csv')]).drop_duplicates('frame_id')
targets=set(pd.read_csv(O/'human_boxes_train.csv').frame_id)|set(pd.read_csv(O/'test.csv').frame_id)
frames=frames[frames.frame_id.isin(targets)]
errors=[]
def fetch(r):
 try:
  p=A/'images'/r.image_path
  if not p.is_file():p=Path(kagglehub.competition_download('rascar-hack-case-1',path=r.image_path,output_dir=str(A/'images')))
  return dict(frame_id=r.frame_id,path=str(p))
 except Exception as e:errors.append(dict(frame_id=r.frame_id,error=str(e)[:300]));return None
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,frames.itertuples()))
pd.DataFrame([r for r in rows if r]).to_csv(A/'image_paths.csv',index=False)
(A/'download_errors.json').write_text(json.dumps(errors,indent=2))
assert not errors,errors[:2]
subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.211'],check=True)
print('V15_PREPARED',len(rows),flush=True)
