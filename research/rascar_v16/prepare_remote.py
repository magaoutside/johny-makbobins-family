from pathlib import Path
import os,sys,json,zipfile,subprocess
os.environ['DISABLE_KAGGLE_CACHE']='true'
from kagglehub.clients import build_kaggle_client
from kagglehub.http_resolver import _build_notebook_download_request
from kagglehub.handle import parse_notebook_handle
import kagglehub,pandas as pd
from concurrent.futures import ThreadPoolExecutor
A=Path('/kaggle/working/rascar_v16');A.mkdir(exist_ok=True)
def download_output(handle,remote,dest):
 dest=Path(dest)
 if dest.is_file():return str(dest)
 req=_build_notebook_download_request(parse_notebook_handle(handle),remote)
 with build_kaggle_client() as client:r=client.kernels.kernels_api_client.download_kernel_output(req)
 r.raise_for_status();dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(r.content);return str(dest)
download_output('shapodev/rascar-v14-geometry-quality-and-learned-ranking/versions/9','rascar_v14_code_metrics_submissions.zip',A/'v14_bundle.zip')
with zipfile.ZipFile(A/'v14_bundle.zip') as z:
 assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(A/'previous')
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
print('DOWNLOAD_ERRORS',len(errors),flush=True)

print('V16_PREPARED',len(rows),'good',sum(r is not None for r in rows),flush=True)
