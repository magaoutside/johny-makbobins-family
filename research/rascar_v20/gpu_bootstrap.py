from pathlib import Path
import os,sys,subprocess,json,hashlib
os.environ['DISABLE_KAGGLE_CACHE']='true'
A=Path('/kaggle/working/rascar_v20');A.mkdir(exist_ok=True)
subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.211','sahi','transformers','scikit-learn','joblib'],check=True)
from kagglehub.clients import build_kaggle_client
from kagglehub.http_resolver import _build_notebook_download_request
from kagglehub.handle import parse_notebook_handle
def download_output(handle,remote,dest):
    dest=Path(dest)
    if dest.is_file():return
    req=_build_notebook_download_request(parse_notebook_handle(handle),remote)
    with build_kaggle_client() as client:r=client.kernels.kernels_api_client.download_kernel_output(req)
    r.raise_for_status();dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(r.content)
    print('RESTORED',str(dest),dest.stat().st_size,flush=True)
for fit in ['2026-10-25','2026-10-26']:
    download_output('shapodev/rascar-v14-geometry-quality-and-learned-ranking/versions/9',f'rascar_v14/yolo26_clean/fit_{fit}/train/weights/last.pt',A/f'weights/multiclass_{fit}.pt')
download_output('shapodev/rascar-v14-geometry-quality-and-learned-ranking/versions/3','rascar_v14/yolo26_clean/final_train/weights/last.pt',A/'weights/multiclass_all.pt')
from ultralytics import YOLO
for p in (A/'weights').glob('multiclass*.pt'):
    m=YOLO(p);print('MODEL',p.name,m.names,flush=True)
print('BOOTSTRAP_DONE',flush=True)
