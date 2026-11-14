import os,sys,json,hashlib,zipfile,subprocess,gzip,base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ['DISABLE_KAGGLE_CACHE']='true'
from kagglehub.clients import build_kaggle_client
from kagglehub.http_resolver import _build_notebook_download_request
from kagglehub.handle import parse_notebook_handle
def download_output(handle_name,remote,dest):
 req=_build_notebook_download_request(parse_notebook_handle(handle_name),remote)
 with build_kaggle_client() as client:r=client.kernels.kernels_api_client.download_kernel_output(req)
 r.raise_for_status();dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(r.content)
 return str(dest),len(r.content)
A=Path('/kaggle/working/rascar_v15');A.mkdir(exist_ok=True)
for ver,remote,local in [('shapodev/rascar-open-vocabulary-rule-based-audit/versions/11','oof_validation_v11/test.csv','test.csv'),('shapodev/rascar-open-vocabulary-rule-based-audit/versions/9','candidate_reproduction_bundle_v8.zip','v8_bundle.zip'),('shapodev/rascar-v14-geometry-quality-and-learned-ranking/versions/9','rascar_v14_stage2_full_reproduction.py','stage2_source.py'),('shapodev/rascar-v14-geometry-quality-and-learned-ranking/versions/9','rascar_local_data.py','local_data.py')]:
 print('AUDIT_RESTORE',download_output(ver,remote,A/local),flush=True)
with zipfile.ZipFile(A/'v8_bundle.zip') as z:
 assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
 z.extractall(A/'v8_replay')
 print('BUNDLE_PATHS',[n for n in z.namelist() if n.endswith('.csv') or n.endswith('.py')],flush=True)
import pandas as pd
test_audit=pd.read_csv(A/'test.csv');print('TEST_SCHEMA',test_audit.shape,test_audit.columns.tolist(),test_audit.head(1).to_dict('records'),flush=True)
print('DATA_SOURCE', (A/'local_data.py').read_text(),flush=True)
