from common56 import *
import zipfile,platform
from importlib.metadata import version
summary=[]
for p in (A/'results').glob('dense_*.csv'):
    d=pd.read_csv(p);print(p.name,d.head(10).to_string(index=False),flush=True);summary.append(d)
if summary:pd.concat(summary).sort_values('ap',ascending=False).to_csv(A/'results/gpu_summary.csv',index=False)
(A/'audit/runtime.json').write_text(json.dumps(dict(python=platform.python_version(),packages={p:version(p) for p in ['torch','transformers','tabicl','scikit-learn','numpy','pandas']}),indent=2))
with zipfile.ZipFile(ROOT/'v56_gpu_results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in A.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or '.cache' in p.parts:continue
        if p.suffix in ['.npy','.ckpt']:continue
        z.write(p,p.relative_to(ROOT))
print('RESULT_ZIP',str(ROOT/'v56_gpu_results.zip'),(ROOT/'v56_gpu_results.zip').stat().st_size,flush=True)
