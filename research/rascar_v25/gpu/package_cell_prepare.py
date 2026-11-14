from pathlib import Path
import pandas as pd
A=Path(__file__).resolve().parent.parent
cups=pd.read_csv(A/'inputs/reviewed_pool_cups.csv');hands=pd.read_csv(A.parent/'rascar_v22/inputs/reviewed_pool_hands.csv');ids=sorted(set(cups.frame_id)|set(hands.frame_id))
code="""import zipfile,base64,hashlib,json,importlib.metadata as md
from IPython.display import HTML,display
from pathlib import Path
A=Path('/kaggle/working/rascar_v25')
assert all(p.poll()==0 for p in pose_jobs.values())
manifest={}
for root in [Path('/root/.cache/rtmlib'),A/'wilor_detector']:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ['.onnx','.pt','.json']:
   manifest[str(p)]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(A/'model_manifest.json').write_text(json.dumps(manifest,indent=2))
(A/'versions.json').write_text(json.dumps({k:md.version(k) for k in ['torch','rtmlib','ultralytics','onnxruntime-gpu','numpy']},indent=2))
paths=json.loads((A/'image_paths.json').read_text())
with zipfile.ZipFile(A/'v25_pose_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for kind in ['rtmw','wilor_detector']:
  for p in (A/kind).glob('*'):
   if p.suffix in ['.csv','.json']:z.write(p,str(p.relative_to(A)))
  z.write(A/(kind+'.log'),kind+'.log')
 for name in ['benchmark.py','queries.csv','persons.csv','versions.json','model_manifest.json']:z.write(A/name,name)
 for fid in POOL_IDS:
  if fid in paths:z.write(paths[fid],'pool_images/'+fid+'.jpg')
print('RESULTS_BYTES',(A/'v25_pose_results.zip').stat().st_size)
display(HTML('<a download="v25_pose_results.zip" href="data:application/zip;base64,'+base64.b64encode((A/'v25_pose_results.zip').read_bytes()).decode()+'">Download V25 pose results</a>'))
""".replace('POOL_IDS',repr(ids))
(A/'gpu/package_cell.py').write_text(code)
