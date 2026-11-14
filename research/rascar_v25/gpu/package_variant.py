import zipfile,base64,hashlib,json,importlib.metadata as md
from IPython.display import HTML,display
from pathlib import Path
A=Path('/kaggle/working/rascar_v25')
print('VARIANT_EXIT',variant_job.poll()); print((A/'rtmw_variants.log').read_text()[-1200:]); assert variant_job.poll()==0
manifest={}
for root in [Path('/root/.cache/rtmlib'),A/'wilor_detector']:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ['.onnx','.pt','.json']:
   manifest[str(p)]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(A/'model_manifest.json').write_text(json.dumps(manifest,indent=2))
(A/'versions.json').write_text(json.dumps({k:md.version(k) for k in ['torch','rtmlib','ultralytics','onnxruntime-gpu','numpy']},indent=2))
paths=json.loads((A/'image_paths.json').read_text())
with zipfile.ZipFile(A/'v25_rtmwx_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for kind in ['rtmwx_bgr','rtmwx_rgb']:
  for p in (A/kind).glob('*'):
   if p.suffix in ['.csv','.json']:z.write(p,str(p.relative_to(A)))
  pass
 for name in ['benchmark.py','queries.csv','persons.csv','versions.json','model_manifest.json']:z.write(A/name,name)
 for fid in ['GENERAL_mishka_1_5_20261025_075300', 'GENERAL_mishka_1_5_20261025_081800', 'GENERAL_mishka_1_5_20261025_085630', 'GENERAL_mishka_1_5_20261025_085800', 'GENERAL_mishka_1_5_20261025_100800', 'GENERAL_mishka_1_5_20261025_100830', 'GENERAL_mishka_1_5_20261025_100900', 'GENERAL_mishka_1_5_20261025_101030', 'GENERAL_mishka_1_5_20261025_111800', 'GENERAL_mishka_1_5_20261025_111900', 'GENERAL_mishka_1_5_20261025_115500', 'GENERAL_mishka_1_5_20261025_125000', 'GENERAL_mishka_1_5_20261025_125100', 'GENERAL_mishka_1_5_20261025_125200', 'GENERAL_mishka_1_5_20261025_125230', 'GENERAL_mishka_1_5_20261025_141030', 'GENERAL_mishka_1_5_20261025_172730', 'GENERAL_mishka_1_5_20261025_172800', 'GENERAL_mishka_1_5_20261026_075200', 'GENERAL_mishka_1_5_20261026_134200', 'GENERAL_mishka_1_5_20261026_153100', 'GENERAL_mishka_1_5_20261026_163830', 'GENERAL_mishka_1_5_20261026_175300']:
  if fid in paths:z.write(paths[fid],'pool_images/'+fid+'.jpg')
print('RESULTS_BYTES',(A/'v25_rtmwx_results.zip').stat().st_size)
display(HTML('<a download="v25_rtmwx_results.zip" href="data:application/zip;base64,'+base64.b64encode((A/'v25_rtmwx_results.zip').read_bytes()).decode()+'">Download V25 RTMWX results</a>'))
