"""Fetch the public checkpoint linked by the model author's hubconf.py.
Downloads only, does not import or execute repository code.
"""
from pathlib import Path
import urllib.request,ssl,certifi,json,hashlib
A=Path(__file__).resolve().parent;D=A/'model_cache/handseg';D.mkdir(parents=True,exist_ok=True);ctx=ssl.create_default_context(cafile=certifi.where())
urls={
 'model.py.txt':'https://raw.githubusercontent.com/guglielmocamporese/hands-segmentation-pytorch/master/model.py',
 'hubconf.py.txt':'https://raw.githubusercontent.com/guglielmocamporese/hands-segmentation-pytorch/master/hubconf.py',
 'checkpoint.ckpt':'https://drive.usercontent.google.com/download?id=1w7dztGAsPHD_fl_Kv_a8qHL4eW92rlQg&export=download&confirm=t'}
manifest=[]
for name,url in urls.items():
 p=D/name
 if not p.exists():
  print('DOWNLOAD',name,flush=True)
  with urllib.request.urlopen(url,context=ctx,timeout=60) as src,p.with_suffix(p.suffix+'.part').open('wb') as dst:
   if name.endswith('.ckpt') and 'text/html' in src.headers.get('Content-Type',''):raise RuntimeError('Checkpoint URL returned HTML, not weights')
   count=0
   while block:=src.read(1024*1024):
    count+=len(block);assert count<1024**3;dst.write(block)
  p.with_suffix(p.suffix+'.part').replace(p)
 manifest.append(dict(file=name,url=url,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(D/'manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',sum(x['bytes'] for x in manifest),flush=True)
