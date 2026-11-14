from pathlib import Path
import zipfile,json,hashlib
A=Path(__file__).resolve().parent;root=A.parent
files=[]
for folder in [root/'rascar_v16',A]:
    for p in folder.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if p.suffix=='.zip' and p!=A/'gpu_payload.zip':continue
        if p.name in ['bundle_manifest.json','bundle_sha256.txt']:continue
        files.append(p)
manifest={'description':'V17 code, models, cached inputs, all experiment results, visual audit; V16 dependencies included. Duplicate transport archives omitted; originals retained locally.',
          'files':{str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)}}
mp=A/'bundle_manifest.json';mp.write_text(json.dumps(manifest,indent=2));files.append(mp)
dest=A/'rascar_v17_code_models_audit.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(files):z.write(p,str(p.relative_to(root)))
with zipfile.ZipFile(dest) as z:assert z.testzip() is None
sha=hashlib.sha256(dest.read_bytes()).hexdigest();(A/'bundle_sha256.txt').write_text(sha+'  '+dest.name+'\n')
print('BUNDLE_OK',len(files),dest.stat().st_size,sha,flush=True)
