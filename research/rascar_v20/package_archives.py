from pathlib import Path
import zipfile,json,hashlib
A=Path(__file__).resolve().parent
excluded={'__pycache__','gpu_payload','replayed'}
paths=[p for p in A.rglob('*') if p.is_file() and not any(x in excluded for x in p.relative_to(A).parts) and p.suffix.lower() not in ['.zip','.pyc']]
manifest=[dict(path=str(p.relative_to(A)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths if p.name!='file_manifest.json']
(A/'file_manifest.json').write_text(json.dumps(manifest,indent=2))
paths=[p for p in paths if p.name!='file_manifest.json']+[A/'file_manifest.json']
for name,images in [('rascar_v20_reproducibility.zip',True),('v20_code_results_no_images.zip',False)]:
    with zipfile.ZipFile(A/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:
            if images or p.suffix.lower() not in ['.jpg','.jpeg','.png','.webp']:
                z.write(p,'rascar_v20/'+str(p.relative_to(A)))
    print(name,round((A/name).stat().st_size/1e6,2),'MB')
