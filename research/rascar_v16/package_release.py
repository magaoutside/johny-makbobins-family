"""Bundle the runnable CPU replay, model caches, images and experiment ledger."""
from pathlib import Path
import hashlib,json,zipfile

A=Path(__file__).resolve().parent
paths=[p for p in sorted(A.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
       and p.suffix not in ['.zip','.pyc'] and p.name not in ['bundle_manifest.json','bundle_receipt.json','.DS_Store']]
manifest={str(p.relative_to(A)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
(A/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
out=A/'rascar_v16_code_data_metrics_submissions.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in paths+[A/'bundle_manifest.json']:z.write(p,'rascar_v16/'+str(p.relative_to(A)))
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
receipt={'archive':out.name,'bytes':out.stat().st_size,'files':len(paths)+1,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(A/'bundle_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
