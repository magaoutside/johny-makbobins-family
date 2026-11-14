"""Download public, non-executable DINOv2 weights from the official HF repo."""
from pathlib import Path
import urllib.request,json,hashlib,ssl,certifi
A=Path(__file__).resolve().parent;D=A/'model_cache/dinov2-small';D.mkdir(parents=True,exist_ok=True)
TLS=ssl.create_default_context(cafile=certifi.where())
with urllib.request.urlopen('https://huggingface.co/api/models/facebook/dinov2-small',timeout=30,context=TLS) as r:meta=json.load(r)
revision=meta['sha'];files=[]
for name in ['config.json','preprocessor_config.json','model.safetensors']:
    path=D/name;url=f'https://huggingface.co/facebook/dinov2-small/resolve/{revision}/{name}'
    if not path.exists():
        print('DOWNLOAD',name,flush=True)
        with urllib.request.urlopen(url,timeout=60,context=TLS) as src,path.with_suffix(path.suffix+'.part').open('wb') as dst:
            while block:=src.read(1024*1024):dst.write(block)
        path.with_suffix(path.suffix+'.part').replace(path)
    files.append(dict(name=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(D/'manifest.json').write_text(json.dumps(dict(repository='facebook/dinov2-small',revision=revision,files=files),indent=2))
print('DONE',sum(x['bytes'] for x in files),revision,flush=True)
