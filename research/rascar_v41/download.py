"""Fetch public model weights only. Never uploads data or uses private tokens."""
from pathlib import Path
import json, hashlib, time
from huggingface_hub import HfApi, snapshot_download

A=Path(__file__).resolve().parent
MODELS={'clip':'openai/clip-vit-base-patch32','vit_b':'facebook/dinov2-base','vit_l':'facebook/dinov2-large'}
manifest=[]
for key, repo in MODELS.items():
    info=HfApi(token=False).model_info(repo)
    files=[x.rfilename for x in info.siblings]
    weights='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
    dest=A/'model_cache'/key
    print('DOWNLOAD',key,repo,info.sha,weights,flush=True)
    snapshot_download(repo,revision=info.sha,token=False,local_dir=str(dest),allow_patterns=[weights,'config.json','preprocessor_config.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt'],max_workers=3)
    row=dict(name=key,repo=repo,revision=info.sha,url=f'https://huggingface.co/{repo}',weights=weights,sha256=hashlib.sha256((dest/weights).read_bytes()).hexdigest())
    manifest.append(row)
    (A/'model_cache/sources.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',key,flush=True)
