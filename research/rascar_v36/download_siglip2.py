"""Download public weights only; no dataset images or credentials are sent."""
from pathlib import Path
import json
from huggingface_hub import HfApi,snapshot_download
p=Path(__file__).resolve().parent/'model_cache/siglip2-base-patch16-224';p.mkdir(parents=True,exist_ok=True)
info=HfApi(token=False).model_info('google/siglip2-base-patch16-224');print('revision',info.sha,flush=True)
(p/'source.json').write_text(json.dumps({'repo_id':info.id,'revision':info.sha},indent=2))
snapshot_download(info.id,revision=info.sha,token=False,local_dir=str(p),allow_patterns=['config.json','model.safetensors','preprocessor_config.json'])
print('Downloaded',flush=True)
