"""Public official pretrained weights only; dataset images never uploaded."""
from pathlib import Path
import os,json,hashlib
A=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(A/'model_cache/hf');os.environ['HF_HUB_CACHE']=str(A/'model_cache/hf/hub');os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1';os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi,snapshot_download
def main():
    p=A/'model_cache/owlv2';p.mkdir(parents=True,exist_ok=True);repo='google/owlv2-base-patch16-ensemble';info=HfApi(token=False).model_info(repo)
    snapshot_download(repo,revision=info.sha,token=False,local_dir=str(p),allow_patterns=['config.json','preprocessor_config.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','tokenizer.json','model.safetensors'])
    manifest=dict(repo=repo,url='https://huggingface.co/'+repo,revision=info.sha,files=[dict(name=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in p.iterdir() if f.is_file()]);(p/'source.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
