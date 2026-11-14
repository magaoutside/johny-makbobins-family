"""Download the two official BPJDet S checkpoints; no project data upload."""
from pathlib import Path
import urllib.request, json, hashlib, ssl, certifi
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'models'
TLS = ssl.create_default_context(cafile=certifi.where())
REPO = 'HoyerChou/BPJDet'
NAMES = ['bh_hand_s_1536_e100_best.pt', 'ch_head_s_1536_e150_best_mMR.pt']

def main():
    with urllib.request.urlopen('https://huggingface.co/api/models/'+REPO, context=TLS, timeout=40) as f:
        meta=json.load(f)
    rev=meta['sha']; available={x['rfilename'] for x in meta['siblings']}
    assert set(NAMES)<=available
    def fetch(name):
        path=DEST/name; url=f'https://huggingface.co/{REPO}/resolve/{rev}/{name}'
        if not path.exists():
            print('DOWNLOAD', name, flush=True)
            with urllib.request.urlopen(url, context=TLS, timeout=90) as src, path.with_suffix('.part').open('wb') as dst:
                while block:=src.read(1024*1024):dst.write(block)
            path.with_suffix('.part').replace(path)
        return dict(name=name, url=url, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=2) as pool:files=list(pool.map(fetch,NAMES))
    (DEST/'download_manifest.json').write_text(json.dumps(dict(repo=REPO,revision=rev,files=files),indent=2))
    print(json.dumps(files,indent=2),flush=True)

if __name__=='__main__':main()
