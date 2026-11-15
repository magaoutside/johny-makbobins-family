"""Download pinned public foundation weights and validate their SHA256 digests."""
import argparse
from pathlib import Path
from rascar.core import ROOT, read_json, write_json, sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=['all'])
    p.add_argument('--out',type=Path,default=ROOT/'cache/weights')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    specs=read_json(ROOT/'weights.json')
    names=list(specs) if a.models==['all'] else a.models
    for name in names:
        if name not in specs: p.error(f'Unknown model {name}; choose from {list(specs)}')
        spec=specs[name]
        if 'patterns' in spec:
            marker=a.out/name/'download_manifest.json'
            if not a.check_only:
                from huggingface_hub import snapshot_download
                snapshot_download(spec['repository'],revision=spec['revision'],
                                  allow_patterns=spec['patterns'],local_dir=a.out/name)
                files={str(p.relative_to(a.out/name)):sha256(p) for p in (a.out/name).rglob('*')
                       if p.is_file() and '.cache' not in p.parts and p.name!='download_manifest.json'}
                write_json(marker,dict(repository=spec['repository'],revision=spec['revision'],files=files))
            saved=read_json(marker)
            if saved['revision']!=spec['revision'] or not saved['files']:
                raise ValueError(f'Wrong or incomplete snapshot: {name}')
            spec=dict(spec,files=saved['files'])
        for filename,digest in spec['files'].items():
            path=a.out/name/filename
            if not path.exists() and not a.check_only:
                if 'urls' in spec:
                    import requests
                    path.parent.mkdir(parents=True,exist_ok=True)
                    temp=path.with_suffix(path.suffix+'.partial')
                    with requests.get(spec['urls'][filename],stream=True,timeout=(30,60)) as response:
                        response.raise_for_status()
                        with temp.open('wb') as f:
                            for chunk in response.iter_content(1<<20):f.write(chunk)
                    if sha256(temp)!=digest:raise ValueError(f'Corrupt download: {filename}')
                    temp.replace(path)
                else:
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(spec['repository'],filename,revision=spec['revision'],local_dir=a.out/name)
            if not path.is_file() or sha256(path)!=digest:
                raise ValueError(f'Missing/corrupt model file: {path}')
        print(f'OK: {name} ({spec["repository"]}@{spec["revision"]})')

if __name__=='__main__': main()
