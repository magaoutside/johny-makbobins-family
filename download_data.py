"""Download the authorized competition archive, or unpack an existing local ZIP."""
import argparse, shutil, zipfile
from pathlib import Path
from rascar.core import ROOT

def unpack(path,out):
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            target=(out/name).resolve()
            if not target.is_relative_to(out.resolve()):
                raise ValueError(f'Unsafe archive path: {name}')
        z.extractall(out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'cache/competition')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.archive:
        source=a.archive
    else:
        import kagglehub
        source=Path(kagglehub.competition_download('rascar-hack-case-1'))
    if source.is_file(): unpack(source,a.out)
    elif source.resolve()!=a.out.resolve(): shutil.copytree(source,a.out,dirs_exist_ok=True)
    for path in list(a.out.rglob('*.zip')): unpack(path,path.parent)
    print(f'Competition files: {a.out}; images: {len(list(a.out.rglob("*.jpg")))}')

if __name__=='__main__': main()
