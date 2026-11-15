"""Regenerate selected features from the original competition JPG/CSV files."""
import argparse
from pathlib import Path
from rascar.core import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['sam3','retrieval','workspace','metadata','semantic-records','semantic','geometry','roi','dense-small','pose','wevis-full','wevis-zoom','wevis-heads','wevis-hands'])
    p.add_argument('--raw',type=Path,default=ROOT/'cache/competition')
    p.add_argument('--detections',type=Path,default=ROOT/'data/detections')
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/rebuilt_features')
    p.add_argument('--weights',type=Path,default=ROOT/'cache/weights')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.stage=='sam3':
        from rascar.sam3 import extract
        extract(a.raw,a.weights,a.out,a.device)
    elif a.stage=='retrieval':
        from rascar.retrieval import extract
        extract(a.data,a.raw,a.weights,a.out,a.device)
    elif a.stage=='workspace':
        import shutil
        if a.out.resolve()==a.data.resolve():p.error('--out must differ from --data')
        shutil.copytree(a.data,a.out,dirs_exist_ok=True)
    elif a.stage=='semantic-records':
        from rascar.synthetic import build_records
        a.out.mkdir(parents=True,exist_ok=True)
        build_records(a.data,a.raw).to_csv(a.out/'dino_records.csv',index=False)
    elif a.stage=='semantic':
        from rascar.synthetic import extract
        extract(a.data,a.raw,a.weights,a.out,a.device)
    elif a.stage=='pose':
        from rascar.detectors import pose
        pose(a.data,a.raw,a.weights,a.out,a.device)
    elif a.stage.startswith('wevis-'):
        from rascar.detectors import wevis
        wevis(a.data,a.raw,a.weights,a.out,a.device,a.stage.split('-')[1])
    elif a.stage=='metadata':
        from rascar.records import prepare
        prepare(a.raw,a.out)
    elif a.stage=='geometry':
        from rascar.geometry import build
        build(a.data,a.detections,a.out)
    elif a.stage=='roi':
        from rascar.vision import extract_roi
        extract_roi(a.data,a.raw,a.weights,a.out,a.device)
    elif a.stage in ['dense','dense-small']:
        from rascar.vision import extract_dense
        extract_dense(a.data,a.raw,a.weights,a.out,a.device,small=a.stage=='dense-small')

if __name__=='__main__': main()
