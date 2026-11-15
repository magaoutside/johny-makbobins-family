"""Recompute the linear occupancy stage, then rebuild its conservative blend."""
import argparse
from pathlib import Path
from rascar.core import ROOT
from rascar.dense_baseline import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--cache',type=Path,default=ROOT/'cache/features')
    p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/rebuilt_features')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--train',action='store_true')
    a=p.parse_args()
    if a.train and a.models.resolve()==(ROOT/'models').resolve():
        p.error('Pass --models outputs/trained_models to keep release weights unchanged')
    run(a.data,a.cache,a.models,a.out,a.device,a.train)
