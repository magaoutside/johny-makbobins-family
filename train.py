"""Train selected models only; original release weights are never overwritten."""
import argparse
from pathlib import Path
from rascar.core import ROOT, read_json, verify_artifacts

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component',choices=['prototype','tabular','spatial','all'],default='prototype')
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--cache',type=Path,default=ROOT/'cache/features')
    p.add_argument('--weights',type=Path,default=ROOT/'cache/weights')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/trained_models')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--threads',type=int,default=3)
    a=p.parse_args(); verify_artifacts()
    if a.out.resolve()==(ROOT/'models').resolve():p.error('Use a separate output directory for retraining')
    if a.component in ['prototype','all']:
        from rascar.prototype import train
        train(a.data,a.out,read_json(ROOT/'config.json'),a.threads)
    if a.component in ['tabular','all']:
        from rascar.tabular import run
        run(a.data,a.out,a.weights/'tabicl/tabicl-classifier-v1.1-20250506.ckpt',a.device,fit=True)
    if a.component in ['spatial','all']:
        from rascar.spatial import train
        train(a.data,a.cache,a.out,a.device)
    print(f'Обучение завершено: {a.out}')

if __name__=='__main__': main()
