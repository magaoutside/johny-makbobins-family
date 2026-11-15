"""Train exactly the selected six CatBoost models; no parameter search."""
import argparse
from pathlib import Path
from rascar.core import ROOT, read_json, verify_artifacts
from rascar.prototype import train

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/trained_models')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    if args.out.resolve()==(ROOT/'models').resolve():parser.error('Use a separate output directory for retraining')
    verify_artifacts()
    train(args.data, args.out, read_json(ROOT / 'config.json'), args.threads)
    print(f'Модели и журнал обучения: {args.out}')

if __name__ == '__main__':
    main()
