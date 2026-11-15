"""Run the six released CatBoost models on the bundled competition features."""
import argparse
from pathlib import Path
from rascar.core import ROOT, queries, read_json, save_predictions, verify_artifacts
from rascar.prototype import predict

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--models', type=Path, default=ROOT / 'models')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    verify_artifacts()
    score = predict(args.data, args.models, read_json(ROOT / 'config.json'))
    path = save_predictions(queries(args.data), score, args.out)
    print(f'Готово: {path} (362 строки, id + score)')
    print(f'OOF: {args.out / "oof.csv"}')

if __name__ == '__main__':
    main()
