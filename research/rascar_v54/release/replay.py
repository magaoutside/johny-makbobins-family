"""Replay V54 on bundled frozen test features; no labels or network required."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='submission_replayed.csv');args=parser.parse_args()
    root=Path(__file__).resolve().parent; manifest=json.loads((root/'manifest.json').read_text())
    base=pd.read_csv(root/'baseline.csv'); rows=pd.read_csv(root/'test_metadata.csv'); preds=[]
    assert np.array_equal(base.id,rows.id)
    for item in manifest['models']:
        features=pd.read_csv(root/item['features_file']).set_index('id').loc[rows.id,item['columns']]
        model=CatBoostClassifier(); model.load_model(str(root/item['weights_file']))
        preds.append(model.predict_proba(features)[:,1])
    p=np.mean(preds,axis=0);result=base.score.to_numpy().copy()
    ix=np.flatnonzero(rows.class_name.eq(manifest['scope']))
    order=np.argsort(result[ix]+manifest['alpha']*p[ix],kind='stable')
    result[ix[order]]=np.sort(result[ix])
    out=base[['id']].assign(score=result); assert out.id.is_unique and out.score.between(0,1).all()
    out.to_csv(args.output,index=False)
    expected=pd.read_csv(root/manifest['submission_file'])
    error=float(np.max(np.abs(expected.score-out.score)))
    assert error<1e-12,error
    print(json.dumps(dict(rows=len(out),max_replay_error=error,output=str(Path(args.output).resolve())),indent=2))


if __name__=='__main__':main()
