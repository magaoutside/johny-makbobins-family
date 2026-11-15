"""Build the released submission; optionally rerun all selected trained heads."""
import argparse
from pathlib import Path
import numpy as np
from rascar.core import ROOT, queries, read_json, aligned_csv, save_predictions, verify_artifacts, DAYS
from rascar.prototype import predict as prototype_predict
from rascar.blend import assemble_scores

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['cached','models'], default='cached')
    p.add_argument('--data', type=Path, default=ROOT/'data')
    p.add_argument('--models', type=Path, default=ROOT/'models')
    p.add_argument('--cache', type=Path, default=ROOT/'cache/features')
    p.add_argument('--weights', type=Path, default=ROOT/'cache/weights')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, default=ROOT/'outputs')
    a=p.parse_args(); verify_artifacts()
    rows=queries(a.data)
    if rows.split.tolist()!=['oof']*198+['test']*362:
        raise ValueError('Expected 198 OOF queries followed by 362 test queries')
    base=prototype_predict(a.data,a.models,read_json(ROOT/'config.json'))
    previous=aligned_csv(a.data/'upstream_scores.csv',rows.id)
    evidence={}
    if a.mode=='cached':
        with np.load(a.data/'expert_predictions.npz',allow_pickle=False) as z:
            evidence={k:z[k] for k in z.files}
    if a.mode=='models':
        from rascar import tabular, spatial
        pred=tabular.run(a.data,a.models,a.weights/'tabicl/tabicl-classifier-v1.1-20250506.ckpt',a.device)
        tab=tabular.aggregate(pred,rows,read_json(a.models/'source_priors.json'))
        evidence.update(p_full=tab['full'],p_crop=tab['crop'],p_anchor_adapted=tab['anchor'])
        dense=spatial.predict(a.data,a.cache,a.models,a.device)
        evidence['new_dense_oof_iou']=dense[dense.split.eq('oof')].set_index('id').loc[rows.loc[rows.split.eq('oof'),'id'],'overlap'].to_numpy()
        evidence['new_dense_test_iou']=np.stack([dense[dense.split.eq('test')&dense.fit_day.eq(d)].set_index('id').loc[rows.loc[rows.split.eq('test'),'id'],'overlap'].to_numpy() for d in DAYS])
        a.out.mkdir(parents=True,exist_ok=True)
        pred.to_csv(a.out/'tabular_predictions.csv',index=False)
        dense.to_csv(a.out/'spatial_predictions.csv',index=False)
    evidence.update(baseline=base,class_name=rows.class_name.to_numpy(),v22=previous.guarded_baseline.to_numpy(),v31=previous.dense_baseline.to_numpy())
    result=save_predictions(rows,assemble_scores(evidence),a.out)
    print(f'Готово: {result} (режим {a.mode})')

if __name__=='__main__': main()
