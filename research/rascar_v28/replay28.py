"""Reproduce the V28 research CSV from saved features and two trained heads."""
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
A=Path(__file__).resolve().parent
for split in ['oof','test']:
    x=pd.read_csv(A/f'features/features_{split}.csv').set_index('id')
    base=pd.read_csv(A/f'inputs/v27_{split}.csv').set_index('id').loc[x.index]
    predictions=[]
    for day in ['2026-10-25','2026-10-26']:
        b=joblib.load(A/f'models/group_coordinates_catboost_shallow_{day}.joblib')
        predictions.append(b['model'].predict_proba(x[b['columns']])[:,1])
    pred=np.mean(predictions,axis=0) if split=='test' else np.where(base.day.eq('2026-10-25'),predictions[1],predictions[0])
    score=.95*base.score.to_numpy()+.05*pred
    if split=='test':
        expected=pd.read_csv(A/'submission_v28_meta_ranker_research.csv').set_index('id').loc[x.index,'score'].to_numpy()
        assert np.max(abs(score-expected))<1e-12
        pd.DataFrame(dict(id=x.index,score=score)).to_csv(A/'replayed_submission_v28.csv',index=False)
    else:print('OOF AP',average_precision_score(base.error,score))
print('Replay matches released research CSV.')
