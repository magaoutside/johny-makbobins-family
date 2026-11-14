"""Independent presence diagnostic, explicitly not competition error AP."""
from common import *
pool=pd.read_csv(ROOT/'rascar_v37/inputs/blind_pool.csv');labels=json.loads((ROOT/'rascar_v37/inputs/blind_review.json').read_text())
pool['target']=np.where(pool.review_index.isin(labels['present']),1,np.where(pool.review_index.isin(labels['absent']),0,np.nan));rows=[];preds=[]
for key in ['clip','vit_b','vit_l']:
 p=A/f'features/{key}_predictions_s4101.csv'
 if not p.exists():continue
 d=pd.read_csv(p);d=d[d.split.eq('pool_holdout')&d.variant.eq('finetuned')&d.day.ne(d.fit_day)].copy();d['id']=d.id.str.removeprefix('hold_');d=d.merge(pool[['id','target','review_index']],on='id');preds.append(d)
 use=d.target.notna();y=d.loc[use,'target'];s=d.loc[use,'presence']
 rows.append(dict(model=key,n=int(use.sum()),presence_ap=float(average_precision_score(y,s)),true_objects_flagged_p_lt_01=int(((y==1)&(s<.1)).sum()),false_objects_confirmed_p_gt_09=int(((y==0)&(s>.9)).sum()),note='Presence labels only; no geometry/duplicates; reused diagnostic set, not fresh independent competition validation.'))
 if use.any():
  for c,g in d[use].groupby('class_name'):
   rows.append(dict(model=key,class_name=c,n=len(g),positive=int(g.target.sum()),true_objects_flagged_p_lt_01=int((g.target.eq(1)&g.presence.lt(.1)).sum()),false_objects_confirmed_p_gt_09=int((g.target.eq(0)&g.presence.gt(.9)).sum())))
pd.DataFrame(rows).to_csv(A/'results/pool_presence.csv',index=False)
if preds:pd.concat(preds).to_csv(A/'audit/pool_presence_predictions.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
