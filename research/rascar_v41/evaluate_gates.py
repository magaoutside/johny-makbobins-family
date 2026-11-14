"""Four-family agreement gates; duplicate-prone query groups abstain.
Fixed thresholds express geometric/class confirmation, not per-test-ID fixes.
"""
from common import *
q,te,h=inputs();bo,bt=baseline(q,te)
text=pd.read_csv(A/'features/clip_text_predictions.csv');yb=pd.concat([pd.read_csv(A/f'features/yolo_fit{x}_predictions.csv') for x in ['25','26']]);geo={k:pd.read_csv(A/f'features/{k}_geometry_predictions.csv') for k in ['vit_b','vit_l']}
def inputs_for(rows,fit):
 t=text[text.fit_day.eq(fit)].set_index('id').loc[rows.id];y=yb[yb.fit_day.eq(fit)].set_index('id').loc[rows.id];b=geo['vit_b'][geo['vit_b'].fit_day.eq(fit)].set_index('id').loc[rows.id];l=geo['vit_l'][geo['vit_l'].fit_day.eq(fit)].set_index('id').loc[rows.id]
 return np.c_[t.presence,y.assigned_iou,y.reference_confidence,b.reference_iou,l.reference_iou]
o=np.zeros((len(q),5));ts=[]
for fit in DAYS:
 m=q.day.ne(fit);o[m]=inputs_for(q[m],fit);ts.append(inputs_for(te,fit))
test=np.stack(ts)
def crowded(rows):
 bad=np.zeros(len(rows),bool)
 for _,g in rows.groupby(['frame_id','class_name']):
  mat=iou(g[B],g[B]);np.fill_diagonal(mat,0);bad[g.index]=mat.max(1)>.5
 return bad
co,ct=crowded(q),crowded(te);results=[];changes=[]
for cut in [.6,.7]:
 og=(o[:,0]>=.75)&(o[:,1]>=cut)&(o[:,2]>=.35)&(o[:,3]>=cut)&(o[:,4]>=cut)&~co
 tg=np.all((test[:,:,0]>=.75)&(test[:,:,1]>=cut)&(test[:,:,2]>=.35)&(test[:,:,3]>=cut)&(test[:,:,4]>=cut),axis=0)&~ct
 ot=np.clip((.85-o[:,1:][:,[0,2,3]].mean(1))/.5,0,1)
 tt=np.clip((.85-test[:,:,[1,3,4]].mean((0,2)))/.5,0,1)
 for weight in [.25,.5]:
  name=f'four_agreement_iou{cut}_rescue{weight}';s=bo-weight*np.maximum(bo-ot,0)*og;t=bt-weight*np.maximum(bt-tt,0)*tg
  q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=t).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
  row=dict(experiment=name,**metrics(q,s),delta=metrics(q,s)['ap']-metrics(q,bo)['ap'],changed_test=int((abs(t-bt)>1e-8).sum()),changed_oof=int((abs(s-bo)>1e-8).sum()));results.append(row)
  ch=te.copy();ch['v31']=bt;ch['new_score']=t;ch['delta']=t-bt;ch['experiment']=name;changes.append(ch[abs(ch.delta)>1e-8])
pd.DataFrame(results).to_csv(A/'results/agreement_experiments.csv',index=False);pd.concat(changes).to_csv(A/'audit/agreement_changes.csv',index=False);print(pd.DataFrame(results).to_string(index=False))
