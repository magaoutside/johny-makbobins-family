"""Diagnostic ablations of the fixed nested ranker; not a leaderboard search."""
from common45 import *
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
torch.set_num_threads(3)
q,te,h=inputs();bo,bt=baseline(q,te);d=pd.read_csv(A/'features/nested_quality.csv')
allcols=[c for c in d if c not in ['id','outer_fit_day','role','day','frame_id','class_name','error','overlap']]
groups={'full':allcols,'no_position':[c for c in allcols if c not in ['cx','cy']], 'no_external_instances':[c for c in allcols if not c.startswith(('external_','anchors_','split_'))], 'metadata_only':[c for c in allcols if c.startswith('class_') and not c.startswith('class_prob_') or c in ['confidence','logw','logh','cx','cy']]}
results=[]
for name,cols in groups.items():
    output=[]
    for fit in DAYS:
        tr=d[d.outer_fit_day.eq(fit)&d.role.eq('meta_train')];ev=d[d.outer_fit_day.eq(fit)&d.role.ne('meta_train')];scale=StandardScaler().fit(tr[cols]);x=torch.tensor(scale.transform(tr[cols]),dtype=torch.float32);y=torch.tensor(tr.error.to_numpy(),dtype=torch.float32);torch.manual_seed(4503);m=nn.Linear(len(cols),1);nn.init.zeros_(m.weight);nn.init.zeros_(m.bias);opt=torch.optim.AdamW(m.parameters(),lr=.02,weight_decay=0.);p=np.flatnonzero(tr.error.to_numpy()==1);n=np.flatnonzero(tr.error.to_numpy()==0)
        for _ in range(500):
            z=m(x)[:,0];loss=F.binary_cross_entropy_with_logits(z,y)+.5*F.softplus(-(z[p,None]-z[None,n])).mean()+.15*m.weight.square().sum();opt.zero_grad();loss.backward();opt.step()
        with torch.inference_mode():pred=m(torch.tensor(scale.transform(ev[cols]),dtype=torch.float32))[:,0].sigmoid().numpy()
        output.append(ev[['id','role','outer_fit_day','overlap']].assign(pred=pred))
    pp=pd.concat(output);values=[]
    for split,r,base in [('oof',q,bo),('test',te,bt)]:
        parts=[g.set_index('id').loc[r.id] for _,g in pp[pp.role.eq(split)].groupby('outer_fit_day')] if split=='test' else [pp[pp.role.eq(split)].set_index('id').loc[r.id]]
        change=np.stack([g.pred.to_numpy()-np.clip((.85-g.overlap.to_numpy())/.5,0,1) for g in parts]);same=(change>=0).all(0)|(change<=0).all(0);ok=np.isfinite(change).all(0)&same&r.class_name.eq('person').to_numpy();delta=np.where((change>=0).all(0),np.min(change,0),np.max(change,0));new=base.copy();new[ok]=np.clip(base[ok]+.1*delta[ok],0,1)
        ix=np.flatnonzero(r.class_name.eq('person'));order=ix[np.argsort(new[ix],kind='stable')];new[order]=np.sort(base[ix]);values.append(new)
    oo,tt=values
    if name=='full':
        exact=pd.read_csv(A/'results/oof/nested_cross_class_rank_person_w10_scale_preserved.csv').set_index('id').loc[q.id,'score'];assert np.max(abs(exact-oo))<1e-12
    q.assign(score=oo).to_csv(A/f'results/oof/ablation_{name}.csv',index=False);te[['id']].assign(score=tt).to_csv(A/f'results/submissions/ablation_{name}.csv',index=False);results.append(dict(name=name,n_features=len(cols),**metrics(q,oo)))
pd.DataFrame(results).to_csv(A/'results/ablation_experiments.csv',index=False);print(pd.DataFrame(results).to_string(index=False))
