from common import *
from scipy.special import expit
def hypotheses(rows,sem,fit=None):
 res=[]
 for r in rows.itertuples():
  if r.class_name!='person':res.append((0.,0.));continue
  pool=h[h.class_name.eq('person')&(True if fit is None else h.day.eq(fit))];b=np.array([getattr(r,c) for c in B]);bs=pool[B].to_numpy();wh=bs[:,2:]-bs[:,:2];qw,qh=b[2:]-b[:2]
  d=((bs[:,0]+bs[:,2]-b[0]-b[2])/2/1280)**2+((bs[:,1]-b[1])/720)**2
  d=d+(.025*np.log(wh[:,0]/qw))**2;ix=np.argsort(d)[:5]
  if np.sqrt(d[ix[0]])>.07:res.append((0.,0.));continue
  weights=np.exp(-d[ix]/(2*.04**2));weights/=weights.sum();risk=float(np.sum(weights*expit((.6-qh/wh[ix,1])/.08)));target=float(np.sum(weights*wh[ix,1]))
  presence=float(sem.loc[r.id,'p_person']);risk*=np.clip((presence-.5)/.35,0,1)
  res.append((risk,target))
 return np.array(res)
sem=pd.read_csv(A/'semantic_oof.csv').set_index('id');semt=pd.read_csv(A/'semantic_test.csv').set_index('id');x=np.zeros((len(q),2))
for fit,held in zip(DAYS,DAYS[::-1]):
 m=q.day.eq(held);x[m]=hypotheses(q[m],sem,fit)
xt=hypotheses(te,semt)
pd.DataFrame({'id':q.id,'extent_distribution_risk':x[:,0],'expected_height':x[:,1]}).to_csv(A/'hypotheses_oof.csv',index=False)
pd.DataFrame({'id':te.id,'extent_distribution_risk':xt[:,0],'expected_height':xt[:,1]}).to_csv(A/'hypotheses_test.csv',index=False)
for w in [.25,.5,.75]:register(f'person_hypotheses_{w:g}',q.score+w*(1-q.score)*x[:,0],current+w*(1-current)*xt[:,0],dict(weight=w,fit='opposite day manual people, top center and width weighted mixture, no fixed extension'))
