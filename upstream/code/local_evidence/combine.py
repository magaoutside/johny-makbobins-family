from common import *
from duplicate_groups import grouped_duplicates
sem=pd.read_csv(A/'semantic_oof.csv').set_index('id').loc[q.id];st=pd.read_csv(A/'semantic_test.csv').set_index('id').loc[te.id]
hy=pd.read_csv(A/'hypotheses_oof.csv').set_index('id').loc[q.id,'extent_distribution_risk'].to_numpy();hyt=pd.read_csv(A/'hypotheses_test.csv').set_index('id').loc[te.id,'extent_distribution_risk'].to_numpy()
ref=pd.read_csv(A/'v8/fused_train_references.csv');reft=pd.read_csv(A/'v8/fused_test_references.csv')
for mode in ['current','guarded','conditional_0.15','conditional_0.3']:
 s=q.score.to_numpy().copy();t=current.copy();m=q.class_name.eq('person').to_numpy();mt=te.class_name.eq('person').to_numpy()
 guard=m if mode=='guarded' else np.zeros(len(q),bool);guardt=mt if mode=='guarded' else np.zeros(len(te),bool)
 if mode.startswith('conditional'):
  cutoff=float(mode.split('_')[1]);guard=m&((hy>=cutoff)|(sem.p_background.to_numpy()>=.35));guardt=mt&((hyt>=cutoff)|(st.p_background.to_numpy()>=.35))
 s[guard]=np.maximum(s[guard],v8[guard]);t[guardt]=np.maximum(t[guardt],v8t[guardt])
 s[m]+=.5*(1-s[m])*np.clip((sem.loc[m,'p_background'].to_numpy()-sem.loc[m,'p_person'].to_numpy()-.1)/.7,0,1)
 t[mt]+=.5*(1-t[mt])*np.clip((st.loc[mt,'p_background'].to_numpy()-st.loc[mt,'p_person'].to_numpy()-.1)/.7,0,1)
 s+=.5*(1-s)*hy;t+=.5*(1-t)*hyt;s,_=grouped_duplicates(q,ref,s);t,_=grouped_duplicates(te,reft,t)
 name='combined_person_'+mode if mode in ['current','guarded'] else 'combined_conditional_guard_'+mode.split('_')[1]
 register(name,s,t,dict(mode=mode,semantic_weight=.5,extent_weight=.5,duplicate_groups=True))
