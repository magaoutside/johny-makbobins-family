from common import *
from duplicate_groups import grouped_duplicates
sem=pd.read_csv(A/'semantic_oof.csv').set_index('id').loc[q.id];st=pd.read_csv(A/'semantic_test.csv').set_index('id').loc[te.id]
hy=pd.read_csv(A/'hypotheses_oof.csv').set_index('id').loc[q.id,'extent_distribution_risk'].to_numpy();hyt=pd.read_csv(A/'hypotheses_test.csv').set_index('id').loc[te.id,'extent_distribution_risk'].to_numpy()
ref=pd.read_csv(A/'v8/fused_train_references.csv');reft=pd.read_csv(A/'v8/fused_test_references.csv')
for mode in ['wide_box','pose_direction']:
 if mode=='pose_direction' and not (A/'pose_references.csv').exists():continue
 gates=[]
 for rows in [q,te]:
  aspect=(rows.x2-rows.x1)/(rows.y2-rows.y1);gate=np.clip((1.8-aspect)/.4,0,1).to_numpy()
  if mode=='pose_direction':
   pp=pd.read_csv(A/'pose_references.csv')
   for ix,r in rows[rows.class_name.eq('person')].iterrows():
    pool=pp[(pp.frame_id==r.frame_id)&(pp.confidence>=.35)]
    if pool.empty:continue
    ov=iou(r[B],pool[B])[0];rr=pool.iloc[int(ov.argmax())]
    if ov.max()<.3:continue
    k=[5,6,11,12]
    if min(rr[f'kp{j}_conf'] for j in k)<.4:continue
    shoulder=np.array([np.mean([rr[f'kp{j}_{c}'] for j in [5,6]]) for c in ['x','y']]);hip=np.array([np.mean([rr[f'kp{j}_{c}'] for j in [11,12]]) for c in ['x','y']]);v=hip-shoulder;vertical=abs(v[1])/max(np.linalg.norm(v),1)
    gate[ix]*=np.clip((vertical-.4)/.35,0,1)
  gates.append(gate)
 hh=hy*gates[0];hht=hyt*gates[1]
 s=q.score.to_numpy().copy();t=current.copy();m=q.class_name.eq('person').to_numpy();mt=te.class_name.eq('person').to_numpy();guard=m&((hh>=.15)|(sem.p_background.to_numpy()>=.35));guardt=mt&((hht>=.15)|(st.p_background.to_numpy()>=.35))
 s[guard]=np.maximum(s[guard],v8[guard]);t[guardt]=np.maximum(t[guardt],v8t[guardt]);s[m]+=.5*(1-s[m])*np.clip((sem.loc[m,'p_background'].to_numpy()-sem.loc[m,'p_person'].to_numpy()-.1)/.7,0,1);t[mt]+=.5*(1-t[mt])*np.clip((st.loc[mt,'p_background'].to_numpy()-st.loc[mt,'p_person'].to_numpy()-.1)/.7,0,1)
 s+=.5*(1-s)*hh;t+=.5*(1-t)*hht;s,_=grouped_duplicates(q,ref,s);t,_=grouped_duplicates(te,reft,t)
 register('combined_orientation_'+mode,s,t,dict(orientation_guard=mode))
