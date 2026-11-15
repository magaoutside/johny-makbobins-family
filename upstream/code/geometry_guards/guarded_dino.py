from common16 import *
def load(name,test=False,column='quality'):
    return pd.read_csv(V16/f'{name}_{"test" if test else "oof"}.csv').set_index('id').loc[te.id if test else q.id,column].to_numpy()

for threshold in [.85,.9]:
    for w in [.2,.35]:
        outputs=[];counts=[]
        for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
            qual=load('quality_nude_hands_linear_tight',test)
            sem=load('semantic_tight',test,'p_nude_hands')
            # Restore no confirmed duplicate; score thresholds intentionally restrict changes.
            if test:
                ids=pd.read_csv(V15/'hand_group_test.csv').query("reason == 'duplicate_loser'").id
            else:
                ids=pd.concat([pd.read_csv(V15/f'hand_group_oof_{day}.csv') for day in DAYS]).query("reason == 'duplicate_loser'").id
            m=rows.class_name.eq('nude_hands').to_numpy()&(base>=threshold)&(qual<=.55)&(sem>=.8)&~rows.id.isin(ids).to_numpy()
            s=base.copy();s[m]=(1-w)*s[m]+w*qual[m];outputs.append(s);counts.append(int(m.sum()))
        register(f'hand_rescue_guarded_s{threshold}_w{w}',*outputs,details=dict(threshold=threshold,quality_max=.55,semantic_min=.8,weight=w,changed_counts=counts))

register('baseline_lb092822',q.score,current,dict(public_lb_user_reported=.92822))
