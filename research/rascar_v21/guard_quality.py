"""Directional, class-aware use of nonlinear quality; abstain on duplicates."""
from common21 import *

def duplicate_veto(rows,test):
    split='test' if test else 'oof'
    old=align(ROOT/f'rascar_v18/replay_inputs/{split}.csv',rows)
    dup=boolean(old.duplicate_veto)&rows.class_name.eq('nude_hands').to_numpy()
    groups=pd.read_csv(ROOT/f'rascar_v15/duplicate_groups_{split}.csv')
    return dup|rows.id.isin(groups.loc[groups.reason.eq('duplicate_loser'),'id']).to_numpy()

if __name__=='__main__':
    for family in ['rbf2','rbf5','ridge2']:
        for cls in CL:
            for weight in [.3,.5]:
                out=[]
                for rows,base,split,test in [(q,q.score.to_numpy(),'oof',False),(te,current,'test',True)]:
                    p=align(A/f'{family}_{split}.csv',rows,'probability')
                    sem=align(ROOT/f'rascar_v16/semantic_both_{split}.csv',rows,'p_'+cls)
                    m=rows.class_name.eq(cls).to_numpy()&(sem>=.8)&(p<=.65)&~duplicate_veto(rows,test)
                    if cls=='person':
                        f=align(ROOT/f'rascar_v20/replay_inputs/{split}.csv',rows)
                        m &= f.mode_front.ge(.6).to_numpy()&f.visible_outside.lt(.01).to_numpy()&f.iou_q75.ge(.55).to_numpy()&f.width_ratio.le(1.5).to_numpy()
                    s=base.copy();s[m]-=weight*np.maximum(s[m]-p[m],0);out.append(s)
                register(f'{family}_{cls}_rescue_w{weight}',*out,dict(weight=weight,class_name=cls,quality_max=.65,semantic_min=.8,directional='lower error only; preserve duplicate losers'))
