"""Retain V21 structural hand penalties; require independent visual veto for rescues."""
from common22 import *
import sys
sys.path.insert(0,str(ROOT/'rascar_v20'))
from evaluate_sahi import old_veto
pdino=pd.read_csv(A/'gpu_results/spatial/predictions.csv')
for variant in ['frozen','control','targeted','window_specialist']:
 if not (A/f'zoom_{variant}_test.csv').exists():continue
 for weight in [.3,.5]:
  out=[]
  for split,rows,base in [('oof',q,q.score.to_numpy()),('test',te,current)]:
   part='oof' if split=='oof' else 'submissions';prefix='' if split=='oof' else 'submission_'
   cup=align(R/part/f'{prefix}dinov2_inside_svm_cup_guarded_w0.3.csv',rows,'score')
   hand=align(ROOT/f'rascar_v21/results/{part}/{prefix}hand_quality_both_w0.75.csv',rows,'score')
   score=base+(cup-base)+(hand-base)
   f=align(A/f'zoom_{variant}_{split}.csv',rows);sem=align(ROOT/f'rascar_v16/semantic_tight_{split}.csv',rows,'p_nude_hands')
   dd=pdino[pdino.representation.eq('spatial')&pdino.split.eq('held' if split=='oof' else 'test')]
   quality=dd.groupby(['id','algorithm']).probability.mean().unstack().loc[rows.id]
   visual_veto=quality.min(axis=1).to_numpy()>=.75
   # A previously detected structural contradiction has priority over detector rescue.
   structural=hand>base+1e-9
   gate=rows.class_name.eq('nude_hands').to_numpy()&boolean(f.assigned)&f.instances.eq(1).to_numpy()&f.overlap.ge(.65).to_numpy()&f.confidence.ge(.5).to_numpy()&f.coherence.ge(.75).to_numpy()&(sem>=.75)&~old_veto(rows,split=='test')&~boolean(f.duplicate_loser)&~visual_veto&~structural
   if split=='test':gate &= f.folds.ge(2).to_numpy()
   target=np.clip((.85-f.overlap.to_numpy())/.5,0,1)
   # The aggressive window specialist only acts inside its intended domain.
   if variant=='window_specialist':gate &= rows.x1.lt(350).to_numpy()&rows.y1.between(150,400).to_numpy()
   score[gate]-=weight*np.maximum(score[gate]-target[gate],0);out.append(score)
   pd.DataFrame(dict(id=rows.id,visual_veto=visual_veto,structural_veto=structural,gate=gate,target=target,score=score)).to_csv(A/f'candidate_{variant}_w{weight}_{split}.csv',index=False)
  register(f'candidate_{variant}_w{weight}',*out,dict(cup='new DINOv2 inside SVM, structure-guarded rescue .3',hand_errors='V21 structural contradiction .75',hand_rescue=variant,weight=weight,person='V20 unchanged; new person experts rejected',visual_veto='both new spatial classifiers >=.75 error',no_test_id_rules=True))
