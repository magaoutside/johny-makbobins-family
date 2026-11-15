from common22 import *
for cup_rule in ['v21','union','agreement']:
 out=[]
 for split,rows,base in [('oof',q,q.score.to_numpy()),('test',te,current)]:
  part='oof' if split=='oof' else 'submissions';prefix='' if split=='oof' else 'submission_'
  oldcup=align(ROOT/f'kernel_quality/results/{part}/{prefix}cup_instance_quality_w0.5.csv',rows,'score')
  newcup=align(R/part/f'{prefix}dinov2_inside_svm_cup_guarded_w0.3.csv',rows,'score')
  candidate=align(R/part/f'{prefix}candidate_targeted_w0.5.csv',rows,'score')
  cup=oldcup if cup_rule=='v21' else np.minimum(oldcup,newcup) if cup_rule=='union' else np.maximum(oldcup,newcup)
  out.append(candidate-(newcup-base)+(cup-base))
 register(f'final_{cup_rule}',*out,dict(cup_rule=cup_rule,hand='V21 structural penalties + V22 targeted YOLO zoom rescue .5 with new DINOv2 visual veto',person='V20 unchanged',leaderboard_submission=False))
