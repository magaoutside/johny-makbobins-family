from common22 import *
out=[]
for split,rows,base in [('oof',q,q.score.to_numpy()),('test',te,current)]:
 part='oof' if split=='oof' else 'submissions';prefix='' if split=='oof' else 'submission_'
 score=align(R/part/f'{prefix}final_union.csv',rows,'score')
 hand=align(ROOT/f'kernel_quality/results/{part}/{prefix}hand_quality_both_w0.75.csv',rows,'score')
 geo=align(ROOT/f'body_and_hands/hand_geometry_features_{split}.csv',rows)
 m=rows.class_name.eq('nude_hands').to_numpy()&geo.center_instances.ge(2).to_numpy()&(score<hand-1e-9)
 score[m]=hand[m];out.append(score)
 pd.DataFrame(dict(id=rows.id,multiple_instance_rescue_veto=m,score=score)).to_csv(A/f'final_reviewed_{split}.csv',index=False)
register('final_reviewed',*out,dict(source='final_union',extra_guard='no new rescue when independent references contain >=2 hand centers',audit='all 40 changes larger than .02 inspected; uncertain clasped hands prompted generic guard, no per-ID overrides',leaderboard_submission=False))
(A/'submission_v22_candidate.csv').write_bytes((R/'submissions/submission_final_reviewed.csv').read_bytes())
control=[]
for split,rows in [('oof',q),('test',te)]:
 part='oof' if split=='oof' else 'submissions';prefix='' if split=='oof' else 'submission_'
 score=align(R/part/f'{prefix}candidate_targeted_w0.5.csv',rows,'score')
 hand=align(ROOT/f'kernel_quality/results/{part}/{prefix}hand_quality_both_w0.75.csv',rows,'score')
 geo=align(ROOT/f'body_and_hands/hand_geometry_features_{split}.csv',rows)
 m=rows.class_name.eq('nude_hands').to_numpy()&geo.center_instances.ge(2).to_numpy()&(score<hand-1e-9)
 score[m]=hand[m];control.append(score)
register('validation_reviewed',*control,dict(cup='new DINOv2 guarded SVM only',extra_guard='independent multi-hand center veto',purpose='higher OOF but fewer frozen diagnostic passes than main candidate',leaderboard_submission=False))
(A/'submission_v22_validation_control.csv').write_bytes((R/'submissions/submission_validation_reviewed.csv').read_bytes())
