from common25 import *
import render23
render23.A=A
for model in ['rtmw','wilor_detector']:
 f=(align(A/f'features/{model}_test_2026-10-25.csv',te)+align(A/f'features/{model}_test_2026-10-26.csv',te))/2
 d=pd.read_csv(A/f'results/reviews/{model}_geometry_both_w0.35.csv')
 print(render23.render(d[d.review_no.isin([98,151,200,248,266,333,354])],model+'_hard',ref=f))
 o=align(A/f'results/oof/{model}_geometry_rescue_w0.35.csv',q,'score');d=q.assign(baseline_v22=base_oof,new_score=o)
 f=pd.concat([pd.read_csv(A/f'features/{model}_held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id]
 print(render23.render(d[d.error.eq(1)&d.new_score.lt(d.baseline_v22-.01)],model+'_bad_rescue',gt=True,ref=f))
