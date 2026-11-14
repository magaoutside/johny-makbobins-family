from common25 import *
import render23
render23.A=A
p=pd.read_csv(A/'results/reviews/person_hidden_legs_strong.csv')
r=align(A/'features/person_probe_test.csv',te)
print(render23.render(p[p.delta.abs()>.02],'person_probe',ref=r))
d=q.assign(baseline_v22=base_oof,new_score=align(A/'results/oof/person_hidden_legs_strong.csv',q,'score'))
print(render23.render(d[d.error.eq(0)&d.new_score.gt(d.baseline_v22+.02)],'person_probe_gt',gt=True,ref=align(A/'features/person_probe_oof.csv',q)))
p=pd.read_csv(A/'results/reviews/tray_hierarchical_edge0.0_w0.5.csv')
f=(align(A/'features/tray_hierarchical_test_2026-10-25.csv',te)+align(A/'features/tray_hierarchical_test_2026-10-26.csv',te))/2
print(render23.render(p[p.class_name.eq('tray')],'tray_all',ref=f))
print(render23.render(audit[audit.class_name.eq('cup')&audit.diagnostic_flag.notna()].assign(baseline_v22=current[audit.class_name.eq('cup')&audit.diagnostic_flag.notna()]),'cup_problem'))
