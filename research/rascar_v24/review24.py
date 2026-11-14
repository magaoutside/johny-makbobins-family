from common24 import *
import render23
render23.A=A

if __name__=='__main__':
    for name in ['context_cb_cup_rescue_w0.5','context_trees_nude_hands_rescue_w0.5','context_cb_person_rescue_w0.25','appearance_seat_max_rescue_w0.5']:
        d=pd.read_csv(A/f'results/reviews/{name}.csv');d=d[d.delta.abs()>.02].sort_values('review_no')
        if 'person_rescue' in name or 'appearance_seat' in name:
            d=d[d.visual_status.isin(['clear_error','likely_error']) & d.delta.lt(-.02)]
        print(name,render23.render(d,name),flush=True)
    name='context_trees_nude_hands_rescue_w0.5'
    f=align(A/f'results/oof/{name}.csv',q);d=q.copy();d['baseline_v22']=base_oof;d['new_score']=f.score.to_numpy()
    d=d[d.class_name.eq('nude_hands') & d.error.eq(1) & d.new_score.lt(d.baseline_v22-.02)]
    print(render23.render(d,'hand_unsafe_oof',gt=True),flush=True)
