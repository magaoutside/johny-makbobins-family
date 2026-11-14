from common21 import *
variants={'main':(.5,.75,0),'conservative':(.3,.5,0),'seat_experiment':(.5,.75,.25)}
for name,(cup_weight,hand_weight,seat_weight) in variants.items():
    out=[]
    for split,rows,base in [('oof',q,q.score.to_numpy()),('test',te,current)]:
        cup=align(R/f'{"oof" if split=="oof" else "submissions"}/{"" if split=="oof" else "submission_"}cup_instance_quality_w{cup_weight}.csv',rows,'score')
        hand=align(R/f'{"oof" if split=="oof" else "submissions"}/{"" if split=="oof" else "submission_"}hand_quality_both_w{hand_weight}.csv',rows,'score')
        score=base+(cup-base)+(hand-base)
        if seat_weight:
            seat=align(R/f'{"oof" if split=="oof" else "submissions"}/{"" if split=="oof" else "submission_"}person_distribution_w{seat_weight}.csv',rows,'score')
            score+=seat-base
        out.append(score)
    register('release_'+name,*out,dict(baseline='V20 LB .95417',cup_weight=cup_weight,hand_weight=hand_weight,person_weight=seat_weight,person_note='seat experiment worsens four officially correct OOF people; optional isolated ablation, not main recommendation' if seat_weight else 'V20 person ranking unchanged'))
    (A/f'submission_v21_{name}.csv').write_bytes((R/'submissions'/f'submission_release_{name}.csv').read_bytes())
assert hashlib.sha256((ROOT/'rascar_v20/submission_v20_main.csv').read_bytes()).hexdigest()=='f9b07bfcb4e92d85bcac64635c0303e2e650a6310cfe47b746d1b9c149c0babe'
