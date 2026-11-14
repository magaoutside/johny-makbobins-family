"""Reproduce final candidates from saved, cross-day predictions; no fitting."""
from common import *

R = A / 'results'
def cached(name, test=False):
    path = R/'submissions'/f'submission_{name}.csv' if test else R/'oof'/f'{name}.csv'
    return pd.read_csv(path).set_index('id').loc[te.id if test else q.id, 'score'].to_numpy()

for hand_weight in [0, .2, .4]:
    for pose_weight in [0, .2]:
        outputs=[]
        for rows, baseline, test in [(q,q.score.to_numpy(),False),(te,current,True)]:
            s=baseline.copy()
            person=rows.class_name.eq('person').to_numpy()
            s[person]=cached('combined_orientation_wide_box',test)[person]
            # Only confidently matched hands can be rescued. DINO alone is insufficient.
            if hand_weight:
                s+=cached(f'hand_specialist_c0.35_w{hand_weight}',test)-baseline
            # Recompute duplicate decisions using the hand specialist, keeping doubtful pairs.
            dup=cached('duplicate_groups_with_specialist',test)
            promoted=dup>baseline+1e-9
            s[promoted]=np.maximum(s[promoted],dup[promoted])
            pose=pd.read_csv(A/f'pose_{"test" if test else "oof"}_features.csv').set_index('id').loc[rows.id,'body_outside'].to_numpy()
            s+=pose_weight*(1-s)*pose
            outputs.append(s)
        register(f'final_hands{hand_weight}_pose{pose_weight}',*outputs,details=dict(
            hand_weight=hand_weight,pose_weight=pose_weight,person='orientation wide-box guard',
            duplicate_matching='one-to-one, specialist references, two-hand abstention',
            test_used_for_fitting=False,test_visuals_used_for_development=True))

selected='final_hands0.2_pose0.2'
(A/'selected_experiment.txt').write_text(selected+'\n')
(A/'submission_v15.csv').write_bytes((R/'submissions'/f'submission_{selected}.csv').read_bytes())
print('SELECTED',selected)
