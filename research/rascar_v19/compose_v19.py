from common19 import *
def cached(name,rows,test):return align(R/('submissions' if test else 'oof')/(('submission_' if test else '')+name+'.csv'),rows,'score')
register('baseline_v18_lb095259',q.score,current,details=dict(user_reported_public_lb=.95259))
for body in [.5,.75,1.]:
    for hands in [.25,.5]:
        outputs=[]
        for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
            p=cached(f'body_uncertainty_fullhead_w{body}',rows,test);hnd=cached(f'landmark_joint_rescue_w{hands}',rows,test)
            assert not ((abs(p-base)>1e-9)&(abs(hnd-base)>1e-9)).any()
            outputs.append(p+hnd-base)
        name=f'combined_body{body}_hands{hands}'
        register(name,*outputs,details=dict(body_weight=body,hand_weight=hands,combination='disjoint classes',new_models='frozen MediaPipe; fit-day body residual distribution'))
        without=align(V18/'results/oof/release_main_no_inherited_contradiction.csv',q,'score')
        register(name+'_without_inherited_contradiction',outputs[0]-(q.score.to_numpy()-without),outputs[1],details=dict(ablation='subtract inherited OOF-only V16 correction',same_submission=name))
