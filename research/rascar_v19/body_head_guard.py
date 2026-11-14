from common19 import *
heads=pd.read_csv(V16/'wevis_heads_references.csv')
def coverage(rows):
    out=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);rr=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy();centers=(rr[:,:2]+rr[:,2:])/2;inside=((centers>=b[:2])&(centers<=b[2:])).all(1)
        if r.class_name!='person' or inside.sum()!=1:out.append(0.);continue
        hb=rr[inside][0];inter=np.maximum(0,np.minimum(b[2:],hb[2:])-np.maximum(b[:2],hb[:2])).prod();out.append(inter/np.maximum((hb[2:]-hb[:2]).prod(),1))
    return np.asarray(out)
for weight in [.25,.5,.75,1.]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        split='test' if test else 'oof';path=R/('submissions' if test else 'oof')/(('submission_' if test else '')+f'body_uncertainty_w{weight}.csv')
        candidate=align(path,rows,'score');c=coverage(rows)
        outputs.append(np.where(c>=.9,candidate,base))
        if weight==.5:pd.DataFrame(dict(id=rows.id,head_coverage=c)).to_csv(A/f'body_head_coverage_{split}.csv',index=False)
    register(f'body_uncertainty_fullhead_w{weight}',*outputs,details=dict(weight=weight,head_coverage_min=.9,reason='body extent prediction needs an entire head within the supplied box'))
