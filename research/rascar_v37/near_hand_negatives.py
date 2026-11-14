"""Mine near-hand negative crops from official training frames only.
Candidate generation excludes every manual hand plus a margin, and mining
uses only a model fitted on that same training day. Visual review required.
"""
from common37 import *
import joblib
from hand_hardneg import vectors

def main():
    path=A/'inputs/near_hand_candidates.csv'
    if not path.exists():
        records=[]
        for i,r in h[h.class_name.eq('nude_hands')].iterrows():
            b=r[B].to_numpy(float);wh=b[2:]-b[:2];center=(b[:2]+b[2:])/2
            others=h[h.frame_id.eq(r.frame_id)&h.class_name.eq('nude_hands')][B].to_numpy(float)
            for k,offset in enumerate([(0,-1.3),(0,1.3),(-1.3,0),(1.3,0),(-1.,-1.),(1.,-1.),(-1.,1.),(1.,1.)]):
                c=center+wh*np.asarray(offset);size=wh*.8;box=np.r_[c-size/2,c+size/2]
                if np.any(box<0) or np.any(box>np.array([1280,720,1280,720])) or min(size)<10:continue
                padded=box+np.array([-4,-4,4,4])
                if iou(padded,others).max(initial=0)>0:continue
                records.append(dict(id=f'near_{i}_{k}',frame_id=r.frame_id,day=r.day,class_name='nude_hands',split='near_negative',anchor_gt_index=i,**dict(zip(B,box))))
        rec=pd.DataFrame(records);rec.to_csv(path,index=False)
    else:rec=pd.read_csv(path)
    # vectors() has an explicit filename override to avoid corrupting the pool cache.
    x=vectors(rec,filename='near_hand_vectors.npy');prob=np.ones(len(rec))
    for fit in DAYS:
        model=joblib.load(ROOT/f'rascar_v36/models/dino_pool_aug_fit{fit[-2:]}.joblib');m=rec.day.eq(fit)
        prob[m]=model.predict_proba(x[m])[:,1]
    rec['mining_p_hand']=prob
    rec['cell']=rec.frame_id+'_'+((rec.x1+rec.x2)//60).astype(int).astype(str)+'_'+((rec.y1+rec.y2)//60).astype(int).astype(str)
    top=rec[rec.mining_p_hand>.5].sort_values('mining_p_hand',ascending=False).drop_duplicates('cell').groupby('frame_id',sort=False).head(2)
    top=top.reset_index().rename(columns={'index':'feature_index'});top['review_index']=np.arange(len(top))
    top.to_csv(A/'inputs/near_hand_review.csv',index=False);gallery(top,A/'audit/near_hand_review',label_columns=['day'])
    (A/'inputs/near_hand_mining_protocol.json').write_text(json.dumps(dict(n_candidates=len(rec),review_candidates=len(top),max_per_frame=2,mining_threshold=.5,
        label_source='Official frame has no hand inside candidate plus 4px margin; visual review additionally required.',
        candidate_model='V36 contextual DINO, own training day only',test_used=False,pool_reviews_in_mining=False),indent=2))
    print('NEAR_NEGATIVES',len(rec),len(top),top.groupby('day').size().to_dict(),flush=True)

if __name__=='__main__':main()
