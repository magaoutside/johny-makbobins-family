from common37 import *

def main():
    # Freeze intended models and rules before evaluating this new pool.
    protocol=dict(seed=3701,training='Official fit-day boxes and fit-day queries only; no pool review in training.',
        features=['SigLIP2 tight1.05 and context1.5','DINOv2 tight1.05 and context1.5'],
        models=['frame-balanced per-class logistic C=.1','nearest class exemplars cosine'],
        corrections='Only increase error for strong absence, or separately test class-consistent geometry confirmation. Presence alone never rescues a box.',
        pool_scope='Blind presence and obvious geometry diagnostic, NOT official competition AP.',
        excluded='Previously selected or reviewed train frames and their 120-second neighborhood. No test labels.',
        threshold_protocol='Prespecified absence cutoffs .05/.15, blend .25; shared feature family agreement, then visual review. Store failures.',
        leaderboard='No automatic uploads. At most distinct candidates after validation and visual checks.')
    (A/'inputs/protocol.json').write_text(json.dumps(protocol,indent=2))
    excluded=set(h.frame_id)
    exclusion_paths=['rascar_v22/inputs/pool_selected.csv','rascar_v25/inputs/reviewed_pool_cups.csv',
        'rascar_v32/inputs/pool_control_records.csv','rascar_v33/inputs/pseudo_records.csv',
        'rascar_v33/inputs/rare_hand_review_candidates.csv','rascar_v34/inputs/semantic_records.csv',
        'rascar_v36/inputs/blind_pool_candidates.csv']
    for rel in exclusion_paths:excluded|=set(pd.read_csv(ROOT/rel,usecols=['frame_id']).frame_id.dropna())
    protocol['exclusion_sources']=exclusion_paths
    (A/'inputs/protocol.json').write_text(json.dumps(protocol,indent=2))
    p=pd.read_csv(ROOT/'rascar_data_full/machine_predictions.csv').rename(columns={'detection_id':'id'})
    p=p[p.class_name.isin(['cup','tray','nude_hands'])].copy();p['day']=pd.to_datetime(p.frame_id.str[-15:-7],format='%Y%m%d').dt.strftime('%Y-%m-%d')
    p['seconds']=pd.to_datetime(p.frame_id.str[-15:],format='%Y%m%d_%H%M%S').astype('int64')//10**9
    stamps=np.array([pd.to_datetime(f[-15:],format='%Y%m%d_%H%M%S').value//10**9 for f in excluded])
    dist=np.full(len(p),np.inf)
    for off in range(0,len(stamps),300):dist=np.minimum(dist,abs(p.seconds.to_numpy()[:,None]-stamps[off:off+300]).min(1))
    p=p.loc[dist>120].copy();p['stratum']=np.where(p.confidence<.65,'low_conf','high_conf')
    p['draw']=np.random.default_rng(3701).random(len(p));p['bucket']=p.day+'_'+(p.seconds//1200).astype(str)+'_'+((p.x1+p.x2)/240).astype(int).astype(str)+'_'+p.class_name
    p=p.sort_values('draw').drop_duplicates('bucket');selected=[];used=set()
    for cls in ['cup','tray','nude_hands']:
        for day in DAYS:
            for st in ['low_conf','high_conf']:
                d=p[p.class_name.eq(cls)&p.day.eq(day)&p.stratum.eq(st)&~p.frame_id.isin(used)].head(6)
                selected.append(d);used|=set(d.frame_id)
    sel=pd.concat(selected,ignore_index=True);sel['review_index']=np.arange(len(sel));sel['split']='blind_pool'
    assert not set(sel.frame_id)&excluded and set(sel.day)<=set(DAYS)
    sel.to_csv(A/'inputs/blind_pool.csv',index=False);gallery(sel,A/'audit/blind_pool',label_columns=['day','stratum'])
    (A/'inputs/selection.json').write_text(json.dumps(dict(n=len(sel),unique_frames=sel.frame_id.nunique(),excluded_frames=len(excluded),test_labels_used=False),indent=2))
    d=q.copy();y=d.error.to_numpy();s=d.score.to_numpy()
    d['inversions']=[int(((s>=s[i])&(y==0)).sum()) if y[i] else int(((s<=s[i])&(y==1)).sum()) for i in range(len(d))]
    d.sort_values('inversions',ascending=False).to_csv(A/'audit/v31_residuals.csv',index=False)
    gallery(d.sort_values('inversions',ascending=False).head(24),A/'audit/oof_residuals',label_columns=['score','error','inversions'],gt=True)
    print('BLIND_POOL',len(sel),sel.groupby(['class_name','day','stratum']).size().to_dict(),flush=True)

if __name__=='__main__':main()
