from common43 import *
q,te,h=inputs();bo,bt=baseline(q,te)
r=pd.read_csv(ROOT/'rascar_v42/inputs/records.csv')
# Reuse the audited sample set, not V42 predictions or test annotations.
old=pd.read_csv(ROOT/'rascar_v33/inputs/pseudo_records.csv')
old=old[old.label_use.eq('geometry_and_presence')]
new=pd.read_csv(ROOT/'rascar_v41/pool_expansion/approved_geometry.csv')
p=pd.concat([old,new[new.decision.eq('weak_geometry')]],ignore_index=True)
p=p.drop_duplicates(['frame_id','class_name']+B)
held_frames=set(pd.read_csv(ROOT/'rascar_v37/inputs/blind_pool.csv').frame_id)
p=p[~p.frame_id.isin(held_frames)]
gh={(f,c):z[B].to_numpy(float) for (f,c),z in h.groupby(['frame_id','class_name'])}
gp={(f,c):z[B].to_numpy(float) for (f,c),z in p.groupby(['frame_id','class_name'])}
targets=[]
for _,x in r.iterrows():
    b=x[B].to_numpy(float)
    if x.source=='test':ov=np.nan;g=np.full(4,np.nan)
    else:
        gt=(gp if x.source=='pool' else gh).get((x.frame_id,x.class_name),np.empty((0,4)))
        if len(gt):
            vals=iou(b,gt)[0];k=int(vals.argmax());ov=vals[k];g=gt[k]
        else:ov=0.;g=b
    targets.append(dict(gt_iou=ov,**dict(zip(['gt_'+c for c in B],g))))
r=pd.concat([r,pd.DataFrame(targets)],axis=1)
assert r[r.source.eq('test')].gt_iou.isna().all()
assert not set(r[r.source.eq('pool')].frame_id)&held_frames
r.to_csv(A/'inputs/records.csv',index=False)
protocol=dict(baseline='V31',public_ap=.96239,baseline_hash=hashlib.sha256((ROOT/'rascar_v31/submission_v31_dense_oof098784.csv').read_bytes()).hexdigest(),
 objective='Supplied ROI -> error probability, IoU quality and instance bbox refinement; frozen detector backbone with fine-tuned localization head.',
 pretrained='FasterRCNN ResNet50 FPN V2 COCO_V1',
 supervision='Official query.error; continuous max same-class human-GT IoU and nearest-instance deltas. Weak pool at reduced weight. No pooled union masks.',
 samples=r.source.value_counts().to_dict(),outer_split='fit UTC25->held26; fit26->held25; all pool same fit day only',
 fixed_variants=['frozen_coco_person_cup','detector_roi_clean','detector_roi_pool'],
 blend_weight=.10,additional_diagnostic_weight=.05,epochs_or_steps=400,seed=4301,
 no_test_labels=True,no_test_pseudolabels=True,not_submitted=True,
 limits='Repeatedly reused development OOF; synthetic single-query labels do not represent duplicate competition. Official query labels retain one-to-one target.')
(A/'inputs/protocol.json').write_text(json.dumps(protocol,indent=2))
print(r.groupby(['source','day']).size().to_string(),flush=True)
