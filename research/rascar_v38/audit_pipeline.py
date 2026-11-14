"""Independent, read-only checks of the active V31 -> V38 data/scoring path."""
from common38 import *
from scipy.optimize import linear_sum_assignment
from PIL import Image
import importlib.util

def independent_iou(a,b):
    out=np.zeros((len(a),len(b)))
    for i,(x,y,u,v) in enumerate(a):
        for j,(xx,yy,uu,vv) in enumerate(b):
            intersection=max(0,min(u,uu)-max(x,xx))*max(0,min(v,vv)-max(y,yy))
            out[i,j]=intersection/max((u-x)*(v-y)+(uu-xx)*(vv-yy)-intersection,1e-12)
    return out

def independent_ap(y,s):
    d=pd.DataFrame({'s':s,'p':y}).groupby('s').p.agg(['sum','size']).sort_index(ascending=False)
    return float((d['sum']/d['sum'].sum()*d['sum'].cumsum()/d['size'].cumsum()).sum())

def main():
    report={};raw=ROOT/'rascar_data_full'
    train=pd.read_csv(raw/'train.csv');test=pd.read_csv(raw/'test.csv');sample=pd.read_csv(raw/'sample_submission.csv')
    gt=pd.read_csv(raw/'human_boxes_train.csv');policy=json.loads((raw/'scoring_policy.json').read_text())
    for d in [train,test,gt]:d['class_name']=d.class_name.replace(policy['aliases'])
    gt[B]=np.stack([(gt.cx-gt.w/2)*1280,(gt.cy-gt.h/2)*720,(gt.cx+gt.w/2)*1280,(gt.cy+gt.h/2)*720],axis=1)
    # A cardinality bonus > maximum possible total IoU enforces lexicographic matching.
    predicted=np.ones(len(train),int);assign=[]
    for (frame,cls),g in train.groupby(['frame_id','class_name']):
        truth=gt[gt.frame_id.eq(frame)&gt.class_name.eq(cls)]
        ov=independent_iou(g[B].to_numpy(),truth[B].to_numpy())
        if ov.size:
            allowed=ov>=policy['correct_iou'];bonus=min(ov.shape)+1
            weight=np.where(allowed,bonus+ov,0.)
            ri,ci=linear_sum_assignment(weight,maximize=True)
            for i,j in zip(ri,ci):
                if allowed[i,j]:predicted[g.index[i]]=0;assign.append(dict(id=g.iloc[i].id,gt_index=int(truth.index[j]),iou=ov[i,j]))
    mismatch=train[predicted!=train.error.to_numpy()].copy();mismatch['recomputed_error']=predicted[predicted!=train.error.to_numpy()]
    mismatch.to_csv(A/'audit/label_mismatches.csv',index=False);pd.DataFrame(assign).to_csv(A/'audit/official_matching.csv',index=False)
    report['official_labels']={'rows':len(train),'non_table':int(train.class_name.ne('table').sum()),'mismatches':len(mismatch),'iou':.6,'lexicographic_matching':True}
    source=train[train.class_name.ne('table')].set_index('id').loc[q.id]
    report['oof_integrity']={'n':len(q),'frames':int(q.frame_id.nunique()),'positives':int(q.error.sum()),'labels_match':bool(np.array_equal(source.error,q.error)),
      'query_max_coordinate_difference':float(abs(source[B].to_numpy()-q[B].to_numpy()).max()),'frame_and_class_match':bool(np.array_equal(source[['frame_id','class_name']].to_numpy(),q[['frame_id','class_name']].to_numpy()))}
    hg=h.reset_index(drop=True);gg=gt.reset_index(drop=True)
    report['manual_conversion']={'rows':len(hg),'source_rows':len(gg),'frame_class_match':bool(np.array_equal(hg[['frame_id','class_name']],gg[['frame_id','class_name']])),'max_coordinate_difference':float(abs(hg[B].to_numpy()-gg[B].to_numpy()).max())}
    test_aligned=test.set_index('id').loc[te.id]
    report['test_integrity']={'rows':len(te),'unique_ids':int(te.id.nunique()),'frames':int(te.frame_id.nunique()),'sample_id_set_match':set(sample.id)==set(te.id),'query_coordinate_difference':float(abs(test_aligned[B].to_numpy()-te[B].to_numpy()).max()),'train_id_overlap':len(set(train.id)&set(te.id)),'train_frame_overlap':len(set(train.frame_id)&set(te.frame_id))}
    sizes=[]
    for frame in sorted(set(train.frame_id)|set(te.frame_id)):
        with Image.open(raw/f'{frame}.jpg') as im:sizes.append({'frame_id':frame,'width':im.width,'height':im.height})
    pd.DataFrame(sizes).to_csv(A/'audit/image_sizes.csv',index=False)
    report['query_image_sizes']=pd.DataFrame(sizes).groupby(['width','height']).size().to_dict().__str__()
    # Rebuild saved V22 scores, with no historical test-audit labels loaded.
    spec=importlib.util.spec_from_file_location('v22_rank',ROOT/'rascar_v22/replay.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    replay={}
    for split,rows in [('oof',q),('test',te)]:
        xx=pd.read_csv(ROOT/f'rascar_v22/replay_inputs/{split}.csv');got=module.rank(xx,test=split=='test').set_index('id').loc[rows.id].score.to_numpy()
        expected=pd.read_csv(ROOT/('rascar_v22/results/oof/final_reviewed.csv' if split=='oof' else 'rascar_v22/submission_v22_candidate.csv')).set_index('id').loc[rows.id].score.to_numpy()
        replay[split]=float(abs(got-expected).max())
    report['v22_replay_max_abs_difference']=replay
    v22o=pd.read_csv(ROOT/'rascar_v22/results/oof/final_reviewed.csv').set_index('id').loc[q.id].score.to_numpy()
    old=pd.read_csv(ROOT/'rascar_v18/results/oof/release_main.csv').set_index('id').loc[q.id].score.to_numpy()
    noold=pd.read_csv(ROOT/'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv').set_index('id').loc[q.id].score.to_numpy()
    oo=np.clip(v22o-(old-noold),0,1);tt=pd.read_csv(ROOT/'rascar_v22/submission_v22_candidate.csv').set_index('id').loc[te.id].score.to_numpy()
    target=lambda x:np.clip((.85-x)/.5,0,1)
    o=target(pd.read_csv(ROOT/'rascar_v31/features/linear_mask_cut0.5_oof_evidence.csv').set_index('id').loc[q.id].overlap.to_numpy())
    ts=np.stack([target(pd.read_csv(ROOT/f'rascar_v31/features/linear_mask_cut0.5_test_fit{d}_evidence.csv').set_index('id').loc[te.id].overlap.to_numpy()) for d in ['25','26']])
    mask=np.isfinite(o);oo[mask]=.75*oo[mask]+.25*o[mask]
    delta=ts-tt;up=np.all(delta>=0,axis=0);down=np.all(delta<=0,axis=0);ok=np.isfinite(ts).all(0)&(up|down)
    conservative=np.where(up,np.min(ts,axis=0),np.max(ts,axis=0));tt[ok]=.75*tt[ok]+.25*conservative[ok]
    report['v31_replay_max_abs_difference']={'oof':float(abs(oo-BASE['oof']).max()),'test':float(abs(tt-BASE['test']).max())}
    report['v31_metrics']=metric(q,oo);report['independent_ap']=independent_ap(q.error,oo)
    assert abs(report['independent_ap']-AP(q.error,oo))<1e-12
    corrected_v22=np.clip(v22o-old+noold,0,1)
    report['historical_oof_only_adjustment']={'rows':int((abs(old-noold)>1e-12).sum()),'max_abs':float(abs(old-noold).max()),'v31_subtracts_it':True,'v22_reported_ap':float(AP(q.error,v22o)),'v22_after_subtraction_ap':float(AP(q.error,corrected_v22))}
    # Check cached target occupancy directly against official boxes, not a model.
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');newrec=pd.read_csv(A/'inputs/multilevel_records.csv');targets=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r')
    maxerr=0.
    for i,r in rec.iterrows():
        expected=np.zeros((4,32,32),np.float32)
        if r.split!='test':
            cb=r[['crop_'+b for b in B]].to_numpy(float);sx=(cb[2]-cb[0])/32;sy=(cb[3]-cb[1])/32;xx=cb[0]+np.arange(32)*sx;yy=cb[1]+np.arange(32)*sy
            for k,c in enumerate(CL):
                for b in gt[gt.frame_id.eq(r.frame_id)&gt.class_name.eq(c)][B].to_numpy():
                    wx=np.clip(np.minimum(xx+sx,b[2])-np.maximum(xx,b[0]),0,None)/sx
                    wy=np.clip(np.minimum(yy+sy,b[3])-np.maximum(yy,b[1]),0,None)/sy
                    expected[k]=np.maximum(expected[k],np.outer(wy,wx))
        maxerr=max(maxerr,float(abs(expected.astype(np.float16).astype(float)-targets[i].astype(float)).max()))
    report['feature_targets']={'rows':len(rec),'v31_v38_records_equal':rec.equals(newrec),'max_target_difference':maxerr,'train_crops':int(rec.split.eq('manual').sum()),'train_frames':int(rec[rec.split.eq('manual')].frame_id.nunique()),'heldout_targets_present_but_never_selected_for_training':True}
    oldx=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r');newx=np.load(A/'features/multilevel_grids.npy',mmap_mode='r')
    maxdiff=0.;meanabs=0.
    for off in range(0,len(rec),8):
        diff=abs(oldx[off:off+8].astype(np.float32)-newx[off:off+8,-384:].astype(np.float32));maxdiff=max(maxdiff,float(diff.max()));meanabs+=float(diff.sum(dtype=np.float64))
    report['last_layer_feature_equivalence']={'max_abs_difference':maxdiff,'mean_abs_difference':meanabs/oldx.size}
    foldchecks=[]
    for version in ['v31','v38']:
        path=ROOT/('rascar_v31/models/training_manifest.json' if version=='v31' else 'rascar_v38/models/multilevel_manifest.json')
        for m in json.loads(path.read_text()):
            frames=m.get('fit_frames',m.get('fit_frame_ids'));fit=m['fit_day'];days=set(f.split('_')[-2] for f in frames)
            valid=days<=({'20261025','20261026'} if fit=='all' else {fit.replace('-','')})
            foldchecks.append(dict(version=version,name=m['name'],fit_day=fit,frames=len(frames),days=sorted(days),day_valid=valid,test_overlap=len(set(frames)&set(te.frame_id))))
    report['fold_checks']=foldchecks
    inherited_checks=[]
    for version in ['v20','v22']:
        for day in DAYS:
            p=ROOT/f'rascar_{version}/inputs/approved_pseudo_boxes_{day}.csv';d=pd.read_csv(p)
            inherited_checks.append(dict(file=str(p.relative_to(ROOT)),rows=len(d),unique_frames=int(d.frame_id.nunique()),invalid_days=int(d.frame_id.str.split('_').str[-2].ne(day.replace('-','')).sum())))
    for day in DAYS:
        p=ROOT/f'rascar_v22/gpu_results/hand_training/{day}/training_ledger.csv';d=pd.read_csv(p)
        inherited_checks.append(dict(file=str(p.relative_to(ROOT)),rows=len(d),unique_frames=int(d.frame_id.nunique()),invalid_days=int(d.frame_id.str.split('_').str[-2].ne(day.replace('-','')).sum())))
    spatial=pd.read_csv(ROOT/'rascar_v22/inputs/spatial_records.csv')
    for day in DAYS:
        tr=spatial[spatial.split.eq('train')&spatial.day.eq(day)];ev=spatial[spatial.split.eq('test')|(spatial.split.eq('held')&spatial.day.ne(day))]
        inherited_checks.append(dict(file='rascar_v22/inputs/spatial_records.csv',fit_day=day,rows=len(tr),unique_frames=int(tr.frame_id.nunique()),invalid_days=int(tr.frame_id.str.split('_').str[-2].ne(day.replace('-','')).sum()),held_or_test_overlap=len(set(tr.frame_id)&set(ev.frame_id))))
    (A/'audit/inherited_training_checks.json').write_text(json.dumps(inherited_checks,indent=2))
    assert all(x['invalid_days']==0 and x.get('held_or_test_overlap',0)==0 for x in inherited_checks)
    pool=pd.read_csv(ROOT/'rascar_v37/inputs/blind_pool.csv');report['v37_pool_count_erratum']={'queries':len(pool),'unique_frames':int(pool.frame_id.nunique()),'previous_report_frames':64}
    hashes={}
    for relative in ['rascar_v22/submission_v22_candidate.csv','rascar_v31/submission_v31_dense_oof098784.csv','rascar_v37/submission_v37_person_refit_RESEARCH.csv']:
        hashes[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    report['frozen_hashes']=hashes
    report['limits']=['V22 replay verifies composition from cached model outputs, not a retraining audit of every inherited model.',
      'Repeated model/threshold selection on these same 198 OOF rows makes them a development set, not untouched validation.',
      'OOF uses one fitting day; test has two fitting days with consensus. Full-fit V38 has no independently measured same-training-size OOF.',
      'Historical visual test status tables exist and were used for diagnostics/model selection; they are not official labels and cannot certify test correctness or be used as training targets.',
      'A single per-crop maximum-IoU dense reference does not enforce global one-to-one assignment and may merge overlapping same-class objects.']
    (A/'audit/pipeline_checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ['fold_checks','limits']},indent=2,ensure_ascii=False))
    assert len(mismatch)==0 and report['manual_conversion']['max_coordinate_difference']<1e-6
    assert max(report['v31_replay_max_abs_difference'].values())<1e-12
    assert all(x['day_valid'] and not x['test_overlap'] for x in foldchecks)

if __name__=='__main__':main()
