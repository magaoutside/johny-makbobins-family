"""Fixed quality gates and diagnostics; synthetic AP is never competition AP."""
from common40 import *
from evaluate_correspondence import guard
from sklearn.metrics import brier_score_loss

def predictions():
    parts=[]
    for p in sorted((A/'features').glob('quality_fit*_predictions.csv')):parts.append(pd.read_csv(p))
    data=pd.concat(parts,ignore_index=True);assert len(parts)==6
    evidence=[]
    for split,rows in [('oof',q),('test',te)]:
        for _,r in rows.iterrows():
            d=data[data.split.eq(split)&data.id.eq(r.id)&data.kind.eq('query')];assert len(d)==(3 if split=='oof' else 6)
            evidence.append(dict(id=r.id,split=split,class_name=r.class_name,day=r.day,p_good_mean=d.p_good.mean(),p_good_min=d.p_good.min(),p_good_max=d.p_good.max(),predicted_iou_mean=d.predicted_iou.mean()))
    ev=pd.DataFrame(evidence);ev.to_csv(A/'features/quality_consensus.csv',index=False)
    return data,ev

def diagnostics(data,ev):
    result=[]
    for field in ['all','day','class_name']:
        for key,part in ([('all',q)] if field=='all' else q.groupby(field)):
            p=ev[ev.split.eq('oof')].set_index('id').loc[part.id].p_good_mean.to_numpy();target=1-part.error.to_numpy()
            result.append(dict(panel='official_query',group=field,value=key,n=len(part),ap_error=float(AP(1-target,1-p)),brier=float(brier_score_loss(target,p)),high_good_predictions=int((p>=.9).sum()),wrong_among_high_good=int(((p>=.9)&(target==0)).sum())))
    panels=pd.concat([pd.read_csv(p) for p in sorted((A/'features').glob('quality_fit*_panel.csv'))],ignore_index=True)
    g=panels.groupby(['fit_day','id','frame_id','class_name']).agg(p_good=('p_good','mean'),predicted_iou=('predicted_iou','mean'),true_iou=('true_iou','first'),good=('good','first')).reset_index()
    g.to_csv(A/'audit/quality_synthetic_predictions.csv',index=False)
    for (fit,cls),part in g.groupby(['fit_day','class_name']):
        p=part.p_good.to_numpy();y=part.good.to_numpy();high=p>=.9
        result.append(dict(panel='synthetic_shifted_box_NOT_competition_AP',group=fit,value=cls,n=len(part),ap_error=float(AP(1-y,1-p)),brier=float(brier_score_loss(y,p)),high_good_predictions=int(high.sum()),wrong_among_high_good=int(((y==0)&high).sum()),mean_iou_absolute_error=float(abs(part.predicted_iou-part.true_iou).mean())))
    pd.DataFrame(result).to_csv(A/'results/quality_diagnostics.csv',index=False)
    # Reference quality can be checked against official OOF GT only.
    refs=pd.read_csv(ROOT/'rascar_v39/features/set_long_identity_matching_evidence.csv');truth=[]
    for _,v in refs[refs.split.eq('oof')].iterrows():
        box=v[['ref_'+b for b in B]].to_numpy(float);gt=h[h.frame_id.eq(v.frame_id)&h.class_name.eq(v.class_name)][B].to_numpy(float)
        pred=data[data.split.eq('oof')&data.id.eq(v.id)&data.kind.eq('v39_reference')&data.proposal_seed.eq(v.seed)]
        if not np.isfinite(box).all() or not len(pred):continue
        true_iou=float(iou(box,gt).max()) if len(gt) else 0.
        truth.append(dict(id=v.id,class_name=v.class_name,frame_id=v.frame_id,proposal_seed=v.seed,actual_reference_iou=true_iou,p_reference_good=pred.p_good.mean(),predicted_reference_iou=pred.predicted_iou.mean()))
    pd.DataFrame(truth).to_csv(A/'audit/v39_reference_quality_oof.csv',index=False)

def main():
    data,ev=predictions();diagnostics(data,ev);results=[];decisions=[]
    old_heads=pd.read_csv(ROOT/'rascar_v16/wevis_heads_references.csv')
    for mode in ['confirm','two_sided']:
        scores={}
        for split,rows in [('oof',q),('test',te)]:
            s=BASE[split].copy();p=ev[ev.split.eq(split)].set_index('id').loc[rows.id];allowed=guard(rows,split)
            for i,r in rows.iterrows():
                lo=p.iloc[i].p_good_min;hi=p.iloc[i].p_good_max;why='uncertain';target=s[i]
                if lo>=.9:
                    if allowed[i]:target=min(s[i],1-lo);why='confirmed_good'
                    else:why='duplicate_or_structure_veto'
                elif mode=='two_sided' and hi<=.1:target=max(s[i],1-hi);why='confirmed_bad_geometry_or_absence'
                if r.class_name=='person' and abs(target-s[i])>1e-10:
                    heads=old_heads[old_heads.frame_id.eq(r.frame_id)][B].to_numpy(float);centers=(heads[:,:2]+heads[:,2:])/2;box=r[B].to_numpy(float)
                    if ((centers>=box[:2])&(centers<=box[2:])).all(1).sum()!=1:target=s[i];why='person_identity_veto'
                s[i]+=.25*(target-s[i]);decisions.append(dict(id=r.id,split=split,mode=mode,reason=why,p_good_min=lo,p_good_max=hi,changed=abs(s[i]-BASE[split][i])>1e-10))
            scores[split]=s
        for cls in ['all']+CL:
            values={split:np.where(rows.class_name.eq(cls) if cls!='all' else np.ones(len(rows),bool),scores[split],BASE[split]) for split,rows in [('oof',q),('test',te)]}
            result=save(f'v40_quality_{mode}_{cls}_RESEARCH',values,'ROI-aligned IoU quality head; all opposite-day seeds P(good)>=.9 for rescue, <=.1 for penalty. Fixed .25 blend and inherited structural veto, one head required for people.');result.update(mode=mode,classes=cls);results.append(result)
    pd.DataFrame(results).to_csv(A/'results/quality_experiments.csv',index=False);pd.DataFrame(decisions).to_csv(A/'audit/quality_gates.csv',index=False)
    # Verify, rather than replace, the V39 person candidate using the same rule.
    values={}
    for split,rows in [('oof',q),('test',te)]:
        parent=ROOT/'rascar_v39'/('results/oof/v39_set_long_full_identity_geometry_person_RESEARCH.csv' if split=='oof' else 'results/submissions/submission_v39_set_long_full_identity_geometry_person_RESEARCH.csv')
        s=pd.read_csv(parent).set_index('id').loc[rows.id].score.to_numpy();p=ev[ev.split.eq(split)].set_index('id').loc[rows.id]
        down=s<BASE[split]-1e-10;up=s>BASE[split]+1e-10;valid=(down&p.p_good_min.ge(.9).to_numpy())|(up&p.p_good_max.le(.1).to_numpy())
        values[split]=np.where(valid,s,BASE[split])
    result=save('v40_v39_person_quality_verified_RESEARCH',values,'V39 long full person proposal, retained only if cross-day ROI quality direction agrees at .9/.1 in all seeds. Cross-day OOF proxy only for full fit.')
    (A/'results/quality_verified.json').write_text(json.dumps(result,indent=2));print(pd.DataFrame(results)[['experiment','ap','2026-10-25','2026-10-26','changed_test']].to_string(index=False))

if __name__=='__main__':main()
