"""Research follow-up: cardinality-first assignment and coordinate meta-quality.

Only frozen pretrained SAM references enter these models. No test review labels,
finetuned YOLO predictions, baseline scores, or held-day labels enter fitting.
This implements the stated matching objective, not an unverified official SDK.
"""
from pathlib import Path
import sys,itertools,json
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'rascar_v25'))
import common25 as common
from common25 import *
from scipy.optimize import linear_sum_assignment
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import joblib
A=ROOT/'rascar_v28';common.A=A
for f in ['inputs','features','models','results/oof','results/submissions','results/reviews','audit','logs']:(A/f).mkdir(parents=True,exist_ok=True)

def match(ov,threshold=.6,forbid=None):
    n,m=ov.shape
    if not n or not m:return {},0,0.
    valid=ov>=threshold
    if forbid is not None:valid[forbid]=False
    # One extra match dominates any possible change in total IoU.
    bonus=min(n,m)+1
    reward=np.zeros((n,m+n));reward[:,:m]=np.where(valid,bonus+ov,-bonus)
    ii,jj=linear_sum_assignment(-reward)
    ans={int(i):int(j) for i,j in zip(ii,jj) if j<m and valid[i,j]}
    return ans,len(ans),sum(float(ov[i,j]) for i,j in ans.items())

def verify_match():
    # A max-sum assignment can choose 0.99+0.59, losing one valid pair.
    case=np.array([[.99,.61],[.61,.59]])
    assert match(case)[1]==2
    rr,cc=linear_sum_assignment(-case);assert int((case[rr,cc]>=.6).sum())==1
    rng=np.random.default_rng(28)
    for n,m in itertools.product(range(1,5),repeat=2):
        for trial in range(10):
            ov=rng.random((n,m));brute=(-1,-1.)
            for assignment in itertools.product(range(-1,m),repeat=n):
                js=[j for j in assignment if j>=0]
                if len(set(js))!=len(js):continue
                if any(ov[i,j]<.6 for i,j in enumerate(assignment) if j>=0):continue
                brute=max(brute,(len(js),sum(ov[i,j] for i,j in enumerate(assignment) if j>=0)))
            _,count,total=match(ov)
            assert count==brute[0] and abs(total-brute[1])<1e-10
    (A/'matching_checks.json').write_text(json.dumps(dict(random_matrices=160,brute_force_equivalent=True,cardinality_counterexample=True)))

def make_features(rows,split):
    ref=pd.read_csv(ROOT/f'rascar_v15/v8/fused_{"train" if split=="oof" else "test"}_references.csv')
    output=[];diagnostics=[]
    for (fid,cls),g in rows.groupby(['frame_id','class_name']):
        rr=ref[ref.frame_id.eq(fid)&ref.class_name.eq(cls)&ref.confidence.ge(.5)].sort_values('confidence',ascending=False)
        # Deduplicate frozen same-model views; never use candidate boxes as refs.
        keep=[]
        for ix,r in rr.iterrows():
            if not keep or iou(r[B],rr.loc[keep,B]).max()<.7:keep.append(ix)
        rr=rr.loc[keep].reset_index(drop=True)
        ov=iou(g[B],rr[B]);exact,count,total=match(ov)
        naive={}
        if len(rr):
            ii,jj=linear_sum_assignment(-ov);naive={i:j for i,j in zip(ii,jj) if ov[i,j]>=.6}
        for k,r in enumerate(g.itertuples()):
            b=np.array([getattr(r,c) for c in B]);wh=np.maximum(b[2:]-b[:2],1);sc=np.tile(wh,2)
            f=dict(id=r.id,confidence=r.confidence,width=wh[0]/1280,height=wh[1]/720,cx=(b[0]+b[2])/2560,cy=(b[1]+b[3])/1440,
                best_iou=0.,second_iou=0.,ref_confidence=0.,assigned=int(k in exact),assignment_iou=0.,
                competitors=0.,rival_margin=0.,essential_pair=0.,alternative_margin=0.,group_size=len(g),reference_count=len(rr))
            f.update({f'class_{c}':int(c==cls) for c in CL})
            f.update({prefix+c:0. for prefix in ['offset_','spread_','mask_offset_'] for c in B})
            if len(rr):
                j=int(ov[k].argmax());best=rr.iloc[j];competitors=np.delete(ov[:,j],k)
                f.update(best_iou=float(ov[k,j]),second_iou=float(np.sort(ov[k])[-2]) if len(rr)>1 else 0.,ref_confidence=float(best.confidence),
                    assignment_iou=float(ov[k,exact[k]]) if k in exact else 0.,competitors=int((competitors>=.6).sum()),
                    rival_margin=float(ov[k,j]-(competitors.max() if len(competitors) else 0)))
                offset=(best[B].to_numpy(float)-b)/sc
                # TTA/view spread is correlated evidence, not independent experts.
                neighbours=ref[ref.frame_id.eq(fid)&ref.class_name.eq(cls)&ref.confidence.ge(.4)]
                neighbours=neighbours.loc[iou(best[B],neighbours[B])[0]>=.65,B]
                spread=neighbours.to_numpy(float).std(0)/sc if len(neighbours)>1 else np.zeros(4)
                mask=(best[['mask_'+c for c in B]].to_numpy(float)-b)/sc
                f.update(dict(zip(['offset_'+c for c in B],offset)),**dict(zip(['spread_'+c for c in B],spread)),**dict(zip(['mask_offset_'+c for c in B],mask)))
                if k in exact:
                    _,c2,t2=match(ov,forbid=(k,exact[k]));f['essential_pair']=int(c2<count);f['alternative_margin']=total-t2 if c2==count else 1.
            output.append(f)
            diagnostics.append(dict(id=r.id,frame_id=fid,class_name=cls,exact_assigned=k in exact,naive_assigned=k in naive,
                exact_cardinality=count,naive_cardinality=len(naive)))
    pd.DataFrame(diagnostics).to_csv(A/f'audit/matching_{split}.csv',index=False)
    result=pd.DataFrame(output).set_index('id').loc[rows.id].fillna(0.)
    result.to_csv(A/f'features/features_{split}.csv');return result

def run():
    verify_match();x=make_features(q,'oof');xt=make_features(te,'test')
    cls=[f'class_{c}' for c in CL]
    basic=['confidence','best_iou','ref_confidence']+cls
    group=['assigned','assignment_iou','competitors','rival_margin','essential_pair','alternative_margin','group_size','reference_count','second_iou']
    coord=['width','height','cx','cy']+['offset_'+c for c in B]+['spread_'+c for c in B]+['mask_offset_'+c for c in B]
    training=[]
    for variant,cols in [('group',basic+group),('coordinates',basic+coord),('group_coordinates',basic+group+coord)]:
        for model_name in ['logreg_003','logreg_01','catboost_shallow']:
            pred=np.zeros(len(q));test=[]
            for fit in DAYS:
                tr=q.day.eq(fit).to_numpy();held=~tr
                assert set(q.loc[tr,'frame_id']).isdisjoint(q.loc[held,'frame_id'])
                if model_name.startswith('logreg'):
                    model=make_pipeline(StandardScaler(),LogisticRegression(C=.03 if model_name.endswith('003') else .1,max_iter=2000))
                else:model=CatBoostClassifier(iterations=100,depth=2,learning_rate=.03,l2_leaf_reg=20,loss_function='Logloss',random_seed=28,verbose=False,thread_count=2,allow_writing_files=False)
                weight=1/q.loc[tr].groupby('frame_id').id.transform('size').to_numpy(float);weight*=len(weight)/weight.sum()
                kwargs={'logisticregression__sample_weight':weight} if model_name.startswith('logreg') else {'sample_weight':weight}
                model.fit(x.iloc[np.flatnonzero(tr)][cols],q.loc[tr,'error'],**kwargs)
                pred[held]=model.predict_proba(x.iloc[np.flatnonzero(held)][cols])[:,1];test.append(model.predict_proba(xt[cols])[:,1])
                joblib.dump(dict(model=model,columns=cols,fit_day=fit,pretrained_references_only=True),A/f'models/{variant}_{model_name}_{fit}.joblib')
                training.append(dict(variant=variant,model=model_name,fit_day=fit,fit_rows=int(tr.sum()),fit_frames=q.loc[tr,'frame_id'].nunique(),columns=len(cols)))
            pt=np.mean(test,axis=0)
            pd.DataFrame(dict(id=q.id,probability=pred)).to_csv(A/f'features/{variant}_{model_name}_oof.csv',index=False)
            pd.DataFrame(dict(id=te.id,probability=pt)).to_csv(A/f'features/{variant}_{model_name}_test.csv',index=False)
            for weight in [.05,.1,.2]:
                register(f'{variant}_{model_name}_w{weight}',(1-weight)*base_oof+weight*pred,(1-weight)*current+weight*pt,
                    dict(type='research_coordinate_group_ablation',variant=variant,model=model_name,weight=weight,primary_target='official error',standalone_oof=metric(q,pred),pretrained_references_only=True))
    pd.DataFrame(training).to_csv(A/'results/training.csv',index=False)
    metrics=[json.loads(f.read_text()) for f in (A/'results').glob('*.json')]
    pd.DataFrame([{k:r[k] for k in ['experiment','pooled_ap','day_25','day_26','changed_test','visual_regressions_gt002','remaining_known']} for r in metrics]).sort_values('pooled_ap',ascending=False).to_csv(A/'results/metrics.csv',index=False)
    for split in ['oof','test']:
        d=pd.read_csv(A/f'audit/matching_{split}.csv');print('MATCHING',split,'assignment status differences',int(d.exact_assigned.ne(d.naive_assigned).sum()),flush=True)
    print(pd.read_csv(A/'results/metrics.csv').head(10).to_string(index=False))
def compare_v27():
    rows=[]
    for (fid,cls),g in q.groupby(['frame_id','class_name']):
        gt=h[h.frame_id.eq(fid)&h.class_name.eq(cls)]
        if not len(gt):continue
        ans,_,_=match(iou(g[B],gt[B]))
        for k,r in enumerate(g.itertuples()):
            rows.append(dict(id=r.id,frame_id=fid,class_name=cls,official_error=r.error,matched_error=int(k not in ans)))
    pd.DataFrame(rows).to_csv(A/'audit/manual_matcher_vs_official.csv',index=False)
    base=align(ROOT/'rascar_v27/results/oof/release_candidate.csv',q,'score')
    test=align(ROOT/'rascar_v27/submission_v27_candidate.csv',te,'score')
    for variant in ['group','coordinates','group_coordinates']:
        for model in ['logreg_003','logreg_01','catboost_shallow']:
            pred=align(A/f'features/{variant}_{model}_oof.csv',q,'probability')
            pt=align(A/f'features/{variant}_{model}_test.csv',te,'probability')
            register(f'v27_{variant}_{model}_w005',.95*base+.05*pred,.95*test+.05*pt,
                dict(baseline='V27 candidate',weight=.05,notes='Fixed small-weight comparison; not promoted merely for better pooled AP'))
if __name__=='__main__':
    run();compare_v27()
