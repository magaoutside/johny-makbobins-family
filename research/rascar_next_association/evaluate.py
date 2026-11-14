"""Two fixed day-safe rankers plus a direct-geometry diagnostic; frozen experts."""
from common import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import joblib

def save(name,values):
    result={'name':name,**metric(q,values['oof']),'submitted':False,'public_ap':None};changes=[]
    for split,rows in [('oof',q),('test',te)]:
        vals=values[split];assert len(vals)==len(rows) and np.isfinite(vals).all() and ((vals>=0)&(vals<=1)).all()
        path=A/(f'results/oof/{name}.csv' if split=='oof' else f'results/submissions/submission_{name}_RESEARCH.csv')
        (rows.assign(score=vals) if split=='oof' else rows[['id']].assign(score=vals)).to_csv(path,index=False)
        delta=vals-BASE[split];sel=abs(delta)>1e-10;result['changed_'+split]=int(sel.sum());changes.append(rows.assign(split=split,old_score=BASE[split],new_score=vals,delta=delta)[sel])
        order=np.argsort(-BASE[split],kind='stable');s=vals[order];result['rank_changes_'+split]=int((np.triu(s[:,None]<s[None,:],1)).sum())
    pd.concat(changes).to_csv(A/f'audit/{name}_changes.csv',index=False)
    return result

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--local',action='store_true');args=p.parse_args();suffix='_local' if args.local else ''
    feat=pd.read_csv(A/f'features/query_features{suffix}.csv');oo=feat[feat.split.eq('oof')].copy();tt=feat[feat.split.eq('test')].copy();features=[c for c in feat if c.startswith('f_')]
    target=q.set_index('id').loc[oo.id,'error'].to_numpy();xo=oo[features].to_numpy();xt=tt[features].to_numpy();assert np.isfinite(xo).all() and np.isfinite(xt).all()
    pred={};fit_manifest=[]
    for kind in ['logreg','catboost']:
        po=np.zeros(len(oo));test_parts=[]
        for day in DAYS:
            train=oo.day.eq(day).to_numpy();held=~train;part=oo[train];counts=part.frame_id.value_counts();weights=np.array([1/counts[f] for f in part.frame_id]);weights*=len(part)/weights.sum()
            if kind=='logreg':model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000));model.fit(xo[train],target[train],logisticregression__sample_weight=weights)
            else:model=CatBoostClassifier(depth=2,iterations=100,l2_leaf_reg=20,learning_rate=.05,random_seed=4101,verbose=False,allow_writing_files=False,thread_count=3);model.fit(xo[train],target[train],sample_weight=weights)
            po[held]=model.predict_proba(xo[held])[:,1];test_parts.append(model.predict_proba(xt)[:,1]);joblib.dump(model,A/f'models/{kind}{suffix}_fit{day[-2:]}.joblib')
            fit_manifest.append(dict(kind=kind,fit_day=day,fit_frames=sorted(part.frame_id.unique()),n=len(part),feature_names=features,base_score_feature=False,test_targets=False))
        pred[kind]={'oof':po,'test':np.mean(test_parts,axis=0)}
        np.savez_compressed(A/f'features/{kind}{suffix}_probabilities.npz',oof_ids=oo.id.to_numpy(str),oof=po,test_ids=tt.id.to_numpy(str),test=pred[kind]['test'],test_fit25=test_parts[0],test_fit26=test_parts[1])
    pred['geometry']={'oof':np.clip((.85-oo.f_ref_iou.to_numpy())/.5,0,1),'test':np.clip((.85-tt.f_ref_iou.to_numpy())/.5,0,1)}
    results=[{'name':'V31',**metric(q,BASE['oof']),'submitted':True,'public_ap':.96239}]
    for kind,predictions in pred.items():
        standalone=AP(target,predictions['oof']);print('STANDALONE',kind,standalone,flush=True)
        for mode in ['rescue_only','two_sided']:
            for cls in ['person','nude_hands','both']:
                vals={}
                for split,rows,f in [('oof',q,oo),('test',te,tt)]:
                    base=rows.id.map(dict(zip(rows.id,BASE[split]))).to_numpy();positions=rows.reset_index().set_index('id').loc[f.id,'index'].to_numpy()
                    gate=f.gate.to_numpy(bool)&(True if cls=='both' else f.class_name.eq(cls).to_numpy())
                    delta=predictions[split]-base[positions]
                    if mode=='rescue_only':delta=np.minimum(delta,0)
                    base[positions]+=.1*gate*delta;vals[split]=np.clip(base,0,1)
                name=f'bpj{suffix}_{kind}_{mode}_{cls}_w010';result=save(name,vals);result['standalone_person_hands_ap']=float(standalone);results.append(result)
    pd.DataFrame(results).to_csv(A/f'results/experiments{suffix}.csv',index=False)
    (A/f'models/fit_manifest{suffix}.json').write_text(json.dumps(fit_manifest,indent=2))
    print(pd.DataFrame(results)[['name','ap','2026-10-25','2026-10-26','changed_oof','changed_test','rank_changes_test']].to_string(index=False))

if __name__=='__main__':main()
