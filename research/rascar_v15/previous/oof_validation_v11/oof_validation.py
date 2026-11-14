from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score

FIXED={'person':'fusion25_iou30','zoom_weight':.15}

def load_data(root):
    policy=json.loads((root/'scoring_policy.json').read_text())
    train=pd.read_csv(root/'train.csv',dtype={'id':str})
    frames=pd.read_csv(root/'train_frames.csv')
    frames['day']=pd.to_datetime(frames.timestamp,utc=True,format='mixed').dt.strftime('%Y-%m-%d')
    train=train.merge(frames[['frame_id','day']],on='frame_id',validate='many_to_one')
    train['class_name']=train.class_name.replace(policy['aliases'])
    train['evaluated']=train.class_name.isin(set(policy['classes'])-set(policy['test_excluded_classes']))
    assert train.day.notna().all() and train.id.is_unique and train.error.isin([0,1]).all()
    assert len(train)==670 and train.evaluated.sum()==198
    test=pd.read_csv(root/'test.csv',dtype={'id':str})
    assert not test.class_name.isin(policy['test_excluded_classes']).any()
    assert set(test.class_name).issubset(set(policy['classes']))
    return train,policy

def score_like_lb(solution,submission):
    if list(submission.columns)!=['id','score']:raise ValueError('Expected id,score only')
    if not solution.id.is_unique or not submission.id.is_unique:raise ValueError('Duplicate ID')
    if set(solution.id)!=set(submission.id):raise ValueError('Missing or unexpected ID')
    if not np.isfinite(submission.score).all() or not submission.score.between(0,1).all():raise ValueError('Invalid score')
    z=solution[['id','error','evaluated']].merge(submission,on='id',validate='one_to_one')
    selected=z[z.evaluated]
    return float(average_precision_score(selected.error,selected.score))

class DayBank:
    def __init__(self,train_rows,source,features,context):
        assert train_rows.day.nunique()==1
        self.day=str(train_rows.day.iloc[0]);self.train_ids=set(train_rows.id)
        indices=np.flatnonzero(source.day.eq(self.day))
        self.records=source.iloc[indices].copy().reset_index(drop=True)
        self.features=features[indices].copy();self.context=context
        assert 'error' not in self.records
        self.records['error']=0
        inputs=self.records.kind.eq('input')
        labels=self.records.loc[inputs,'id'].map(train_rows.set_index('id').error)
        assert labels.notna().all()
        self.records.loc[inputs,'error']=labels.to_numpy(dtype=int)
        assert self.records.day.eq(self.day).all()
        self.fingerprint=hashlib.sha256(self.records[['record_id','error']].to_csv(index=False).encode()).hexdigest()

    def predict(self,queries,qfeatures,inner=False):
        assert 'error' not in queries and len(queries)==len(qfeatures)
        if not inner:assert queries.day.ne(self.day).all()
        result=[]
        for i,(_,q) in enumerate(queries.iterrows()):
            excluded={q.frame_id}
            if inner and q.frame_id in self.context.index:
                excluded.update(self.context.loc[q.frame_id,['previous','next']].dropna().tolist())
            bank=self.records[self.records.class_name.eq(q.class_name)&~self.records.frame_id.isin(excluded)]
            if not inner:assert q.frame_id not in set(self.records.frame_id)
            row={'id':q.id,'fit_day':self.day,'query_day':q.day,'fallback':False}
            similarities={}
            for label,name in [(0,'good'),(1,'bad')]:
                neighbors=bank[bank.error.eq(label)].copy()
                neighbors['similarity']=self.features[neighbors.index]@qfeatures[i]
                neighbors=neighbors.sort_values('similarity',ascending=False).drop_duplicates('frame_id').head(3)
                row[name+'_frames']='|'.join(neighbors.frame_id)
                row[name+'_records']='|'.join(neighbors.record_id)
                if len(neighbors):similarities[name]=float(neighbors.similarity.mean())
            if len(similarities)==2:
                row['knn_score']=float(expit((similarities['bad']-similarities['good'])/.05))
            else:
                # Only remaining training rows enter this fallback; no query label.
                row['knn_score']=float((bank.error.sum()+1)/(len(bank)+2));row['fallback']=True
            result.append(row)
        return pd.DataFrame(result)

def combine(comp,knn,config):
    k=knn.set_index('id').loc[comp.id,'knn_score'].to_numpy()
    person=comp.class_name.eq('person').to_numpy()
    z=float(config['zoom_weight'])
    scores=(1-z)*(.9*comp.sam_original.to_numpy()+.1*k)+z*comp.zoom_score.to_numpy()
    scores[person]=.65*comp.loc[person,config['person']].to_numpy()+.25*comp.loc[person,'wevis_full'].to_numpy()+.1*k[person]
    scores[comp.duplicate.to_numpy(dtype=bool)]=1.
    assert np.isfinite(scores).all()
    return pd.DataFrame({'id':comp.id.to_numpy(),'score':scores.clip(0,1)})

class Validator:
    def __init__(self,root):
        self.root=root
        self.train,self.policy=load_data(root)
        self.components=pd.read_csv(root/'frozen_components.csv',dtype={'id':str})
        assert 'error' not in self.components
        self.source=pd.read_csv(root/'bank_records_without_labels.csv',dtype={'id':str})
        self.features=np.load(root/'bank_features.npy')
        self.qindex=pd.read_csv(root/'query_index.csv',dtype={'id':str})
        self.qfeatures=np.load(root/'query_features.npy')
        self.query_lookup=pd.Series(np.arange(len(self.qindex)),index=self.qindex.id)
        self.context=pd.read_csv(root/'train_context.csv').set_index('frame_id')
        self.unscored=pd.read_csv(root/'unscored_frozen_scores.csv',dtype={'id':str}).set_index('id').unscored_sam_score
        self.plan=json.loads((root/'fold_selection_plan.json').read_text())
        assert set(self.components.id)==set(self.train.loc[self.train.evaluated,'id'])
        assert set(self.unscored.index)==set(self.train.loc[~self.train.evaluated,'id'])

    def query_features(self,queries):
        return self.qfeatures[self.query_lookup.loc[queries.id].to_numpy()]

    def fold(self,fit_day,hold_day,labels=None):
        data=self.train if labels is None else labels
        fitting=data[data.day.eq(fit_day)].copy()
        held=data[data.day.eq(hold_day)].copy()
        bank=DayBank(fitting,self.source,self.features,self.context)
        train_frames=set(fitting.frame_id)
        held_frames=set(held.frame_id)
        held_context=set(self.context.loc[list(held_frames),['previous','next']].stack())|held_frames
        assert train_frames.isdisjoint(held_context)
        assert set(bank.records.frame_id).isdisjoint(held_context)
        train_queries=fitting.loc[fitting.evaluated,['id','frame_id','class_name','day']]
        held_queries=held.loc[held.evaluated,['id','frame_id','class_name','day']]
        inner=bank.predict(train_queries,self.query_features(train_queries),inner=True)
        outer=bank.predict(held_queries,self.query_features(held_queries),inner=False)
        train_comp=self.components.set_index('id').loc[train_queries.id].reset_index()
        held_comp=self.components.set_index('id').loc[held_queries.id].reset_index()
        trials=[]
        for j,candidate in enumerate(self.plan['candidates']):
            pred=combine(train_comp,inner,candidate)
            y=fitting.set_index('id').loc[pred.id,'error']
            trials.append(dict(candidate=j,**candidate,training_inner_ap=float(average_precision_score(y,pred.score))))
        # All candidate selection is complete before held-day labels are scored.
        chosen=max(trials,key=lambda r:(r['training_inner_ap'],-r['candidate']))
        configs={'fixed_v8':FIXED,'train_day_selected':{'person':chosen['person'],'zoom_weight':chosen['zoom_weight']}}
        predictions={}
        for mode,config in configs.items():
            ranked=combine(held_comp,outer,config)
            full=held[['id','frame_id','class_name','day','evaluated']].merge(ranked,on='id',how='left',validate='one_to_one')
            inactive=~full.evaluated
            full.loc[inactive,'score']=full.loc[inactive,'id'].map(self.unscored)
            full['fit_day']=fit_day
            assert np.isfinite(full.score).all() and full.id.is_unique
            predictions[mode]=full
        info={'fit_day':fit_day,'held_day':hold_day,'fit_rows':len(fitting),'held_rows':len(held),'fit_scored':len(train_queries),'held_scored':len(held_queries),'bank_input':int(bank.records.kind.eq('input').sum()),'bank_gt':int(bank.records.kind.eq('gt').sum()),'bank_fingerprint':bank.fingerprint,'held_context_overlap':0,'outer_fallbacks':int(outer.fallback.sum()),'inner_fallbacks':int(inner.fallback.sum()),'selected_config':configs['train_day_selected'],'selected_training_inner_ap':chosen['training_inner_ap']}
        return predictions,info,pd.DataFrame(trials),inner,outer

    def run(self):
        all_predictions={'fixed_v8':[],'train_day_selected':[]};folds=[];tests={}
        for fit_day,hold_day in self.plan['folds']:
            preds,info,trials,inner,outer=self.fold(fit_day,hold_day)
            # Poison every held-day label and refit; predictions and choices must not change.
            poisoned=self.train.copy();m=poisoned.day.eq(hold_day)
            poisoned.loc[m,'error']=1-poisoned.loc[m,'error']
            again,again_info,_,_,_=self.fold(fit_day,hold_day,poisoned)
            assert info==again_info
            for mode in preds:
                assert np.array_equal(preds[mode].score.to_numpy(),again[mode].score.to_numpy())
                all_predictions[mode].append(preds[mode])
            trials.to_csv(self.root/f'coefficient_selection_fit_{fit_day}.csv',index=False)
            inner.to_csv(self.root/f'inner_neighbors_fit_{fit_day}.csv',index=False)
            outer.to_csv(self.root/f'oof_neighbors_predict_{hold_day}.csv',index=False)
            folds.append(info)
        tests['heldout_label_poisoning_invariant']=True
        metrics=[];full_outputs={}
        for mode,parts in all_predictions.items():
            oof=pd.concat(parts,ignore_index=True)
            assert oof.id.is_unique and len(oof)==len(self.train) and set(oof.id)==set(self.train.id)
            assert oof.fit_day.ne(oof.day).all()
            oof=self.train[['id','error']].merge(oof,on='id',validate='one_to_one')
            oof.to_csv(self.root/f'{mode}_oof_all_rows.csv',index=False)
            scored=oof[oof.evaluated]
            scored[['id','score']].to_csv(self.root/f'{mode}_oof_scored.csv',index=False)
            full_outputs[mode]=oof
            pooled=score_like_lb(self.train,oof[['id','score']])
            by_day={d:float(average_precision_score(g.error,g.score)) for d,g in scored.groupby('day')}
            by_class={c:float(average_precision_score(g.error,g.score)) for c,g in scored.groupby('class_name')}
            metrics.append({'mode':mode,'pooled_oof_ap':pooled,'mean_fold_ap_diagnostic_only':float(np.mean(list(by_day.values()))),'by_day':by_day,'by_class':by_class,'n_scored':len(scored),'n_all_rows':len(oof)})
        fixed=full_outputs['fixed_v8'];pred=fixed[['id','score']]
        value=score_like_lb(self.train,pred)
        assert value==score_like_lb(self.train,pred.sample(frac=1,random_state=11))
        tests['id_order_invariant']=True
        changed_labels=self.train.copy();m=~changed_labels.evaluated
        changed_labels.loc[m,'error']=1-changed_labels.loc[m,'error']
        changed_scores=pred.copy();changed_scores.loc[m,'score']=np.arange(m.sum())%2
        assert score_like_lb(changed_labels,changed_scores)==value
        tests['excluded_tables_do_not_affect_ap']=True
        const=pred.assign(score=.5)
        assert abs(score_like_lb(self.train,const)-self.train.loc[self.train.evaluated,'error'].mean())<1e-15
        assert score_like_lb(self.train,const)==score_like_lb(self.train,const.sample(frac=1,random_state=17))
        tests['constant_and_tied_scores_match_sklearn']=True
        bads=[pd.concat([pred,pred.iloc[[0]]]),pred.iloc[:-1],pred.assign(score=np.nan),pred.assign(score=1.1)]
        extra=pred.copy();extra.loc[0,'id']='unexpected_id';bads.append(extra)
        for bad in bads:
            try:score_like_lb(self.train,bad)
            except ValueError:pass
            else:raise AssertionError('Invalid submission accepted')
        tests['invalid_ids_and_scores_rejected']=True
        previous=pd.read_csv(self.root/'baseline_previous_oof.csv',dtype={'id':str}).set_index('id')
        delta=fixed[fixed.evaluated].set_index('id').score-previous.score
        assert delta.abs().max()<1e-7, float(delta.abs().max())
        assert abs(score_like_lb(self.train,pred)-average_precision_score(previous.error,previous.score))<1e-12
        tests['fixed_v8_reproduced']=True
        tests['one_oof_prediction_per_id']=True
        tests['training_day_and_context_disjoint']=True
        baseline=score_like_lb(self.train,self.train[['id']].assign(score=1-self.train.confidence))
        result={'metric':'sklearn.metrics.average_precision_score on concatenated evaluated OOF rows','test_excluded_classes':self.policy['test_excluded_classes'],'baseline_confidence_pooled_ap':baseline,'fixed_v8_max_score_difference':float(delta.abs().max()),'cross_environment_absolute_tolerance':1e-7,'ap_match_tolerance':1e-12,'folds':folds,'metrics':metrics,'checks':tests,'history_warning':self.plan['historical_limit']}
        (self.root/'validation_results.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
        return result

if __name__=='__main__':
    Validator(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent).run()
