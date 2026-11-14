from infer import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import joblib
sys.path.insert(0,str(ROOT/'rascar_v46'))
from common46 import baseline,metrics,DAYS
def main():
    q,te,h=inputs();bo,bt=baseline(q,te);rows=[];references=[]
    for split in ['oof','test']:
        path=A/f'features/{split}.jsonl'
        if not path.exists():continue
        for d in map(json.loads,path.read_text().splitlines()):
            rows.append({k:d[k] for k in ['id','frame_id','day','split','class_name']}|d['features']);ref=max([r for r in d['references'] if r['positive']>=.1],key=lambda x:x['overlap'],default=None);references.append(dict(id=d['id'],frame_id=d['frame_id'],day=d['day'],split=split,source='query' if split=='oof' else 'test',fit_day='2026-10-26' if d['day']=='2026-10-25' else '2026-10-25',**dict(zip(['ref_'+v for v in B],ref['xyxy'] if ref else [np.nan]*4))))
    data=pd.DataFrame(rows);assert data.id.is_unique;assert set(q.id)<=set(data.id);data['error']=data.id.map(q.set_index('id').error);columns=[c for c in data if c not in ['id','frame_id','day','split','class_name','error']]
    for c in CL:data['class_'+c]=data.class_name.eq(c).astype(float);columns.append('class_'+c)
    data.to_csv(A/'features/features.csv',index=False);pd.DataFrame(references).to_csv(A/'features/references.csv',index=False);test_ready=set(te.id)<=set(data.id);predictions=[]
    for fit in DAYS:
        tr=data[data.split.eq('oof')&data.day.eq(fit)];ev=data[(data.split.eq('oof')&data.day.ne(fit))|data.split.eq('test')]
        for kind in ['logreg','catboost']:
            if kind=='logreg':
                scale=StandardScaler().fit(tr[columns]);model=LogisticRegression(C=.1,max_iter=2000).fit(scale.transform(tr[columns]),tr.error);pred=model.predict_proba(scale.transform(ev[columns]))[:,1];joblib.dump((columns,scale,model),A/f'models/{kind}_fit{fit[-2:]}.joblib')
            else:
                model=CatBoostClassifier(iterations=150,depth=2,learning_rate=.03,l2_leaf_reg=10,random_seed=5001,verbose=False,thread_count=3,allow_writing_files=False);model.fit(tr[columns],tr.error);pred=model.predict_proba(ev[columns])[:,1];model.save_model(str(A/f'models/{kind}_fit{fit[-2:]}.cbm'));(A/'models/catboost_columns.json').write_text(json.dumps(columns))
            predictions.append(ev[['id','split','day','class_name','frame_id']].assign(fit_day=fit,kind=kind,p_error=pred))
    pp=pd.concat(predictions);pp.to_csv(A/'features/predictions.csv',index=False);results=[]
    def save(name,s,t):
        q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False)
        if t is not None:te[['id']].assign(score=t).to_csv(A/f'results/submissions/{name}.csv',index=False)
        results.append(dict(name=name,**metrics(q,s),changed_test=int((abs(t-bt)>1e-9).sum()) if t is not None else None))
    for kind,g in pp.groupby('kind'):
        p=g[g.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();dt=None
        if test_ready:
            tp=np.stack([z.set_index('id').loc[te.id].p_error.to_numpy() for _,z in g[g.split.eq('test')].groupby('fit_day')]);up=(tp>=bt).all(0);down=(tp<=bt).all(0);dt=np.where(up,tp.min(0),tp.max(0))-bt;dt[~(up|down)]=0
        for scope in ['all']+CL:
            om=np.ones(len(q),bool) if scope=='all' else q.class_name.eq(scope).to_numpy();tm=np.ones(len(te),bool) if scope=='all' else te.class_name.eq(scope).to_numpy()
            for w in [.05,.1]:
                s=bo.copy();s[om]+=w*(p[om]-bo[om]);t=None
                if test_ready:t=bt.copy();t[tm]+=w*dt[tm]
                save(f'owl_{kind}_{scope}_w{int(w*100):02}',s,t)
    for scope in ['all']+CL:
        s=bo.copy();o=data.set_index('id').loc[q.id];om=(o.missing_c01.eq(0)&(o.class_name.eq(scope) if scope!='all' else True)).to_numpy();target=np.clip((.85-o.iou_c01.to_numpy())/.5,0,1);s[om]+=.1*(target[om]-bo[om]);t=None
        if test_ready:
            f=data.set_index('id').loc[te.id];tm=(f.missing_c01.eq(0)&(f.class_name.eq(scope) if scope!='all' else True)).to_numpy();target=np.clip((.85-f.iou_c01.to_numpy())/.5,0,1);t=bt.copy();t[tm]+=.1*(target[tm]-bt[tm])
        save(f'owl_geometry_{scope}_w10',s,t)
    result=pd.DataFrame(results).sort_values('ap',ascending=False);result.to_csv(A/'results/experiments.csv',index=False);print(result.to_string(index=False));print('test_ready',test_ready,flush=True)
if __name__=='__main__':main()
