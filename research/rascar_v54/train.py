"""Paired cross-day augmentation ablations with parent-balanced auxiliary loss."""
from common54 import *
import argparse, time
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit
from sklearn.metrics import average_precision_score as AP


def configurations():
    families = {'geometry':['geometry'], 'visual':['geometry','proto'],
                'occupancy':['geometry','dense'], 'all':['geometry','proto','dense']}
    specs = {}
    for family, groups in families.items():
        for depth in [2, 4]:
            for budget in [0, .25, 1.]:
                name = f'{family}_d{depth}_all_w{int(budget*100):03}'
                specs[name] = dict(groups=groups, depth=depth, policy='all', budget=budget)
    for family in ['occupancy', 'all']:
        for depth in [2, 4]:
            for policy in ['mild', 'no_gt']:
                for budget in [.25, 1.]:
                    name = f'{family}_d{depth}_{policy}_w{int(budget*100):03}'
                    specs[name] = dict(groups=families[family], depth=depth, policy=policy, budget=budget)
    return specs


def fitting_rows(rows, fit, spec):
    original = rows.split.eq('oof') & rows.day.eq(fit)
    aug = rows.split.eq('augmentation') & rows.day.eq(fit)
    if spec['budget'] == 0:
        aug &= False
    if spec['policy'] == 'mild':
        aug &= rows.strength.le(.12) & rows.family.ne('toward_gt')
    if spec['policy'] == 'no_gt':
        aug &= rows.family.ne('toward_gt')
    train = original | aug
    w = pd.Series(1., index=rows.index[train])
    if aug.any():
        count = rows[aug].groupby('parent_id').size()
        w.loc[aug[aug].index] = spec['budget'] / rows.loc[aug, 'parent_id'].map(count)
        assert abs(w.loc[aug[aug].index].sum() - spec['budget'] * len(count)) < 1e-7
    return train, w


def assess(q, t, oo, tt, name, spec, control):
    bo, bt = incumbent(q,t); results = []
    for scope in ['person', 'nude_hands', 'all']:
        om = q.class_name.eq(scope).to_numpy() if scope != 'all' else np.ones(len(q), bool)
        tm = t.class_name.eq(scope).to_numpy() if scope != 'all' else np.ones(len(t), bool)
        for alpha in [.05, .1, .2]:
            s, z = bo.copy(), bt.copy()
            s[om] = (bo[om] + alpha * oo[om])/(1+alpha)
            z[tm] = (bt[tm] + alpha * tt[tm])/(1+alpha)
            tag = f'{name}_{scope}_a{int(alpha*100):02}'
            q.assign(score=s).to_csv(A / f'results/oof/{tag}.csv', index=False)
            t[['id']].assign(score=z).to_csv(A / f'results/submissions/{tag}.csv', index=False)
            results.append(dict(name=tag, model=name, scope=scope, alpha=alpha, control=control,
                                **spec, **metrics(q,s), delta=AP(q.error,s)-AP(q.error,bo),
                                standalone_ap=AP(q.error,oo), standalone_person_ap=AP(q[q.class_name.eq('person')].error,oo[q.class_name.eq('person')]),
                                changed_test=int((abs(z-bt)>1e-10).sum())))
    return results


def main():
    p = argparse.ArgumentParser(); p.add_argument('--names', nargs='*'); p.add_argument('--seeds', nargs='+', type=int, default=[23])
    p.add_argument('--control', choices=['actual','repeat_parent'], default='actual'); p.add_argument('--tag',default='initial')
    p.add_argument('--target', choices=['error','quality'], default='error')
    args = p.parse_args()
    rows = pd.read_csv(A / 'inputs/records.csv'); groups = json.loads((A / 'features/groups.json').read_text())
    banks = {fit:pd.read_pickle(A / f'features/bank_fit{fit[-2:]}.pkl') for fit in DAYS}
    q,t,h = inputs(); results = []; manifests = []; importances = []
    specs = configurations(); original_lookup = rows[rows.family.eq('original')].reset_index().set_index('id')['index']
    for name,spec in specs.items():
        if args.names and name not in args.names:
            continue
        start = time.time(); cols = list(dict.fromkeys(c for g in spec['groups'] for c in groups[g])) + ['class_name']
        pred = []
        for seed in args.seeds:
            for fit in DAYS:
                tr,w = fitting_rows(rows,fit,spec)
                ev = rows.split.eq('test') | (rows.split.eq('oof') & rows.day.ne(fit))
                assert set(rows.loc[tr,'frame_id']).isdisjoint(rows.loc[ev,'frame_id'])
                target_column = 'error' if args.target == 'error' else 'gt_iou'
                X = banks[fit].loc[tr,cols].copy(); y = rows.loc[tr,target_column].copy()
                if args.control == 'repeat_parent':
                    ix = original_lookup.loc[rows.loc[tr,'parent_id']].to_numpy()
                    X.iloc[:] = banks[fit].loc[ix,cols].to_numpy()
                    y.iloc[:] = rows.loc[ix,target_column].to_numpy()
                target_suffix = '_quality' if args.target == 'quality' else ''
                modelname = f'{name}_{args.control}{target_suffix}_s{seed}_fit{fit[-2:]}'
                model_class = CatBoostClassifier if args.target == 'error' else CatBoostRegressor
                model = model_class(iterations=250 if spec['depth']==2 else 400, depth=spec['depth'],
                         learning_rate=.03, l2_leaf_reg=10 if spec['depth']==2 else 20,
                         random_seed=seed, thread_count=2, loss_function='Logloss' if args.target=='error' else 'RMSE', verbose=False, allow_writing_files=False)
                model.fit(X,y,cat_features=['class_name'],sample_weight=w)
                prob = model.predict_proba(banks[fit].loc[ev,cols])[:,1] if args.target=='error' else expit((.6-np.clip(model.predict(banks[fit].loc[ev,cols]),0,1))/.1)
                model.save_model(str(A/f'models/{modelname}.cbm'))
                pred.append(rows.loc[ev,['id','split','day','class_name']].assign(seed=seed,fit_day=fit,p_error=prob))
                manifests.append(dict(name=modelname, **spec, control=args.control, target=args.target, seed=seed, fit_day=fit, features=cols,
                     original_rows=int((tr & rows.split.eq('oof')).sum()), augmented_rows=int((tr & rows.split.eq('augmentation')).sum()),
                     total_weight=float(w.sum()), train_frames=sorted(rows.loc[tr,'frame_id'].unique()), validation_frames=sorted(q.loc[q.day.ne(fit),'frame_id'].unique()),
                     train_id_sha256=hashlib.sha256('\n'.join(rows.loc[tr,'id']).encode()).hexdigest()))
                importances.extend(dict(model=modelname, feature=c, importance=float(v)) for c,v in zip(cols,model.feature_importances_))
        modeltag = name + '_' + args.control + '_' + args.tag
        predictions = pd.concat(pred); predictions.to_csv(A/f'features/{modeltag}_predictions.csv',index=False)
        oo = predictions[predictions.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy()
        tt = predictions[predictions.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
        new = assess(q,t,oo,tt,modeltag,spec,args.control); results.extend(new)
        pd.DataFrame(results).to_csv(A/f'results/experiments_{args.tag}_{args.control}.csv',index=False)
        pd.DataFrame(importances).to_csv(A/f'audit/importances_{args.tag}_{args.control}.csv',index=False)
        (A/f'models/manifest_{args.tag}_{args.control}.json').write_text(json.dumps(manifests,indent=2))
        print(name,args.control,'AP',round(AP(q.error,oo),6),'bestblend',round(max(r['ap'] for r in new),6),'seconds',round(time.time()-start,1),flush=True)
    print(pd.DataFrame(results).sort_values('ap',ascending=False).head(15).drop(columns='groups').to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
