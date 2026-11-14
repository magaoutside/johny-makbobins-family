from common20 import *
raw=[]
for variant in ['clean','expanded']:
    po=pd.concat([pd.read_csv(A/f'gpu_results/quality/{fit}/{variant}_held.csv') for fit in DAYS]).set_index('id').loc[q.id]
    tdays=[pd.read_csv(A/f'gpu_results/quality/{fit}/{variant}_test.csv').set_index('id').loc[te.id] for fit in DAYS];pt=(tdays[0]+tdays[1])/2
    po.reset_index().to_csv(A/f'quality_{variant}_oof.csv',index=False);pt.reset_index().to_csv(A/f'quality_{variant}_test.csv',index=False)
    raw.append(dict(variant=variant,**metric(q,po.quality_error.to_numpy())))
    for classes in [CL,['person'],['nude_hands'],['cup','tray']]:
        tag='all' if classes==CL else '_'.join(classes)
        for weight in [.1,.2]:
            outputs=[]
            for rows,base,pred in [(q,q.score.to_numpy(),po),(te,current,pt)]:
                m=rows.class_name.isin(classes).to_numpy();s=base.copy();s[m]=(1-weight)*s[m]+weight*pred.quality_error.to_numpy()[m];outputs.append(s)
            register(f'quality_{variant}_{tag}_w{weight}',*outputs,dict(model='mask-conditioned ResNet18',classes=classes,weight=weight,training=variant))
    # Selective rescue needs agreement of the two trained models on test;
    # held-day examples have only their legitimate opposite-day model.
    for weight in [.25,.5]:
        outputs=[]
        for rows,base,pred,test in [(q,q.score.to_numpy(),po,False),(te,current,pt,True)]:
            split='test' if test else 'oof';old=align(ROOT/f'rascar_v18/replay_inputs/{split}.csv',rows)
            dup=boolean(old.duplicate_veto)
            groups=pd.read_csv(ROOT/f'rascar_v15/duplicate_groups_{split}.csv')
            dup |= rows.id.isin(groups.loc[groups.reason.eq('duplicate_loser'),'id']).to_numpy()
            valid=(pred.quality_error.to_numpy()<.2)&(pred.presence.to_numpy()>.8)&(pred.multiple.to_numpy()<.2)
            if test:
                valid &= (np.maximum(tdays[0].quality_error,tdays[1].quality_error)<.3)&(np.minimum(tdays[0].presence,tdays[1].presence)>.7)
            eligible=rows.class_name.isin(['nude_hands','cup','tray']).to_numpy()&~dup&valid&(base>.6)
            s=base.copy();s[eligible]=(1-weight)*s[eligible]+weight*pred.quality_error.to_numpy()[eligible];outputs.append(s)
        register(f'quality_{variant}_guarded_rescue_w{weight}',*outputs,dict(weight=weight,quality_max=.2,presence_min=.8,multiple_max=.2,duplicate_veto=True))
pd.DataFrame(raw).to_csv(A/'quality_standalone_metrics.csv',index=False)
print('STANDALONE',pd.DataFrame(raw).to_string(index=False))
