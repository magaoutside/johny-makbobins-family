"""Zoom/continuation/targeted ablations; global one-to-one instance assignment."""
from common22 import *
import sys
sys.path.insert(0,str(ROOT/'rascar_v20'))
from evaluate_sahi import clusters,evidence,old_veto,detection_metrics
for variant in (sys.argv[1:] or ['frozen','control','targeted']):
    raw={fit:pd.read_csv(A/f'gpu_results/hand_training/{fit}/{variant}_zoom_references.csv').rename(columns={'zoom':'tile'}) for fit in DAYS}
    fs={};refout=[]
    for split,rows in [('oof',q),('test',te)]:
        if split=='oof':
            pieces=[]
            for fit,held in zip(DAYS,DAYS[::-1]):
                rr=clusters(raw[fit][raw[fit].split.eq('held')],2);pieces.append(evidence(rows[rows.day.eq(held)],rr));refout.append(rr)
            f=pd.concat(pieces).set_index('id').loc[rows.id].reset_index()
        else:
            rr=clusters(pd.concat([g[g.split.eq('test')] for g in raw.values()],ignore_index=True),2);f=evidence(rows,rr)
        fs[split]=f;f.to_csv(A/f'zoom_{variant}_{split}.csv',index=False)
        if split=='test':rr.to_csv(A/f'zoom_{variant}_refs_test.csv',index=False)
    pd.concat(refout).to_csv(A/f'zoom_{variant}_refs_oof.csv',index=False)
    (A/f'zoom_{variant}_detection_metrics.json').write_text(json.dumps(detection_metrics(pd.concat(refout),q,['nude_hands']),indent=2))
    for confidence in [.5,.7]:
      for weight in [.3,.5]:
        outputs=[]
        for split,rows,base in [('oof',q,q.score.to_numpy()),('test',te,current)]:
            f=fs[split];sem=align(ROOT/f'rascar_v16/semantic_tight_{split}.csv',rows,'p_nude_hands')
            # No test labels or per-ID rules. Do not rescue competing or merged instances.
            gate=rows.class_name.eq('nude_hands').to_numpy()&boolean(f.assigned)&f.instances.eq(1).to_numpy()&f.overlap.ge(.65).to_numpy()&f.confidence.ge(confidence).to_numpy()&f.coherence.ge(.75).to_numpy()&(sem>=.75)&~old_veto(rows,split=='test')&~boolean(f.duplicate_loser)
            if split=='test':gate &= f.folds.ge(2).to_numpy()
            target=np.clip((.85-f.overlap.to_numpy())/.5,0,1);s=base.copy();s[gate]-=weight*np.maximum(s[gate]-target[gate],0);outputs.append(s)
            pd.DataFrame(dict(id=rows.id.to_numpy(),gate=gate,target=target,score=s)).to_csv(A/f'zoom_{variant}_c{confidence}_w{weight}_{split}.csv',index=False)
        register(f'zoom_{variant}_c{confidence}_w{weight}',*outputs,dict(model=variant,confidence=confidence,weight=weight,overlap=.65,scales=[96,160,256],semantic=.75,one_to_one=True,test_fold_agreement=True))
