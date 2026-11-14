"""Require independent geometric evidence before a new learned penalty."""
from common27 import *

def pretrained_features(prefix,split,rows):
    paths=[ROOT/f'rascar_v25/features/{prefix}_{"held" if split=="oof" else "test"}_{fit}.csv' for fit in DAYS]
    ff=[pd.read_csv(p).set_index('id') for p in paths]
    return pd.concat(ff).loc[rows.id] if split=='oof' else (ff[0].loc[rows.id]+ff[1].loc[rows.id])/2

def main():
    for mode in ['quality_agreement','geometric_agreement','both_agreements']:
        for weight in [.3,.6]:
            output=[]
            for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
                p=align(A/f'features/contrastive_pair_visual_{split}.csv',rows)
                z=align(V22/f'replay_inputs/{split}.csv',rows)
                f=pretrained_features('wilor_detector_padwide',split,rows)
                anatomy=pretrained_features('rtmw_padwide',split,rows)
                quality=np.minimum(z.quality_logreg,z.quality_svm).to_numpy()>.65
                # A concrete contradictory box, not mere detector absence.
                geo=z.zoom_confidence.ge(.6).to_numpy()&z.zoom_overlap.between(.15,.55).to_numpy()
                geo&=f.reference_conf.ge(.4).to_numpy()&f.best_iou.between(.15,.55).to_numpy()
                geo&=anatomy.best_iou.lt(.6).to_numpy()
                gate=quality if mode=='quality_agreement' else geo if mode=='geometric_agreement' else quality&geo
                gate&=rows.class_name.eq('nude_hands').to_numpy()&p.probability.gt(.7).to_numpy()
                if split=='test':gate&=p[['p25','p26']].min(axis=1).gt(.6).to_numpy()
                score=base+weight*gate*np.maximum(p.probability.to_numpy()-base,0)
                pd.DataFrame(dict(id=rows.id,gate=gate,quality_agreement=quality,geometry_agreement=geo,target=p.probability.to_numpy(),score=score)).to_csv(A/f'features/agreement_{mode}_{weight}_{split}.csv',index=False)
                output.append(score)
            register(f'agreement_{mode}_w{weight}',*output,dict(type='pairwise_quality_with_independent_evidence',mode=mode,weight=weight))
if __name__=='__main__':main()
