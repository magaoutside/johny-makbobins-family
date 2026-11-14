from common17 import *
from scipy.special import expit

for mode in ['bench_distribution','bench_classifier','bench_consensus']:
    for weight in [.4,.65]:
        predictions=[]
        for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
            split='test' if test else 'oof';f=align(A/('seat_features_test_all.csv' if test else 'seat_features_oof.csv'),rows)
            qual=align(A/f'seat_catboost_{split}.csv',rows,'quality');aspect=(rows.x2-rows.x1)/(rows.y2-rows.y1)
            eligible=rows.class_name.eq('person').to_numpy()&(f.head_count.to_numpy()==1)&(f.template_n.to_numpy()>=3)&(f.seat_distance.to_numpy()<1.2)&f.back.to_numpy().astype(bool)&(aspect.to_numpy()<1.6)
            height=(f.height_ratio_q50.to_numpy()<.8)&(f.top_offset.abs().to_numpy()<.2)
            if mode=='bench_distribution':signal=expit((.6-f.iou_q75.to_numpy())/.07);eligible&=height
            elif mode=='bench_classifier':signal=qual;eligible&=(aspect.to_numpy()>.85)&(qual>=.65)
            else:
                distribution=expit((.6-f.iou_q75.to_numpy())/.07)
                signal=np.minimum(qual,distribution);eligible&=height&(qual>=.6)
            s=base.copy()
            if mode=='bench_classifier':s[eligible]+=weight*np.maximum(signal[eligible]-base[eligible],0)
            else:s[eligible]+=weight*(1-s[eligible])*signal[eligible]
            predictions.append(s)
        register(f'{mode}_w{weight}',*predictions,details=dict(weight=weight,mode=mode,bench_only=True,aspect_max=1.6,head_count=1,minimum_templates=3))
