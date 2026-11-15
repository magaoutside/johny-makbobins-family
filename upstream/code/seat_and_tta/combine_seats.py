"""Directional union of independently evaluated seat evidence; never sum it twice."""
from common17 import *

for weight in [.4,.65]:
    outputs=[]
    for rows,split in [(q,'oof'),(te,'submissions')]:
        prefix='' if split=='oof' else 'submission_'
        components=[align(R/split/f'{prefix}{mode}_w{weight}.csv',rows,'score')
                    for mode in ['bench_distribution','bench_classifier']]
        outputs.append(np.maximum(*components))
    register(f'seat_union_w{weight}',*outputs,details=dict(weight=weight,
        combination='maximum of two directional corrections, no additive double counting',
        scope='back-bench only; one head; >=3 fit-day templates; aspect <1.6'))
