"""Verify the two delivered rankings from frozen model outputs/features."""
from common26 import *

def main():
    prefix='tray_query_oriented_monotone_grouped'
    of=pd.concat([pd.read_csv(A/f'features/{prefix}_held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id]
    tf=[align(A/f'features/{prefix}_test_{fit}.csv',te) for fit in DAYS]
    ft=(tf[0]+tf[1])/2
    diffs={}
    for split,rows,base,f in [('oof',q,base_oof,of),('test',te,current,ft)]:
        hand=align(ROOT/f'rascar_v25/release/expected/hand_{split}.csv',rows,'score')
        gate=guard(split,rows,'tray')&rows.class_name.eq('tray').to_numpy()&f.support.ge(.5).to_numpy()&f.geometry_iou.ge(.8).to_numpy()
        if split=='test':gate &= np.logical_and.reduce([(z.support.ge(.5)&z.geometry_iou.ge(.8)).to_numpy() for z in tf])
        error=np.clip((.9-f.geometry_iou.to_numpy())/.5,0,1)
        tray=np.clip(base+.2*gate*(np.minimum(base,error)-base),0,1)
        scores=np.clip(hand+tray-base,0,1)
        expected=align(A/('submission_v26_hands_and_monotone_trays_research.csv' if split=='test' else 'results/oof/hands_and_monotone_trays_research.csv'),rows,'score')
        diff=float(np.max(abs(scores-expected)));assert diff<1e-12
        diffs[split]=diff
    (A/'replay_verification.json').write_text(json.dumps(diffs,indent=2))
    print(diffs)

if __name__=='__main__':main()
