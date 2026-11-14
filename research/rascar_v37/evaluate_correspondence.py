from common37 import *

def guard(rows,split):
    # Read numerical automatic evidence only, never the historical test review.
    table=pd.read_csv(ROOT/f'rascar_v21/object_structure_{split}.csv').set_index('id')
    old=pd.read_csv(ROOT/f'rascar_v22/replay_inputs/{split}.csv').set_index('id')
    allowed=np.ones(len(rows),bool)
    yes=lambda x:str(x).lower() in ['true','1','1.0']
    for i,r in rows.iterrows():
        if r.class_name in ['cup','tray']:
            z=table.loc[r.id];allowed[i]=(z.distinct_instances<=1 and not yes(z.ambiguous_duplicate))
        elif r.class_name=='nude_hands':
            z=old.loc[r.id];allowed[i]=not(z.independent_hand_centers>=2 or yes(z.old_duplicate_veto) or yes(z.zoom_duplicate_loser) or min(z.quality_logreg,z.quality_svm)>=.75 or z.structural_hand>z.baseline+1e-9)
    return allowed

def main(prefix=''):
    pred=pd.read_csv(A/f'features/{prefix}correspondence.csv');appearance=pd.read_csv(A/'features/appearance_predictions.csv');evidence=[];F={}
    for split,rows in ROWS.items():
        entries=[]
        for _,r in rows.iterrows():
            d=pred[pred.id.eq(r.id)]
            z=dict(id=r.id,valid=False,min_iou=0.,max_iou=1.,support=0,semantic=0.)
            if len(d) and d.valid.all():
                boxes=d[['ref_'+b for b in B]].to_numpy();agree=iou(boxes,boxes).min()
                if agree>=.65:
                    pp=appearance[appearance.id.eq(r.id)&appearance.target_class.eq(r.class_name)]
                    z.update(valid=True,min_iou=float(d.ref_iou.min()),max_iou=float(d.ref_iou.max()),support=int(d.support.min()),semantic=float(pp.p_present.min()),**dict(zip(['ref_'+b for b in B],boxes.mean(0))))
            entries.append(z)
        f=pd.DataFrame(entries);F[split]=f;f.assign(split=split).to_csv(A/f'features/{prefix}correspondence_{split}_gates.csv',index=False)
    results=[]
    for classes,label in [(['cup'],'cup'),(['tray'],'tray'),(['nude_hands'],'hands'),(['cup','tray'],'tableware')]:
        for direction in ['confirm','two_sided']:
            for weight in [.25,.5]:
                scores={}
                for split,rows in ROWS.items():
                    f=F[split];base=BASE[split];valid=f.valid.to_numpy()&rows.class_name.isin(classes).to_numpy()&f.semantic.ge(.9).to_numpy()
                    rescue=valid&f.min_iou.ge(.65).to_numpy()&guard(rows,split)
                    penalty=valid&f.max_iou.lt(.5).to_numpy() if direction=='two_sided' else np.zeros(len(rows),bool)
                    target=np.clip((.85-f.min_iou.to_numpy())/.5,0,1)
                    delta=np.minimum(target-base,0)*rescue+np.maximum(target-base,0)*penalty
                    scores[split]=base+weight*delta
                results.append(save(f'{prefix}transfer_{label}_{direction}_w{weight}',scores,'Fit-day GT transferred through mutual correspondences; >=2 template frames; test daily reference agreement; semantic and duplicate veto.'))
    pd.DataFrame(results).to_csv(A/f'results/{prefix}correspondence_experiments.csv',index=False)

if __name__=='__main__':
    import sys
    main('sift_' if '--sift' in sys.argv else '')
