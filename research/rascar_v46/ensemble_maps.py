"""Exploratory fixed 50:50 probability-map fusion with the original backbone."""
from common46 import *
from train import decode
import argparse
def main():
    p=argparse.ArgumentParser();p.add_argument('key',choices=['vit_b','vit_l','siglip2']);args=p.parse_args();rec=records().set_index('id');output=[]
    for fit in DAYS:
        original=np.load(A/f'features/small_control_s3101_fit{fit[-2:]}_maps.npz');other=np.load(A/f'features/{args.key}_s3101_fit{fit[-2:]}_maps.npz');look={x:i for i,x in enumerate(other['ids'])}
        original_maps=original['maps'];other_maps=other['maps']
        for j,ident in enumerate(original['ids']):
            r=rec.loc[ident].copy();r['id']=ident;prob=(original_maps[j].astype(np.float32)+other_maps[look[ident]].astype(np.float32))*.5
            output.append(dict(id=ident,origin_id=r.origin_id,frame_id=r.frame_id,day=r.day,split=r.split,source='query' if r.split=='oof' else 'test',class_name=r.class_name,fit_day=fit,key=f'small_{args.key}_mean',seed=3101,**decode(prob,r)))
    pd.DataFrame(output).to_csv(A/f'features/small_{args.key}_mean_evidence.csv',index=False)
if __name__=='__main__':main()
