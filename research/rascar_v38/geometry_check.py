"""Held-day local geometry diagnostic on newly reviewed TRAIN pool regions.

Positive-selected, approximate annotations. Not an additional competition AP.
No annotation is used to fit or select thresholds. Each reference is evaluated
against reviewed same-class objects; class coverage is only local to the ROI.
"""
from common38 import *
from multilevel_train import Head
from refit import Head as LastHead,decode
import torch,cv2,os
from transformers import AutoModel
torch.set_num_threads(3);cv2.setNumThreads(1)
os.environ['TOKENIZERS_PARALLELISM']='false'

def bound_iou(query,box,tol):
    query=np.asarray(query,float);box=np.asarray(box,float)
    inner=box+np.array([tol,tol,-tol,-tol]);outer=box+np.array([-tol,-tol,tol,tol])
    inter=lambda b:np.maximum(0,np.minimum(query[2:],b[2:])-np.maximum(query[:2],b[:2])).prod()
    amin=np.maximum(0,inner[2:]-inner[:2]).prod();amax=np.maximum(0,outer[2:]-outer[:2]).prod();aq=np.maximum(0,query[2:]-query[:2]).prod()
    low=inter(inner)/max(aq+amax-inter(inner),1e-9);high=min(1.,inter(outer)/max(aq+amin-inter(outer),1e-9))
    return low,high

def main():
    notes=json.loads((A/'inputs/train_geometry_notes.json').read_text())['records'];pool=pd.read_csv(A/'inputs/geometry_review_frames.csv').set_index('id');rr=[]
    for note in notes:
        r=pool.loc[note['id']];b=r[B].to_numpy(float);cc=(b[:2]+b[2:])/2;size=np.maximum(b[2:]-b[:2],16)*2.4
        cb=np.r_[np.maximum(0,np.floor(cc-size/2)),np.minimum([1280,720],np.ceil(cc+size/2))]
        z=r.to_dict();z.update(dict(zip(['crop_'+b for b in B],cb)));z['id']=note['id'];rr.append(z)
    rec=pd.DataFrame(rr);rec.to_csv(A/'inputs/geometry_model_records.csv',index=False)
    backbone=AutoModel.from_pretrained(str(ROOT/'rascar_v31/model_cache/dinov2-small'),local_files_only=True).eval()
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1);features=[]
    for off in range(0,len(rec),4):
        batch=[]
        for _,r in rec.iloc[off:off+4].iterrows():
            raw=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));cb=r[['crop_'+b for b in B]].to_numpy(int)
            batch.append(cv2.resize(raw[cb[1]:cb[3],cb[0]:cb[2],::-1],(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        with torch.inference_mode():
            z=backbone(pixel_values=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std,output_hidden_states=True)
            features.append(torch.cat([backbone.layernorm(z.hidden_states[l])[:,1:].transpose(1,2).reshape(-1,384,32,32) for l in [3,6,9,12]],1))
    feat=torch.cat(features);del backbone
    outputs=[]
    for kind in ['v31','all4','early2']:
        for fit in DAYS:
            for seed in ([3101] if kind=='v31' else [3101,3102,3103]):
                model=LastHead() if kind=='v31' else Head(kind)
                path=ROOT/f'rascar_v31/models/linear_mask_fit{fit[-2:]}.pt' if kind=='v31' else A/f'models/multilevel_{kind}_fit{fit[-2:]}_s{seed}.pt'
                model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
                ix=rec.index[rec.day.ne(fit)].to_numpy();assert all(rec.loc[ix,'day'].isin(DAYS))
                with torch.inference_mode():p=model(feat[ix,-384:] if kind=='v31' else feat[ix]).sigmoid().numpy()
                for i,prob in zip(ix,p):
                    r=rec.loc[i];note=notes[i];z=decode(prob,r);reference=np.array([z.get('ref_'+b,np.nan) for b in B])
                    manual=np.array(note['boxes']);nominal=iou(r[B],manual).ravel();bounds=[bound_iou(r[B],box,note['edge_uncertainty_px']) for box in manual]
                    if np.isfinite(reference).all():
                        roi=np.array(note['roi']);contained=bool((reference[:2]>=roi[:2]).all() and (reference[2:]<=roi[2:]).all())
                        geom=float(iou(reference,manual).max()) if contained else np.nan
                    else:contained=True;geom=0.
                    outputs.append(dict(**z,kind=kind,seed=seed,fit_day=fit,frame_id=r.frame_id,day=r.day,class_name=r.class_name,review_index=note['review_index'],reference_inside_reviewed_roi=contained,local_reference_iou=geom,
                        query_nominal_iou=float(nominal.max()),query_iou_lower=max(v[0] for v in bounds),query_iou_upper=max(v[1] for v in bounds)))
    d=pd.DataFrame(outputs);d.to_csv(A/'audit/geometry_pool_predictions.csv',index=False)
    # Average seeds within query, not as independent observations; flag coverage.
    per=d.groupby(['kind','class_name','id']).agg(iou=('local_reference_iou','mean'),fully_in_roi=('reference_inside_reviewed_roi','all'),query_iou=('query_nominal_iou','first')).reset_index()
    per.loc[~per.fully_in_roi,'iou']=np.nan
    summary=[]
    for (kind,cls),g in per.groupby(['kind','class_name']):
        summary.append(dict(kind=kind,class_name=cls,queries=len(g),in_scope=int(g.iou.notna().sum()),nominal_mean_iou=g.iou.mean(),nominal_fraction_iou06=g.iou.dropna().ge(.6).mean()))
    pd.DataFrame(summary).to_csv(A/'results/geometry_pool_summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    target=d.drop_duplicates('id');print('QUERY BOUNDS',int(target.query_iou_upper.lt(.6).sum()),'robust geometry errors;',int(target.query_iou_lower.ge(.6).sum()),'robust geometry support; remaining ambiguous. Not duplicate-aware labels.')

if __name__=='__main__':main()
