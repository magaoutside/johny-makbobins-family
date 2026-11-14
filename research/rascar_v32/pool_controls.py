"""External, assistant-reviewed positives: auxiliary controls, not official GT/AP.
Models are frozen and predictions always use the opposite day's training fold.
"""
from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32';V31=ROOT/'rascar_v31'
os.environ.setdefault('MPLCONFIGDIR',str(V31/'model_cache/mpl'))
sys.path.insert(0,str(V31));sys.path.insert(0,str(A))
from dense_train import Head,torch,np,pd,cv2,CL,B,DAYS,iou,get_boxes
from refine_instances import peak_decode,fuse_centers
from watershed_boxes import separate
from transformers import AutoModel

def main():
 h=pd.read_csv(ROOT/'rascar_v22/inputs/reviewed_pool_hands.csv');c=pd.read_csv(ROOT/'rascar_v25/inputs/reviewed_pool_cups.csv');d=pd.concat([h,c],ignore_index=True)
 paths={p.stem:p for p in (ROOT/'rascar_v25/gpu/results/pool_images').glob('*.jpg')};assert set(d.frame_id)<=set(paths)
 manual=pd.read_csv(ROOT/'rascar_v22/inputs/manual_boxes.csv');assert not set(d.frame_id)&set(manual.frame_id)
 rows=[];crops=[]
 for _,r in d.iterrows():
  b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2;cb=np.r_[np.maximum(0,np.floor(cc-1.2*wh)),np.minimum([1280,720],np.ceil(cc+1.2*wh))].astype(int)
  im=cv2.resize(cv2.imread(str(paths[r.frame_id])),(1280,720));im=im[cb[1]:cb[3],cb[0]:cb[2],::-1];crops.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
  rows.append(dict(id=r.id,frame_id=r.frame_id,day=r.day,class_name=r.class_name,**dict(zip(B,b)),**dict(zip(['crop_x1','crop_y1','crop_x2','crop_y2'],cb))))
 rows=pd.DataFrame(rows);rows.to_csv(A/'inputs/pool_control_records.csv',index=False)
 backbone=AutoModel.from_pretrained(str(V31/'model_cache/dinov2-small'),local_files_only=True,trust_remote_code=False).eval();mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None];features=[]
 for off in range(0,len(crops),4):
  x=(torch.from_numpy(np.stack(crops[off:off+4])).float()/255-mean)/std
  with torch.inference_mode():features.append(backbone(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32).numpy())
 features=np.concatenate(features);np.save(A/'features/pool_control_grids.npy',features.astype(np.float16));output=[]
 for fit in DAYS:
  mask=Head('linear_mask');mask.load_state_dict(torch.load(V31/f'models/linear_mask_fit{fit[-2:]}.pt',weights_only=True));mask.eval()
  det=Head('conv_detector');det.load_state_dict(torch.load(V31/f'models/conv_detector_fit{fit[-2:]}.pt',weights_only=True));det.eval()
  ids=rows[rows.day.ne(fit)].index.to_numpy()
  with torch.inference_mode():mp=mask(torch.from_numpy(features[ids]));dp=det(torch.from_numpy(features[ids]))
  for j,i in enumerate(ids):
   r=rows.loc[i];raw=get_boxes(tuple(z[j] for z in dp),r,'conv_detector');peak=peak_decode(tuple(z[j] for z in dp),r);fused=fuse_centers(peak)
   sets={'linear_mask':get_boxes(mp[j],r,'linear_mask'),'conv_detector':raw,'peak_detector':peak.to_dict('records'),'fused_detector':fused.to_dict('records'),'watershed_fused':separate(mp[j].sigmoid().numpy(),fused,r)}
   for kind,boxes in sets.items():
    b=pd.DataFrame(boxes)
    for cut in ([.35,.5,.65] if kind=='linear_mask' or kind.startswith('watershed') else [.15,.3,.5]):
     z=b[b.class_name.eq(r.class_name)] if len(b) else pd.DataFrame(columns=['class_name','confidence','threshold']+B)
     z=z[z.threshold.eq(cut)] if kind=='linear_mask' or kind.startswith('watershed') else z[z.confidence.ge(cut)]
     val=iou(r[B],z[B]).ravel();best=z.iloc[val.argmax()] if len(val) else None
     output.append(dict(id=r.id,frame_id=r.frame_id,held_day=r.day,fit_day=fit,class_name=r.class_name,head=kind,cut=cut,overlap=float(val.max()) if len(val) else np.nan,confidence=float(best.confidence) if best is not None else 0,**{f'ref_{b}':float(best[b]) if best is not None else np.nan for b in B}))
 out=pd.DataFrame(output);out.to_csv(A/'audit/pool_control_evidence.csv',index=False)
 summary=out.groupby(['head','cut','class_name']).agg(n=('id','size'),coverage=('overlap',lambda x:x.notna().mean()),mean_iou=('overlap',lambda x:x.fillna(0).mean()),fraction_iou06=('overlap',lambda x:x.ge(.6).mean()))
 summary.to_csv(A/'results/pool_control_summary.csv');print(summary.to_string())
 (A/'audit/pool_control_protocol.json').write_text(json.dumps(dict(controls=len(rows),frames=int(rows.frame_id.nunique()),used_for_training=False,official_ground_truth=False,selection_bias='Previously reviewed positives chosen using older models. Not an unbiased sample; no negatives; cannot compute AP.',evaluation='Opposite day models only; object-centered crops, auxiliary geometry check'),indent=2))
if __name__=='__main__':main()
