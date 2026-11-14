"""Compare frozen DINO and SigLIP classifiers on newly reviewed train-pool crops."""
from siglip_semantic import *
from transformers import AutoModel

def dino_features(rec):
 path=A/'features/blind_pool_dino.npy'
 if path.exists():return np.load(path)
 model=AutoModel.from_pretrained(str(V31/'model_cache/dinov2-small'),local_files_only=True).eval();out=[]
 mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1);yy,xx=np.mgrid[:32,:32];xx=(xx+.5)/32;yy=(yy+.5)/32
 for off in range(0,len(rec),4):
  rr=rec.iloc[off:off+4];batch=[];crops=[]
  for _,r in rr.iterrows():
   b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2;cb=np.r_[np.maximum(0,np.floor(cc-1.2*wh)),np.minimum([1280,720],np.ceil(cc+1.2*wh))].astype(int);crops.append(cb)
   im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));batch.append(cv2.resize(im[cb[1]:cb[3],cb[0]:cb[2],::-1],(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
  x=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std
  with torch.inference_mode():grids=model(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32).numpy()
  for g,cb,(_,r) in zip(grids,crops,rr.iterrows()):
   b=(r[B].to_numpy(float)-np.r_[cb[:2],cb[:2]])/np.r_[cb[2:]-cb[:2],cb[2:]-cb[:2]];m=(xx>=b[0])&(xx<=b[2])&(yy>=b[1])&(yy<=b[3]);inside=g[:,m].mean(1);out.append(np.r_[inside,inside-g[:,~m].mean(1)])
 x=np.array(out);np.save(path,x);return x

def main():
 rec=pd.read_csv(A/'inputs/blind_pool_candidates.csv');labels_path=A/'inputs/blind_pool_presence_review.json';labels=json.loads(labels_path.read_text());label_hash=hashlib.sha256(labels_path.read_bytes()).hexdigest();rec['presence']=np.nan
 rec.loc[rec.review_index.isin(labels['present_review_indices']),'presence']=1;rec.loc[rec.review_index.isin(labels['absent_review_indices']),'presence']=0
 vectors={'siglip':extract(rec,'blind_pool_siglip.npy'),'dino':dino_features(rec)};rows=[];summary=[]
 for family,x in vectors.items():
  manifest=json.loads((A/'models/siglip_manifest.json' if family=='siglip' else ROOT/'rascar_v34/models/semantic_manifest.json').read_text())
  for model in manifest:assert not set(model['fit_frames'])&set(rec.frame_id)
  for cval in [.03,.1]:
   for fit in DAYS:
    path=A/f'models/siglip_hand_c{cval}_fit{fit[-2:]}.joblib' if family=='siglip' else ROOT/f'rascar_v34/models/hand_presence_c{cval}_fit{fit[-2:]}.joblib'
    use=rec.day.ne(fit);m=joblib.load(path);d=rec.loc[use,['id','review_index','day','presence']].copy();d['p_hand']=m.predict_proba(x[use])[:,1];d['family']=family;d['C']=cval;d['fit_day']=fit;rows.append(d)
    known=d[d.presence.notna()];z=dict(family=family,C=cval,held_day=known.day.iloc[0],n=len(known),nonhands=int(known.presence.eq(0).sum()),presence_detection_ap=float(AP(1-known.presence,1-known.p_hand)))
    for cutoff in [.05,.15]:
     z[f'true_hands_flagged_p{cutoff}']=int((known.presence.eq(1)&known.p_hand.lt(cutoff)).sum());z[f'nonhands_flagged_p{cutoff}']=int((known.presence.eq(0)&known.p_hand.lt(cutoff)).sum())
    summary.append(z)
 pd.concat(rows).to_csv(A/'audit/blind_pool_model_predictions.csv',index=False);pd.DataFrame(summary).to_csv(A/'results/blind_pool_presence.csv',index=False)
 (A/'results/blind_pool_integrity.json').write_text(json.dumps(dict(label_sha256=label_hash,model_thresholds_fixed_before_evaluation=True,pool_not_used_for_fit=True,official_competition_error_labels=False),indent=2));print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
