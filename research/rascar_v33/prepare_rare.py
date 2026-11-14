"""Visual presence labels for rare TRAIN-pool hands; no boundary targets."""
from prepare import *

# Indices refer ONLY to the saved, manually inspected training-pool contact sheets.
# No inference-time rule refers to these IDs or to any test ID.
ACCEPT=[1,2,4,5,6,7,9,11,15,16,18,20,22,23,25,29,30,31,32,33,34,35,36,37,39,41,43,46,50]
FALSE_FABRIC=[13,24,28]

def main():
    candidates=pd.read_csv(A/'inputs/rare_hand_review_candidates.csv');ledger=candidates.copy()
    ledger['decision']=np.where(ledger.review_index.isin(ACCEPT),'hand_presence_only',np.where(ledger.review_index.isin(FALSE_FABRIC),'reject_fabric','exclude_ambiguous_or_redundant'))
    ledger['source']='assistant_visual_review_nonofficial';ledger['geometry_verified']=False
    ledger.to_csv(A/'audit/rare_hand_visual_decisions.csv',index=False)
    rows=[]
    for _,r in candidates[candidates.review_index.isin(ACCEPT)].iterrows():
        b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);center=(b[:2]+b[2:])/2
        lo=np.maximum(0,np.floor(center-wh*1.2));hi=np.minimum([1280,720],np.ceil(center+wh*1.2))
        rows.append(dict(id=f'rare33_{r.detection_id}',frame_id=r.frame_id,day=r.day,class_name='nude_hands',split='pool_rare_presence',origin_id=r.detection_id,review_index=r.review_index,label_use='presence_only_not_boundary_ground_truth',**dict(zip(B,b)),**dict(zip(CB,np.r_[lo,hi]))))
    rec=pd.DataFrame(rows);rec['feature_index']=np.arange(len(rec));rec.to_csv(A/'inputs/rare_records.csv',index=False)
    assert not set(rec.frame_id)&set(h.frame_id)
    torch.set_num_threads(3);cv2.setNumThreads(1)
    model=AutoModel.from_pretrained(str(V31/'model_cache/dinov2-small'),local_files_only=True,trust_remote_code=False).eval()
    dst=np.lib.format.open_memmap(A/'features/rare_grids.npy',mode='w+',dtype=np.float16,shape=(len(rec),384,32,32))
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    for off in range(0,len(rec),4):
        batch=[]
        for _,r in rec.iloc[off:off+4].iterrows():
            im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));b=r[CB].to_numpy(int);im=im[b[1]:b[3],b[0]:b[2],::-1]
            batch.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        with torch.inference_mode():z=model(pixel_values=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32)
        dst[off:off+len(batch)]=z.numpy().astype(np.float16)
    dst.flush();(A/'inputs/rare_manifest.json').write_text(json.dumps(dict(reviewed=60,accepted_presence=len(rec),rejected_or_unused=60-len(rec),explicit_fabric_false_positives=FALSE_FABRIC,exact_boundary_labels=0,test_labels_used=False,days=rec.day.value_counts().to_dict(),weight=.3),indent=2))
    print('DONE',len(rec),rec.day.value_counts().to_dict(),flush=True)
if __name__=='__main__':main()
