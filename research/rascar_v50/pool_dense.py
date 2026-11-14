"""Reused 36-crop presence diagnostic, not a fresh leaderboard holdout."""
from infer import *
from transformers import AutoModel
sys.path.insert(0,str(ROOT/'rascar_v46'))
from train import Dense
def main():
    torch.set_num_threads(3);cv2.setNumThreads(1);rec=pd.read_csv(ROOT/'rascar_v34/inputs/fresh_pool_candidates.csv');labels=json.loads((ROOT/'rascar_v34/inputs/fresh_pool_presence_review.json').read_text());rec['presence']=np.nan;rec.loc[rec.review_index.isin(labels['present_review_indices']),'presence']=1;rec.loc[rec.review_index.isin(labels['absent_review_indices']),'presence']=0
    b=rec[B].to_numpy(float);wh=np.maximum(b[:,2:]-b[:,:2],16)*2.4;cc=(b[:,2:]+b[:,:2])/2;lo=np.maximum(0,np.floor(cc-wh/2));hi=np.minimum([1280,720],np.ceil(cc+wh/2));rec[CB]=np.c_[lo,hi];rec.to_csv(A/'inputs/pool_records.csv',index=False)
    model=AutoModel.from_pretrained(str(ROOT/'rascar_v31/model_cache/dinov2-small'),local_files_only=True).eval();mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1);features=[]
    for off in range(0,len(rec),4):
        images=[]
        for _,r in rec.iloc[off:off+4].iterrows():
            im=cv2.resize(cv2.imread(str(D/f'{r.frame_id}.jpg')),(1280,720));bb=r[CB].to_numpy(int);im=im[bb[1]:bb[3],bb[0]:bb[2],::-1];images.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        with torch.inference_mode():z=model(pixel_values=(torch.tensor(np.stack(images)).float()/255-mean)/std).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32)
        features.append(z.numpy())
    features=np.concatenate(features);out=[]
    for fit in ['2026-10-25','2026-10-26']:
        head=Dense(384);head.load_state_dict(torch.load(ROOT/f'rascar_v31/models/linear_mask_fit{fit[-2:]}.pt',weights_only=True));head.eval()
        with torch.inference_mode():prob=head(torch.tensor(features)).sigmoid().numpy()[:,1]
        for j,(_,r) in enumerate(rec.iterrows()):
            cb=r[CB].to_numpy(float);bb=r[B].to_numpy(float);yy,xx=np.mgrid[:32,:32];xx=cb[0]+(xx+.5)*(cb[2]-cb[0])/32;yy=cb[1]+(yy+.5)*(cb[3]-cb[1])/32;inside=(xx>=bb[0])&(xx<=bb[2])&(yy>=bb[1])&(yy<=bb[3]);out.append(dict(id=r.id,fit_day=fit,day=r.day,frame_id=r.frame_id,coverage=float((prob[j][inside]>=.5).mean()),mean=float(prob[j][inside].mean()),presence=r.presence))
    pd.DataFrame(out).to_csv(A/'features/pool_dense.csv',index=False);print('POOL DENSE',len(rec),flush=True)
if __name__=='__main__':main()
