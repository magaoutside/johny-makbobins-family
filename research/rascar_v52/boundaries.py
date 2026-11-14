"""Query-border features from strictly nested occupancy heads and frozen grids."""
from train import *
import torch
from torch import nn
import torch.nn.functional as F
class Dense(nn.Module):
    def __init__(self,d):super().__init__();self.d=d;self.net=nn.Conv2d(d,4,1)
    def forward(self,x):return self.net(F.normalize(x,dim=1)*np.sqrt(self.d))
def describe(z,prob,r):
    cb=r[CB].to_numpy(float);b=r[B].to_numpy(float);wh=b[2:]-b[:2];yy,xx=np.mgrid[:32,:32];x=cb[0]+(xx+.5)*(cb[2]-cb[0])/32;y=cb[1]+(yy+.5)*(cb[3]-cb[1])/32;k=CL.index(r.class_name);p=prob[k];inside=(x>=b[0])&(x<=b[2])&(y>=b[1])&(y<=b[3]);out={};mean=z[:,inside].mean(1);mean/=max(np.linalg.norm(mean),1e-8)
    for side in ['left','right','top','bottom']:
        horiz=side in ['left','right'];pos=x if horiz else y;cross=(y>=b[1])&(y<=b[3]) if horiz else (x>=b[0])&(x<=b[2]);edge=b[0] if side=='left' else b[2] if side=='right' else b[1] if side=='top' else b[3];direction=-1 if side in ['left','top'] else 1;distance=(pos-edge)*direction;band=max((cb[2]-cb[0] if horiz else cb[3]-cb[1])/32*1.1,wh[0 if horiz else 1]*.15);mi=cross&(distance<=0)&(distance>=-band);mo=cross&(distance>0)&(distance<=band)
        for label,m in [('in',mi),('out',mo)]:
            out[f'{side}_{label}_missing']=float(not m.any());out[f'{side}_{label}_prob']=float(p[m].mean()) if m.any() else 0.;out[f'{side}_{label}_coverage']=float((p[m]>.5).mean()) if m.any() else 0.
            if m.any():v=z[:,m].mean(1);v/=max(np.linalg.norm(v),1e-8);out[f'{side}_{label}_visual_cos']=float(v@mean)
            else:out[f'{side}_{label}_visual_cos']=0.
        out[f'{side}_prob_out_minus_in']=out[f'{side}_out_prob']-out[f'{side}_in_prob']
    zz=z[:,inside];zz/=np.maximum(np.linalg.norm(zz,axis=0),1e-8);similarity=mean@zz
    for a in [.1,.5,.9]:out[f'inner_similarity_q{int(a*100)}']=float(np.quantile(similarity,a))
    _,labels,stats,_=cv2.connectedComponentsWithStats((p>=.5).astype(np.uint8),8);refs=[]
    for j,(a,c,w,hh,area) in enumerate(stats[1:],1):
        if area>=4:refs.append(np.array([a,c,a+w,c+hh])/32*np.tile(cb[2:]-cb[:2],2)+np.tile(cb[:2],2))
    from common import iou
    if refs:
        overlaps=iou(b,refs)[0];ref=refs[overlaps.argmax()];d=(ref-b)/np.tile(np.maximum(wh,1),2);out.update(dict(zip(['ref_left','ref_top','ref_right','ref_bottom'],d)));out['ref_iou']=float(overlaps.max());out['matching_components']=float((overlaps>.1).sum())
    else:out.update({c:0. for c in ['ref_left','ref_top','ref_right','ref_bottom','ref_iou','matching_components']})
    return out
def extract():
    torch.set_num_threads(2);q,t,h=inputs();rows=pd.read_csv(A/'inputs/rows.csv');rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');rl=rec.reset_index().set_index('id')['index'];man=json.loads((ROOT/'rascar_v48/models/inner_manifest.json').read_text());out=[]
    for key in ['small_control','vit_b','vit_l','siglip2']:
        path=ROOT/'rascar_v31/features/dense_grids.npy' if key=='small_control' else ROOT/f'rascar_v46/features/{key}_grids.npy';f=np.load(path,mmap_mode='r')
        for fit in DAYS:
            maps=np.load(ROOT/f'rascar_v46/features/{key}_s3101_fit{fit[-2:]}_maps.npz');look={id:i for i,id in enumerate(maps['ids'])};inner={}
            for m in [a for a in man if a['key']==key and a['outer_fit_day']==fit]:
                path=ROOT/f'rascar_v45/models/inner_dense_fit{fit[-2:]}_block{m["inner_block"]}.pt' if key=='small_control' else ROOT/f'rascar_v48/models/{key}_fit{fit[-2:]}_block{m["inner_block"]}.pt';model=Dense(f.shape[1]);model.load_state_dict(torch.load(path,weights_only=True));model.eval()
                for fid in m['held_frames']:inner[fid]=model
            for _,r in rows.iterrows():
                i=rl.loc[r.id];rr=rec.loc[i];z=np.array(f[i],np.float32)
                if r.split=='oof' and r.day==fit:
                    assert r.frame_id in inner
                    with torch.inference_mode():pr=inner[r.frame_id](torch.from_numpy(z[None]))[0].sigmoid().numpy()
                else:pr=maps['maps'][look[r.id]].astype(np.float32)
                out.append(dict(id=r.id,fit_day=fit,key=key,**describe(z,pr,rr)))
            print('BOUNDARY',key,fit,flush=True)
    d=pd.DataFrame(out);d.to_csv(A/'features/nested_boundary_long.csv',index=False)
def main():
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
    if not (A/'features/nested_boundary_long.csv').exists():extract()
    rows,banks,groups,_=load_bank();q,t,h=inputs();bo,bt=baseline(q,t);d=pd.read_csv(A/'features/nested_boundary_long.csv');cols=[c for c in d if c not in ['id','fit_day','key']];wide=d.pivot(index=['id','fit_day'],columns='key',values=cols);wide.columns=[f'boundary_{k}_{c}' for c,k in wide.columns];wide=wide.reset_index();bc=wide.columns[2:].tolist()
    for fit in DAYS:
        x=wide[wide.fit_day.eq(fit)].set_index('id').loc[rows.id]
        for c in bc:banks[fit][c]=x[c].to_numpy()
    out=[];manifest=[]
    for name,base,keys in [('small',['base23'],['small_control']),('all',['base23'],['small_control','vit_b','vit_l','siglip2']),('siglip',['base23','embed_siglip2'],['small_control','vit_b','vit_l','siglip2']),('only',[],['small_control','vit_b','vit_l','siglip2']),('frozen',['frozen_all','owl','clip_text'],['small_control','vit_b','vit_l','siglip2'])]:
        cc=[c for c in bc if any(c.startswith('boundary_'+k+'_') for k in keys)]+list(dict.fromkeys(c for g in base for c in groups[g]))+['class_name']
        for depth in [2,4]:
            stem=f'boundary_{name}_d{depth}';pp=[]
            for fit in DAYS:
                tr=rows.split.eq('oof')&rows.day.eq(fit);ev=~tr;X=banks[fit][cc];model=CatBoostClassifier(iterations=250 if depth==2 else 400,depth=depth,learning_rate=.03,l2_leaf_reg=10 if depth==2 else 20,random_seed=23,thread_count=3,verbose=False,allow_writing_files=False);model.fit(X.loc[tr],rows.loc[tr,'error'],cat_features=['class_name']);pp.append(rows.loc[ev,['id','frame_id','day','class_name','split']].assign(p_error=model.predict_proba(X.loc[ev])[:,1],fit_day=fit));model.save_model(str(A/f'models/{stem}_fit{fit[-2:]}.cbm'));manifest.append(dict(model=stem,fit_day=fit,features=cc,train_ids=rows.loc[tr,'id'].tolist(),nested_heads=True))
            pp=pd.concat(pp);pp.to_csv(A/f'features/{stem}_predictions.csv',index=False);oo=pp[pp.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();tt=pp[pp.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
            for a in [.05,.1,.2,.5,1000000.]:
                tag='standalone' if a>100 else f'a{int(a*100):02}';s=oo if a>100 else (bo+a*oo)/(1+a);st=tt if a>100 else (bt+a*tt)/(1+a);exp=stem+'_'+tag;q.assign(score=s).to_csv(A/f'results/oof/{exp}.csv',index=False);t[['id']].assign(score=st).to_csv(A/f'results/submissions/{exp}.csv',index=False);out.append(dict(name=exp,model=stem,alpha=a,**metrics(q,s)))
            pd.DataFrame(out).to_csv(A/'results/boundary_experiments.csv',index=False);(A/'models/boundary_manifest.json').write_text(json.dumps(manifest,indent=2));print(stem,AP(q.error,oo),flush=True)
if __name__=='__main__':main()
