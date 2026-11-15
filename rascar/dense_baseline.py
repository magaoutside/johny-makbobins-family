"""Original two-day linear occupancy model on frozen DINOv2-S grids."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from .core import BBOX, CROP, DAYS, CLASSES, queries, aligned_csv, write_json
from .occupancy import occupancy, overlap


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Conv2d(384,4,1)
    def forward(self,x):
        return self.net(F.normalize(x,dim=1)*np.sqrt(384))


def run(data, cache, models, out, device, fit=False):
    torch.set_num_threads(2)
    data,cache,models,out=map(Path,[data,cache,models,out])
    rec=pd.read_csv(data/'dense_records.csv')
    from .core import read_json
    if read_json(cache/'dense_small_complete.json')['ids']!=rec.id.tolist():
        raise ValueError('Incomplete or reordered small DINO cache')
    x=np.load(cache/'dense_small.npy',mmap_mode='r')
    if x.shape!=(len(rec),384,32,32): raise ValueError('Invalid dense grid shape')
    models.mkdir(parents=True,exist_ok=True)
    if fit:
        h=pd.read_csv(data/'manual_boxes.csv')
        targets=np.stack([occupancy(r,h) if r.split!='test' else np.zeros((4,32,32))
                          for _,r in rec.iterrows()]).astype(np.float16)
    results=[]
    for day in DAYS:
        model=Head().to(device)
        path=models/f'linear_day{day[-2:]}.pt'
        if fit:
            torch.manual_seed(3101);rng=np.random.default_rng(3101)
            model=Head().to(device)
            tr=rec.index[rec.split.eq('manual')&rec.day.eq(day)].to_numpy()
            counts=rec.loc[tr].groupby('frame_id').size()
            prob=np.array([1/counts[f] for f in rec.loc[tr,'frame_id']]);prob/=prob.sum()
            opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
            for _ in range(450):
                ix=rng.choice(tr,8,replace=True,p=prob)
                xx=torch.from_numpy(np.array(x[ix],np.float32)).to(device)
                yy=torch.from_numpy(np.array(targets[ix],np.float32)).to(device)
                loss=F.binary_cross_entropy_with_logits(model(xx),yy,weight=1+3*yy)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            torch.save({k:v.cpu() for k,v in model.state_dict().items()},path)
        else:
            model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        model.eval()
        ev=rec.index[rec.split.eq('test')|(rec.split.eq('oof')&rec.day.ne(day))].to_numpy()
        maps=[]
        with torch.inference_mode():
            for start in range(0,len(ev),16):
                maps.append(model(torch.from_numpy(np.array(x[ev[start:start+16]],np.float32)).to(device)).sigmoid().cpu().numpy())
        results.append(rec.loc[ev,['id','split','day']].assign(fit_day=day,overlap=overlap(np.concatenate(maps),rec.loc[ev])))
        print(f'Dense baseline: fit {day}',flush=True)
    df=pd.concat(results,ignore_index=True)
    rows=queries(data)
    guarded=aligned_csv(data/'guarded_scores.csv',rows.id).score.to_numpy()
    scores=guarded.copy()
    oo=rows.split.eq('oof').to_numpy();tt=~oo
    ov=df[df.split.eq('oof')].set_index('id').loc[rows.loc[oo,'id'],'overlap'].to_numpy()
    target=np.clip((.85-ov)/.5,0,1);valid=np.isfinite(target)
    s=scores[oo];s[valid]+=.25*(target[valid]-s[valid]);scores[oo]=s
    test=np.stack([df[df.split.eq('test')&df.fit_day.eq(day)].set_index('id').loc[rows.loc[tt,'id'],'overlap'].to_numpy() for day in DAYS])
    target=np.clip((.85-test)/.5,0,1);delta=target-guarded[tt][None]
    valid=np.isfinite(target).all(0)&((delta>=0).all(0)|(delta<=0).all(0))
    conservative=np.where((delta>=0).all(0),np.min(target,axis=0),np.max(target,axis=0))
    s=scores[tt];s[valid]+=.25*(conservative[valid]-s[valid]);scores[tt]=s
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'linear_overlaps.csv',index=False)
    rows[['id']].assign(dense_baseline=scores,guarded_baseline=guarded).to_csv(out/'upstream_scores.csv',index=False)
    write_json(out/'linear_model.json',dict(seed=3101,steps=450,learning_rate=.0007,weight_decay=.01,fit=fit))
