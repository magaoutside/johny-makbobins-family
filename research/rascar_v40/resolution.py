"""Matched 16/32-grid continuation of the V39 set head, fitting day only.

Same initializer, batches, optimizer and steps; only memory resolution differs.
This avoids attributing extra training to the higher-resolution representation.
"""
from common40 import *
import set_detector as old
from set_detector import torch,nn,F,math,Data,SEEDS,CB,set_loss,decode
import argparse,time

class SetHead(old.SetHead):
    def __init__(self,grid=32):
        super().__init__();assert grid in [16,32];self.grid=grid
        yy,xx=torch.meshgrid((torch.arange(grid)+.5)/grid,(torch.arange(grid)+.5)/grid,indexing='ij')
        freq=2*math.pi*2**torch.arange(24).float().div(4)
        self.position=torch.cat([(xx[...,None]*freq).sin(),(xx[...,None]*freq).cos(),(yy[...,None]*freq).sin(),(yy[...,None]*freq).cos()],-1).reshape(1,grid*grid,96)
    def forward(self,x):
        x=F.normalize(x,dim=1)*math.sqrt(384)
        if self.grid==16:x=F.avg_pool2d(x,2)
        memory=self.project(x).flatten(2).transpose(1,2)+self.position
        z=self.queries[None].expand(len(x),-1,-1);out=[]
        for layer in self.layers:
            z=layer(z,memory);v=self.norm(z);out.append((self.cls(v),self.box(v).sigmoid()))
        return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--grids',nargs='+',type=int,default=[16,32]);p.add_argument('--seeds',nargs='+',type=int,default=SEEDS);p.add_argument('--steps',type=int,default=1000);p.add_argument('--fit-both',action='store_true');args=p.parse_args()
    data=Data();rec=data.rec;fits=['all'] if args.fit_both else DAYS
    protocol=dict(grids=args.grids,seeds=args.seeds,steps=args.steps,fit_days=fits,init='V39 set_long corresponding day and seed',lr=[.0001,.00001],train_splits=['manual','empty'],test_targets=False)
    (A/f'inputs/resolution_{"full" if args.fit_both else "daily"}_protocol.json').write_text(json.dumps(protocol,indent=2))
    for grid in args.grids:
        run=f'refine{grid}'+('_full' if args.fit_both else '');manifest_path=A/f'models/{run}_manifest.json'
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        for fit in fits:
            fitmask=rec.day.isin(DAYS) if fit=='all' else rec.day.eq(fit)
            tr=rec.index[fitmask&rec.split.isin(['manual','empty'])].to_numpy()
            ev=rec.index[rec.split.eq('test')|((rec.day.ne(fit)&rec.split.isin(['manual','empty','oof'])) if fit!='all' else False)].to_numpy()
            assert not set(rec.loc[tr,'frame_id'])&set(rec.loc[ev,'frame_id'])
            groups=[tr[rec.loc[tr,'split'].eq(s).to_numpy()] for s in ['manual','empty']];probs=[]
            for group in groups:
                counts=rec.loc[group].frame_id.value_counts();w=np.array([1/counts[f] for f in rec.loc[group,'frame_id']]);probs.append(w/w.sum())
            for seed in args.seeds:
                suffix=f'{fit if fit=="all" else fit[-2:]}_seed{seed}';name=f'{run}_fit{suffix}'
                if any(m['name']==name for m in manifest):print('EXISTS',name,flush=True);continue
                init=ROOT/f'rascar_v39/models/{"set_long_full" if fit=="all" else "set_long"}_fit{suffix}.pt'
                torch.manual_seed(seed);rng=np.random.default_rng(seed+40000);model=SetHead(grid)
                state=torch.load(init,map_location='cpu',weights_only=True);state.pop('position');missing=model.load_state_dict(state,strict=False);assert missing.missing_keys==['position'] and not missing.unexpected_keys
                opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01);losses=[];start=time.time()
                for step in range(args.steps):
                    ids=np.r_[rng.choice(groups[0],6,p=probs[0]),rng.choice(groups[1],2,p=probs[1])]
                    for group in opt.param_groups:group['lr']=.0001*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
                    loss=set_loss(model(data.features(ids)),[data.targets[i] for i in ids]);assert torch.isfinite(loss)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
                    if step%200==0:print('TRAIN',name,step,round(np.mean(losses[-100:]),4),round(time.time()-start,1),flush=True)
                torch.save(model.state_dict(),A/f'models/{name}.pt');model.eval();out=[]
                for off in range(0,len(ev),16):
                    ids=ev[off:off+16]
                    with torch.inference_mode():logits,boxes=model(data.features(ids))[-1]
                    for j,i in enumerate(ids):
                        r=rec.loc[i]
                        for z in decode(logits[j],boxes[j],r):out.append(dict(record_id=r.id,frame_id=r.frame_id,split=r.split,fit_day=fit,seed=seed,**z))
                pd.DataFrame(out).to_csv(A/f'features/{name}_boxes.csv',index=False)
                manifest.append(dict(name=name,grid=grid,fit_day=fit,seed=seed,steps=args.steps,initializer=str(init.relative_to(ROOT)),initializer_sha256=hashlib.sha256(init.read_bytes()).hexdigest(),fit_frames=sorted(rec.loc[tr,'frame_id'].unique()),last_loss=float(np.mean(losses[-100:])),seconds=time.time()-start))
                manifest_path.write_text(json.dumps(manifest,indent=2));print('DONE',name,len(out),round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
