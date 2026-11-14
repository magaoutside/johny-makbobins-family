"""Check ranking changes by frame and render all changed test queries.
Test images are qualitative diagnostics only; no test labels are manufactured.
"""
from temporal_context import *
from PIL import Image,ImageDraw

def plot(name,split):
    changes=pd.read_csv(A/f'audit/{name}_changes.csv')
    changes=changes[changes.split.eq(split)].sort_values('delta',key=lambda x:abs(x),ascending=False)
    agg=pd.read_csv(A/'audit/aggregated_evidence.csv')
    support='current_and_neighbor' if 'current_and_neighbor' in name else 'two'
    agg=agg[agg.split.eq(split)&agg.support_mode.eq(support)]
    human=pd.read_csv(ROOT/'rascar_v22/inputs/manual_boxes.csv') if split=='oof' else None
    for off in range(0,len(changes),12):
        rr=changes.iloc[off:off+12];w=470;hh=340
        canvas=Image.new('RGB',(w*3,hh*int(np.ceil(len(rr)/3))),'#202020');draw=ImageDraw.Draw(canvas)
        for k,(_,r) in enumerate(rr.iterrows()):
            b=r[B].to_numpy(float);pad=np.maximum((b[2:]-b[:2])*.6,40)
            lo=np.maximum(0,np.floor(b[:2]-pad)).astype(int);hi=np.minimum([1280,720],np.ceil(b[2:]+pad)).astype(int)
            im=Image.fromarray(image(r.frame_id)[lo[1]:hi[1],lo[0]:hi[0],::-1]);scale=min((w-6)/im.width,(hh-67)/im.height)
            im=im.resize((int(im.width*scale),int(im.height*scale)))
            x=k%3*w;y=k//3*hh;canvas.paste(im,(x,y+63))
            def box(bb,color,width=2):
                if not np.isfinite(bb).all():return
                c=(np.asarray(bb)-np.r_[lo,lo])*scale+[x,y+63,x,y+63]
                c[[0,2]]=np.clip(c[[0,2]],x,x+im.width-1);c[[1,3]]=np.clip(c[[1,3]],y+63,y+63+im.height-1)
                if c[2]>c[0] and c[3]>c[1]:draw.rectangle(tuple(c),outline=color,width=width)
            if human is not None:
                h=human[human.frame_id.eq(r.frame_id)&human.class_name.eq(r.class_name)]
                for bb in h[B].to_numpy():
                    if iou(b,bb)[0,0]>.01:box(bb,'#77ee66',3)
            for _,g in agg[agg.id.eq(r.id)&agg.eligible].iterrows():
                box(g[['ref_'+b for b in B]].to_numpy(float),'#ffcc33')
                old=pd.read_csv(V31/f'features/linear_mask_cut0.5_{"oof" if split=="oof" else "test_fit"+g.fit_day[-2:]}_evidence.csv').set_index('id').loc[r.id]
                box(old[['ref_'+b for b in B]].to_numpy(float),'#55ccff')
            box(b,'#ff5555',3)
            title=f'{r.id[:8]} {r.class_name}'
            if split=='oof':title+=f' GTerror={int(r.error)}'
            draw.text((x+4,y+3),title,fill='white')
            draw.text((x+4,y+20),f'V31 {r.v31_score:.4f} -> {r.new_score:.4f} ({r.delta:+.4f})',fill='white')
            draw.text((x+4,y+37),'red=query cyan=current yellow=temporal'+(' green=GT' if split=='oof' else ''),fill='white')
        canvas.save(A/f'audit/{name}_{split}_{off//12+1}.jpg')

def main():
    ex=pd.read_csv(A/'results/experiments.csv');y=q.error.to_numpy();base=BASE['oof']
    names=ex[ex.experiment.str.startswith('temporal')].head(3).experiment.tolist()
    names+=['temporal_two_guarded_w0.125','temporal_two_guarded_w0.25','temporal_two_nonhands_both_w0.25','temporal_current_and_neighbor_nonhands_both_w0.25',
            'placebo_two_w0.25','placebo_current_and_neighbor_w0.25']
    names=list(dict.fromkeys(names));rng=np.random.default_rng(3511);frames=q.frame_id.unique()
    ids={f:np.flatnonzero(q.frame_id.eq(f)) for f in frames};boot=[]
    for _ in range(1500):
        bb=[]
        for day in DAYS:
            ff=q[q.day.eq(day)].frame_id.unique();bb.extend(ids[f] for f in rng.choice(ff,len(ff),replace=True))
        boot.append(np.concatenate(bb))
    summary=[];loo=[]
    for name in names:
        s=pd.read_csv(A/f'results/oof/{name}.csv').set_index('id').loc[q.id,'score'].to_numpy()
        bt=[AP(y[ix],s[ix])-AP(y[ix],base[ix]) for ix in boot];ll=[]
        for f in frames:
            mask=~q.frame_id.eq(f).to_numpy();delta=AP(y[mask],s[mask])-AP(y[mask],base[mask]);ll.append(delta)
            loo.append(dict(experiment=name,omitted_frame=f,delta_ap=delta))
        p=np.flatnonzero(y==1);n=np.flatnonzero(y==0)
        old=np.sign(base[p,None]-base[n]);new=np.sign(s[p,None]-s[n]);within=q.day.to_numpy()[p,None]==q.day.to_numpy()[n]
        summary.append(dict(experiment=name,**metrics(q,s),bootstrap_low=np.quantile(bt,.025),bootstrap_high=np.quantile(bt,.975),
                            loo_delta_min=min(ll),loo_delta_max=max(ll),improved_pairs=int((new>old).sum()),worsened_pairs=int((new<old).sum()),
                            within_day_improved=int(((new>old)&within).sum()),within_day_worsened=int(((new<old)&within).sum()),
                            caveat='Repeatedly used development OOF, post-selection; not an unbiased generalization interval.'))
    pd.DataFrame(summary).to_csv(A/'results/stability.csv',index=False)
    pd.DataFrame(loo).to_csv(A/'audit/leave_one_frame_out.csv',index=False)
    checks=[]
    for _,r in ex.iterrows():
        t=pd.read_csv(A/f'results/submissions/submission_{r.experiment}.csv')
        assert list(t.columns)==['id','score'] and len(t)==362 and t.id.is_unique and set(t.id)==set(te.id)
        assert t.score.between(0,1).all() and np.isfinite(t.score).all()
        s=pd.read_csv(A/f'results/oof/{r.experiment}.csv').set_index('id').loc[q.id,'score'].to_numpy()
        assert abs(AP(y,s)-r.pooled_ap)<1e-12
        checks.append(r.experiment)
    frozen=json.loads((A/'inputs/frozen_hashes.json').read_text())
    assert frozen=={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in PATHS.items()}
    evidence=pd.read_csv(A/'features/context_evidence.csv');oo=evidence[evidence.split.eq('oof')]
    assert all(f[-15:-7]!=fit.replace('-','') for f,fit in zip(oo.query_frame,oo.fit_day))
    assert all(a[-15:-7]==b[-15:-7] for a,b in zip(evidence.query_frame,evidence.context_frame))
    for name in ['temporal_two_guarded_w0.125','temporal_two_nonhands_both_w0.25']:
        plot(name,'oof');plot(name,'test')
    (A/'results/checks.json').write_text(json.dumps(dict(validated_experiments=len(checks),frozen_v31_unchanged=True,context_same_day=True,fit_day_separation=True,leaderboard_uploaded=False),indent=2))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
