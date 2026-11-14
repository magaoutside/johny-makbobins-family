"""User's table-anchored extension hypothesis, including conservative ablations."""
from features23 import frame_objects, heads
from common23 import *
from functools import lru_cache

def anchor(r):
    b=np.array([getattr(r,c) for c in B]);hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy();hc=(hh[:,:2]+hh[:,2:])/2
    inside=((hc>=b[:2])&(hc<=b[2:])).all(1)
    if inside.sum()!=1:return None
    hb=hh[inside][0];head=(hb[:2]+hb[2:])/2
    rr=frame_objects(r.frame_id)['table'];rr=rr[rr.confidence.ge(.55)]
    if not len(rr):return None
    tb=rr[B].to_numpy();tc=(tb[:,:2]+tb[:,2:])/2;tw=tb[:,2:]-tb[:,:2]
    j=int(np.linalg.norm((tc-head)/tw,axis=1).argmin());table=tb[j];wh=tw[j];rel=(head-tc[j])/wh
    if abs(rel[0])>.7 or not -1.3<rel[1]<2.1 or not 60<wh[1]<180:return None
    side='far' if rel[1]<-.15 else 'near' if rel[1]>.15 else 'uncertain'
    return b,hb,head,table,wh,rel,side

@lru_cache(None)
def templates(fit):
    records=[]
    for r in h[h.day.eq(fit)&h.class_name.eq('person')].itertuples():
        a=anchor(r)
        if a is None:continue
        b,hb,head,tb,wh,rel,side=a
        records.append(dict(frame_id=r.frame_id,side=side,head_x=head[0],rel_x=rel[0],rel_y=rel[1],
            bottom=(b[3]-tb[3])/wh[1],top=(b[1]-tb[1])/wh[1],height=(b[3]-b[1])/wh[1]))
    return pd.DataFrame(records)

def make_features(rows,fit):
    bank=templates(fit);out=[]
    for r in rows.itertuples():
        f=dict(id=r.id,eligible=0,far=0,near=0,table_x1=0.,table_y1=0.,table_x2=0.,table_y2=0.,head_x=0.,head_y=0.,
            supports=0,learned_bottom_q10=0.,learned_bottom_q50=0.,learned_top_q90=0.,learned_height_q10=0.)
        a=anchor(r) if r.class_name=='person' else None
        if a is None:out.append(f);continue
        b,hb,head,tb,wh,rel,side=a
        f.update(eligible=int(side!='uncertain'),far=int(side=='far'),near=int(side=='near'),head_x=head[0],head_y=head[1],**dict(zip(['table_'+v for v in B],tb)))
        g=bank[bank.side.eq(side)&bank.frame_id.ne(r.frame_id)]
        dist=np.linalg.norm((g[['rel_x','rel_y']].to_numpy()-rel),axis=1)+abs(g.head_x.to_numpy()-head[0])/600
        g=g.loc[dist<.8];f['supports']=len(g)
        if len(g)>=3:
            f.update(learned_bottom_q10=float(g.bottom.quantile(.1)),learned_bottom_q50=float(g.bottom.median()),
                learned_top_q90=float(g.top.quantile(.9)),learned_height_q10=float(g.height.quantile(.1)))
        out.append(f)
    return pd.DataFrame(out)

if __name__=='__main__':
    os=[];ts=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        templates(fit).to_csv(A/f'inputs/table_body_templates_{fit}.csv',index=False)
        o=make_features(q[q.day.eq(held)],fit);t=make_features(te,fit)
        o.to_csv(A/f'features/table_zone_held_{fit}.csv',index=False);t.to_csv(A/f'features/table_zone_test_{fit}.csv',index=False)
        os.append(o);ts.append(t.set_index('id').loc[te.id])
    fo=pd.concat(os).set_index('id').loc[q.id];ft=(ts[0]+ts[1])/2
    for extension in ['edge','plus15','plus30','learned_min','learned_median']:
        for guard in ['far','far_visible','both_learned']:
            if guard=='both_learned' and extension not in ['learned_min','learned_median']:continue
            for weight in [.25,.5]:
                outputs=[]
                for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                    bb=rows[B].to_numpy().copy();height=f.table_y2.to_numpy()-f.table_y1.to_numpy()
                    value=0. if extension=='edge' else .15 if extension=='plus15' else .3 if extension=='plus30' else f['learned_bottom_q10' if extension=='learned_min' else 'learned_bottom_q50'].to_numpy()
                    bottom=f.table_y2.to_numpy()+value*height
                    bb[:,3]=np.maximum(bb[:,3],bottom)
                    gate=rows.class_name.eq('person').to_numpy()&f.eligible.eq(1).to_numpy()&f.far.eq(1).to_numpy()
                    if extension.startswith('learned'):gate &= f.supports.ge(3).to_numpy()
                    if guard=='both_learned':
                        gate=rows.class_name.eq('person').to_numpy()&f.eligible.eq(1).to_numpy()&f.supports.ge(3).to_numpy()
                        near=f.near.eq(1).to_numpy();bb[near,1]=np.minimum(bb[near,1],(f.table_y1+f.learned_top_q90*height).to_numpy()[near])
                    if split=='test' and extension.startswith('learned'):
                        gate &= np.minimum(ts[0].supports,ts[1].supports).to_numpy()>=3
                    cf=pd.concat([pd.read_csv(A/f'features/held_{fit}.csv') for fit in DAYS]).set_index('id').loc[rows.id] if split=='oof' else sum([pd.read_csv(A/f'features/test_{fit}.csv').set_index('id').loc[rows.id] for fit in DAYS])/2
                    if guard=='far_visible':gate &= (cf.fg_bottom.to_numpy()>.25)&(cf.body_pose_views.to_numpy()>=2)
                    # If any similar complete GT supports the current box, do not override it.
                    if extension=='learned_min':gate &= cf.seat_iou_max.to_numpy()<.65
                    wh=rows[['x2','y2']].to_numpy()-rows[['x1','y1']].to_numpy();ov=wh.prod(1)/np.maximum((bb[:,2:]-bb[:,:2]).prod(1),1)
                    strength=np.clip((.7-ov)/.3,0,1);score=base+weight*(1-base)*strength*gate
                    pd.DataFrame(dict(id=rows.id,gate=gate,extension_iou=ov,strength=strength,score=score,ref_y1=bb[:,1],ref_y2=bb[:,3])).to_csv(A/f'features/table_{extension}_{guard}_{weight}_{split}.csv',index=False)
                    outputs.append(score)
                register(f'table_{extension}_{guard}_w{weight}',*outputs,dict(type='table_anchored_extension',extension=extension,guard=guard,weight=weight,
                    meaning='extend reference, keep original query coordinates; apply bounded error penalty',test_rules_per_id=False))
