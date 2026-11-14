"""Ablation using only bench templates consistent with the whole-person policy.
Official validation labels are never edited or filtered.
"""
from common17 import *
from scipy.special import expit
bank=pd.read_csv(V16/'head_geometry_templates.csv');heads=pd.read_csv(V16/'wevis_heads_references.csv')
maps=json.loads((A/'seat_maps.json').read_text())

def seat(center,tables):
    w=tables[:,2]-tables[:,0];hh=tables[:,3]-tables[:,1];cx=(tables[:,0]+tables[:,2])/2
    anchors=np.vstack([np.c_[cx,tables[:,1]-.45*hh],np.c_[cx,tables[:,3]+.1*hh]])
    scale=np.vstack([np.c_[w,hh],np.c_[w,hh]]);dist=np.linalg.norm((anchors-center)/scale,axis=1);j=int(dist.argmin());return j,float(dist[j])

def templates(fit):
    pool=bank if fit=='all' else bank[bank.day.eq(fit)]
    records=[]
    for r in pool.itertuples():
        # Filtering only uses training-day annotations from the same frame.
        tt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq('table')].sort_values('x1')[B].to_numpy()
        if len(tt)!=6:continue
        j,dist=seat([r.head_x,r.head_y],tt);t=tt[j%6];head_ratio=(r.y2-r.y1)/r.head_w
        full=(j<6)&(dist<1.2)&(head_ratio>=3)&(r.y2>=t[3]-.15*(t[3]-t[1]))
        records.append(dict(**r._asdict(),seat=j,distance=dist,head_height_ratio=head_ratio,table_bottom=t[3],accepted=bool(full)))
    return pd.DataFrame(records)

def evidence(rows,fit):
    table=np.array(maps[fit]['tables']);pool=templates(fit);pool.to_csv(A/f'policy_template_ledger_{fit}.csv',index=False);valid=pool[pool.accepted];out=[]
    for r in rows.itertuples():
        d=dict(id=r.id,policy_risk=0.,policy_iou=1.,support=0,borrowed=False)
        if r.class_name!='person':out.append(d);continue
        b=np.array([getattr(r,c) for c in B]);hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy();cc=(hh[:,:2]+hh[:,2:])/2;inside=((cc>=b[:2])&(cc<=b[2:])).all(1)
        if inside.sum()!=1:out.append(d);continue
        head=hh[inside][0];center=(head[:2]+head[2:])/2;hw=head[2]-head[0];j,dist=seat(center,table)
        if j>=6 or dist>1.2:out.append(d);continue
        sub=valid[valid.seat.eq(j)];borrowed=False
        if len(sub)<2:
            # Borrow only from adjacent upright bench positions, excluding the
            # oblique window seat; head-size scaling adjusts perspective.
            if j==0:out.append(d);continue
            sub=valid[valid.seat.between(max(1,j-1),min(5,j+1))];borrowed=True
        if len(sub)<2:out.append(d);continue
        pc=sub[['head_x','head_y']].to_numpy();scale=np.clip(hw/sub.head_w.to_numpy(),.65,1.5)
        pb=sub[B].to_numpy();pred=(pb-np.tile(pc,(1,2)))*scale[:,None]+np.tile(center,(len(pb),2))
        ov=iou(b,pred)[0];high=float(np.quantile(ov,.75));aspect=(b[2]-b[0])/(b[3]-b[1])
        # Abstain on wide sideways poses. It is a reference-quality signal only.
        risk=float(expit((.6-high)/.07))*float(aspect<1.6)
        d.update(policy_risk=risk,policy_iou=high,support=len(sub),borrowed=borrowed);out.append(d)
    return pd.DataFrame(out)
fo=pd.concat([evidence(q[q.day.eq(held)],fit) for fit,held in zip(DAYS,DAYS[::-1])]).set_index('id').loc[q.id].reset_index();ft=evidence(te,'all')
fo.to_csv(A/'policy_evidence_oof.csv',index=False);ft.to_csv(A/'policy_evidence_test.csv',index=False)
for weight in [.35,.6]:
    register(f'policy_templates_w{weight}',q.score+weight*(1-q.score)*fo.policy_risk.to_numpy(),current+weight*(1-current)*ft.policy_risk.to_numpy(),dict(weight=weight,policy='whole body including parts behind table',validation_labels_unchanged=True,template_filter='fit-day boxes extend to table bottom and height at least 3 head widths'))
