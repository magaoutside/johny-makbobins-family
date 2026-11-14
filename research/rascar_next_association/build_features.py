"""Extract independent frozen BPJDet features. Targets are not consulted here."""
from common import *
from scipy.optimize import linear_sum_assignment

def inside(point,box):return float(box[0]<=point[0]<=box[2] and box[1]<=point[1]<=box[3])

def references(rec,cls):
    if cls=='nude_hands':
        result=[]
        for part in rec['parts']:
            links=part['associations'];body=rec['bodies'][links[0]['body_id']] if len(links)==1 else None
            result.append(dict(xyxy=part['xyxy'],confidence=part['confidence'],part_confidence=part['confidence'],body_confidence=body['confidence'] if body else 0,linked=int(len(links)==1),body_id=body['instance_id'] if body else -1,anchor=None,link_distance=links[0]['center_distance'] if body else 999))
        return result
    result=[]
    for body in rec['bodies']:
        heads=[p for p in rec['parts'] if any(v['body_id']==body['instance_id'] for v in p['associations'])]
        head=max(heads,key=lambda p:p['confidence']) if heads else None
        result.append(dict(xyxy=body['xyxy'],confidence=body['confidence'],body_confidence=body['confidence'],part_confidence=head['confidence'] if head else 0,linked=int(head is not None),body_id=body['instance_id'],anchor=(np.asarray(head['xyxy'][:2])+head['xyxy'][2:])/2 if head else None,link_distance=min(v['center_distance'] for v in head['associations'] if v['body_id']==body['instance_id']) if head else 999))
    return result

def extract(rows,split,pred):
    out=[]
    for (frame,cls),group in rows.groupby(['frame_id','class_name'],sort=False):
        if cls not in ['person','nude_hands']:continue
        rec=pred['head' if cls=='person' else 'hand'][frame];refs=references(rec,cls)
        rb=np.asarray([r['xyxy'] for r in refs]).reshape(-1,4);qb=group[B].to_numpy(float);ov=iou(qb,rb)
        assigned={}
        if ov.size:
            weight=np.where(ov>=.6,min(ov.shape)+1+ov,0)
            ri,ci=linear_sum_assignment(weight,maximize=True)
            assigned={int(i):float(ov[i,j]) for i,j in zip(ri,ci) if ov[i,j]>=.6}
        for k,(_,r) in enumerate(group.iterrows()):
            box=qb[k];wh=np.maximum(box[2:]-box[:2],1);center=(box[:2]+box[2:])/2
            feat=dict(hand=float(cls=='nude_hands'),confidence=float(r.confidence),cx=center[0]/1280,cy=center[1]/720,log_w=float(np.log(wh[0]/1280)),log_h=float(np.log(wh[1]/720)),n_refs=len(refs),n_queries=len(group),assigned_iou=assigned.get(k,0.))
            gate=False;reason='no_reference';refbox=np.full(4,np.nan);selected=-1
            if len(refs):
                # A head inside the query is an identity anchor, not a class label.
                head_inside=np.array([inside(v['anchor'],box) if v['anchor'] is not None else 0 for v in refs])
                eligible=(head_inside>0)&np.array([v['part_confidence']>=.3 for v in refs]) if cls=='person' else np.array([bool(v['linked']) and v['body_confidence']>=.3 for v in refs])
                ranking=ov[k]+.25*eligible;selected=int(np.argmax(ranking));ref=refs[selected];refbox=rb[selected];rw=np.maximum(refbox[2:]-refbox[:2],1)
                d=(refbox-box)/np.r_[wh,wh];rc=(refbox[:2]+refbox[2:])/2
                other=np.delete(ov[k],selected);competitor=np.delete(ov[:,selected],k)
                feat.update(ref_iou=float(ov[k,selected]),max_iou=float(ov[k].max()),second_iou=float(other.max()) if len(other) else 0.,body_confidence=ref['body_confidence'],part_confidence=ref['part_confidence'],linked=float(ref['linked']),head_inside=float(head_inside[selected]),heads_inside=float(head_inside.sum()),log_area_ratio=float(np.log(rw.prod()/wh.prod())),center_distance=float(np.linalg.norm((rc-center)/wh)),link_distance=float(ref['link_distance']/max(rw)),competitor_iou=float(competitor.max()) if len(competitor) else 0.,candidate_count=float((ov[k]>=.1).sum()),**{f'edge_{j}':float(v) for j,v in enumerate(np.clip(d,-5,5))})
                gate=bool(ov[k,selected]>=.1 and ref['linked'] and ref['part_confidence']>=.3 and ref['body_confidence']>=.3 and (cls=='nude_hands' or head_inside[selected]))
                reason='eligible' if gate else ('low_overlap' if ov[k,selected]<.1 else 'uncertain_identity_or_confidence')
            else:
                feat.update({n:0. for n in ['ref_iou','max_iou','second_iou','body_confidence','part_confidence','linked','head_inside','heads_inside','log_area_ratio','center_distance','link_distance','competitor_iou','candidate_count','edge_0','edge_1','edge_2','edge_3']})
            out.append(dict(id=r.id,frame_id=frame,day=r.day,split=split,class_name=cls,gate=gate,gate_reason=reason,selected_reference=selected,**{'ref_'+b:float(v) for b,v in zip(B,refbox)},**{'f_'+n:v for n,v in feat.items()}))
    return pd.DataFrame(out).set_index('id').loc[rows[rows.class_name.isin(['person','nude_hands'])].id].reset_index()

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--local',action='store_true');args=p.parse_args();suffix='_local' if args.local else ''
    pred={'hand':load_predictions('hand'+suffix),'head':load_predictions('head')};frames=set(h.frame_id)|set(te.frame_id)
    for k in pred:assert frames<=set(pred[k]),(k,len(frames-set(pred[k])))
    oo=extract(q,'oof',pred);tt=extract(te,'test',pred)
    pd.concat([oo,tt],ignore_index=True).to_csv(A/f'features/query_features{suffix}.csv',index=False)
    rows=[]
    for kind,cls,field in [('head','person','bodies'),('hand','person','bodies'),('hand','nude_hands','parts')]:
        for _,g in h[h.class_name.eq(cls)].iterrows():
            candidates=pred[kind][g.frame_id][field];gt=g[B].to_numpy(float)
            vals=iou(gt,[v['xyxy'] for v in candidates]).ravel() if candidates else np.array([])
            rows.append(dict(kind=kind,frame_id=g.frame_id,day=g.day,class_name=cls,best_iou=float(vals.max()) if len(vals) else 0,gt_index=int(g.name)))
    d=pd.DataFrame(rows);d.to_csv(A/f'audit/gt_reference_quality{suffix}.csv',index=False)
    summary=d.groupby(['kind','day','class_name']).agg(n=('best_iou','size'),mean_iou=('best_iou','mean'),recall06=('best_iou',lambda s:float((s>=.6).mean())))
    summary.to_csv(A/f'audit/gt_reference_summary{suffix}.csv');print(summary.to_string())
    print(pd.concat([oo,tt]).groupby(['split','class_name','gate']).size().to_string())

if __name__=='__main__':main()
