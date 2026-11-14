"""Replace a merged dense reference only with one independently anchored instance.

No fitted thresholds or per-ID exceptions. Reuses exact original V31 composition.
"""
from common import *
from build_features import inside
from evaluate import save

def target(x):return np.clip((.85-x)/.5,0,1)
def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])

def evidence(rows,old,head,hand):
    out=[]
    for _,r in rows.iterrows():
        d=old.loc[r.id];oldbox=d[['ref_'+b for b in B]].to_numpy(float);box=r[B].to_numpy(float);frame=r.frame_id
        heads=[p for p in head[frame]['parts'] if p['confidence']>=.3];hp=[(np.array(p['xyxy'][:2])+p['xyxy'][2:])/2 for p in heads]
        hands=[p for p in hand[frame]['parts'] if p['confidence']>=.3 and len(p['associations'])==1 and hand[frame]['bodies'][p['associations'][0]['body_id']]['confidence']>=.3]
        candidates=[];anchors=[]
        if r.class_name=='person':
            anchors=hp
            for b in hand[frame]['bodies']:
                if b['confidence']<.3:continue
                ih=[p for p in hp if inside(p,b['xyxy'])]
                if len(ih)==1 and inside(ih[0],box):candidates.append(b['xyxy'])
        elif r.class_name=='nude_hands':
            candidates=[p['xyxy'] for p in hands];anchors=[(np.array(p['xyxy'][:2])+p['xyxy'][2:])/2 for p in hands]
        n_anchors=sum(inside(p,oldbox) for p in anchors) if np.isfinite(oldbox).all() else 0
        best_iou=0.;newbox=np.full(4,np.nan);gate=False;ratio=0.
        if candidates:
            ov=iou(box,candidates).ravel();j=int(np.argmax(ov));best_iou=float(ov[j]);newbox=np.asarray(candidates[j]);ratio=area(oldbox)/max(area(newbox),1)
            gate=bool(np.isfinite(oldbox).all() and n_anchors>=2 and best_iou>=.65 and d.overlap<.6 and ratio>=1.5)
        out.append(dict(id=r.id,frame_id=frame,class_name=r.class_name,old_iou=float(d.overlap),new_iou=best_iou,old_anchors=n_anchors,area_ratio=ratio,gate=gate,**{'ref_'+b:float(v) for b,v in zip(B,newbox)}))
    return pd.DataFrame(out).set_index('id').loc[rows.id].reset_index()

def main():
    protocol=dict(new_reference_min_query_iou=.65,old_reference_max_query_iou=.6,min_old_reference_anchors=2,min_old_new_area_ratio=1.5,body_and_part_min_conf=.3,person_reference='BPJ hand-local body with exactly one confident BPJ head in body and query',hand_reference='separate BPJ hand-local part with one confident body association',effect='Replace only merged dense evidence, then recompute original V31 consensus composition',test_labels=False)
    (A/'inputs/refine_merged_protocol.json').write_text(json.dumps(protocol,indent=2))
    head=load_predictions('head');hand=load_predictions('hand_local')
    v22o=pd.read_csv(ROOT/'rascar_v22/results/oof/final_reviewed.csv',usecols=['id','score']).set_index('id').loc[q.id].score.to_numpy()
    prior=pd.read_csv(ROOT/'rascar_v18/results/oof/release_main.csv',usecols=['id','score']).set_index('id').loc[q.id].score.to_numpy()
    corrected=pd.read_csv(ROOT/'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv',usecols=['id','score']).set_index('id').loc[q.id].score.to_numpy()
    v22={'oof':np.clip(v22o-prior+corrected,0,1),'test':pd.read_csv(ROOT/'rascar_v22/submission_v22_candidate.csv').set_index('id').loc[te.id].score.to_numpy()}
    old={};ev={}
    for split,rows in [('oof',q),('test',te)]:
        old[split]=[];ev[split]=[]
        for tail in (['oof'] if split=='oof' else ['test_fit25','test_fit26']):
            d=pd.read_csv(ROOT/f'rascar_v31/features/linear_mask_cut0.5_{tail}_evidence.csv').set_index('id').loc[rows.id]
            e=evidence(rows,d,head,hand);e.to_csv(A/f'audit/merged_evidence_{tail}.csv',index=False);old[split].append(d.overlap.to_numpy());ev[split].append(e)
    def compose(split,overlaps):
        base=v22[split].copy();t=np.stack([target(v) for v in overlaps])
        if split=='oof':
            m=np.isfinite(t[0]);base[m]=.75*base[m]+.25*t[0,m]
        else:
            diff=t-base;up=np.all(diff>=0,axis=0);down=np.all(diff<=0,axis=0);ok=np.isfinite(t).all(0)&(up|down)
            conservative=np.where(up,np.min(t,axis=0),np.max(t,axis=0));base[ok]=.75*base[ok]+.25*conservative[ok]
        return base
    for split in old:assert np.max(abs(compose(split,old[split])-BASE[split]))<1e-12
    results=[]
    for cls in ['person','nude_hands','both']:
        values={}
        for split in old:
            revised=[]
            for overlaps,e in zip(old[split],ev[split]):
                gate=e.gate.to_numpy()&(True if cls=='both' else e.class_name.eq(cls).to_numpy());new=overlaps.copy();new[gate]=e.loc[gate,'new_iou'];revised.append(new)
            values[split]=compose(split,revised)
        name=f'bpj_merged_reference_{cls}';result=save(name,values);results.append(result)
    pd.DataFrame(results).to_csv(A/'results/merged_experiments.csv',index=False);print(pd.DataFrame(results).to_string(index=False))

if __name__=='__main__':main()
