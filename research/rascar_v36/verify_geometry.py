"""Check native coordinate target/decode roundtrip and identity-switch veto.
These checks catch coordinate/class mix-ups which AP alone could conceal.
"""
from dense_votes import *
from evaluate_votes import collect

def main():
 rec=pd.read_csv(V31/'inputs/dense_records.csv');tested=[]
 for _,r in rec[rec.split.eq('manual')&rec.class_name.isin(CL)].iloc[::11].iterrows():
  y,dist,ctr,w=targets(r,'dense');cl=CL.index(r.class_name)
  if w[cl].sum()==0:continue
  logits=torch.logit(torch.tensor(y).clamp(.0001,.9999));center=torch.logit(torch.tensor(ctr).clamp(.0001,.9999))
  boxes=pd.DataFrame(decode((logits,torch.tensor(dist),center),r));boxes=boxes[boxes.class_name.eq(r.class_name)&boxes.decoder.eq('vote')]
  actual=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy(float)
  gt=actual[iou(r[B],actual).argmax()]
  # Baseline NMS intentionally removes overlapping proposals. Roundtrip below
  # tests isolated boxes; a separate real overlapping-hands regression follows.
  own=iou(gt,actual).ravel();own[own.argmax()]=0
  if own.max()>=.45:continue
  cb=r[CB].to_numpy(float)
  if (gt[:2]<cb[:2]).any() or (gt[2:]>cb[2:]).any():continue
  overlap=float(iou(gt,boxes[B]).max());assert overlap>.99,(r.id,overlap)
  tested.append(dict(id=r.id,class_name=r.class_name,decoded_gt_iou=overlap))
 assert len(tested)>20
 r=rec[rec.id.eq('manual_477_1')].iloc[0];y,dist,ctr,w=targets(r,'dense')
 pred=(torch.logit(torch.tensor(y).clamp(.0001,.9999)),torch.tensor(dist),torch.logit(torch.tensor(ctr).clamp(.0001,.9999)))
 exact={}
 for threshold in [.45,.85]:
  boxes=pd.DataFrame(decode(pred,r,suppression_iou=threshold));boxes=boxes[boxes.class_name.eq(r.class_name)&boxes.decoder.eq('vote')]
  exact[str(threshold)]=float(iou(r[B],boxes[B]).max())
 assert exact['0.45']<.6 and exact['0.85']>.99
 # Verify the actual observed neighboring-person failure is refused by identity.
 test_id='52bbde202c7ea8f7';rows=te[te.id.eq(test_id)];checks=[]
 for fit in DAYS:
  boxes=pd.read_csv(A/f'features/dense_fit{fit[-2:]}_boxes.csv')
  raw=collect(boxes,rows,'vote',.55,False).iloc[0];guarded=collect(boxes,rows,'vote',.55,True).iloc[0]
  checks.append(dict(fit_day=fit,raw_valid=bool(raw.valid),identity_valid=bool(guarded.valid),query_heads=int(guarded.query_heads)))
 assert not all(z['identity_valid'] for z in checks)
 (A/'results/geometry_checks.json').write_text(json.dumps(dict(roundtrip_samples=tested,overlapping_hands_regression=exact,neighbor_switch_guard=checks),indent=2));print('Geometry checks passed:',len(tested),'isolated target roundtrips; overlapping-hands suppression and neighboring-person guard verified.')

if __name__=='__main__':main()
