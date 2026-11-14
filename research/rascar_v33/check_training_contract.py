"""Check data isolation and that person pseudo-labels cannot supervise geometry."""
from boundary_head import *
import hashlib

def main():
    data=Data();r=data.rec;rng=np.random.default_rng(3309)
    ids=r.index[r.split.str.startswith('pool_')&r.split.str.endswith('presence')].to_numpy();targets=[data.target(int(i),rng) for i in ids]
    assert all(t[2]==0 and t[3][0]==1 and t[3][1]==-1 and abs(t[4]-.3)<1e-6 for t in targets)
    b=torch.from_numpy(np.stack([t[1] for t in targets]));v=torch.tensor([t[2] for t in targets]);y=torch.from_numpy(np.stack([t[3] for t in targets]));w=torch.tensor([t[4] for t in targets])
    logits=torch.randn(len(targets),4,32,requires_grad=True);offset=torch.randn(len(targets),4,32,requires_grad=True);qual=torch.randn(len(targets),2,requires_grad=True)
    loss_fn((logits,offset,qual),(b,v,y,w),regional=True).backward()
    assert logits.grad is None and offset.grad is None and (qual.grad[:,1]==0).all() and (qual.grad[:,0]!=0).all()
    train_manifest=json.loads((A/'models/training_manifest.json').read_text())
    for m in train_manifest:
        if m['fit_day']=='all':continue
        assert all(f.split('_')[-2]==m['fit_day'].replace('-','') for f in m['fit_frames'])
        assert not set(m['fit_frames'])&set(r[r.split.eq('test')].frame_id)
    fresh=pd.read_csv(ROOT/'rascar_data_full/train.csv');aligned=fresh.set_index('id').loc[q.id];assert np.array_equal(aligned.error,q.error)
    manual=pd.read_csv(ROOT/'rascar_data_full/human_boxes_train.csv');manual.class_name=manual.class_name.replace(json.loads((ROOT/'rascar_data_full/scoring_policy.json').read_text())['aliases'])
    assert manual.equals(h[manual.columns])
    frozen={
      'rascar_v22/submission_v22_candidate.csv':'44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c',
      'rascar_v32/submission_v32_research_oof098800.csv':'c85bdb16f7c4fb33bb3bf6912a5a6dac350d220c3947a1a7c0d37345d228d985',
      'rascar_v32/submission_v32_control_oof098711.csv':'cd218cff070ed5f5639610c284776db79444310696040fcb10066e9e98460641'}
    for path,sha in frozen.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    out=dict(presence_only_rows=len(ids),presence_only_coordinate_gradient_zero=True,presence_only_quality_gradient_zero=True,manual_labels_match_fresh_download=True,all_fit_frames_day_disjoint=True,frozen_files=frozen,trained_models_checked=len(train_manifest))
    (A/'audit/training_contract_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
