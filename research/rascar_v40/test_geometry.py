"""Numerical ROI alignment and same-resolution checkpoint equivalence."""
from quality import *
from resolution import SetHead
from set_detector import SetHead as Original

def main():
    t=(torch.arange(32)+.5)/32;yy,xx=torch.meshgrid(t,t,indexing='ij');feat=torch.stack([xx,yy])[None]
    boxes=torch.tensor([[.25,.25,.75,.75],[0.,0.,1.,1.]])
    out=sample_roi(feat.repeat(2,1,1,1),boxes,4)
    for k,b in enumerate(boxes):
        tx=b[0]+(torch.arange(4)+.5)/4*(b[2]-b[0]);ty=b[1]+(torch.arange(4)+.5)/4*(b[3]-b[1])
        assert torch.allclose(out[k,0],tx[None].expand(4,4),atol=1e-6)
        assert torch.allclose(out[k,1],ty[:,None].expand(4,4),atol=1e-6)
    state=torch.load(ROOT/'rascar_v39/models/set_long_fit25_seed3901.pt',weights_only=True,map_location='cpu')
    old=Original().eval();old.load_state_dict(state);new=SetHead(16).eval();new.load_state_dict(state)
    x=Data().features([0])
    with torch.inference_mode():a=old(x);b=new(x)
    assert all(torch.equal(u,v) for pair,other in zip(a,b) for u,v in zip(pair,other))
    data=QualityData();rng=np.random.default_rng(40001);idx=data.rec.index[data.rec.split.isin(['manual','empty'])].to_numpy()
    for i in rng.choice(idx,200):
        box,c,y=data.proposal(int(i),rng);assert (box[:2]<box[2:]).all() and box.min()>=0 and box.max()<=1 and y[0]==float(y[1]>=.6)
    try:data.proposal(int(data.rec.index[data.rec.split.eq('test')][0]),rng)
    except AssertionError:pass
    else:raise AssertionError('Test targets must be prohibited')
    (A/'audit/behavior_checks.json').write_text(json.dumps(dict(roi_affine_coordinates=True,checkpoint_grid16_bitwise_equivalent=True,synthetic_targets_valid=True,test_targets_rejected=True),indent=2));print('Four geometry/target checks passed')

if __name__=='__main__':main()
