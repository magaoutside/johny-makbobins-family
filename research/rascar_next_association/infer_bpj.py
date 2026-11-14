"""Frozen official BPJDet inference. No training, test labels or online calls.

Uses authors' preprocessing/NMS and exact association function (AST-extracted
to avoid unrelated COCO evaluation imports). All exported coordinates 1280x720.
"""
from pathlib import Path
import os,sys,json,time,ast,argparse,hashlib,subprocess
A=Path(__file__).resolve().parent;ROOT=A.parent;V=A/'vendor/BPJDet'
os.environ['MPLCONFIGDIR']=str(A/'features/mpl');sys.path.insert(0,str(V))
import numpy as np,pandas as pd,torch,cv2
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression,scale_coords

def author_decoder():
    tree=ast.parse((V/'val.py').read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['cal_inside_iou','post_process_batch']]
    assert len(selected)==2
    ns=dict(np=np,torch=torch,Path=Path,scale_coords=scale_coords)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(V/'val.py'),'exec'),ns)
    return ns['post_process_batch']

def infer(model,im,kind,decode,imgsz,coordinate_shape=(720,1280)):
    stride=int(model.stride.max());imx,ratio,pad=letterbox(im,imgsz,auto=True,stride=stride)
    x=torch.from_numpy(np.ascontiguousarray(imx[:,:,::-1].transpose(2,0,1))).float()[None]/255
    offsets=4 if kind=='hand' else 2
    with torch.inference_mode():out=model(x,augment=False)[0]
    bodies=non_max_suppression(out,.05,.6,classes=[0],num_offsets=offsets)
    parts=non_max_suppression(out,.10,.3,classes=[1],num_offsets=offsets)
    rawpart=parts[0].clone();rawbody=bodies[0].clone()
    rawpart[:,:4]=scale_coords(x.shape[2:],rawpart[:,:4].clone(),coordinate_shape)
    rawbody[:,:4]=scale_coords(x.shape[2:],rawbody[:,:4].clone(),coordinate_shape)
    rawbody[:,6:]=scale_coords(x.shape[2:],rawbody[:,6:].clone(),coordinate_shape)
    config={'dataset':'BodyHands' if kind=='hand' else 'CrowdHuman','num_offsets':offsets,'match_iou_thres':.6}
    bb,pp,ss,_,_,_=decode(config,x,[],[[coordinate_shape]],bodies,parts)
    instances=[];assigned=set();links=[]
    for body_id,(box,points,score) in enumerate(zip(bb,pp,ss)):
        instances.append(dict(instance_id=body_id,xyxy=np.asarray(box).tolist(),confidence=float(score),predicted_part_centers=rawbody[body_id,6:].reshape(-1,2).tolist()))
        for slot,point in enumerate(points):
            if point[2]<=0:continue
            candidates=np.max(abs(rawpart[:,:4].numpy()-np.asarray(point[3:])),axis=1)
            j=int(np.argmin(candidates));assert candidates[j]<1e-3
            assigned.add(j)
            anchor=rawbody[body_id,6:].reshape(-1,2)[slot].numpy()
            links.append(dict(body_id=body_id,slot=slot,part_id=j,center_distance=float(np.linalg.norm(anchor-np.asarray(point[:2])))))
    ps=[dict(part_id=j,xyxy=p[:4].tolist(),confidence=float(p[4]),part_type=kind,associations=[v for v in links if v['part_id']==j]) for j,p in enumerate(rawpart)]
    return dict(bodies=instances,parts=ps,unassigned_part_ids=[j for j in range(len(ps)) if j not in assigned],input_shape=list(x.shape),native_shape=list(im.shape[:2]),coordinate_shape=list(coordinate_shape),letterbox_ratio=list(ratio),letterbox_pad=list(pad))

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['hand','head'],required=True);p.add_argument('--limit',type=int);p.add_argument('--imgsz',type=int,default=1536);args=p.parse_args()
    torch.set_num_threads(3)
    q=pd.read_csv(ROOT/'rascar_v31/results/oof/linear_mask_cut0.5_all_two_sided_w0.25.csv');h=pd.read_csv(ROOT/'rascar_v22/inputs/manual_boxes.csv');te=pd.read_csv(ROOT/'rascar_v22/inputs/test_queries.csv')
    # Process all official manual frames, including class-absent frames, and test query frames.
    frames=sorted(set(h.frame_id)|set(te.frame_id))
    outpath=A/f'features/{args.kind}.jsonl';done=set()
    if outpath.exists():done={json.loads(line)['frame_id'] for line in outpath.read_text().splitlines()}
    frames=[f for f in frames if f not in done]
    if args.limit:frames=frames[:args.limit]
    fname='bh_hand_s_1536_e100_best.pt' if args.kind=='hand' else 'ch_head_s_1536_e150_best_mMR.pt'
    weights=A/'models'/fname;manifest=json.loads((A/'models/download_manifest.json').read_text());entry=next(x for x in manifest['files'] if x['name']==fname)
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==entry['sha256']
    model=attempt_load(str(weights),map_location='cpu');decode=author_decoder();started=time.time()
    for k,frame in enumerate(frames):
        im=cv2.imread(str(ROOT/'rascar_data_full'/f'{frame}.jpg'));assert im is not None,frame;t=time.time()
        rec=infer(model,im,args.kind,decode,args.imgsz)
        rec.update(frame_id=frame,kind=args.kind,seconds=time.time()-t,weights_sha256=entry['sha256'],imgsz=args.imgsz)
        with outpath.open('a') as f:f.write(json.dumps(rec)+'\n')
        print(args.kind,k+1,len(frames),frame,'bodies',len(rec['bodies']),'parts',len(rec['parts']),'sec',round(rec['seconds'],2),flush=True)
    (A/f'features/{args.kind}_runtime.json').write_text(json.dumps(dict(torch=torch.__version__,device='cpu',threads=3,repo_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=V,text=True).strip(),seconds=time.time()-started,frames_this_run=len(frames),imgsz=args.imgsz,augment=False,body_conf=.05,body_nms=.6,part_conf=.1,part_nms=.3,association_inside_ratio=.6),indent=2))

if __name__=='__main__':main()
