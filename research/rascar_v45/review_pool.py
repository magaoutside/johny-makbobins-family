"""Assistant visual review of fresh machine-pool images, NOT official labels.

Only unambiguous local boundaries are recorded. Hidden person geometry and
ambiguous hands are explicitly unlabelled; all new records remain out of fit.
"""
from common45 import *
from PIL import Image,ImageDraw,ImageFont
from itertools import product
# Coordinates relative to the displayed canonical-resolution crop, not resized GUI.
# These are coarse silver boxes with +/-4 canonical pixels (two native pixels).
BOXES={21:[[33,35,49,61],[63,37,80,60]],22:[[37,19,67,42]],23:[[35,34,59,58],[68,32,79,53]],26:[[47,37,68,56]]}
NOTES={19:'Two touching hands; boundaries/wrist and separate-instance ownership uncertain.',20:'Blurred/overexposed skin and object; reliable individual hand boundaries unavailable.',21:'Query covers much of phone between two separate hands; review both hands.',22:'Query mostly phone below visible hand; hand is above the query.',23:'Query spans phone and left fingers; separate hands, boundary uncertainty retained.',24:'Hands around object; motion blur and overlap prevent reliable instance bounds.',25:'Hands around light object; separate finger extents uncertain.',26:'Visible green cup; query includes substantial bright surrounding area.'}

def bounds(query,box,tol):
    options=[]
    for k in range(4):
        lo,hi=box[k]-tol,box[k]+tol
        options.append(sorted(set([lo,box[k],hi]+[v for v in query[[k%2,k%2+2]] if lo<=v<=hi])))
    samples=np.array(list(product(*options)));samples=samples[((samples[:,2:]-samples[:,:2])>0).all(1)];v=iou(query,samples)[0]
    return float(v.min()),float(v.max())

def main():
    d=pd.read_csv(A/'pool_review/candidates.csv');reviews=[];boxes=[]
    for _,r in d.iterrows():
        n=int(r.review_index);cb=r[CB].to_numpy(float);bb=[np.array(b)+np.tile(cb[:2],2) for b in BOXES.get(n,[])];status='presence_only' if r.class_name=='person' else 'uncertain';lo=hi=np.nan
        if bb:
            limits=[bounds(r[B].to_numpy(float),b,4) for b in bb];lo=max(v[0] for v in limits);hi=max(v[1] for v in limits)
            status='robust_local_geometry_error' if hi<.6 else ('geometry_supported_not_one_to_one_verified' if lo>=.6 else 'boundary_ambiguous')
            for i,b in enumerate(bb):boxes.append(dict(review_index=n,id=r.id,frame_id=r.frame_id,day=r.day,class_name=r.class_name,partition=r.partition,instance=i,edge_uncertainty_px=4,**dict(zip(B,b))))
        note=NOTES.get(n,'Person visibly present; hidden/full-body annotation extent is not recoverable precisely from this crop. No bbox error label assigned.')
        reviews.append(dict(review_index=n,id=r.id,frame_id=r.frame_id,class_name=r.class_name,partition=r.partition,status=status,iou_lower=lo,iou_upper=hi,error_silver=1 if status=='robust_local_geometry_error' else np.nan,used_for_fit=False,note=note))
    pd.DataFrame(reviews).to_csv(A/'pool_review/reviews.csv',index=False);pd.DataFrame(boxes).to_csv(A/'pool_review/boxes.csv',index=False)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
    sheet=Image.new('RGB',(1120,1200),'#161b24');draw=ImageDraw.Draw(sheet)
    for j,n in enumerate(BOXES):
        r=d.iloc[n];cb=r[CB].to_numpy(float);x=j%2*560;y=j//2*600;im=Image.open(D/f'{r.frame_id}.jpg').resize((1280,720)).crop(tuple(cb)).resize((505,505),Image.Resampling.NEAREST);sheet.paste(im,(x+25,y+75));scale=np.tile(505/(cb[2:]-cb[:2]),2);offset=np.tile([x+25,y+75],2)
        draw.rectangle(tuple((r[B].to_numpy(float)-np.tile(cb[:2],2))*scale+offset),outline='yellow',width=2)
        for b in BOXES[n]:draw.rectangle(tuple(np.array(b)*scale+offset),outline='#4ff0a0',width=2)
        rr=reviews[n];draw.text((x+15,y+8),f'#{n} {rr["status"]}',font=font,fill='white');draw.text((x+15,y+30),f'IoU interval [{rr["iou_lower"]:.3f}, {rr["iou_upper"]:.3f}]',font=font,fill='white')
    sheet.save(A/'pool_review/silver_overlay.jpg')
    (A/'pool_review/review_summary.json').write_text(json.dumps(dict(reviewed=len(d),reserved=int(d.partition.eq('reserved').sum()),status_counts=pd.Series([r['status'] for r in reviews]).value_counts().to_dict(),annotated_instances=len(boxes),test_labels=False,used_for_training=False,scope='Local silver geometry only; not complete frame matching; no reliable correct/error mix for AP',resolution='640x360 native, 1280x720 coordinate system; +/-4 coordinate-pixel bounds'),indent=2))
    print(pd.DataFrame(reviews)[['review_index','status','iou_lower','iou_upper']].to_string(index=False))
if __name__=='__main__':main()
