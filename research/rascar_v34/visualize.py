"""Inspect model changes. Green GT is drawn ONLY on official training frames."""
from experiments import *
from PIL import Image,ImageDraw
from sklearn.metrics import average_precision_score
import cv2

def plot(rows,name,official=False,cols=3,perpage=12):
    rb=['ref_'+c for c in B]
    for off in range(0,len(rows),perpage):
        rr=rows.iloc[off:off+perpage];w=510;hh=340
        canvas=Image.new('RGB',(cols*w,int(np.ceil(len(rr)/cols))*hh),'#181818');dr=ImageDraw.Draw(canvas)
        for k,(_,r) in enumerate(rr.iterrows()):
            path=ROOT/'rascar_data_full'/f'{r.frame_id}.jpg'
            im=cv2.imread(str(path));assert im is not None;im=cv2.resize(im,(1280,720))
            b=r[B].to_numpy(float);wh=b[2:]-b[:2]
            lo=np.maximum(0,np.floor(b[:2]-np.maximum(wh*.55,35))).astype(int)
            hi=np.minimum([1280,720],np.ceil(b[2:]+np.maximum(wh*.55,35))).astype(int)
            roi=Image.fromarray(im[lo[1]:hi[1],lo[0]:hi[0],::-1]);scale=min((w-8)/roi.width,(hh-65)/roi.height)
            roi=roi.resize((int(roi.width*scale),int(roi.height*scale)))
            x=(k%cols)*w;y=(k//cols)*hh;canvas.paste(roi,(x,y+60))
            def box(bb,color,width=2):
                if not np.isfinite(bb).all():return
                c=(np.array(bb)-np.r_[lo,lo])*scale+np.array([x,y+60,x,y+60])
                # Clip overlays to their own tile.
                c[[0,2]]=np.clip(c[[0,2]],x,x+roi.width-1)
                c[[1,3]]=np.clip(c[[1,3]],y+60,y+60+roi.height-1)
                if c[2]>c[0] and c[3]>c[1]:dr.rectangle(tuple(c),outline=color,width=width)
            if official:
                h=pd.read_csv(ROOT/'rascar_v22/inputs/manual_boxes.csv')
                for bb in h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy():
                    if iou(b,bb)[0,0]>.01:box(bb,'#48ef53',3)
            fits=[d for d in DAYS if d!=r.day] if official else DAYS
            for fit in fits:
                side=align(ROOT/f'rascar_v33/features/side_pool_regional_balanced_fit{fit[-2:]}.csv',pd.DataFrame([r]))
                box(side.iloc[0][rb].to_numpy(float),'#32ceee')
            box(b,'#ff4444',3)
            title=f"{r.get('review_no',r.id[:8])} {r.class_name}"
            if official:title+=f" GT error={int(r.error)}"
            dr.text((x+4,y+3),title,fill='white')
            dr.text((x+4,y+19),f"V31 {r.v31_score:.4f}  new {r.new_score:.4f}",fill='white')
            dr.text((x+4,y+35),'red=query cyan=predicted ref'+(' green=official GT' if official else ''),fill='white')
        path=A/f'audit/{name}_{off//perpage+1}.jpg';canvas.save(path);print(path)

def main():
    # IDs and display indices only: no old manual test judgments are consumed.
    numbers=pd.read_csv(ROOT/'rascar_v29/audit/all_362_fresh_audit.csv',usecols=['id','review_no'])
    for name in ['person_confirm_floor0.4_restore0.5','nude_hands_boundary_identity0_w0.1']:
        d=pd.read_csv(A/f'audit/{name}_changes.csv');d=d[d.split.eq('test')].merge(numbers,on='id')
        plot(d.sort_values('review_no'),name)
    o=q.assign(score=BASE['oof'],v31_score=BASE['oof'],new_score=BASE['oof']).copy()
    ap=average_precision_score(o.error,o.score);gain=[]
    for i,r in o.iterrows():
        s=o.score.copy();s.iloc[i]=-1 if r.error==0 else 2
        gain.append(average_precision_score(o.error,s)-ap)
    o['oracle_gain']=gain
    top=o.nlargest(12,'oracle_gain')
    top.to_csv(A/'audit/oof_priority.csv',index=False);plot(top,'oof_priority',official=True)

if __name__=='__main__':main()
