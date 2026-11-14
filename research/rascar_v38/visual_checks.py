"""Render model geometry for inspection; no human test labels are written."""
from common38 import *
from PIL import Image,ImageDraw
import cv2

def render(rows,path,oracle=False):
    for start in range(0,len(rows),8):
        batch=rows.iloc[start:start+8];canvas=Image.new('RGB',(1600,850),'#202020');dr=ImageDraw.Draw(canvas)
        for j,(_,r) in enumerate(batch.iterrows()):
            b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],20);cc=(b[:2]+b[2:])/2;lo=np.maximum(0,np.floor(cc-wh*1.2)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+wh*1.2)).astype(int)
            raw=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));im=Image.fromarray(raw[lo[1]:hi[1],lo[0]:hi[0],::-1]);scale=min(390/im.width,345/im.height);im=im.resize((int(im.width*scale),int(im.height*scale)))
            x=j%4*400;y=j//4*425;canvas.paste(im,(x,y+72))
            def box(bb,color):
                bb=(np.asarray(bb)-np.r_[lo,lo])*scale+[x,y+72,x,y+72];bb[[0,2]]=np.clip(bb[[0,2]],x,x+im.width-1);bb[[1,3]]=np.clip(bb[[1,3]],y+72,y+72+im.height-1)
                if np.isfinite(bb).all() and bb[2]>bb[0] and bb[3]>bb[1]:dr.rectangle(tuple(bb),outline=color,width=2)
            if oracle:
                for truth in h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy():box(truth,'#39ff76')
                box(r[['oracle_'+b for b in B]].to_numpy(float),'#00cfff')
            else:
                box(r[['ref_'+b for b in B]].to_numpy(float),'#00cfff')
            box(b,'#ff5050')
            dr.text((x+3,y+2),f'{r.class_name} {r.id[:12]}',fill='white')
            dr.text((x+3,y+18),'RED query; CYAN predicted reference',fill='white')
            if oracle:dr.text((x+3,y+34),f'GT error={r.error}; GREEN manual GT',fill='white')
            else:dr.text((x+3,y+34),f'V31 {r.v31_score:.3f}; V38 {r.new_score:.3f}; no test GT',fill='white')
        canvas.save(A/f'audit/{path}_{start//8+1}.jpg')

def main():
    oo=pd.read_csv(A/'audit/occupancy_oracle.csv');bad=oo[oo.oracle_disagrees].merge(q[['id']+B],on='id')
    render(bad,'official_oracle_failure',oracle=True)
    changes=pd.read_csv(A/'audit/v38_all4_all_daily_RESEARCH_changes.csv');d=changes[changes.split.eq('test')].copy();d['absolute_delta']=abs(d.delta)
    selected=d.sort_values('absolute_delta',ascending=False).drop_duplicates(['class_name','frame_id']).groupby('class_name').head(6)
    pred=pd.read_csv(A/'features/multilevel_predictions.csv');pred=pred[pred.kind.eq('all4')&pred.split.eq('test')&pred.fit_day.eq('2026-10-25')&pred.seed.eq(3101)]
    # A fixed single reference for illustration, not a new rule or human label.
    selected=selected.merge(pred[['id']+['ref_'+b for b in B]],on='id').sort_values(['class_name','absolute_delta'],ascending=[True,False])
    selected.to_csv(A/'audit/test_geometry_inspection_selection.csv',index=False);render(selected,'test_model_geometry')

if __name__=='__main__':main()
