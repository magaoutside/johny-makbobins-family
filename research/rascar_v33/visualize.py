"""Review outputs on pre-existing hard cases and largest adverse OOF changes."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v33'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import B,q,te,align
from features23 import image
import numpy as np,pandas as pd
from PIL import Image,ImageDraw

def main():
    audit=pd.read_csv(ROOT/'rascar_v29/audit/all_362_fresh_audit.csv');rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv').set_index('id')
    kinds=['regional_hardneg','pool_regional'] if '--pool' in sys.argv else ['manual','hardneg']
    if '--final' in sys.argv:kinds=['pool_regional_balanced','pool_regional_balanced_rare']
    evidence={(k,f):pd.read_csv(A/f'features/side_{k}_fit{f}.csv').set_index('id') for k in kinds for f in ['25','26']}
    experiments=pd.read_csv(A/'results/experiments.csv');best=experiments.iloc[0].experiment
    if '--final' in sys.argv:best='v33_positive_confirmation_only'
    new=align(A/f'results/submissions/submission_{best}.csv',te,'score');old=align(ROOT/'rascar_v32/submission_v32_research_oof098800.csv',te,'score');scores=dict(zip(te.id,zip(old,new)))
    sets=[([112,185,266,145,146,204,260,267],'hands'),([133,65,67,1,157,136,73,142],'people_cups_tray')]
    if '--pool' in sys.argv:sets=[([112,185,266,204,296,220,200,27],'pool_hands'),([292,178,133,65,157,136,73,142],'pool_people_cups_tray')]
    if '--final' in sys.argv:sets=[([112,185,266,296,220,27],'final_hands'),([133,36,301,263,159,276],'final_people')]
    for numbers,name in sets:
        canvas=Image.new('RGB',(1200,280*len(numbers)),'#171717');draw=ImageDraw.Draw(canvas)
        for row,number in enumerate(numbers):
            ar=audit[audit.review_no.eq(number)].iloc[0];r=rec.loc[ar.id];cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(int)
            roi=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2],::-1];scale=min(390/roi.shape[1],205/roi.shape[0]);size=(int(roi.shape[1]*scale),int(roi.shape[0]*scale));base=Image.fromarray(roi).resize(size)
            for col,kind in enumerate(['original']+kinds):
                x=400*col;y=280*row+45;canvas.paste(base,(x,y));draw.text((x+3,y-39),f'#{number} {r.class_name}: {kind}',fill='white')
                bs,ns=scores[r.name];draw.text((x+3,y-23),f'V32 {bs:.3f} -> exploratory {ns:.3f}',fill='white')
                def box(b,color):
                    b=(np.asarray(b)-np.r_[cb[:2],cb[:2]])*scale+np.array([x,y,x,y]);draw.rectangle(tuple(b),outline=color,width=2)
                box(r[B].to_numpy(float),'#ff3333')
                if kind!='original':
                    text=[]
                    for fit,color in [('25','#40ff80'),('26','#36beff')]:
                        e=evidence[(kind,fit)].loc[r.name]
                        if e.valid:box(e[['ref_'+k for k in B]].to_numpy(float),color)
                        text.append(f'{fit}:IoU{e.ref_iou:.2f} Q{e.predicted_quality:.2f} P{e.presence:.2f}')
                    draw.text((x+3,y+size[1]+2),' | '.join(text),fill='white')
        canvas.save(A/f'audit/boundary_{name}.jpg')
    oo=align(A/f'results/oof/{best}.csv',q,'score');oldoo=align(ROOT/'rascar_v32/results/oof/compose_watershed1_confirmed_hand_protection2.csv',q,'score');d=q.copy();d['old_v32']=oldoo;d['new']=oo;d['delta']=oo-oldoo
    d['adverse_change']=np.where(d.error==1,-d.delta,d.delta);d.sort_values('adverse_change',ascending=False).to_csv(A/'audit/oof_changes.csv',index=False)
    dd=audit.set_index('id').loc[te.id].reset_index();dd['v32']=old;dd['new']=new;dd['delta']=new-old
    for k,f in evidence:
        e=evidence[k,f].loc[te.id]
        for c in ['presence','predicted_quality','ref_iou','optimistic_iou']:dd[f'{k}_{f}_{c}']=e[c].to_numpy()
    dd.to_csv(A/'audit/test_score_changes.csv',index=False)
    (A/'audit/reviewed_experiment.json').write_text(json.dumps(dict(experiment=best,selection='Highest primary OOF before visual review; not a release decision.',query_box='red',fit25='green',fit26='blue',numbers_are_existing_audit_ids=True),indent=2))
    print(best);print(d.sort_values('adverse_change',ascending=False)[['id','class_name','error','old_v32','new','delta']].head(10).to_string(index=False))
if __name__=='__main__':main()
