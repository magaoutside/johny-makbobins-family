"""Diagnostic plots; never used as model input."""
from perspective_tray import *
from PIL import Image,ImageDraw,ImageFont
FONT=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',17)

def make(mode='physical',policy='checked'):
    for day in DAYS:
        im=Image.fromarray(cv2.cvtColor(image(CALIBRATION[day]['frame']),cv2.COLOR_BGR2RGB));dr=ImageDraw.Draw(im)
        for j,quad in enumerate(CALIBRATION[day]['quads']):
            dr.line([tuple(p) for p in quad]+[tuple(quad[0])],fill='#00ff80',width=3)
            dr.text(tuple(np.mean(quad,axis=0)),f'T{j+1}',font=FONT,fill='yellow',stroke_width=2,stroke_fill='black')
        im.save(A/f'audit/table_calibration_{day}.jpg',quality=95)
    day=DAYS[0];model=json.loads((A/f'models/tray_size_{mode}_{day}.json').read_text());cam=CameraMetric(day,mode)
    im=Image.fromarray(cv2.cvtColor(image(CALIBRATION[day]['frame']),cv2.COLOR_BGR2RGB));dr=ImageDraw.Draw(im);rows=[]
    for x in [200,400,600,800,1000,1200]:
        for y in [210,300,390]:
            poly=cam.rectangle([x,y],model['length'],model['short'],np.array([0.]))[0]
            b=bounds(poly);wh=b[2:]-b[:2]
            color={210:'#00FFFF',300:'#FFFF00',390:'#FF77FF'}[y]
            dr.line([tuple(p) for p in poly]+[tuple(poly[0])],fill=color,width=2)
            dr.text((x-32,y+25),f'{wh[0]:.0f}x{wh[1]:.0f}',font=FONT,fill=color,stroke_width=2,stroke_fill='black')
            rows.append(dict(x=x,y=y,orientation_degrees=0,width_px=wh[0],height_px=wh[1]))
    dr.rectangle((0,0,1280,45),fill='black');dr.text((15,12),'Same tray at hypothetical positions on tabletop-height plane: position-dependent pixel size',font=FONT,fill='white')
    suffix='' if mode=='physical' else '_'+mode
    im.save(A/f'audit/tray_perspective_scale{suffix}.jpg',quality=95)
    pd.DataFrame(rows).to_csv(A/f'results/position_size_examples{suffix}.csv',index=False)
    # Completion diagnostic: show all intact changes and a balanced set of
    # improved/degraded 35% crops, keeping the selected row IDs in a ledger.
    d=pd.read_csv(A/f'results/tray_completion_benchmark_{mode}_{policy}.csv')
    d['delta']=d.after-d.before
    selected=pd.concat([d[d.fraction.eq(0)&d.changed],d[d.fraction.eq(.35)].nlargest(5,'delta'),d[d.fraction.eq(.35)].nsmallest(5,'delta')])
    selected.to_csv(A/f'audit/completion_visual_selection{suffix}.csv',index=False)
    for start in range(0,len(selected),6):
        page=Image.new('RGB',(1200,760),'white');dr=ImageDraw.Draw(page)
        for k,r in enumerate(list(selected.itertuples())[start:start+6]):
            gt=np.array([getattr(r,'gt_'+c) for c in B]);center=(gt[:2]+gt[2:])/2;size=max(140,(gt[2:]-gt[:2]).max()*1.7)
            cr=np.r_[center-size/2,center+size/2];ox=(k%3)*400;oy=(k//3)*380
            im=Image.open(ROOT/f'rascar_v16/images/{r.frame_id}.jpg').convert('RGB');page.paste(im.crop(tuple(cr)).resize((390,310)),(ox+5,oy+65))
            dr.text((ox+5,oy+4),f'GT {r.gt_index}, crop {r.fraction:.0%}',font=FONT,fill='black')
            dr.text((ox+5,oy+28),f'IoU {r.before:.3f} -> {r.after:.3f}',font=FONT,fill='black')
            for prefix,color in [('gt_','lime'),('old_','yellow'),('','cyan')]:
                box=np.array([getattr(r,prefix+c) for c in B]);xy=(box-np.tile(cr[:2],2))*np.tile([390/size,310/size],2)+[ox+5,oy+65,ox+5,oy+65]
                dr.rectangle(tuple(xy),outline=color,width=2)
        page.save(A/f'audit/completion_examples{suffix}_{start//6}.jpg',quality=95)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

if __name__=='__main__':make(*(sys.argv[1:] or []))
