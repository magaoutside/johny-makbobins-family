from common21 import *
from PIL import Image,ImageDraw,ImageFont
r=pd.read_csv(R/'reviews/release_main.csv')
prior=r.diagnostic_flag.isin(['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'])
sel=r[prior|r.delta.abs().gt(1e-9)|r.review_no.isin([139,157,160,161,288,66,234,272,128,308])].sort_values('review_no')
dest=A/'audit/final_review';dest.mkdir(exist_ok=True)
sel.to_csv(dest/'cases.csv',index=False)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
for page,start in enumerate(range(0,len(sel),12)):
    out=Image.new('RGB',(1200,1110),'white');dr=ImageDraw.Draw(out)
    for j,rr in enumerate(sel.iloc[start:start+12].itertuples()):
        x=(j%4)*300;y=(j//4)*370;b=np.array([getattr(rr,c) for c in B]);center=(b[:2]+b[2:])/2;size=max(float((b[2:]-b[:2]).max())*1.5,125)
        l=max(0,min(1280-size,center[0]-size/2));t=max(0,min(720-size,center[1]-size/2))
        im=Image.open(ROOT/'rascar_v16/test_images'/f'{rr.frame_id}.jpg').convert('RGB').resize((1280,720)).crop((l,t,l+size,t+size)).resize((296,296))
        ImageDraw.Draw(im).rectangle(tuple((b-np.array([l,t,l,t]))*296/size),outline='yellow',width=2)
        out.paste(im,(x,y+70));dr.text((x+3,y+3),f'#{rr.review_no} {rr.class_name}',fill='black',font=font)
        dr.text((x+3,y+23),f'{rr.baseline_v20:.3f} -> {rr.new_score:.3f}',fill='black',font=font)
        dr.text((x+3,y+43),str(rr.issue)[:30],fill='black',font=font)
    out.save(dest/f'page_{page+1}.jpg',quality=92)
print('RENDERED',len(sel),'test cases')
