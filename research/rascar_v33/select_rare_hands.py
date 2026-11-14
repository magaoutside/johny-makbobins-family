"""Select rare spatial/standing hand contexts from TRAIN DAYS only for review."""
from prepare import *
from PIL import Image,ImageDraw

def main():
    p=pd.read_csv(ROOT/'rascar_data_full/machine_predictions.csv');p['day']=pd.to_datetime(p.frame_id.str[-15:-7],format='%Y%m%d').dt.strftime('%Y-%m-%d')
    old=pd.read_csv(A/'inputs/pseudo_records.csv');exclude=set(old.frame_id)|set(h.frame_id)
    hand=p[p.class_name.eq('nude_hands')&p.confidence.ge(.55)&~p.frame_id.isin(exclude)].copy()
    hand['cx']=(hand.x1+hand.x2)/2;hand['cy']=(hand.y1+hand.y2)/2;hand['min_side']=np.minimum(hand.x2-hand.x1,hand.y2-hand.y1)
    hand=hand[hand.min_side.ge(14)&hand.min_side.le(100)]
    people={f:z[z.class_name.eq('person')&z.confidence.ge(.7)] for f,z in p.groupby('frame_id')}
    def standing(r):
        z=people[r.frame_id];inside=(z.x1<r.cx)&(z.x2>r.cx)&(z.y1<r.cy)&(z.y2>r.cy)
        return bool((((z.y2-z.y1)/(z.x2-z.x1))>1.8)[inside].any())
    hand['standing_context']=[standing(r) for r in hand.itertuples()]
    hand=hand[hand.standing_context|hand.cy.ge(400)|hand.cx.ge(950)]
    stamp=pd.to_datetime(hand.frame_id.str[-15:],format='%Y%m%d_%H%M%S').astype('int64')//(600*10**9)
    hand['bucket']=hand.day+'_'+stamp.astype(str)+'_'+(hand.cx//80).astype(int).astype(str)+'_'+(hand.cy//80).astype(int).astype(str)
    # Select varied examples, not the same motion repeated every 30 seconds.
    hand['priority']=hand.standing_context.astype(float)+hand.cy.ge(400).astype(float)+.3*hand.confidence
    hand=hand.sort_values('priority',ascending=False).drop_duplicates('bucket').drop_duplicates('frame_id')
    selected=pd.concat([hand[hand.day.eq(day)].head(30) for day in DAYS],ignore_index=True);selected['review_index']=np.arange(len(selected))
    selected.to_csv(A/'inputs/rare_hand_review_candidates.csv',index=False)
    for page,start in enumerate(range(0,len(selected),20)):
        canvas=Image.new('RGB',(1200,1000),'#151515');draw=ImageDraw.Draw(canvas)
        for j,(_,r) in enumerate(selected.iloc[start:start+20].iterrows()):
            im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);center=(b[:2]+b[2:])/2
            lo=np.maximum(0,np.floor(center-wh*1.5)).astype(int);hi=np.minimum([1280,720],np.ceil(center+wh*1.5)).astype(int);roi=Image.fromarray(im[lo[1]:hi[1],lo[0]:hi[0],::-1]);scale=min(295/roi.width,195/roi.height);roi=roi.resize((int(roi.width*scale),int(roi.height*scale)));x=(j%4)*300;y=(j//4)*200;canvas.paste(roi,(x,y+30));coords=(b-np.r_[lo,lo])*scale+np.array([x,y+30,x,y+30]);draw.rectangle(tuple(coords),outline='#ff4444',width=2);draw.text((x+2,y+1),f'{r.review_index} {r.frame_id[-15:]} conf {r.confidence:.2f}',fill='white')
        canvas.save(A/f'audit/rare_hand_candidates_{page+1}.jpg')
    print(selected.groupby('day').size());print(selected[['review_index','frame_id','cx','cy','standing_context']].to_string(index=False))
if __name__=='__main__':main()
