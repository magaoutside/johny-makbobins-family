"""Assistant visual annotation of MACHINE POOL ONLY; unofficial, uncertain edges.
36 candidates inspected at enlarged native pixels. Test annotations are forbidden.
Coordinates are canonical 1280x720 (source JPEG is 640x360).
"""
from pathlib import Path
import pandas as pd,numpy as np,json,sys
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v43'))
from common43 import D,B,inputs,iou
# All visible hands in each accepted local window, including clipped neighbours.
BOXES={
0:[[913,212,939,231]],
2:[[1246,276,1280,298]],
3:[],
4:[[550,219,568,240],[570,222,598,251]],
5:[[971,228,1004,260],[1010,232,1025,246]],
8:[[1126,406,1154,434]],
9:[[719,181,733,205],[747,208,773,231]],
10:[[529,247,544,269],[552,222,572,245]],
11:[[513,218,535,235]],
12:[[800,213,828,233],[857,218,878,238]],
13:[[842,263,884,311]],
14:[[1156,215,1184,256]],
15:[[966,181,985,206],[989,240,1014,258]],
16:[[156,345,181,363]],
17:[],
18:[[1189,215,1214,239],[1218,226,1236,256]],
21:[[558,232,579,253]],
22:[[1236,191,1259,215],[1219,232,1238,246]],
24:[[561,316,590,350]],
25:[[550,349,575,370]],
27:[[1228,221,1247,244],[1201,250,1218,284]],
28:[[1000,203,1015,221],[941,208,961,227]],
31:[[557,194,576,212],[540,225,560,243]],
32:[[1207,233,1227,257],[1240,249,1257,263]],
33:[[147,278,172,298]],
}
REJECT={1:'Edge-clipped blurred wrist/hand boundary',6:'Blurred hand versus pink clothing',7:'Interlocked hands: cannot separate reliably',19:'Tiny blurred fingers',20:'Overlapping blurred hands',23:'Tiny blurred fingers',26:'Rejected during overlay QA: overlapping hand boundaries not reliable',29:'Hand versus sleeve/face boundary',30:'Hand and cup heavily blurred',34:'Hand boundary and second hand ambiguous',35:'Rejected during overlay QA: page texture versus fingers ambiguous'}
# Reserved BEFORE annotation. Rejected images remain reserved and are not replaced.
HOLDOUT={0,4,8,12,18,22,26,30}
def main():
    r=pd.read_csv(A/'pool_review/candidates.csv');q,te,h=inputs();assert not set(r.frame_id)&set(te.frame_id)
    reviews=[];labels=[]
    for _,v in r.iterrows():
        k=int(v.review_index);accepted=k in BOXES
        role='reserved_holdout' if k in HOLDOUT else 'train'
        reviews.append(dict(**v.to_dict(),status='accepted_silver' if accepted else 'uncertain_excluded',role=role,notes=REJECT.get(k,'Pink backpack inside queried ROI: negative supervision only inside query plus 3px; remaining crop ignored' if k in [3,17] else 'Visible hands bounded; approximate edges, NOT organizer GT'),edge_uncertainty_px=4 if k in [0,8,18,22,26,33] else 3))
        for j,b in enumerate(BOXES.get(k,[])):
            labels.append(dict(review_index=k,instance=j,frame_id=v.frame_id,day=pd.to_datetime(str(v.day)).strftime('%Y-%m-%d'),class_name='nude_hands',role=role,source='assistant_manual_silver',uncertainty_px=4 if k in [0,8,18,22,26,33] else 3,**dict(zip(B,b))))
    pd.DataFrame(reviews).to_csv(A/'pool_review/reviews.csv',index=False);pd.DataFrame(labels).to_csv(A/'pool_review/boxes.csv',index=False)
    manifest=dict(candidates=len(r),accepted_frames=len(BOXES),rejected_frames=len(REJECT),annotated_hands=len(labels),negative_frames=[3,17],heldout_indices=sorted(HOLDOUT),status='UNOFFICIAL_MANUAL_SILVER',uncertainty='3 or 4 canonical pixels per edge; ambiguous boundary bands excluded from training loss',test_frames_used=False,selection='Random shuffled machine hand candidates; area650..6000, w>=22,h>=18; >=10min apart within day; old reviewed frames and train/test/context frames excluded',holdout_limit='Only 6 accepted reserved frames; not representative of full competition, exact IoU labels may remain uncertain',overlay_review='All 36 overlays inspected; 26 and 35 additionally excluded before fitting; 8,9,14,22,27,32 boundaries corrected')
    (A/'pool_review/manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
