"""Recompute the selected frozen WeVis geometry features."""
from common16 import *
from rascar.geometry import wevis_features
heads=pd.read_csv(V16/'wevis_heads_references.csv')
hands=pd.read_csv(V16/'wevis_hands_references.csv')
pose=pd.read_csv(V16/'multiview_pose.csv')
for rows,split in [(q,'oof'),(te,'test')]:
    wevis_features(rows,heads,hands,pose).to_csv(V16/f'wevis_features_{split}.csv',index=False)
