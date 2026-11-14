from pathlib import Path
import sys, os, json, hashlib
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rascar_v41'))
from common import inputs,baseline,iou,metrics,B,CL,DAYS,D
A=Path(__file__).resolve().parent

def detector(device='cpu'):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    m=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None)
    m.load_state_dict(torch.load(A/'model_cache/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth',map_location='cpu',weights_only=True))
    return m.eval().to(device)

def decode(box,delta):
    import torch
    wh=(box[:,2:]-box[:,:2]).clamp(min=1);cc=(box[:,:2]+box[:,2:])/2
    center=cc+delta[:,:2]*wh;size=wh*delta[:,2:].clamp(-2,2).exp()
    return torch.cat([center-size/2,center+size/2],1)

def geometry(box,gt):
    wh=np.maximum(box[:,2:]-box[:,:2],1);gw=np.maximum(gt[:,2:]-gt[:,:2],1)
    return np.c_[((gt[:,:2]+gt[:,2:])-(box[:,:2]+box[:,2:]))/(2*wh),np.log(gw/wh)]
