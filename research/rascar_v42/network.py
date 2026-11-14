from base42 import *
import torch,copy
from torch import nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny
torch.set_num_threads(3)
def backbone():
 net=convnext_tiny(weights=None);net.load_state_dict(torch.load(A/'model_cache/convnext_tiny-983f1562.pth',map_location='cpu',weights_only=True));return net
def query_mask(metadata,grid=7):
 r=metadata[:,9:13];lo=torch.arange(grid,device=r.device)/grid;hi=(torch.arange(grid,device=r.device)+1)/grid
 wx=(torch.minimum(r[:,2,None],hi)-torch.maximum(r[:,0,None],lo)).clamp_min(0)*grid
 wy=(torch.minimum(r[:,3,None],hi)-torch.maximum(r[:,1,None],lo)).clamp_min(0)*grid
 return wy[:,:,None]*wx[:,None,:]
def pool(z,mask):return (z*mask[:,None]).sum((2,3))/mask.sum((1,2)).clamp_min(.01)[:,None]
def vectors(z,metadata,mode,norm):
 tight=F.layer_norm(z[:,0].mean((2,3)),(768,),norm.weight,norm.bias,norm.eps)
 if mode=='tight':return torch.cat([tight,metadata[:,:9]],1)
 context=z[:,1];m=query_mask(metadata);inside=pool(context,m);outside=pool(context,1-m)
 # Four thin bands along query edges keep cut-off fingers/legs visible.
 g=7;xy=(torch.arange(g,device=z.device)+.5)/g;r=metadata[:,9:13];edge=[]
 for axis,bound in [(0,0),(1,1),(0,2),(1,3)]:
  distance=(xy[None]-r[:,bound,None]).abs();band=(distance<=1/g).float()
  mm=band[:,None,:].expand(-1,g,-1) if axis==0 else band[:,:,None].expand(-1,-1,g)
  edge.append(pool(context,mm))
 vec=[tight]+[F.layer_norm(x,(768,),norm.weight,norm.bias,norm.eps) for x in [inside,outside]+edge]
 return torch.cat(vec+[metadata],1)
class Classifier(nn.Module):
 def __init__(self,base,mode):
  super().__init__();self.mode=mode;self.last=copy.deepcopy(base.features[6:]);self.norm=copy.deepcopy(base.classifier[0]);dim=768+9 if mode=='tight' else 768*7+18
  self.head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,192),nn.GELU(),nn.Dropout(.3),nn.Linear(192,1))
 def forward(self,x,metadata):
  n=len(x);nv=1 if self.mode=='tight' else 2;z=self.last(x[:,:nv].reshape(n*nv,384,14,14)).reshape(n,nv,768,7,7)
  return self.head(vectors(z,metadata,self.mode,self.norm)).squeeze(1)

