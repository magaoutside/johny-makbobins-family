from common import *
import torch,copy,cv2
from torch import nn
from transformers import AutoModel,CLIPVisionModel
torch.set_num_threads(3);cv2.setNumThreads(1)
def load(key):
 p=A/'model_cache'/key
 model=(CLIPVisionModel if key=='clip' else AutoModel).from_pretrained(str(p),local_files_only=True,attn_implementation='eager')
 return model.eval()
def parts(model,key):
 return (model.vision_model.encoder.layers[-1],model.vision_model.post_layernorm) if key=='clip' else (model.encoder.layer[-1],model.layernorm)
def aux(rec,grid):
 b=rec[B].to_numpy(float);cb=rec[CB].to_numpy(float);size=cb[:,2:]-cb[:,:2]
 rel=(b-np.tile(cb[:,:2],2))/np.tile(size,2);lo=np.arange(grid)/grid;hi=(np.arange(grid)+1)/grid
 wx=np.maximum(0,np.minimum(rel[:,2,None],hi)-np.maximum(rel[:,0,None],lo))*grid
 wy=np.maximum(0,np.minimum(rel[:,3,None],hi)-np.maximum(rel[:,1,None],lo))*grid
 mask=(wy[:,:,None]*wx[:,None,:]).reshape(len(rec),-1)
 wh=b[:,2:]-b[:,:2];cc=(b[:,:2]+b[:,2:])/2
 geo=np.c_[rel,wh/[1280,720],cc/[1280,720],np.log(np.maximum(wh[:,0],1)/np.maximum(wh[:,1],1))]
 cls=np.eye(4)[rec.class_name.map(CL.index)]
 return torch.tensor(mask,dtype=torch.float32),torch.tensor(np.c_[geo,cls],dtype=torch.float32)
def pooled(z,mask,extra):
 v=z[:,1:];m=mask.unsqueeze(-1);inside=(v*m).sum(1)/m.sum(1).clamp_min(.01);outside=(v*(1-m)).sum(1)/(1-m).sum(1).clamp_min(.01)
 return torch.cat([z[:,0],inside,inside-outside,extra],1)
class QualityModel(nn.Module):
 def __init__(self,model,key):
  super().__init__();self.key=key;block,norm=parts(model,key);self.block=copy.deepcopy(block);self.norm=copy.deepcopy(norm)
  d=(model.config.vision_config.hidden_size if hasattr(model.config,'vision_config') else model.config.hidden_size)
  self.head=nn.Sequential(nn.LayerNorm(d*3+13),nn.Linear(d*3+13,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,3))
 def forward(self,x,mask,extra):
  z=self.block(x,attention_mask=None,causal_attention_mask=None)[0] if self.key=='clip' else self.block(x)[0]
  return self.head(pooled(self.norm(z),mask,extra))
