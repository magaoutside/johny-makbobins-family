"""Keep NMS on CPU on Apple GPUs; the default GPU loop can hit a time limit.
This changes execution device, not confidence, IoU, or max-detection thresholds.
"""
import torch
from ultralytics.utils import nms
_original=nms.non_max_suppression
def complete_nms(prediction,*args,**kwargs):
 p=prediction[0] if isinstance(prediction,(list,tuple)) else prediction
 device=p.device
 if device.type!='mps':return _original(prediction,*args,**kwargs)
 kwargs['max_time_img']=30.
 result=_original(p.detach().cpu(),*args,**kwargs)
 def restore(value):
  if isinstance(value,torch.Tensor):return value.to(device)
  if isinstance(value,tuple):return tuple(restore(x) for x in value)
  return [restore(x) for x in value]
 return restore(result)
nms.non_max_suppression=complete_nms
