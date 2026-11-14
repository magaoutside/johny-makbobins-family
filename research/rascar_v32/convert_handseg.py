"""Extract tensors with restricted torch loading; never enable arbitrary pickle.
The old Lightning checkpoint names its callback class in metadata. Replace
that one name with an inert local class; no third-party callback is imported.
"""
from pathlib import Path
import torch,json,hashlib
from safetensors.torch import save_file
A=Path(__file__).resolve().parent;D=A/'model_cache/handseg';p=D/'checkpoint.ckpt'
class InertCheckpointMetadata:pass
allowed='pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'
assert torch.serialization.get_unsafe_globals_in_checkpoint(p)==[allowed]
with torch.serialization.safe_globals([(InertCheckpointMetadata,allowed)]):x=torch.load(p,map_location='cpu',weights_only=True)
state={k.removeprefix('deeplab.'):v.contiguous() for k,v in x['state_dict'].items() if k.startswith('deeplab.')}
assert state and all(isinstance(v,torch.Tensor) for v in state.values())
save_file(state,str(D/'weights.safetensors'))
from torchvision.models.segmentation import deeplabv3_resnet50
m=deeplabv3_resnet50(weights=None,weights_backbone=None,num_classes=2,aux_loss=any(k.startswith('aux_classifier.') for k in state))
m.load_state_dict(state,strict=True)
meta=dict(tensors=len(state),keys=list(x),hyper_parameters=str(x.get('hyper_parameters')),safe_loading=True,sha256=hashlib.sha256((D/'weights.safetensors').read_bytes()).hexdigest(),architecture='torchvision deeplabv3_resnet50, num_classes=2',normalization='ImageNet RGB mean/std, as official main.py')
(D/'conversion.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
