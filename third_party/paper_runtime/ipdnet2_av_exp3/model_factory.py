from model import *
from crossband_model import CrossBandNet
from masked_model import masked_from_state
def load_model(checkpoint,device='cuda'):
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False);state=ck['model']
 if 'masked_norm_version' in state:net=masked_from_state(state)
 else:
  net=(CrossBandNet if any(k.startswith('crossband') for k in state) else AVNet)(False)
  missing=net.load_state_dict(state,strict=False)
  assert not missing.unexpected_keys and all(k.startswith('align_') for k in missing.missing_keys)
 return net.to(device),ck
