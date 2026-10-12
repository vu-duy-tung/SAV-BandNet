from model import *
from crossband_model import CrossBandNet
import weakref
class ValidGroupNorm(torch.nn.GroupNorm):
 def __init__(self,previous,owner):
  super().__init__(previous.num_groups,previous.num_channels,eps=previous.eps,affine=True)
  self._owner=weakref.ref(owner)
 def forward(self,x):
  b,c,t=x.shape;mask=self._owner()._visual_valid[:,:t].to(x.dtype)[:,None,None]
  grouped=x.reshape(b,self.num_groups,c//self.num_groups,t)
  count=(mask.sum(-1,keepdim=True)*(c//self.num_groups)).clamp_min(1)
  mean=(grouped*mask).sum((2,3),keepdim=True)/count
  var=((grouped-mean).square()*mask).sum((2,3),keepdim=True)/count
  normalized=((grouped-mean)*torch.rsqrt(var+self.eps)).reshape(b,c,t)
  return (normalized*self.weight[None,:,None]+self.bias[None,:,None])*mask[:,0]
class MaskedMixin:
 def setup(self):
  self.visual[1]=ValidGroupNorm(self.visual[1],self)
  self.register_buffer('masked_norm_version',torch.tensor(1))
 def forward(self,wave,visual,valid):
  self._visual_valid=valid
  return super().forward(wave,visual,valid)
class MaskedAVNet(MaskedMixin,AVNet):
 def __init__(self,initialize=False):super().__init__(initialize);self.setup()
class MaskedCrossBandNet(MaskedMixin,CrossBandNet):
 def __init__(self,initialize=False):super().__init__(initialize);self.setup()
def masked_from_state(state):
 net=(MaskedCrossBandNet if any(k.startswith('crossband') for k in state) else MaskedAVNet)(False)
 missing=net.load_state_dict(state,strict=False)
 assert not missing.unexpected_keys and all(k.startswith(('align_','masked_norm')) for k in missing.missing_keys)
 return net
