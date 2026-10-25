import torch,sys,importlib.util
from torch import nn
from .paths import VENDOR,BUNDLE
class FrozenDAVE(nn.Module):
 def __init__(self):
  super().__init__();spec=importlib.util.spec_from_file_location("sav_dave_upstream",VENDOR/"dave/inference.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);self.net=m.TIGER(**m.TIGER_M_CFG);state=torch.load(BUNDLE/"dave.pt",map_location="cpu",weights_only=True)["state_dict"];from .dave_noise import NoiseGateDAVE;self.net=NoiseGateDAVE(self.net,source_gates=state["noise_heads.0.weight"].shape[0]==2*self.net.band_width[0]) if any(k.startswith("noise_heads.") for k in state) else self.net;self.net.load_state_dict(state,strict=True);self.net.eval().requires_grad_(False);self._latent=None
