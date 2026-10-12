"""Visual target conditioning with an unconditional spatial auxiliary head."""
from pathlib import Path
import sys,copy,json,os
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'ipdnet2_ft'))
import common as audio
from common import torch,np,features
class AVNet(torch.nn.Module):
 def __init__(self,initialize=True):
  super().__init__();self.audio=audio.model()
  if initialize:
   raise ValueError('Load the complete SAV-BandNet checkpoint instead')
  self.visual=torch.nn.Sequential(torch.nn.Conv1d(256,128,1),torch.nn.GroupNorm(8,128),torch.nn.SiLU(),torch.nn.Conv1d(128,128,5,padding=2),torch.nn.SiLU())
  self.film=torch.nn.ModuleList([torch.nn.Conv1d(128,192,1) for _ in range(2)])
  self.residual=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(96,96,3,padding=1,groups=8),torch.nn.GELU(),torch.nn.Conv2d(96,96,1)) for _ in range(2)])
  for layer in self.film:torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
  for layer in self.residual:torch.nn.init.zeros_(layer[-1].weight);torch.nn.init.zeros_(layer[-1].bias)
  self.target_inverse=copy.deepcopy(self.audio.freq_inverse);self.target_decoder=copy.deepcopy(self.audio.decoder)
  self.temporal=torch.nn.GRU(96+128+16,96,batch_first=True,bidirectional=True)
  self.temporal_out=torch.nn.Linear(192,96)
  torch.nn.init.zeros_(self.temporal_out.weight);torch.nn.init.zeros_(self.temporal_out.bias)
  self.mask_head=torch.nn.Linear(192,16)
  self.align_query=torch.nn.Sequential(torch.nn.LayerNorm(352),torch.nn.Linear(352,64))
  self.align_key=torch.nn.Sequential(torch.nn.LayerNorm(128),torch.nn.Linear(128,64))
  self.align_output=torch.nn.Linear(128,128)
  self.align_bias=torch.nn.Parameter(torch.tensor([-2.,-1.,2.,-1.,-2.]))
  torch.nn.init.zeros_(self.align_output.weight);torch.nn.init.zeros_(self.align_output.bias)
  self.audio.freq_inverse.register_forward_pre_hook(self.capture)
 def capture(self,module,args):self.hidden=args[0]
 def forward(self,wave,visual,valid):
  auxiliary=audio.forward(self.audio,wave)
  h=self.hidden;v=self.visual(visual.transpose(1,2));v=v[:,:,:h.shape[2]];valid=valid[:,:h.shape[2]][:,None]
  b,c,t,f=h.shape
  z=features(wave);mag=(z[:,:6].square()+z[:,6:].square()+1e-8).sqrt()[:,0,:,:t*5]
  mag=torch.log1p(10*mag).reshape(b,256,t,5).mean(-1)
  query=self.align_query(torch.cat([h.mean(-1),mag],1).transpose(1,2))
  vt=v.transpose(1,2);keys=self.align_key(vt)
  keys=torch.nn.functional.pad(keys,(0,0,2,2)).unfold(1,5,1).permute(0,1,3,2)
  values=torch.nn.functional.pad(vt,(0,0,2,2)).unfold(1,5,1).permute(0,1,3,2)
  available=torch.nn.functional.pad(valid[:,0],(2,2)).unfold(1,5,1)
  logits=(query[:,:,None]*keys).sum(-1)/8+self.align_bias
  weights=logits.masked_fill(~available.bool(),-1e4).softmax(-1)
  attended=(weights[:,:,:,None]*values).sum(2)
  v=v+self.align_output(attended-vt).transpose(1,2)*valid
  self.alignment_weights=weights
  for film,residual in zip(self.film,self.residual):
   gain,bias=(film(v)*valid).chunk(2,dim=1)
   h=h*(1+.25*gain.tanh().unsqueeze(-1))+.25*bias.unsqueeze(-1)
   h=h+.1*residual(h)
  b,c,t,f=h.shape
  mag=mag.reshape(b,16,16,t).permute(0,1,3,2)
  sequence=torch.cat([h.permute(0,3,2,1),v.transpose(1,2)[:,None].expand(-1,f,-1,-1),mag],-1).reshape(b*f,t,-1)
  sequence,_=self.temporal(sequence)
  self.mask_logits=self.mask_head(sequence).reshape(b,f,t,16).permute(0,2,1,3).reshape(b,t,256)
  residual=self.temporal_out(sequence).reshape(b,f,t,96).permute(0,3,2,1)
  h=h+residual
  x=self.target_inverse(h).permute(0,3,2,1);x=self.target_decoder(x)
  b,f,t,_=x.shape
  x=x.permute(0,2,1,3).reshape(b,t,f,2,-1).permute(0,1,3,2,4)
  x=x.reshape(b,t,2,f*2,-1).permute(0,1,3,4,2)
  return x[...,0],auxiliary
