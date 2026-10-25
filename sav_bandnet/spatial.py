import torch,importlib,sys
from torch import nn
from .paths import VENDOR,BUNDLE
from .features import FrozenTalkNet
from .alignment import align_visual
from .memory import PreTemporalMemory
PARENT=None
class FrozenTalkImage(nn.Module):
 def __init__(self):
  super().__init__();old=FrozenTalkNet();self.image=old.net.visualFrontend;self.requires_grad_(False);self.eval()
 def train(self,mode=True):return super().train(False)
 @torch.no_grad()
 def forward(self,pixels):
  assert pixels.shape[1:]==(25,112,112)
  x=(pixels.float()/255-.4161)/.1688;B,T=x.shape[:2]
  x=self.image.frontend3D(x[:,None]);x=x.transpose(1,2).contiguous();x=x.reshape(B*T,*x.shape[2:]);return self.image.resnet(x).reshape(B,T,512)

class SpatialAdapter(nn.Module):
 def __init__(self):
  super().__init__();self.norm=nn.LayerNorm(512);self.proj=nn.Linear(512,256);self.depth=nn.Conv1d(256,256,3,padding=1,groups=256);self.out=nn.Conv1d(256,256,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,features,valid):
  assert features.shape[1:]==(25,512) and valid.shape[1]==25
  t=torch.arange(10,device=features.device)*.1+.04;v,ok=align_visual(features,valid,torch.arange(25,device=features.device)*.04+.02,t)
  x=self.proj(self.norm(v));x=x.masked_fill(~ok[...,None],0);x=x+.1*self.out(torch.nn.functional.gelu(self.depth(x.transpose(1,2)))).transpose(1,2)
  return torch.nn.functional.layer_norm(x,(256,)).masked_fill(~ok[...,None],0),ok

class TalkSpatial(nn.Module):
 def __init__(self,parent=PARENT):
  super().__init__();sys.path[:0]=[str(VENDOR/'ipdnet2_av_exp3'),str(VENDOR/'ipdnet2_ft')];factory=importlib.import_module('masked_model');state=torch.load(BUNDLE/'spatial.pt',map_location='cpu',weights_only=True)['model'];self.net=factory.masked_from_state({k.removeprefix('net.'):v for k,v in state.items() if k.startswith('net.')});self.net.requires_grad_(True);self.adapter=SpatialAdapter()
 def train(self,mode=True):
  super().train(mode)
  for m in self.net.modules():
   if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
  return self
 def forward(self,wave,features,valid):
  visual,ok=self.adapter(features,valid);pred,aux=self.net(wave,visual,ok)
  return pred,aux,ok,self.net.mask_logits
def condition(net,h,mag,visual,valid):
 """Exact AVNet conditioning path, with pretrained parameters and externally supplied audio features."""
 net._visual_valid=valid;v=net.visual(visual.transpose(1,2));v=v[:,:,:h.shape[2]];valid=valid[:,:h.shape[2]][:,None];b,c,t,f=h.shape
 query=net.align_query(torch.cat([h.mean(-1),mag],1).transpose(1,2));vt=v.transpose(1,2);keys=net.align_key(vt);keys=torch.nn.functional.pad(keys,(0,0,2,2)).unfold(1,5,1).permute(0,1,3,2);values=torch.nn.functional.pad(vt,(0,0,2,2)).unfold(1,5,1).permute(0,1,3,2);available=torch.nn.functional.pad(valid[:,0],(2,2)).unfold(1,5,1);logits=(query[:,:,None]*keys).sum(-1)/8+net.align_bias;weights=logits.masked_fill(~available.bool(),-1e4).softmax(-1);attended=(weights[:,:,:,None]*values).sum(2);v=v+net.align_output(attended-vt).transpose(1,2)*valid
 for film,residual in zip(net.film,net.residual):
  gain,bias=(film(v)*valid).chunk(2,dim=1);h=h*(1+.25*gain.tanh().unsqueeze(-1))+.25*bias.unsqueeze(-1);h=h+.1*residual(h)
 mag=mag.reshape(b,16,16,t).permute(0,1,3,2);sequence=torch.cat([h.permute(0,3,2,1),v.transpose(1,2)[:,None].expand(-1,f,-1,-1),mag],-1).reshape(b*f,t,-1);sequence,_=net.temporal(sequence);mask=net.mask_head(sequence).reshape(b,f,t,16).permute(0,2,1,3).reshape(b,t,256);h=h+net.temporal_out(sequence).reshape(b,f,t,96).permute(0,3,2,1);x=net.target_inverse(h).permute(0,3,2,1);x=net.target_decoder(x);b,f,t,_=x.shape;x=x.permute(0,2,1,3).reshape(b,t,f,2,-1).permute(0,1,3,2,4);phase=x.reshape(b,t,2,f*2,-1).permute(0,1,3,4,2)[...,0]
 net.mask_logits=mask;net.alignment_weights=weights
 return dict(phase=phase,hidden=h,visual=v*valid,valid=valid[:,0],mask_logits=mask,alignment_weights=weights)

class SpatialMemory:
 def __init__(self,checkpoint,device='cuda',encoder=None):
  assert encoder is not None;self.encoder=encoder.to(device).eval();self.net=self.encoder.net;self.module=importlib.import_module('model');self.memory=PreTemporalMemory();zero=self.audio_features(torch.zeros(1,6,16000,device=device));self.h_pad=zero['h'][:,:,4:5].clone();self.mag_pad=zero['mag'][:,:,4:5].clone()
 def reset(self):self.memory=PreTemporalMemory()
 @torch.inference_mode()
 def audio_features(self,wave):
  self.module.audio.forward(self.net.audio,wave);h=self.net.hidden;z=self.module.features(wave);b,_,t,_=h.shape;mag=(z[:,:6].square()+z[:,6:].square()+1e-8).sqrt()[:,0,:,:t*5];mag=torch.log1p(10*mag).reshape(b,256,t,5).mean(-1);return dict(h=h,mag=mag)
 @torch.inference_mode()
 def step(self,h,mag,visual,valid):
  assert h.shape[2]==mag.shape[2]==visual.shape[1]==valid.shape[1]==10
  specs=dict(h=(h,2),mag=(mag,2),visual=(visual,1),valid=(valid,1));hist={k:self.memory.push(k,v,(4,8),40,axis,self.h_pad if k=='h' else self.mag_pad if k=='mag' else 0) for k,(v,axis) in specs.items()};full=condition(self.net,**hist)
  out={k:(v[:,:,30:] if k in ['hidden','visual'] else v[:,30:]) for k,v in full.items()};out['times']=torch.arange(10,device=h.device)*.1+.04
  self.encoder._target=self.encoder._visual=None
  return out
