import torch,numpy as np,cv2,sys,importlib,types
from torch import nn
from pathlib import Path
from .paths import VENDOR,BUNDLE
UPSTREAM=VENDOR/"talknet"
DEPS=UPSTREAM/"deps"
TALK=BUNDLE/"talknet.pt"
class FrozenModule(nn.Module):
    def train(self, mode=True):
        # Parent model.train() must not reactivate dropout/batch statistics.
        return super().train(False)
def talknet_class():
 """Scope legacy imports during construction; restore spatial encoder namespace."""
 names=lambda k:k=='model' or k.startswith('model.')
 saved={k:v for k,v in sys.modules.items() if names(k)};paths=sys.path.copy()
 try:
  for k in saved:del sys.modules[k]
  sys.path[:0]=[str(UPSTREAM),str(DEPS)]
  # TalkNet's model directory is a namespace package. Explicitly bind it so a
  # later legacy model.py on sys.path cannot take precedence during resolution.
  package=types.ModuleType('model');package.__path__=[str(UPSTREAM/'model')]
  package.__package__='model';sys.modules['model']=package
  return importlib.import_module('model.talkNetModel').talkNetModel
 finally:
  for k in list(sys.modules):
   if names(k):del sys.modules[k]
  sys.modules.update(saved);sys.path[:]=paths

class FrozenTalkNet(FrozenModule):
 def __init__(self):
  super().__init__();self.net=talknet_class()();self.fc=nn.Linear(256,2)
  ck=torch.load(TALK,map_location='cpu',weights_only=True);ck={k.removeprefix('module.'):v for k,v in ck.items()}
  self.net.load_state_dict({k.removeprefix('model.'):v for k,v in ck.items() if k.startswith('model.')},strict=True)
  self.fc.load_state_dict({k.removeprefix('lossAV.FC.'):v for k,v in ck.items() if k.startswith('lossAV.FC.')},strict=True)
  sys.path.insert(0,str(DEPS));self.mfcc=importlib.import_module('python_speech_features').mfcc
  self.requires_grad_(False).eval()
 def visual_frontend(self,pixels,return_frames=False):
  # Upstream flattens B*T before its temporal Conv3d, joining separate clips.
  # Retain the batch axis through Conv3d; weights and single-clip behavior match.
  B,T,H,W=pixels.shape
  x=(pixels/255-.4161)/.1688
  x=self.net.visualFrontend.frontend3D(x[:,None])
  x=x.transpose(1,2).contiguous();x=x.reshape(B*T,*x.shape[2:])
  x=self.net.visualFrontend.resnet(x).reshape(B,T,512).transpose(1,2)
  out=self.net.visualConv1D(self.net.visualTCN(x)).transpose(1,2)
  return (out,x.transpose(1,2)) if return_frames else out
 @torch.no_grad()
 def forward(self,waves,mouths,return_features=False,return_frontends=False,return_voice=False):
  B=len(waves);dev=waves.device;N=waves.shape[-1];V=mouths.shape[1];T=N//1600
  mf=np.stack([self.mfcc(np.pad(x,(0,160))*32768,16000,numcep=13,winlen=.025,winstep=.01)[:N//160] for x in waves.cpu().numpy().reshape(-1,N)]).astype(np.float32)
  mf_tensor=torch.from_numpy(mf).to(dev)
  ae=self.net.forward_audio_frontend(mf_tensor)
  voice=None
  if return_voice:
   assert N==16000 and not return_frontends
   aa_voice=ae[:,10:20];cc_voice=mf_tensor[:,40:80,1:]
   voice=torch.cat([aa_voice.mean(1),aa_voice.std(1,unbiased=False),cc_voice.mean(1),cc_voice.std(1,unbiased=False)],-1).reshape(B,2,280)
  pixels=np.stack([cv2.resize(f,(112,112)) for f in mouths.cpu().numpy().reshape(-1,96,96)]).reshape(B,V,112,112).astype(np.float32)
  ve0=self.visual_frontend(torch.from_numpy(pixels).to(dev));ve=ve0[:,None].expand(-1,2,-1,-1).reshape(B*2,V,-1)
  aa,vv=self.net.forward_cross_attention(ae,ve)
  features=self.net.forward_audio_visual_backend(aa,vv).reshape(B,2,V,256)
  lg=self.fc(features);scores=lg[...,1]-lg[...,0]
  if return_features:
   t=torch.arange(T,device=dev)*2.5+.5;j=t.floor().long();r=t-j
   aligned=features[:,:,j]*(1-r)[None,None,:,None]+features[:,:,j+1]*r[None,None,:,None]
   if getattr(self,'full_rate',False):aligned=features
   if return_voice:return scores,aligned,voice
   return (scores,aligned,ae.reshape(B,2,V,128),ve0) if return_frontends else (scores,aligned)
  return scores

def phase_compatibility(wave,candidates,phase):
 B=len(wave);T=wave.shape[-1]//1600;window=torch.hann_window(512,device=wave.device)
 z=torch.stft(wave.flatten(0,1),512,320,window=window,return_complex=True).reshape(B,6,257,-1)[:,:,1:,:T*5]
 cross=z[:,0,None]*z[:,1:].conj();unit=cross/(cross.abs()+1e-8)
 unit=unit.reshape(B,5,256,T,5).mean(-1).permute(0,3,2,1)
 pred=torch.complex(phase[:,:,:256],phase[:,:,256:]);agreement=(pred*unit.conj()).real.mean(-1)
 sy=torch.stft(candidates.flatten(0,1),512,320,window=window,return_complex=True)[:,1:,:T*5].abs().square().reshape(B,2,256,T,5).mean(-1).permute(0,1,3,2)
 wt=sy/(sy.sum(1,keepdim=True)+1e-8)
 return (agreement[:,None]*wt).reshape(B,2,T,16,16).sum(-1)/wt.reshape(B,2,T,16,16).sum(-1).clamp_min(1e-8)

def candidate_energy(waves):
 p=waves.square().reshape(len(waves),2,-1,1600).mean(-1)
 scale=p.sum(1).mean(-1).clamp_min(1e-8)[:,None,None]
 return torch.stack([torch.log10(p/scale+1e-8).clamp(-8,3),p/(p.sum(1,keepdim=True)+1e-8)],-1)
