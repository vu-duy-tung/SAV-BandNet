import torch,numpy as np,cv2,json
from pathlib import Path
from torch import nn
from .paths import BUNDLE
from .features import FrozenTalkNet,phase_compatibility,candidate_energy
from .spatial import TalkSpatial,SpatialMemory,PARENT
from .selector import FrontendRanker,set_emission
from .separator import FrozenDAVE
from .memory import VoiceProjection,VoiceMemory,TemporalSelection
from .utils import state_sha
LOCALIZER=BUNDLE/"spatial.pt"
DAVE=BUNDLE/"dave.pt"
class TalkSpatialBranch(nn.Module):
 def __init__(self,checkpoint=LOCALIZER,device='cuda'):
  super().__init__();self.localizer=TalkSpatial();ck=torch.load(checkpoint,map_location='cpu',weights_only=False);self.localizer.load_state_dict(ck['model'],strict=True);self.localizer.to(device).eval().requires_grad_(False);self.expected_image_sha=ck['image_sha'];self.memory=SpatialMemory(PARENT,device,encoder=self.localizer);self.targets=1
 def reset(self):self.memory.reset()
 def set_targets(self,n):
  assert n in [1,2]
  if self.memory.memory.state:assert n==self.targets
  self.targets=n
 @torch.inference_mode()
 def forward(self,wave,frames,valid):
  visual,ok=self.localizer.adapter(frames,valid);n=self.targets;assert len(wave)%n==0;group=wave.reshape(-1,n,6,16000)
  if n==2:assert torch.equal(group[:,0],group[:,1])
  acoustic=self.memory.audio_features(group[:,0]);pieces=[]
  for i in range(len(group)):
   sl=slice(i*n,(i+1)*n);pieces.append(self.memory.step(h=acoustic['h'][i:i+1].expand(n,-1,-1,-1),mag=acoustic['mag'][i:i+1].expand(n,-1,-1),visual=visual[sl],valid=ok[sl]))
  out={k:torch.cat([p[k] for p in pieces]) for k in ['phase','hidden','visual','valid']};out['adapter_visual']=visual;return out

class SharedImageModel(nn.Module):
 def __init__(self,spatial_checkpoint=LOCALIZER,dave_checkpoint=DAVE,device='cuda'):
  super().__init__();self.talknet=FrozenTalkNet();self.spatial=TalkSpatialBranch(spatial_checkpoint,device);self.separator=FrozenDAVE();self.separator.net.load_state_dict(torch.load(dave_checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True);self.voice_features=False
  assert state_sha({'image.'+k:v for k,v in self.talknet.net.visualFrontend.state_dict().items()})==self.spatial.expected_image_sha
  self.to(device).eval().requires_grad_(False)
 @torch.inference_mode()
 def separate(self,mono):
  out=self.separator.net(mono);self.separator._latent=None;return out
 @torch.inference_mode()
 def image_frames(self,mouth):
  b,t=mouth.shape[:2];pixels=np.stack([cv2.resize(f,(112,112)) for f in mouth.cpu().numpy().reshape(-1,96,96)]).reshape(b,t,112,112)
  x=(torch.from_numpy(pixels).to(mouth.device).float()/255-.4161)/.1688;net=self.talknet.net.visualFrontend;x=net.frontend3D(x[:,None]).transpose(1,2).contiguous();return net.resnet(x.reshape(b*t,*x.shape[2:])).reshape(b,t,512)
 @torch.inference_mode()
 def features(self,wave,mouth,valid,candidates):
  assert wave.shape[1:]==(6,16000) and mouth.shape[1:]==(25,96,96)
  frames=self.image_frames(mouth);sp=self.spatial(wave,frames,valid);b=len(wave);net=self.talknet.net
  mf=np.stack([self.talknet.mfcc(np.pad(x,(0,160))*32768,16000,numcep=13,winlen=.025,winstep=.01)[:100] for x in candidates.cpu().numpy().reshape(-1,16000)]).astype(np.float32);mf=torch.from_numpy(mf).to(wave.device);ae=net.forward_audio_frontend(mf)
  ve=net.visualConv1D(net.visualTCN(frames.transpose(1,2))).transpose(1,2);vv=ve[:,None].expand(-1,2,-1,-1).reshape(b*2,25,128);aa,vv=net.forward_cross_attention(ae,vv);f=net.forward_audio_visual_backend(aa,vv).reshape(b,2,25,256);lg=self.talknet.fc(f);t=torch.arange(10,device=wave.device)*2.5+.5;j=t.floor().long();r=t-j;rich=(f[:,:,j]*(1-r)[None,None,:,None]+f[:,:,j+1]*r[None,None,:,None]).half().float()
  latent=torch.cat([sp['hidden'].mean(-1).transpose(1,2),sp['visual'].transpose(1,2),sp['adapter_visual'],sp['valid'][...,None].float()],-1)
  out=dict(mfcc=mf.reshape(b,2,100,13),vf=frames,ae=ae.reshape(b,2,25,128),ve=ve,wave=candidates,phase=sp['phase'],latent=latent,talk=lg[...,1]-lg[...,0],talk_features=rich,compat=phase_compatibility(wave,candidates,sp['phase']),valid=valid,energy=candidate_energy(candidates))
  if self.voice_features:
   aa=ae[:,10:20];cc=mf[:,40:80,1:];out['voice']=torch.cat([aa.mean(1),aa.std(1,unbiased=False),cc.mean(1),cc.std(1,unbiased=False)],-1).reshape(b,2,280)
  return out
class EngineState:
 def reset(self):
   self.voice_memory=VoiceMemory(self.voice_config) if self.voice_config else None
   self.state=None;self.previous=None;self.smoother=TemporalSelection(2,.2,6400)
   if self.spatial_memory:self.m.spatial.reset()
 def encode(self,wave,mouth,valid):
   B,targets=mouth.shape[:2];assert targets in [1,2] and len(wave)==B
   if self.spatial_memory:self.m.spatial.set_targets(targets)
   cand=self.m.separate(wave[:,0]);f=self.m.features(wave.repeat_interleave(targets,0),mouth.flatten(0,1),valid.flatten(0,1),cand.repeat_interleave(targets,0));return {k:v.reshape(B,targets,*v.shape[1:]) for k,v in f.items()}
 def select(self,f,raw=False):
   if self.fine_temporal:f=fine_features(f)
   targets=len(f['latent']);assert targets in [1,2];short=f['wave'][0,:,-16000:];perm=torch.arange(2,device=short.device);reliability=0.
   if self.previous is not None:perm,reliability=TemporalSelection.alignment(self.previous,short,6400)
   short=short[perm];self.previous=short.clone();x={k:f[k][None] for k in ['latent','talk','compat','valid','talk_features']}
   if targets==1:x={k:v.expand(-1,2,*v.shape[2:]) for k,v in x.items()}
   for k in ['talk','compat','talk_features']:x[k]=x[k].index_select(2,perm)
   x['energy']=f['energy'][0,perm][None];x['reliability']=torch.tensor([float(reliability)],device=short.device);scores,self.state=self.ranker.step(x,self.state);scores=scores[0,:targets]
   if self.voice_memory is not None:
    embedding=self.voice_projection(f['voice'][0,perm]).cpu().numpy();energy=f['energy'][0,perm,4:8,1].mean(-1).cpu().numpy();valid=f['valid'][:,10:20].float().mean(-1).cpu().numpy();scores=torch.from_numpy(self.voice_memory.step(scores.cpu().numpy(),embedding,energy,valid)).to(short.device)
   used=scores if raw else self.smoother.step(scores,short);choice=used.argmax(-1)
   return dict(waveform=short[choice,6400:12800],choice=choice,scores=scores,permutation=perm,reliability=float(reliability))
 def step(self,wave,mouth,valid,raw=False):
   f=self.encode(wave,mouth[None],valid[None]);return self.select({k:v[0] for k,v in f.items()},raw)
class StreamingEngine(EngineState):
 def __init__(self,bundle=None,device='cuda'):
  if bundle is not None and Path(bundle).resolve()!=BUNDLE.resolve():raise ValueError('Set SAV_BUNDLE before importing sav_bandnet')
  ck=torch.load(BUNDLE/'selector.pt',map_location='cpu',weights_only=True);self.context=1;self.fine_temporal=False
  self.m=SharedImageModel(LOCALIZER,DAVE,device);ranker=FrontendRanker({'config':ck['parent_config']},{'args':ck['ranker_args']}).to(device).eval();ranker.load_state_dict(ck['model'],strict=True)
  for k in ['crossA2V','crossV2A','selfAV','audioEncoder','visualTCN','visualConv1D']:setattr(self.m.talknet.net,k,getattr(ranker,k))
  self.m.talknet.fc=ranker.fc;self.ranker=ranker.ranker;set_emission(self.ranker.head,.4,.4);self.spatial_memory=True
  v=torch.load(BUNDLE/'voice.pt',map_location='cpu',weights_only=True);self.voice_projection=VoiceProjection(v['model']['mean'],v['model']['std']).to(device).eval();self.voice_projection.load_state_dict(v['model'],strict=True)
  self.voice_config=json.loads((BUNDLE/'config.json').read_text())['voice'];self.voice_metadata={'config':self.voice_config};self.m.voice_features=True
  self.m.eval().requires_grad_(False);self.ranker.eval().requires_grad_(False);self.reset()
