import torch,numpy as np
from torch import nn
from .features import FrozenTalkNet
class SpatialCorrection(nn.Module):
 def __init__(self,clip=2.,gate=2.,spatial=True,scope='central',rich=False,frozen_base=None,residual_phase_only=False):
  super().__init__();self.config=dict(clip=clip,gate=gate,spatial=spatial,scope=scope,rich=rich)
  if frozen_base is not None:
   assert rich and frozen_base['rich'] and not frozen_base['spatial'] and 'frozen_base' not in frozen_base
   self.config['frozen_base']=frozen_base;self.config['residual_phase_only']=residual_phase_only
   self.frozen_base=SpatialCorrection(**frozen_base);self.frozen_base.requires_grad_(False)
   self.residual=SpatialCorrection(clip=clip,gate=0,spatial=spatial,scope=scope,rich=False)
   return
  self.latent=nn.Sequential(nn.LayerNorm(481),nn.Linear(481,48),nn.SiLU())
  self.candidate=nn.Sequential(nn.Linear(20,32),nn.SiLU())
  self.temporal=nn.GRU(80,48,batch_first=True,bidirectional=True)
  self.out=nn.Linear(96,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
  if rich:
   self.teacher_projection=nn.Linear(256,32,bias=False);nn.init.zeros_(self.teacher_projection.weight)
 def forward(self,latent,talk,compat,valid,energy=None,shuffle=False,talk_features=None):
  if 'frozen_base' in self.config:
   self.frozen_base.train(self.training and getattr(self,"adapt_base",False))
   with torch.set_grad_enabled(torch.is_grad_enabled() and getattr(self,"adapt_base",False)):base_scores,raw_scores,_=self.frozen_base(latent,talk,compat,valid,energy,talk_features=talk_features)
   spatial_latent=latent.clone();spatial_latent[...,224:480]=0
   if self.config['residual_phase_only']:spatial_latent[...,:480]=0
   _,_,correction=self.residual(spatial_latent,talk,compat,valid,energy,shuffle=shuffle)
   scores=base_scores+correction
   return scores,raw_scores,scores-raw_scores
  # target,candidate axes independent; no other-mouth feature in either query.
  B=len(latent);cfg=self.config;T=latent.shape[-2];V=valid.shape[-1]
  lo,hi=getattr(self,"emission",(T*2//5,T*4//5));vlo=lo*V//T;vhi=hi*V//T
  vv=valid.float();weights=vv if cfg['scope']=='full' else vv.clone()
  if cfg['scope']!='full':weights[:,:,:vlo]=0;weights[:,:,vhi:]=0
  baseline=(talk*weights[:,:,None]).sum(-1)/weights.sum(-1).clamp_min(1)[:,:,None]
  latent=latent.clone()
  if not cfg['spatial']:latent[...,:224]=0;compat=torch.zeros_like(compat)
  if shuffle:latent[...,:224]=latent.roll(1,0)[...,:224];compat=compat.roll(1,0)
  f=(torch.arange(T,device=talk.device).float() if T==V else torch.arange(T,device=talk.device)*2.5+.5);j=f.floor().long();r=f-j
  small=talk[...,j]*(1-r)+talk[...,(j+1).clamp_max(V-1)]*r
  rival=small.flip(2)
  energy=torch.zeros(B,2,T,2,device=talk.device) if energy is None else energy
  energy=energy[:,None].expand(-1,2,-1,-1,-1)
  cand=torch.cat([compat,small[...,None],rival[...,None],energy],-1)
  candidate=self.candidate[0](cand)
  if cfg['rich']:
   if talk_features is None:raise ValueError('This checkpoint requires frozen TalkNet audiovisual features')
   candidate=candidate+self.teacher_projection(talk_features.to(cand.dtype))
  candidate=self.candidate[1](candidate)
  la=self.latent(latent)[:,:,None].expand(-1,-1,2,-1,-1)
  x=torch.cat([la,candidate],-1).reshape(B*4,T,80)
  x,_=self.temporal(x);delta=self.out(x).reshape(B,2,2,T)
  # The final latent channel is the authoritative100ms visual validity.
  w=latent[...,-1].clone()
  if cfg['scope']!='full':w[:,:,:lo]=0;w[:,:,hi:]=0
  # Missing target video: use time context rather than silently disabling spatial correction.
  w=torch.where(w.sum(-1,keepdim=True)>0,w,torch.ones_like(w))
  if cfg['scope']!='full':w[:,:,:lo]=0;w[:,:,hi:]=0
  delta=(delta*w[:,:,None]).sum(-1)/w.sum(-1).clamp_min(1)[:,:,None]
  delta=cfg['clip']*torch.tanh(delta)
  gate=(baseline[...,0]-baseline[...,1]).abs()<cfg['gate'] if cfg['gate']>0 else torch.ones_like(baseline[...,0],dtype=torch.bool)
  delta=delta*gate[:,:,None]
  return baseline+delta,baseline,delta

def set_emission(model,history,emit,fps=10):
 for module in model.modules():
  if isinstance(module,SpatialCorrection):module.emission=(round(history*fps),round((history+emit)*fps))
class Ranker(nn.Module):
 def __init__(self,ck,memory=False,spatial=True):
  super().__init__();self.head=SpatialCorrection(**ck['config']);self.head.load_state_dict(ck['model']) if 'model' in ck else None;self.head.requires_grad_(True)
  for layer in self.head.modules():
   if isinstance(layer,SpatialCorrection):layer.adapt_base=True
  self.memory=memory;self.spatial=spatial
  if memory:
   self.visual=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,32),nn.SiLU())
   self.latent=nn.Sequential(nn.LayerNorm(481),nn.Linear(481,32),nn.SiLU())
   self.candidate=nn.Sequential(nn.Linear(22,32),nn.SiLU())
   self.gru=nn.GRU(96,64,batch_first=True);self.output=nn.Linear(64,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def encode(self,flat):
  if not self.spatial:
   flat=dict(flat);flat['latent']=flat['latent'].clone();flat['latent'][...,:224]=0;flat['compat']=torch.zeros_like(flat['compat'])
  scores,base,_=self.head(*[flat[k] for k in ['latent','talk','compat','valid']],flat['energy'],talk_features=flat['talk_features'])
  if not self.memory:return scores,None
  lat=self.latent(flat['latent'].mean(-2))[:,:,None].expand(-1,-1,2,-1)
  av=self.visual(flat['talk_features'].mean(-2))
  en=flat['energy'].mean(-2)[:,None].expand(-1,2,-1,-1)
  rel=flat['reliability'][:,None,None,None].expand(-1,2,2,1)
  tiny=torch.cat([flat['compat'].mean(-2),en,scores[...,None],scores.flip(-1)[...,None],base[...,None],rel],-1)
  return scores,torch.cat([lat,av,self.candidate(tiny)],-1)
 def forward(self,x):
  B,L=x['latent'].shape[:2];flat={k:v.flatten(0,1) for k,v in x.items() if k not in ['length','mask']};scores,f=self.encode(flat)
  if not self.memory:return scores.reshape(B,L,2,2)
  f=f.reshape(B,L,2,2,96).permute(0,2,3,1,4).reshape(B*4,L,96)
  h,_=self.gru(f);delta=self.output(h).reshape(B,2,2,L).permute(0,3,1,2)
  return scores.reshape(B,L,2,2)+delta
 def step(self,x,state=None):
  """One canonical candidate pair per target; reuse recurrent state without replaying history."""
  B=len(x['latent']);scores,f=self.encode(x)
  if not self.memory:return scores,None
  h,state=self.gru(f.reshape(B*4,1,96),state)
  return scores+self.output(h).reshape(B,2,2),state

def batch(records,ids,device):
 n=max(len(records[i]['latent']) for i in ids);result={}
 for k in records[ids[0]]:
  shape=(len(ids),n,*records[ids[0]][k].shape[1:]);value=torch.zeros(shape,dtype=records[ids[0]][k].dtype)
  for j,i in enumerate(ids):value[j,:len(records[i][k])]=records[i][k]
  result[k]=value.to(device)
 result['mask']=torch.arange(n,device=device)[None]<torch.tensor([len(records[i]['latent']) for i in ids],device=device)[:,None]
 return result

def objective(scores,x):
 err=x['error'];lab=err.argmin(-1);margin=(err.amax(-1)-err.amin(-1))/err.sum(-1).clamp_min(1e-8);weight=margin*x['mask'][...,None]
 if 'objective_weight' in x:weight=weight*x['objective_weight']
 ce=torch.nn.functional.cross_entropy(scores.reshape(-1,2),lab.reshape(-1),reduction='none').reshape_as(weight)
 loss=(ce*weight).sum()/weight.sum().clamp_min(1)
 return loss

def evaluate(m,ds,device):
 m.eval();regret=weight=correct=total=0
 for start in range(0,len(ds),8):
  x=batch(ds,list(range(start,min(start+8,len(ds)))),device);s=m(x);err=x['error'];gap=(err.amax(-1)-err.amin(-1))/err.sum(-1).clamp_min(1e-8);mask=x['mask'][...,None];chosen=err.gather(-1,s.argmax(-1)[...,None])[...,0]
  regret+=(((chosen-err.amin(-1))/err.sum(-1).clamp_min(1e-8))*mask).sum().item();weight+=(gap*mask).sum().item();correct+=((s.argmax(-1)==err.argmin(-1))*mask*gap).sum().item();total+=mask.sum().item()*2
 return dict(regret=regret/max(weight,1e-8),weighted_accuracy=correct/max(weight,1e-8),queries=total)
def fine_features(x):
 y=dict(x);V=x['valid'].shape[-1]
 for key in ['latent','compat','energy']:
  old=x[key];T=old.shape[-2];center=.05 if key=='energy' else .04;t=(torch.arange(V,device=old.device)*.04+.02-center)/.1;t=t.clamp(0,T-1);j=t.floor().long();r=t-j;shape=[1]*(old.ndim-2)+[V,1];r=r.reshape(shape);y[key]=old[...,j,:]*(1-r)+old[...,(j+1).clamp_max(T-1),:]*r
 y['latent']=y['latent'].clone();y['latent'][...,-1]=((y['latent'][...,-1]>.5)&x['valid'].bool()).float()
 return y

class BackendRanker(nn.Module):
 def __init__(self,parent,ranker):
  super().__init__();self.fine_temporal=False;self.history=ranker['args'].get('history',.4);self.ranker=Ranker(parent,ranker['args']['memory'],not ranker['args']['no_spatial']);self.ranker.load_state_dict(ranker['model']) if 'model' in ranker else None;set_emission(self.ranker.head,ranker['args'].get('history',.4),.4)
  net=FrozenTalkNet();self.crossA2V=net.net.crossA2V;self.crossV2A=net.net.crossV2A;self.selfAV=net.net.selfAV;self.fc=net.fc
  for name in ['crossA2V','crossV2A','selfAV','fc']:getattr(self,name).requires_grad_(True)
 def configure_fine(self,enabled):
  self.fine_temporal=enabled;set_emission(self.ranker.head,self.history,.4,fps=25 if enabled else 10)
 def prepare(self,x):
  B,L=x['ae'].shape[:2];V=x['ae'].shape[-2];T=x['latent'].shape[-2];ae=x['ae'][:,:,None].expand(-1,-1,2,-1,-1,-1).reshape(B*L*4,V,128);ve=x['ve'][:,:,:,None].expand(-1,-1,-1,2,-1,-1).reshape(B*L*4,V,128)
  aa=self.crossA2V(ae,ve);vv=self.crossV2A(ve,ae);z=torch.cat([aa,vv],-1);z=self.selfAV(z,z).reshape(B,L,2,2,V,256);logits=self.fc(z);scores=logits[...,1]-logits[...,0]
  t=torch.arange(T,device=z.device)*2.5+.5;j=t.floor().long();r=t-j
  features=z[...,j,:]*(1-r)[None,None,None,None,:,None]+z[...,j+1,:]*r[None,None,None,None,:,None]
  if self.fine_temporal:features=z
  features=features.half().float() # Original frozen feature interface and straight-through cast.
  y=fine_features(x) if self.fine_temporal else dict(x);y['talk']=scores;y['talk_features']=features
  return y
 def forward(self,x):return self.ranker(self.prepare(x))
 def step(self,x,state=None):
  y=self.prepare({k:v[:,None] for k,v in x.items()});return self.ranker.step({k:v[:,0] for k,v in y.items()},state)
class FrontendRanker(BackendRanker):
 def __init__(self,parent,ranker):
  super().__init__(parent,ranker);old=FrozenTalkNet()
  for name in ['audioEncoder','visualTCN','visualConv1D']:setattr(self,name,getattr(old.net,name));getattr(self,name).requires_grad_(True)
 def train(self,mode=True):
  super().train(mode)
  # Preserve pretrained running statistics and avoid padding-dependent batch statistics.
  for name in ['audioEncoder','visualTCN','visualConv1D']:
   for m in getattr(self,name).modules():
    if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
  return self
 def temporal_frontends(self,x):
  B,L=x['mfcc'].shape[:2];V=x['vf'].shape[-2];mf=x['mfcc'].reshape(B*L*2,V*4,13);vf=x['vf'].reshape(B*L*2,V,512);ae=self.audioEncoder(mf[:,None].transpose(2,3)).reshape(B,L,2,V,128);ve=self.visualConv1D(self.visualTCN(vf.transpose(1,2))).transpose(1,2).reshape(B,L,2,V,128);return ae,ve
 def clean_logits(self,x):
  B,L=x['clean_mfcc'].shape[:2];V=x['vf'].shape[-2];mf=x['clean_mfcc'].reshape(B*L*2,V*4,13);ae=self.audioEncoder(mf[:,None].transpose(2,3)).reshape(B,L,2,V,128);vf=x['vf'].reshape(B*L*2,V,512);ve=self.visualConv1D(self.visualTCN(vf.transpose(1,2))).transpose(1,2).reshape(B,L,2,V,128);return super().prepare(dict(x,ae=ae,ve=ve))['talk']
 def prepare(self,x):
  ae,ve=self.temporal_frontends(x);y=dict(x,ae=ae,ve=ve);return super().prepare(y)
