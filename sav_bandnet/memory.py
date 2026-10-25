import torch,numpy as np
from torch import nn
class PreTemporalMemory:
 def __init__(self):self.state={}
 def push(self,key,current,central,history,axis,pad=0):
  past_length=history-(current.shape[axis]-central[0]);shape=list(current.shape);shape[axis]=past_length;past=current.new_empty(shape);past[...] = torch.as_tensor(pad,device=current.device,dtype=current.dtype);previous=self.state.get(key)
  if previous is not None:
   take=min(past_length,previous.shape[axis]);past.narrow(axis,past_length-take,take).copy_(previous.narrow(axis,previous.shape[axis]-take,take))
  emitted=current.narrow(axis,central[0],central[1]-central[0]);combined=emitted if previous is None else torch.cat([previous,emitted],axis);keep=min(past_length,combined.shape[axis]);self.state[key]=combined.narrow(axis,combined.shape[axis]-keep,keep).clone()
  return torch.cat([past,current.narrow(axis,central[0],current.shape[axis]-central[0])],axis)
 def bytes(self):return sum(v.numel()*v.element_size() for v in self.state.values())
class TemporalSelection:
 def __init__(self,strength=1.,transition=.2,hop=25600):
  assert strength>=0 and 0<transition<=.5
  self.hop=hop;self.strength=strength;self.transition=transition;self.previous_wave=None;self.previous_probability=None
 @staticmethod
 def alignment(previous,current,hop=25600):
  # Adjacent queries advance1.6s; their acoustic overlap is2.4s.
  a=previous[:,hop:];b=current[:,:current.shape[-1]-hop];a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
  corr=(a@b.T).abs()/(a.square().sum(-1).sqrt()[:,None]*b.square().sum(-1).sqrt()[None]).clamp_min(1e-12)
  same=corr.diagonal().sum();swap=corr.flip(-1).diagonal().sum();permutation=torch.tensor([1,0] if swap>same else [0,1],device=current.device)
  reliability=((same-swap).abs()/2).clamp(0,1)
  return permutation,reliability
 def step(self,scores,waves):
  assert scores.shape[-1]==2 and waves.ndim==2 and waves.shape[0]==2
  updated=scores
  if self.previous_wave is not None and self.strength:
   perm,reliability=self.alignment(self.previous_wave,waves,self.hop)
   old=self.previous_probability[...,perm]
   transition=.5-reliability*(.5-self.transition)
   prior=old*(1-transition)+old.flip(-1)*transition
   updated=scores+self.strength*(prior.clamp_min(1e-8).log()+0.6931471805599453)
  self.previous_wave=waves.detach().clone();self.previous_probability=updated.softmax(-1).detach()
  return updated
class VoiceProjection(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('mean',mean);self.register_buffer('std',std);self.layers=nn.Sequential(nn.Linear(len(mean),128),nn.GELU(),nn.Linear(128,64))
 def forward(self,x):return torch.nn.functional.normalize(self.layers((x-self.mean)/self.std),dim=-1)

class VoiceMemory:
 def __init__(self,config):self.config=config;self.positive=self.negative=self.mass_pos=self.mass_neg=None
 def step(self,scores,embedding,energy,valid):
  s=np.asarray(scores,np.float32);z=np.asarray(embedding,np.float32);B=len(s);cfg=self.config
  if self.positive is None:
   self.positive=np.zeros((B,z.shape[-1]),np.float32);self.negative=np.zeros_like(self.positive);self.mass_pos=np.zeros(B,np.float32);self.mass_neg=np.zeros(B,np.float32)
  result=s.copy();pos=self.positive/np.maximum(np.linalg.norm(self.positive,axis=-1,keepdims=True),1e-8);neg=self.negative/np.maximum(np.linalg.norm(self.negative,axis=-1,keepdims=True),1e-8);delta=pos@z.T-(neg@z.T)*(self.mass_neg>=1)[:,None];gate=(self.mass_pos>=1)&(abs(s[:,1]-s[:,0])<cfg['gate']);result+=cfg['gain']*delta*gate[:,None]
  decay=np.exp(-.4/cfg['tau']);self.positive*=decay;self.negative*=decay;self.mass_pos*=decay;self.mass_neg*=decay;p=np.exp(s-s.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
  for target in range(B):
   c=int(p[target].argmax());confidence=float(p[target,c]);weight=2*confidence-1
   if confidence<cfg['confidence'] or valid[target]<.5 or energy[c]<.05:continue
   self.positive[target]+=weight*z[c];self.mass_pos[target]+=weight
   if energy[1-c]>=.05:self.negative[target]+=weight*z[1-c];self.mass_neg[target]+=weight
  return result
