"""Native-six adaptation, exact-delay supervision, and geometry-aware decoding."""
from pathlib import Path
import sys,io,contextlib,json
P=Path(__file__).resolve().parent;BASE=P.parent
sys.path[:0]=[str(BASE/'ipdnet2_runtime'),str(BASE/'ipdnet2_upstream/IPDnet2')]
import numpy as np
import torch
from scipy.special import j0
from IPDnet2 import OnlineSpatialNet
from utils_ import forgetting_norm
POS=np.c_[.035*np.cos(np.arange(6)*np.pi/3),.035*np.sin(np.arange(6)*np.pi/3),np.zeros(6)]
def model(channels=6):
 return OnlineSpatialNet(dim_input=channels*2,dim_output=(channels-1)*4,num_layers=8,dim_hidden=96,num_heads=4,kernel_size=(5,3),conv_groups=(8,8),norms=['LN','LN','GN','LN','LN','LN'],dim_squeeze=8,num_freqs=256,attention='mamba(16,4)',rope=False,time_compression_layer=0,fre_compression_ratio=16,time_compression_ratio=5)
def canonicalize(wave,channel_ids):
 ids=list(channel_ids)
 if sorted(ids)!=list(range(6)):raise ValueError('Need each verified physical channel ID0..5 exactly once')
 return wave[...,np.argsort(ids),:]
def features(wave):
 b,c,n=wave.shape
 z=torch.stft(wave.reshape(b*c,n),512,320,window=torch.hann_window(512,device=wave.device),center=True,return_complex=True).reshape(b,c,257,-1)
 norm=forgetting_norm(z.abs(),sample_length=249)
 return torch.cat([z.real/(norm+1e-6),z.imag/(norm+1e-6)],1)[:,:,1:,:]
def forward(net,wave):
 with contextlib.redirect_stdout(io.StringIO()):return net(features(wave))
def targets(tdoa,active):
 f=torch.arange(1,257,device=tdoa.device)*16000/512
 phase=2*np.pi*f[None,None,:,None,None]*tdoa[:,:,:,1:].permute(0,1,3,2)[:,:,None]
 target=torch.cat([phase.cos(),phase.sin()],2)
 dist=np.linalg.norm(POS[1:]-POS[0],axis=1)
 null=np.concatenate([j0(2*np.pi*np.arange(1,257)[:,None]*16000/512*dist[None]/343),np.zeros((256,5))],axis=0)
 null=torch.tensor(null,dtype=target.dtype,device=target.device)[None,None,:,:,None]
 return torch.where(active[:,:,None,None,:],target,null)
def pit_loss(pred,target):
 a=(pred-target).square().mean((2,3,4));b=(pred-target.flip(-1)).square().mean((2,3,4))
 return torch.minimum(a,b).mean()
class Decoder:
 def __init__(self,device):
  self.angles=np.linspace(-180,180,360)
  elevations=np.deg2rad(np.arange(0,76,5));az=np.deg2rad(self.angles)
  directions=np.array([[np.cos(a)*np.cos(e),np.sin(a)*np.cos(e),np.sin(e)] for e in elevations for a in az])
  delay=-(directions@(POS[1:]-POS[0]).T)/343
  phase=2*np.pi*np.arange(1,257)[None,:,None]*16000/512*delay[:,None,:]
  self.template=torch.tensor(np.concatenate([np.cos(phase),np.sin(phase)],axis=1).reshape(-1,2560),dtype=torch.float32,device=device)
  self.angle_grid=np.tile(self.angles,len(elevations))
 def __call__(self,pred):
  shape=pred.shape[:2];v=pred.permute(0,1,4,2,3).reshape(-1,2560)
  mse=v.square().mean(-1)[:,None]+self.template.square().mean(-1)[None]-2*(v@self.template.T)/2560
  score,idx=mse.min(-1)
  return self.angle_grid[idx.detach().cpu().numpy()].reshape(*shape,2),score.detach().cpu().numpy().reshape(*shape,2)
