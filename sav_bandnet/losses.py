import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from scipy.special import j0
class Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        pos=np.c_[.035*np.cos(np.arange(6)*np.pi/3),.035*np.sin(np.arange(6)*np.pi/3),np.zeros(6)]
        angles=np.linspace(-180,180,360);elev=np.deg2rad(np.arange(0,76,5))
        dirs=np.array([[np.cos(a)*np.cos(e),np.sin(a)*np.cos(e),np.sin(e)] for e in elev for a in np.deg2rad(angles)])
        delay=-(dirs@(pos[1:]-pos[0]).T)/343
        phase=2*np.pi*np.arange(1,257)[None,:,None]*16000/512*delay[:,None]
        template=np.concatenate([np.cos(phase),np.sin(phase)],axis=1).reshape(-1,2560)
        dist=np.linalg.norm(pos[1:]-pos[0],axis=1)
        null=np.concatenate([j0(2*np.pi*np.arange(1,257)[:,None]*16000/512*dist[None]/343),np.zeros((256,5))])
        self.register_buffer("template",torch.tensor(template,dtype=torch.float32))
        self.register_buffer("template_norm",self.template.square().mean(-1))
        self.register_buffer("angles",torch.tensor(angles,dtype=torch.float32))
        self.register_buffer("null",torch.tensor(null,dtype=torch.float32))
        self.register_buffer("freq",torch.arange(1,257)*16000/512)
    def targets(self,tdoa,active):
        phase=2*torch.pi*self.freq[None,None,:,None,None]*tdoa[:,:,:,1:].permute(0,1,3,2)[:,:,None]
        target=torch.cat([phase.cos(),phase.sin()],2)
        return torch.where(active[:,:,None,None,:],target,self.null[None,None,:,:,None])
    def scores(self,pred):
        flat=pred.reshape(-1,2560)
        return 2*(flat@self.template.T)/2560-self.template_norm
    @torch.no_grad()
    def decode(self,pred):
        idx=self.scores(pred).argmax(-1)%360
        return self.angles[idx].reshape(pred.shape[:2])
    def angular(self,pred,azimuth,weight):
        keep=weight>0
        if not keep.any():return pred.sum()*0
        selected=pred[keep];az=azimuth[keep];w=weight[keep]
        # Chunk the template operation to bound its activation memory.
        total=pred.new_zeros(())
        for start in range(0,len(selected),512):
            score=self.scores(selected[start:start+512]).reshape(-1,16,360).amax(1)/.02
            delta=(self.angles[None]-az[start:start+512,None]+180)%360-180
            soft=(-.5*(delta/3).square()).softmax(-1)
            total+=(-(soft*score.log_softmax(-1)).sum(-1)*w[start:start+512]).sum()
        return total/w.sum().clamp_min(1)

def masked_mean(x,w):return (x*w).sum()/w.sum().clamp_min(1)
def loss_fn(net,pred,aux,d,geo):
 T=pred.shape[1];ix=torch.arange(len(pred),device=pred.device);ident=d['identity'];targets=geo.targets(d['tdoa'],d['active']);chosen=targets[ix,:,:,:,ident];binding=masked_mean((pred-chosen).square().mean((2,3)),d['valid']);first=(aux-targets).square().mean((2,3,4));second=(aux-targets.flip(-1)).square().mean((2,3,4));fv=d['sample_valid'].reshape(len(pred),T,1600).float().mean(-1);spatial=masked_mean(torch.minimum(first,second),fv)
 with torch.no_grad():
  win=torch.hann_window(512,device=pred.device)
  def power(w):return torch.stft(w,512,320,window=win,return_complex=True)[:,1:,:T*5].abs().square().reshape(len(w),256,T,5).mean(-1).transpose(1,2)
  p=power(d['reference']);q=power(d['wave'][:,0]-d['reference']);irm=p/(p+q+1e-10)
 spectral=masked_mean(torch.nn.functional.binary_cross_entropy_with_logits(net.mask_logits,irm,reduction='none').mean(-1),d['valid']);direction=d['direction'][ix,:,ident];az=torch.atan2(direction[...,1],direction[...,0])*180/torch.pi;weight=d['valid']*d['active'][ix,:,ident]*(1+d['active'].all(-1).float());angular=geo.angular(pred,az,weight)
 return binding+.5*spatial+.3*spectral+.02*angular
def pit(y,r,valid,mixture):
 mask=valid[:,None].float();den=mask.sum(-1).clamp_min(1);yp=(y*mask).sum(-1)/den;rp=(r*mask).sum(-1)/den;y0=(y-yp[...,None])*mask;r0=(r-rp[...,None])*mask;rr=r0.square().sum(-1);yy=y0.square().sum(-1);cross=(y0*r0).sum(-1);proj=cross.square()/rr.clamp_min(1e-9);si=10*torch.log10((proj+1e-9)/(yy-proj).clamp_min(1e-9));energy=(r.square()*mask).sum(-1);mse=((y-r).square()*mask).sum(-1)/(energy+((mixture.square()*valid).sum(-1)[:,None])*.1+1e-7);active=energy/den>1e-7
 return (torch.where(active,-si.clamp(-40,40)/10,torch.log1p(mse))+.2*torch.log1p(mse)).mean(-1)

def aligned_loss(y,r,valid,mixture):
 a=pit(y,r,valid,mixture);b=pit(y.flip(1),r,valid,mixture);flip=b<a
 return torch.minimum(a,b).mean(),torch.where(flip[:,None,None],y.flip(1),y)