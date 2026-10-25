"""Small shared noise gate that relaxes DAVE's forced mixture reconstruction."""
import torch
from torch import nn
class NoiseGateDAVE(nn.Module):
    def __init__(self,base,source_gates=False):
        super().__init__();self.base=base;self.source_gates=source_gates
        self.noise_heads=nn.ModuleList([nn.Conv1d(base.feature_dim,int(bw)*(2 if source_gates else 1),1) for bw in base.band_width])
        for head in self.noise_heads:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.features=None
        self.base.separator.register_forward_hook(self.capture)
    def capture(self,module,inputs,output):
        self.features=output
    @property
    def separator(self):
        return self.base.separator
    def forward(self,x):
        y=self.base(x)
        feat=self.features
        if self.source_gates:
            logits=torch.cat([head(feat[:,i]).reshape(y.shape[0],2,int(self.base.band_width[i]),-1)
                for i,head in enumerate(self.noise_heads)],dim=2)
        else:
            logits=torch.cat([head(feat[:,i]) for i,head in enumerate(self.noise_heads)],dim=1)[:,None]
        # Starts exactly at one; may suppress noise without forcing it into another speech stream.
        gate=2*torch.sigmoid(logits)
        b,s,n=y.shape;win=torch.hann_window(self.base.win,device=y.device,dtype=y.dtype)
        z=torch.stft(y.reshape(b*s,n),self.base.win,self.base.stride,
            window=win,return_complex=True).reshape(b,s,gate.shape[-2],-1)
        z=z*gate
        out=torch.istft(z.reshape(b*s,z.shape[-2],z.shape[-1]),self.base.win,
            self.base.stride,window=win,length=n).reshape(b,s,n)
        return out
