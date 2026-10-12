from model import *
class CrossBandNet(AVNet):
 def __init__(self,initialize=False):
  super().__init__(initialize)
  self.crossband=torch.nn.TransformerEncoderLayer(d_model=192,nhead=4,dim_feedforward=384,dropout=0.,batch_first=True,norm_first=True)
  self.crossband_output=torch.nn.Linear(192,192)
  torch.nn.init.zeros_(self.crossband_output.weight);torch.nn.init.zeros_(self.crossband_output.bias)
  self.temporal.register_forward_hook(self.mix_bands)
 def mix_bands(self,module,args,output):
  sequence,hidden=output;b=self.hidden.shape[0];f=self.hidden.shape[-1];t=sequence.shape[1]
  x=sequence.reshape(b,f,t,192).permute(0,2,1,3).reshape(b*t,f,192)
  x=x+self.crossband_output(self.crossband(x))
  return x.reshape(b,t,f,192).permute(0,2,1,3).reshape(b*f,t,192),hidden
