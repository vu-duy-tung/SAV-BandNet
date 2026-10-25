"""SAVE packed-scene I/O and explicit zero-padded streaming windows."""
from pathlib import Path
from functools import lru_cache
import io,json,os,struct,zipfile,hashlib
import numpy as np,soundfile as sf,torch

def crop(x,start,n,axis=-1):
 shape=list(x.shape);shape[axis]=n;y=np.zeros(shape,x.dtype);lo=max(0,start);hi=min(x.shape[axis],start+n)
 if hi>lo:
  a=[slice(None)]*x.ndim;b=a.copy();a[axis]=slice(lo,hi);b[axis]=slice(lo-start,hi-start);y[tuple(b)]=x[tuple(a)]
 return y
class SaveData:
 def __init__(self,root,split):
  self.root=Path(root);self.split=split;self.rows=[json.loads(s) for s in (self.root/'manifests'/f'{split}.jsonl').read_text().splitlines()]
 @lru_cache(maxsize=2)
 def scene(self,i):
  r=self.rows[i];p=str(r['path']);p=p if Path(p).is_absolute() else str(self.root/p)
  if '.pack@' in p:
   archive,off=p.rsplit('@',1)
   with open(archive,'rb') as f:f.seek(int(off));n=struct.unpack('<Q',f.read(8))[0];payload=f.read(n)
   assert len(payload)==n
   if 'record_sha256' in r:assert hashlib.sha256(payload).hexdigest()==r['record_sha256'],r['path']
   z=zipfile.ZipFile(io.BytesIO(payload))
   get=lambda name:z.read(name)
  else:get=lambda name:(Path(p)/name).read_bytes()
  def audio(name):
   x,sr=sf.read(io.BytesIO(get(name)),dtype='float32',always_2d=True);assert sr==16000;return x.T
  wave=audio('mixture.flac');refs=np.stack([audio(f'source_{l}.flac')[0] for l in 'ab']);direct=audio('direct_reference.flac');mouth=[];valid=[]
  for l in 'ab':
   with np.load(io.BytesIO(get(f'mouth_{l}.npz'))) as q:mouth.append(q['data'].copy());valid.append((q['valid']&q['present']).copy())
  with np.load(io.BytesIO(get('localization.npz'))) as q:labels={k:q[k].copy() for k in q.files}
  assert wave.shape[0]==6 and refs.shape[1]==wave.shape[1]
  return dict(wave=wave,reference=refs,direct=direct,mouth=mouth,valid=valid,labels=labels,row=r)
 def window(self,i,start):
  s=self.scene(i);wave=crop(s['wave'],start,16000);reference=crop(s['reference'],start,16000);direct=crop(s['direct'],start,16000).T
  times=start/16000+.04+np.arange(10)*.1;timevalid=(times>=0)&(times<s['wave'].shape[-1]/16000)
  active=(np.log10(np.square(abs(np.fft.rfft(direct.reshape(10,1600,2),axis=1)[:,:800])).sum(1)+1e-10)>=-2.5)&timevalid[:,None]
  z=s['labels'];ix=np.clip(np.rint(times*25-.5).astype(int),0,z['unit_direction'].shape[1]-1)
  return dict(wave=wave,reference=reference,mouth=np.stack([crop(x,start//640,25,0) for x in s['mouth']]),valid=np.stack([crop(x,start//640,25,0) for x in s['valid']]),tdoa=z['tdoa_to_reference_s'][:,ix].transpose(1,0,2),direction=z['unit_direction'][:,ix].transpose(1,0,2),active=active,sample_valid=(np.arange(16000)+start>=0)&(np.arange(16000)+start<s['wave'].shape[-1]))
class Windows(torch.utils.data.Dataset):
 def __init__(self,root,split,limit=0,hop=16000):
  self.data=SaveData(root,split);self.ids=list(range(len(self.data.rows)))[:limit or None];self.plan=[]
  for i in self.ids:
   n=round(self.data.rows[i]['seconds']*16000);self.plan.extend((i,s) for s in [-6400]+list(range(0,n,hop)))
 def __len__(self):return len(self.plan)
 def __getitem__(self,i):
  scene,start=self.plan[i];x=self.data.window(scene,start);x['scene']=scene;x['start']=start;return x
