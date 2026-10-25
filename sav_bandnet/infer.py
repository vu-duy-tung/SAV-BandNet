"""Full-scene inference, exact tail trimming and full-scene SI-SNR."""
from pathlib import Path
import json,time,hashlib
import numpy as np,torch,soundfile as sf
from .model import StreamingEngine
from .data import SaveData,crop
from .utils import sha

def si(y,r):
 y=y.astype(np.float64);r=r.astype(np.float64);y-=y.mean();r-=r.mean();e=np.dot(r,r)
 if e<1e-12:return None
 p=r*np.dot(y,r)/e;return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
def run(args):
 torch.set_num_threads(2)
 from .paths import BUNDLE
 ds=SaveData(args.data,args.split);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);engine=StreamingEngine(device=args.device);results=[];begin=time.time()
 ids=list(range(len(ds.rows)))[:args.limit or None][args.rank::args.world]
 for i in ids:
  engine.reset();s=ds.scene(i);L=s['wave'].shape[-1];parts=[];choices=[]
  with torch.inference_mode():
   for pos in range(0,L,6400):
    wave=torch.from_numpy(crop(s['wave'],pos-6400,16000)[None]).to(args.device);mouth=torch.from_numpy(np.stack([crop(x,(pos-6400)//640,25,0) for x in s['mouth']])).to(args.device);valid=torch.from_numpy(np.stack([crop(x,(pos-6400)//640,25,0) for x in s['valid']])).to(args.device)
    y=engine.step(wave,mouth,valid);a=y['waveform'].cpu().numpy();assert a.shape==(2,6400) and np.isfinite(a).all();parts.append(a[:,:min(6400,L-pos)]);choices.append(y['choice'].cpu().tolist())
  y=np.concatenate(parts,axis=1);folder=out/f'{i:06d}';folder.mkdir(exist_ok=True);files=[]
  for t,l in enumerate('ab'):
   p=folder/f'target_{l}.wav';sf.write(p,y[t],16000,subtype='FLOAT');files.append(dict(path=str(p),sha256=sha(p)))
  r=dict(scene=i,source_uid=ds.rows[i]['uid'],samples=L,si_snr=[si(y[t],s['reference'][t]) for t in range(2)],choices=choices,files=files);(folder/'complete.json').write_text(json.dumps(r));results.append(r);print('SCENE',i,'complete',flush=True)
 summary=dict(scenes=len(results),targets=2*len(results),si_snr=float(np.mean([v for r in results for v in r['si_snr'] if v is not None])),seconds=time.time()-begin,window=[.4,.4,.2],bundle_sha=sha(BUNDLE/'PROVENANCE.json'),manifest_sha=sha(Path(args.data)/'manifests'/f'{args.split}.jsonl'))
 (out/f'COMPLETE_rank{args.rank}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
