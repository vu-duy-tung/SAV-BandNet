"""Full-coverage fine-tuning with frozen image features and cached attribution latents.

Stages: visually conditioned IPDNet2; short-context DAVE; frozen feature cache;
SAV selector/temporal frontends; target-voice projection. No test-set selection.
"""
from pathlib import Path
import json,time,copy,os
import numpy as np,torch,cv2
from torch import nn
from torch.utils.data import DataLoader
from .data import SaveData,Windows,crop
from .paths import BUNDLE
from .model import StreamingEngine
from .spatial import FrozenTalkImage,TalkSpatial
from .selector import FrontendRanker,batch,objective,evaluate
from .separator import FrozenDAVE
from .memory import TemporalSelection,VoiceProjection
from .losses import Geometry,loss_fn,aligned_loss
from .utils import sha,state_sha

def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');torch.save(obj,tmp);tmp.replace(p)
def record(out,stage,**kwargs):
 p=Path(out)/'progress.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(dict(stage=stage,epoch_time=time.time(),**kwargs),indent=2));q.replace(p);print(stage,kwargs,flush=True)
def update(loss,m,opt):
 assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5);assert torch.isfinite(norm);opt.step();return float(loss.detach())
def localize(args):
 image=FrozenTalkImage().to(args.device).eval();m=TalkSpatial().to(args.device);saved=torch.load(BUNDLE/'spatial.pt',map_location='cpu',weights_only=True);m.load_state_dict(saved['model'],strict=True);geo=Geometry().to(args.device)
 opt=torch.optim.AdamW([{'params':m.net.parameters(),'lr':1e-5},{'params':m.adapter.parameters(),'lr':1e-4}],weight_decay=1e-4);best=float('inf');steps=0
 for epoch in range(args.localization_epochs):
  for split in ['tr','cv']:
   m.train(split=='tr');ds=Windows(args.data,split,args.limit);loader=DataLoader(ds,batch_size=args.batch,shuffle=split=='tr',num_workers=args.workers);losses=[]
   for k,b in enumerate(loader):
    if args.smoke_steps and k>=args.smoke_steps:break
    n=len(b['wave']);device=args.device;pixels=np.stack([cv2.resize(x,(112,112)) for x in b['mouth'].numpy().reshape(-1,96,96)]).reshape(n*2,25,112,112)
    with torch.no_grad():features=image(torch.from_numpy(pixels).to(device))
    d={key:b[key].to(device).repeat_interleave(2,0) for key in ['wave','tdoa','direction','active','sample_valid']};d.update(reference=b['reference'].to(device).flatten(0,1),identity=torch.arange(2,device=device).repeat(n))
    with torch.set_grad_enabled(split=='tr'):
     pred,aux,valid,_=m(d['wave'],features,b['valid'].to(device).flatten(0,1));d['valid']=valid;loss=loss_fn(m.net,pred,aux,d,geo)
     losses.append(update(loss,m,opt) if split=='tr' else float(loss))
    steps+=split=='tr'
   score=float(np.mean(losses));record(args.output,'localization',epoch=epoch+1,split=split,loss=score,steps=steps)
   if split=='cv' and score<best:
    best=score;atomic(BUNDLE/'spatial.pt',dict(model=m.state_dict(),image_sha=saved['image_sha']));atomic(Path(args.output)/'localization_state.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,best=best))
 return dict(steps=steps,best_cv_loss=best)
def separator(args):
 net=FrozenDAVE().net.to(args.device);teacher=copy.deepcopy(net).eval().requires_grad_(False);net.requires_grad_(True);opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=1e-4);best=float('inf');steps=0
 for epoch in range(args.separator_epochs):
  for split in ['tr','cv']:
   net.train(split=='tr');ds=Windows(args.data,split,args.limit,hop=6400);loader=DataLoader(ds,batch_size=args.batch,shuffle=split=='tr',num_workers=0);losses=[]
   for k,b in enumerate(loader):
    if args.smoke_steps and k>=args.smoke_steps:break
    long=torch.from_numpy(np.stack([crop(ds.data.scene(int(i))['wave'][0],int(s)-19200,64000) for i,s in zip(b['scene'],b['start'])])).to(args.device)
    x=b['wave'][:,0].to(args.device);ref=b['reference'].to(args.device);valid=b['sample_valid'].to(args.device)
    with torch.no_grad():old=teacher(long)[:,:,19200:35200]
    with torch.set_grad_enabled(split=='tr'):
     y=net(x);rl,_=aligned_loss(y,ref,valid,x);tl,_=aligned_loss(y,old,valid,x);loss=.5*rl+.5*tl;losses.append(update(loss,net,opt) if split=='tr' else float(loss))
    steps+=split=='tr'
   score=float(np.mean(losses));record(args.output,'separator',epoch=epoch+1,split=split,loss=score,steps=steps)
   if split=='cv' and score<best:
    best=score;atomic(BUNDLE/'dave.pt',dict(state_dict=net.state_dict()));atomic(Path(args.output)/'separator_state.pt',dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,best=best))
 return dict(steps=steps,best_cv_loss=best)
def cache(args):
 engine=StreamingEngine(device=args.device);root=Path(args.output)/'cache';root.mkdir(exist_ok=True);ident=dict(localizer=sha(BUNDLE/'spatial.pt'),separator=sha(BUNDLE/'dave.pt'),selector=sha(BUNDLE/'selector.pt'))
 for split in ['tr','cv']:
  ds=SaveData(args.data,split)
  for i in range(min(args.limit or len(ds.rows),len(ds.rows))):
   path=root/split/f'{i:06d}.pt';path.parent.mkdir(exist_ok=True)
   if path.exists():
    ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['identity']==ident;continue
   engine.reset();s=ds.scene(i);L=s['wave'].shape[-1];records=[];previous=None
   with torch.inference_mode():
    for pos in range(0,L,6400):
     if args.smoke_steps and len(records)>=max(2,args.smoke_steps):break
     w=ds.window(i,pos-6400);wave=torch.from_numpy(w['wave'][None]).to(args.device);mouth=torch.from_numpy(w['mouth']).to(args.device);valid=torch.from_numpy(w['valid']).to(args.device);cand=engine.m.separate(wave[:,0]);engine.m.spatial.set_targets(2)
     f=engine.m.features(wave.repeat_interleave(2,0),mouth,valid,cand.repeat_interleave(2,0));perm=torch.arange(2,device=args.device);rel=0.
     if previous is not None:perm,rel=TemporalSelection.alignment(previous,cand[0],6400)
     current=cand[0,perm];previous=current.clone();n=min(6400,L-pos);ref=torch.from_numpy(s['reference'][:,pos:pos+n]).to(args.device);error=((current[None,:,6400:6400+n]-ref[:,None])**2).sum(-1)
     r={key:f[key] for key in ['latent','talk','compat','valid','talk_features','vf','ve']}
     for key in ['talk','compat','talk_features']:r[key]=r[key].index_select(1,perm)
     r.update(energy=f['energy'][0,perm],mfcc=f['mfcc'][0,perm],ae=f['ae'][0,perm],error=error,reliability=torch.tensor(float(rel),device=args.device),active=torch.from_numpy(w['active'][4:8].mean(0)>=.5).to(args.device))
     # Clean target pairs support the same auxiliary active-speaker objective.
     clean=[]
     for target in range(2):clean.append(engine.m.talknet.mfcc(np.pad(w['reference'][target],(0,160))*32768,16000,numcep=13,winlen=.025,winstep=.01)[:100])
     r['clean_mfcc']=torch.tensor(np.stack(clean),dtype=torch.float32,device=args.device);r['clean_active']=torch.from_numpy(w['active'][np.minimum((np.arange(25)*.04/.1).astype(int),9)].T.astype(np.float32)).to(args.device)
     records.append({key:value.detach().cpu() for key,value in r.items()})
   packed={key:torch.stack([r[key] for r in records]) for key in records[0]};atomic(path,dict(identity=ident,data=packed,speakers=ds.rows[i]['speakers'],scene=i));record(args.output,'cache',split=split,scenes=i+1,total=min(args.limit or len(ds.rows),len(ds.rows)))
 return ident

def load_head(device):
 ck=torch.load(BUNDLE/'selector.pt',map_location='cpu',weights_only=True);m=FrontendRanker({'config':ck['parent_config']},{'args':ck['ranker_args']}).to(device);m.load_state_dict(ck['model'],strict=True);m.configure_fine(False);return m,ck

def extraction(args):
 root=Path(args.output)/'cache';paths={s:sorted((root/s).glob('*.pt')) for s in ['tr','cv']};m,ck=load_head(args.device);m.requires_grad_(True);opt=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=1e-4);best=float('inf');steps=0
 def read(p):
  d=torch.load(p,map_location='cpu',weights_only=True)['data'];return {k:v for k,v in d.items() if k!='active'}
 for ep in range(args.extraction_epochs):
  m.train();order=np.random.default_rng(161026+ep).permutation(len(paths['tr']));losses=[]
  for k in range(0,len(order),4):
   if args.smoke_steps and k//4>=args.smoke_steps:break
   data=[read(paths['tr'][i]) for i in order[k:k+4]];x=batch(data,list(range(len(data))),args.device);loss=objective(m(x),x);clean=m.clean_logits(x);lab=torch.eye(2,device=args.device)[None,None,:,:,None].expand_as(clean);weight=x['valid'][:,:,:,None,:].float()*x['clean_active'][:,:,None,:,:]*x['mask'][:,:,None,None,None]
   aux=(torch.nn.functional.binary_cross_entropy_with_logits(clean,lab,reduction='none')*weight).sum()/weight.sum().clamp_min(1);loss=loss+.1*min(1.,(steps+1)/200)*aux;losses.append(update(loss,m,opt));steps+=1
  m.eval();regret=weighttotal=0.
  with torch.no_grad():
   for p in paths['cv']:
    x=batch([read(p)],[0],args.device);scores=m(x);err=x['error'];gap=(err.amax(-1)-err.amin(-1))/err.sum(-1).clamp_min(1e-8);chosen=err.gather(-1,scores.argmax(-1)[...,None])[...,0];regret+=float(((chosen-err.amin(-1))/err.sum(-1).clamp_min(1e-8)).sum());weighttotal+=float(gap.sum())
  score=regret/max(weighttotal,1e-8);record(args.output,'selector',epoch=ep+1,loss=float(np.mean(losses)),cv_regret=score,steps=steps)
  if score<best:
   best=score;atomic(BUNDLE/'selector.pt',dict(model=m.state_dict(),parent_config=ck['parent_config'],ranker_args=ck['ranker_args']));atomic(Path(args.output)/'selector_state.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=ep+1,best=best))
 return dict(steps=steps,best_cv_regret=best)

def identities(args):
 m,_=load_head(args.device);m.eval();xs=[];labels=[];sceneids=[];classes={};paths=sorted((Path(args.output)/'cache/tr').glob('*.pt'))
 with torch.no_grad():
  for p in paths:
   ck=torch.load(p,map_location='cpu',weights_only=True);d=ck['data'];Q=len(d['mfcc']);mf=d['mfcc'].to(args.device).reshape(-1,100,13);ae=m.audioEncoder(mf[:,None].transpose(2,3));a=ae[:,10:20];c=mf[:,40:80,1:];f=torch.cat([a.mean(1),a.std(1,unbiased=False),c.mean(1),c.std(1,unbiased=False)],-1).reshape(Q,2,280).cpu();err=d['error'];swap=err[:,0,1]+err[:,1,0]<err[:,0,0]+err[:,1,1];gap=(err.amax(-1)-err.amin(-1))/err.sum(-1).clamp_min(1e-8)
   for q in range(Q):
    for candidate in range(2):
     target=1-candidate if swap[q] else candidate
     if not d['active'][q,target] or gap[q,target]<=.2 or d['energy'][q,candidate,4:8,1].mean()<.05 or err[q,target].argmin()!=candidate:continue
     speaker=ck['speakers'][target];classes.setdefault(speaker,len(classes));xs.append(f[q,candidate]);labels.append(classes[speaker]);sceneids.append(ck['scene'])
 if len(set(sceneids))<2 or len(classes)<2:raise RuntimeError('Insufficient eligible TRAIN scenes/speakers for identity fitting; no fabricated completion')
 x=torch.stack(xs).to(args.device);y=torch.tensor(labels,device=args.device);unique=sorted(set(sceneids));validation=set(np.random.default_rng(161026).choice(unique,max(1,len(unique)//10),replace=False).tolist());mask=torch.tensor([s in validation for s in sceneids],device=args.device);tr=(~mask).nonzero().flatten();cv=mask.nonzero().flatten();prior=torch.load(BUNDLE/'voice.pt',map_location='cpu',weights_only=True)['model'];v=VoiceProjection(prior['mean'],prior['std']).to(args.device);v.load_state_dict(prior,strict=True);classifier=nn.Linear(64,len(classes),bias=False).to(args.device);opt=torch.optim.AdamW(list(v.parameters())+list(classifier.parameters()),lr=1e-3,weight_decay=1e-4);best=float('inf')
 class_weight=torch.bincount(y[tr],minlength=len(classes)).float().clamp_min(1).rsqrt();class_weight/=class_weight.mean()
 if not args.limit:assert set(y[cv].tolist())<=set(y[tr].tolist()), 'Identity validation includes a speaker absent from fitting data'
 for ep in range(args.identity_epochs):
  v.train()
  for ids in tr[torch.randperm(len(tr),device=args.device)].split(1024):
   loss=nn.functional.cross_entropy(16*(v(x[ids])@nn.functional.normalize(classifier.weight,dim=-1).T),y[ids],weight=class_weight);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(v.parameters())+list(classifier.parameters()),5);opt.step()
  v.eval()
  with torch.no_grad():score=float(nn.functional.cross_entropy(16*(v(x[cv])@nn.functional.normalize(classifier.weight,dim=-1).T),y[cv]))
  record(args.output,'speaker_identity',epoch=ep+1,validation_loss=score)
  if score<best:best=score;atomic(BUNDLE/'voice.pt',dict(model=v.state_dict()))
 return dict(training_windows=len(tr),validation_windows=len(cv),speakers=len(classes),epochs=args.identity_epochs,best_validation_loss=best)

def run(args):
 torch.manual_seed(161026);np.random.seed(161026);torch.set_num_threads(2);cv2.setNumThreads(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);counts={s:len(SaveData(args.data,s).rows) for s in ['tr','cv','tt']}
 if not args.limit:assert counts=={'tr':32768,'cv':1000,'tt':2000},counts
 protocol=dict(recipe='Full coverage continued fine-tuning; original loss definitions, fresh AdamW states; fixed CV loss/regret selection',counts=counts,manifest_hashes={s:sha(Path(args.data)/'manifests'/f'{s}.jsonl') for s in counts},smoke_only=bool(args.limit or args.smoke_steps),args=vars(args),test_used_for_training=False,voice_gate_policy='Retained fixed hyperparameters; projection fitted using TRAIN internal scene holdout',initial_bundle=sha(Path(args.bundle)/'PROVENANCE.json'))
 marker=out/'TRAINING_PROTOCOL.json'
 if marker.exists():assert json.loads(marker.read_text())==protocol, 'Resume requires identical data, bundle and arguments'
 else:marker.write_text(json.dumps(protocol,indent=2))
 results={}
 for name,fn in [('localization',localize),('dave',separator),('cache',cache),('selector',extraction),('speaker_identity',identities)]:
  marker=out/f'{name}_COMPLETE.json'
  if marker.exists():results[name]=json.loads(marker.read_text());continue
  results[name]=fn(args);marker.write_text(json.dumps(results[name],indent=2))
 config=json.loads((BUNDLE/'config.json').read_text());config.update(training_scenes=min(args.limit or counts['tr'],counts['tr']),smoke_only=protocol['smoke_only']);(BUNDLE/'config.json').write_text(json.dumps(config,indent=2))
 provenance=dict(training=protocol,stages=results,files={p.name:sha(p) for p in BUNDLE.glob('*') if p.is_file() and p.name!='PROVENANCE.json'});(BUNDLE/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2));(out/'COMPLETE.json').write_text(json.dumps(results,indent=2))
