
# -*- coding: utf-8 -*-
# Six-loss DDP trainer for the TIGER-M separation backbone:
#   L = SI-SDR(PIT) + l_cer*CE + l_spk*(1-cos) + l_stoi*STOI + l_pesq*PESQ + l_ut2*UTMOS
# - CER: frozen Fun-ASR-Nano teacher, teacher-forced cross-entropy on the estimated audio.
# - Speaker: evaluation-aligned WeSpeaker cnceleb_resnet34_LM (frozen); the estimate side
#   runs compute_fbank+resnet outside no_grad so the cosine term stays differentiable.
# - STOI/PESQ: differentiable surrogates, applied only to clean-reference samples.
# - UTMOS: differentiable naturalness critic; scores are clamped to prevent optimization
#   into the critic's unreliable high-score region (see the LAMBDA_UT2 note below).
# Auxiliary losses index sample 0 only — run with per-GPU batch = 1.
import os, sys, glob, time, warnings, random
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, soundfile as sf, torch
import torch.distributed as dist
import torch.nn.functional as TF
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
# Bundled modules (src/models): TIGER-M backbone and the WeSpeaker ResNet34 extractor —
# no external separation repo is required for training.
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from models import TIGER
from models.wespeaker_resnet34 import WeSpeakerResNet34

SR=16000; MAXLEN=96000; MINLEN=32000
BATCH=int(os.environ.get("BATCH","1"))
EPOCHS=int(os.environ.get("EPOCHS","3"))
PEAK_LR=float(os.environ.get("PEAK_LR","1e-4"))
WARM=int(os.environ.get("WARMUP","200"))
LAMBDA=float(os.environ.get("LAMBDA_CER","0.1"))
LAMSPK=float(os.environ.get("LAMBDA_SPK","0.5"))
LAMUT=float(os.environ.get("LAMBDA_UT","0"))    # optional torch.hub UTMOS22 (SpeechMOS); 0 in the final training configuration
LAMST=float(os.environ.get("LAMBDA_STOI","0.3"))
LAMPQ=float(os.environ.get("LAMBDA_PESQ","0.05"))
SDRT=os.environ.get("SDR_TYPE","sisdr")
R2REP=int(os.environ.get("R2_REPEAT","1"))
MBCAP=int(os.environ.get("MB_CAP","0"))
LAMUT2=float(os.environ.get("LAMBDA_UT2","0"))  # differentiable UTMOS critic; 1.0 in the final training configuration (see the note where the critic is constructed)
DATA_CAP=int(os.environ.get("DATA_CAP","0"))
SPEECHMOS_DIR=os.environ.get("SPEECHMOS_DIR","/path/to/REAL-AVSE/ops/SpeechMOS")
OFFSET=int(os.environ.get("STEP_OFFSET","300000"))
SAVE_EVERY=int(os.environ.get("SAVE_EVERY","2000"))
SMOKE=int(os.environ.get("SMOKE","0"))
# Initialization: the released checkpoint by default. The original training started
# from the TIGER-speech pretrained model (JusperLee/TIGER-speech) instead.
RESUME=os.environ.get("RESUME_CKPT","model/real_avse_tiger_m.ckpt")
SPK_CKPT=os.environ.get("SPK_CKPT","/path/to/wespeaker_cnceleb_resnet34/model_5.pt")  # cnceleb-resnet34-LM checkpoint (WeSpeaker release)
CKDIR=os.environ.get("CKPT_DIR","checkpoints")

class CerSet(Dataset):
    def __init__(self):
        items=[]
        for pat in ("/path/to/data/meeting_best/mb_*/s1.txt",
                    "/path/to/REAL-AVSE/data/r2syn/*/s1.txt",
                    "/path/to/data/misp_mix/mm_*/s1.txt",
                    "/path/to/data/mbpure/mp_*/s1.txt"):
            for t1 in glob.glob(pat):
                d=os.path.dirname(t1)
                t2=d+"/s2.txt"
                if not os.path.exists(t2): continue
                x1=open(t1,encoding="utf-8").read().strip()
                x2=open(t2,encoding="utf-8").read().strip()
                if len(x1)<2 or len(x2)<2: continue
                items.append((d,x1,x2))
        keep=[]
        for d,x1,x2 in items:
            try:
                fr=sf.info(d+"/mix.wav").frames
                if MINLEN<=fr<=MAXLEN: keep.append((d,x1,x2))
            except Exception: pass
        if MBCAP>0:
            mb=[it for it in keep if "meeting_best" in it[0]]
            oth=[it for it in keep if "meeting_best" not in it[0]]
            random.seed(31); random.shuffle(mb)
            keep=oth+mb[:MBCAP]
        if R2REP>1:
            extra=[it for it in keep if "/r2syn/" in it[0]]
            for _ in range(R2REP-1):
                keep.extend(extra)
        random.seed(13); random.shuffle(keep)
        if DATA_CAP>0: keep=keep[:DATA_CAP]
        self.items=keep
    def __len__(self): return len(self.items)
    def __getitem__(self,k):
        d,x1,x2=self.items[k]
        m,_=sf.read(d+"/mix.wav",dtype="float32")
        a,_=sf.read(d+"/s1.wav",dtype="float32")
        b,_=sf.read(d+"/s2.wav",dtype="float32")
        if m.ndim>1: m=m.mean(1)
        if a.ndim>1: a=a.mean(1)
        if b.ndim>1: b=b.mean(1)
        L=min(len(m),len(a),len(b))
        return torch.from_numpy(m[:L]),torch.from_numpy(a[:L]),torch.from_numpy(b[:L]),x1,x2,d

def neg_sdr(e,r):
    # negative SI-SDR (SDR_TYPE=sisdr, zero-mean, scale-invariant) or negative SNR
    if SDRT=="sisdr":
        e=e-e.mean(-1,keepdim=True); r=r-r.mean(-1,keepdim=True)
        a=(e*r).sum(-1,keepdim=True)/((r*r).sum(-1,keepdim=True)+1e-8)
        t=a*r; n=e-t
    else:
        t=r; n=e-r
    return -10.0*torch.log10(((t*t).sum(-1)+1e-8)/((n*n).sum(-1)+1e-8))

def pit2(est,refs):
    # Permutation-invariant training for two sources: evaluate both stream orders and
    # keep the better one; also return the estimate reordered to match the references.
    lA=(neg_sdr(est[:,0],refs[:,0])+neg_sdr(est[:,1],refs[:,1]))/2
    lB=(neg_sdr(est[:,0],refs[:,1])+neg_sdr(est[:,1],refs[:,0]))/2
    swap=lB<lA
    loss=torch.where(swap,lB,lA).mean()
    est_ord=torch.stack([est[b].flip(0) if swap[b] else est[b] for b in range(est.shape[0])])
    return loss,est_ord

def main():
    rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    model=TIGER(sample_rate=SR,out_channels=256,in_channels=512,num_blocks=12,upsampling_depth=5,win=640,stride=160,num_sources=2)
    sd=torch.load(RESUME,map_location="cpu")["state_dict"]
    model.load_state_dict(sd,strict=False); model.cuda()
    model=DDP(model,device_ids=[rank],find_unused_parameters=True)
    from funasr import AutoModel
    # ASR_MODEL: modelscope ID (auto-downloads ~2.1 GB on first run) or a local model dir
    am=AutoModel(model=os.environ.get("ASR_MODEL","FunAudioLLM/Fun-ASR-Nano-2512"),
        trust_remote_code=True,remote_code="/path/to/REAL-AVSE/Baseline/Fun-ASR/model.py",
        disable_update=True,device="cuda:%d"%rank,disable_pbar=True)
    asr=am.model; tok=am.kwargs["tokenizer"]; fe=am.kwargs["frontend"]
    for p in asr.parameters(): p.requires_grad_(False)
    asr.eval()
    spk=WeSpeakerResNet34(ckpt_path=SPK_CKPT,sample_rate=SR).eval().cuda()
    for p in spk.parameters(): p.requires_grad_(False)
    ut2=None
    if LAMUT2>0:
        # Differentiable UTMOSv2 critic (our wrapper around the public UTMOSv2 weights).
        # The wrapper module is not distributed with this repo — plug in your own builder
        # returning a callable wav->score, or keep LAMBDA_UT2=0.
        sys.path.insert(0,"/path/to/REAL-AVSE/ops")
        from diff_v2loss import build as _v2build
        ut2,_um=_v2build("cuda")
    from torch_stoi import NegSTOILoss
    from torch_pesq import PesqLoss
    st_loss=NegSTOILoss(sample_rate=SR).cuda()
    pq_loss=PesqLoss(0.5,sample_rate=SR).cuda()
    ut22=None
    if LAMUT>0:
        ut22=torch.hub.load(SPEECHMOS_DIR,"utmos22_strong",source="local",trust_repo=True).cuda().eval()
        for p in ut22.parameters(): p.requires_grad_(False)
    spk_dev=[True]  # GPU fbank availability; falls back to CPU on first failure
    if rank==0: print("[train] ready RESUME=%s lam_cer=%.2f lam_spk=%.2f lam_st=%.2f lam_pq=%.2f cap=%d"%(os.path.basename(RESUME),LAMBDA,LAMSPK,LAMST,LAMPQ,DATA_CAP),flush=True)
    if rank==0: print("[train] SDR_TYPE=%s"%SDRT,flush=True)
    opt=torch.optim.Adam(model.parameters(),lr=PEAK_LR)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=0.5,patience=2)
    ds=CerSet()
    if rank==0: print("[train] trainable pairs =",len(ds),flush=True)
    sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,drop_last=True)
    dl=DataLoader(ds,batch_size=BATCH,sampler=sampler,num_workers=3,pin_memory=True,drop_last=True)
    gstep=0
    def cer_ce(est_ord,clean,t1,t2):
        rows=[]
        for si,txt in ((0,t1),(1,t2)):
            try:
                if not txt: continue
                contents={"system":["You are a helpful assistant."],
                          "user":[["语音转写:<|startofspeech|>!!<|endofspeech|>",clean[0,si].numpy()]],
                          "assistant":[txt]}
                bd=asr.data_load_speech(contents,tok,fe,max_token_length=4096)
                if not torch.is_tensor(bd["speech"]): continue
                w=est_ord[0,si].float().cpu()
                feats,fl=fe(w[None,:],torch.tensor([w.numel()],dtype=torch.int32))
                if feats.shape!=bd["speech"].shape: continue
                bd["speech"]=feats
                rows.append(bd)
            except Exception:
                continue
        if not rows: return None
        ml=max(r["input_ids"].shape[1] for r in rows)
        def padto(x,v,l):
            if x.dim()==1: x=x[None,:]
            if x.shape[1]<l: x=torch.cat([x,torch.full((1,l-x.shape[1]),v,dtype=x.dtype)],dim=1)
            return x
        speech=torch.cat([r["speech"] for r in rows]).cuda()
        slens=torch.cat([r["speech_lengths"] for r in rows]).cuda()
        iid=torch.cat([padto(r["input_ids"],0,ml) for r in rows]).cuda()
        att=torch.cat([padto(r["attention_mask"],0,ml) for r in rows]).cuda()
        lab=torch.cat([padto(r["labels_ids"],-100,ml) for r in rows]).cuda()
        fbm=torch.cat([padto(r["fbank_mask"],0,ml) for r in rows]).cuda()
        fbg=torch.cat([r["fbank_beg"] for r in rows]).cuda()
        ftl=torch.cat([r["fake_token_len"] for r in rows]).cuda()
        out=asr(speech=speech,speech_lengths=slens,input_ids=iid,attention_mask=att,
                labels_ids=lab,fbank_beg=fbg,fbank_mask=fbm,fake_token_len=ftl)
        l=out[0] if isinstance(out,(tuple,list)) else out
        return l.float()
    def spk_sim_loss(est_ord,refs):
        tot=None; cs=0.0; cnt=0
        for si in range(2):
            try:
                w=est_ord[0,si].float()
                if spk_dev[0]:
                    try:
                        feat=spk.compute_fbank(w)
                    except Exception:
                        spk_dev[0]=False
                        feat=spk.compute_fbank(w.cpu()).cuda()
                else:
                    feat=spk.compute_fbank(w.cpu()).cuda()
                _,e=spk.resnet(feat.unsqueeze(0))
                e=TF.normalize(e,p=2,dim=1)[0]
                with torch.no_grad():
                    try:
                        r=spk(refs[0,si].float())
                    except Exception:
                        r=spk(refs[0,si].float().cpu())
                c=torch.dot(e,r)
                tot=(1.0-c) if tot is None else tot+(1.0-c)
                cs+=float(c.detach()); cnt+=1
            except Exception:
                continue
        if tot is None: return None,0.0
        return tot/cnt, cs/cnt
    def ut_loss(est_ord):
        if ut22 is None: return None,0.0
        tot=None; ss=0.0; cnt=0
        for si in range(2):
            try:
                w=est_ord[0,si].float()
                sc=ut22(w[None,:],16000)
                sc=sc.reshape(())
                tot=(5.0-sc)/4.0 if tot is None else tot+(5.0-sc)/4.0
                ss+=float(sc.detach()); cnt+=1
            except Exception:
                continue
        if tot is None: return None,0.0
        return tot/cnt, ss/cnt
    skipped=0
    for ep in range(EPOCHS):
        sampler.set_epoch(ep)
        model.train(); tl=0.0; tc=0.0; ts=0.0; tcos=0.0; tut=0.0; tst=0.0; tpq=0.0; tu2=0.0; n=0; t0=time.time()
        for ms,as_,bs,t1s,t2s,dps in dl:
            ms=ms.cuda(non_blocking=True)
            refs=torch.stack([as_,bs],dim=1).cuda(non_blocking=True)
            est=model(ms)
            si_loss,est_ord=pit2(est,refs)
            cl=cer_ce(est_ord,torch.stack([as_,bs],dim=1),t1s[0],t2s[0])
            sl,mcos=spk_sim_loss(est_ord,refs)
            ul,mut=ut_loss(est_ord)
            l_st=None; l_pq=None; l_u2=None
            if ut2 is not None:
                try:
                    w=est_ord[0,gstep%2].float()
                    if w.shape[-1]>16000:
                        _st=int(torch.randint(0,w.shape[-1]-16000,(1,)).item())
                        w=w[_st:_st+16000]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        sc=ut2(w)
                    sc=sc.float().clamp(max=4.2)
                    l_u2=(5.0-sc)/4.0
                except Exception:
                    l_u2=None
            if "misp_mix" not in dps[0]:
                try:
                    if LAMST>0 or LAMPQ>0:
                        l_st=st_loss(est_ord[0],refs[0]).mean()
                        l_pq=pq_loss(refs[0],est_ord[0]).mean()
                    else:
                        with torch.no_grad():
                            l_st=st_loss(est_ord[0],refs[0]).mean()
                            l_pq=pq_loss(refs[0],est_ord[0]).mean()
                except Exception:
                    l_st=None; l_pq=None; l_u2=None
            loss=si_loss
            if cl is not None: loss=loss+LAMBDA*cl
            if sl is not None: loss=loss+LAMSPK*sl
            if ul is not None: loss=loss+LAMUT*ul
            if l_st is not None: loss=loss+LAMST*l_st
            if l_pq is not None: loss=loss+LAMPQ*l_pq
            if l_u2 is not None: loss=loss+LAMUT2*l_u2
            bad=(~torch.isfinite(loss.detach())).float().reshape(1).cuda()
            dist.all_reduce(bad)
            if bad.item()>0:
                skipped+=1
                opt.zero_grad(set_to_none=True)
                if rank==0 and skipped%10==1: print("[train] non-finite loss, step skipped on all ranks total=%d"%skipped,flush=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            if gstep<WARM:
                for g in opt.param_groups: g["lr"]=PEAK_LR*(gstep+1)/WARM
            opt.step()
            gstep+=1
            tl+=float(si_loss.detach()); tc+=0.0 if cl is None else float(cl.detach())
            ts+=0.0 if sl is None else float(sl.detach()); tcos+=mcos; tut+=mut
            tst+=0.0 if l_st is None else float(l_st.detach()); tpq+=0.0 if l_pq is None else float(l_pq.detach())
            tu2+=0.0 if l_u2 is None else 5.0-4.0*float(l_u2.detach()); n+=1
            if rank==0 and gstep%50==0:
                el=time.time()-t0
                print("ep%d step%d si=%.3f ce=%.3f spk=%.3f cos=%.3f st=%.3f pq=%.3f u2=%.2f lr=%.1e %.2fs/it"%(
                    ep,gstep,tl/n,tc/n,ts/n,tcos/n,tst/n,tpq/n,tu2/n,opt.param_groups[0]["lr"],el/n),flush=True)
            if SMOKE and gstep>=SMOKE:
                break
            if rank==0 and gstep%SAVE_EVERY==0:
                torch.save({"state_dict":model.module.state_dict(),"gstep":gstep,"ep":ep},
                           "%s/step%d.ckpt"%(CKDIR,OFFSET+gstep))
        if SMOKE and gstep>=SMOKE:
            if rank==0: print("SMOKE_OK steps=%d si=%.3f ce=%.3f spk=%.3f cos=%.3f st=%.3f pq=%.3f u2=%.2f"%(gstep,tl/max(n,1),tc/max(n,1),ts/max(n,1),tcos/max(n,1),tst/max(n,1),tpq/max(n,1),tu2/max(n,1)),flush=True)
            break
        if rank==0:
            torch.save({"state_dict":model.module.state_dict(),"gstep":gstep,"ep":ep},
                       "%s/step%d.ckpt"%(CKDIR,OFFSET+gstep))
            print("ep%d done si=%.4f ce=%.4f spk=%.4f st=%.4f time=%.1fmin"%(ep,tl/max(n,1),tc/max(n,1),ts/max(n,1),tst/max(n,1),(time.time()-t0)/60),flush=True)
        if gstep>=WARM: sched.step(tl/max(n,1))
        dist.barrier()
    dist.destroy_process_group()

if __name__=="__main__":
    main()
