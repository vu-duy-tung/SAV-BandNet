# -*- coding: utf-8 -*-
# Two-speaker mixture synthesis: 50% direct-sum mixtures + 50% RIR-convolved mixtures
# (simulated rooms, RT60 0.2-0.4 s). Among the candidate recipes evaluated, this
# configuration minimized the synthetic-to-real performance gap of the baseline
# separation model (10.18 dB SI-SDR on synthetic vs 10.70 dB on real development
# mixtures).
# Usage: synthesize.py <shard> <nshards> <total_pairs>
import os, sys, json, glob, random, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf
import pyroomacoustics as pra
# The official challenge Baseline repository is used here only for the speaker-embedding
# gate that rejects same-speaker pairs (cosine similarity > 0.5).
sys.path.insert(0,"/path/to/REAL-AVSE/Baseline")
from look2hear.metrics.real_metrics import RealMetricsTracker
rm=RealMetricsTracker(metrics=("objective",),device="cuda",
    wespeaker_ckpt="/path/to/REAL-AVSE/Baseline/pretrained/wespeaker_cnceleb_resnet34/model_5.pt")
def emb(x):
    e=rm.speaker_embedding(x)
    if hasattr(e,"cpu"): e=e.detach().cpu().numpy()
    e=np.asarray(e,np.float32).reshape(-1); return e/(np.linalg.norm(e)+1e-9)
SH,NSH,TOTAL=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
# clean_sources.json: build it from your local corpora —
#   {"AliMeeting": ["/abs/path/x.wav", ...], "MISP": [...], "AISHELL-4": [...]}
# (lists of clean single-speaker utterance wavs, 16 kHz)
clean=json.load(open("/path/to/REAL-AVSE/ops/clean_sources.json"))
srcs=clean["AliMeeting"]+clean["MISP"]
random.seed(100+SH); random.shuffle(srcs); srcs=srcs[:1500]
cache={}
for f in srcs:
    try:
        x,_=sf.read(f,dtype="float32")
        if x.ndim>1: x=x.mean(1)
        if len(x)>=8000: cache[f]=(x,emb(x))
    except Exception: pass
pool=list(cache.keys())
print(f"shard {SH}: cached {len(pool)} sources",flush=True)
noises=glob.glob("/path/to/REAL-AVSE/data/noise_bank/*.wav")+glob.glob("/path/to/data/musan/musan/noise/*/*.wav")
OUT="/path/to/data/meeting_best"; os.makedirs(OUT,exist_ok=True)
def fit(x,need): return x[:need] if len(x)>=need else np.tile(x,int(np.ceil(need/len(x))))[:need]
def load_noise(need,idx):
    r=random.Random(idx*131+SH); buf=[]
    while sum(len(b) for b in buf)<need:
        try:
            y,_=sf.read(r.choice(noises),dtype="float32")
            if y.ndim>1: y=y.mean(1)
            buf.append(y if len(y)>0 else np.zeros(1600,np.float32))
        except Exception: buf.append(np.zeros(1600,np.float32))
    return np.concatenate(buf)[:need]
def rir_conv(sig,dim,absorp,mo,mic,pos):
    room=pra.ShoeBox(dim,fs=16000,materials=pra.Material(absorp),max_order=mo)
    room.add_microphone_array(pra.MicrophoneArray(mic,16000))
    room.add_source(pos,signal=sig)
    room.simulate()
    return room.mic_array.signals[0]
per=TOTAL//NSH; manifest=[]; n=0; i=0
while n<per and i<per*4:
    idx=SH*per*4+i; i+=1
    rng=random.Random(idx*7919+13)
    fa,fb=rng.sample(pool,2)
    if float(cache[fa][1]@cache[fb][1])>0.5: continue
    dur=rng.uniform(4.0,8.0); need=int(dur*16000)
    a=fit(cache[fa][0],need).copy(); b=fit(cache[fb][0],need).copy()
    a=a/(np.sqrt((a**2).mean())+1e-9); b=b/(np.sqrt((b**2).mean())+1e-9)
    # 50% direct sum, 50% RIR convolution
    if rng.random()<0.5:
        ar,br=a,b
    else:
        try:
            dim=[rng.uniform(3,8),rng.uniform(3,7),rng.uniform(2.5,3.5)]
            rt60=rng.uniform(0.2,0.4)
            absorp,mo=pra.inverse_sabine(rt60,dim); mo=int(min(mo,8))
            mic=np.array([[rng.uniform(1,dim[0]-1)],[rng.uniform(1,dim[1]-1)],[1.5]])
            pa=[rng.uniform(0.5,dim[0]-0.5),rng.uniform(0.5,dim[1]-0.5),rng.uniform(1.2,1.8)]
            pb=[rng.uniform(0.5,dim[0]-0.5),rng.uniform(0.5,dim[1]-0.5),rng.uniform(1.2,1.8)]
            ya=rir_conv(a,dim,absorp,mo,mic,pa); yb=rir_conv(b,dim,absorp,mo,mic,pb)
            ar=(ya[:need] if len(ya)>=need else np.pad(ya,(0,need-len(ya)))).astype(np.float32)
            br=(yb[:need] if len(yb)>=need else np.pad(yb,(0,need-len(yb)))).astype(np.float32)
        except Exception:
            ar,br=a,b
    br=br*10**(rng.uniform(-3,3)/20)
    mix=ar+br
    snr=rng.uniform(12,30)
    nz=load_noise(len(mix),idx)
    ps=float((mix**2).mean())+1e-9; pn=float((nz**2).mean())+1e-9
    mix=mix+np.sqrt(ps/(pn*(10**(snr/10))))*nz
    pk=np.abs(mix).max()
    if pk>0.99: mix=mix/pk*0.99; ar=ar/pk*0.99; br=br/pk*0.99
    key=f"mb_{SH}_{n:06d}"; od=f"{OUT}/{key}"; os.makedirs(od,exist_ok=True)
    sf.write(f"{od}/mix.wav",mix.astype(np.float32),16000)
    sf.write(f"{od}/s1.wav",ar.astype(np.float32),16000)
    sf.write(f"{od}/s2.wav",br.astype(np.float32),16000)
    manifest.append((f"{od}/mix.wav",f"{od}/s1.wav",f"{od}/s2.wav",len(mix)))
    n+=1
    if n%500==0: print(f"shard {SH}: {n}",flush=True)
json.dump(manifest,open(f"{OUT}/manifest_{SH}.json","w"))
print(f"SYNTH_DONE shard={SH} produced={n}",flush=True)
