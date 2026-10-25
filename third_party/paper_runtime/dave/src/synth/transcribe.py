# -*- coding: utf-8 -*-
# Transcript labeling for synthesized training pairs. Runs the frozen Fun-ASR-Nano
# teacher over s1.wav / s2.wav in each pair directory and writes s1.txt / s2.txt.
# These transcripts are the targets of the teacher-forced cross-entropy (CER) loss in
# src/train.py; the trainer skips any pair that lacks either transcript.
# Usage: transcribe.py "<pair_glob>" ["<pair_glob>" ...]
#   e.g. transcribe.py "/path/to/data/meeting_best/mb_*" "/path/to/data/misp_mix/mm_*"
# Environment: ASR_MODEL = modelscope ID (default, auto-downloads ~2.1 GB on first run)
#              or a local model directory; WORKERS/WORKER shard the file list for
#              running several GPU workers in parallel.
import os, sys, glob, time, warnings
warnings.filterwarnings("ignore")
from funasr import AutoModel
am=AutoModel(model=os.environ.get("ASR_MODEL","FunAudioLLM/Fun-ASR-Nano-2512"),
    trust_remote_code=True,remote_code="/path/to/REAL-AVSE/Baseline/Fun-ASR/model.py",
    disable_update=True,device="cuda",disable_pbar=True)
W=int(os.environ.get("WORKERS","1")); K=int(os.environ.get("WORKER","0"))
jobs=[]
for pat in sys.argv[1:]:
    for d in sorted(glob.glob(pat)):
        for st in ("s1","s2"):
            w=os.path.join(d,st+".wav"); t=os.path.join(d,st+".txt")
            if os.path.exists(w) and not os.path.exists(t): jobs.append((w,t))
jobs=jobs[K::W]
print("worker %d/%d: %d files to transcribe"%(K,W,len(jobs)),flush=True)
t0=time.time(); n=0
for w,t in jobs:
    try:
        r=am.generate(input=[w],batch_size=1,disable_pbar=True)
        txt=(r[0].get("text") or "").strip()
        with open(t,"w",encoding="utf-8") as f: f.write(txt+"\n")
        n+=1
    except Exception as e:
        print("SKIP %s %s"%(w[-40:],repr(e)[:40]),flush=True)
    if n and n%500==0:
        el=time.time()-t0
        print("progress %d/%d rate=%.2f/s"%(n,len(jobs),n/el),flush=True)
print("TRANSCRIBE_DONE n=%d"%n,flush=True)
