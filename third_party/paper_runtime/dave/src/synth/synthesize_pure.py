#!/usr/bin/env python3
# Direct-sum mixture synthesis with clean references: two RMS-normalized utterances from
# different speakers are summed without simulated RIR or added noise, so the target
# residual is exactly zero. Transcripts are inherited from sibling .txt files of the
# source utterances (produce those with transcribe.py first if absent).
# Environment: N_PAIRS = number of pairs to produce (default 20000).
import os, json, random, re
import numpy as np, soundfile as sf
OUT = "/path/to/data/mbpure"
N_PAIRS = int(os.environ.get("N_PAIRS", "20000"))
random.seed(52)
# clean_sources.json: same format as in synthesize.py —
#   {"AliMeeting": ["/abs/path/x.wav", ...], "MISP": [...], "AISHELL-4": [...]}
c = json.load(open("/path/to/REAL-AVSE/ops/clean_sources.json"))
srcs = c["AliMeeting"] + c["MISP"]
def spk_of(p):
    m = re.search(r"(SPK\d+)", p)
    if m:
        return m.group(1)
    return os.path.basename(p).split("_")[0]
pool = []
for p in srcs:
    t = os.path.splitext(p)[0] + ".txt"
    try:
        i = sf.info(p)
        d = i.frames / i.samplerate
        if 1.0 <= d <= 15.0:
            pool.append((p, t, spk_of(p), d))
    except Exception:
        pass
print("source pool=%d" % len(pool), flush=True)
os.makedirs(OUT, exist_ok=True)
def fit(x, L):
    if len(x) >= L:
        st = random.randint(0, len(x) - L)
        return x[st:st + L]
    rep = (L + len(x) - 1) // len(x)
    return np.tile(x, rep)[:L]
made = 0
tries = 0
while made < N_PAIRS and tries < N_PAIRS * 6:
    tries += 1
    (pa, ta, sa, da), (pb, tb, sb, db) = random.sample(pool, 2)
    if sa == sb:
        continue
    if not (os.path.exists(ta) and os.path.exists(tb)):
        continue
    L = int(min(6.0, max(2.0, max(da, db))) * 16000)
    try:
        xa, _ = sf.read(pa, dtype="float32")
        xb, _ = sf.read(pb, dtype="float32")
        if xa.ndim > 1:
            xa = xa.mean(1)
        if xb.ndim > 1:
            xb = xb.mean(1)
        xa = fit(xa, L)
        xb = fit(xb, L)
        ra = np.sqrt((xa ** 2).mean()) + 1e-9
        rb = np.sqrt((xb ** 2).mean()) + 1e-9
        xa = xa / ra * 0.05
        xb = xb / rb * 0.05
        g_db = random.uniform(-3, 3)
        xb = xb * (10 ** (g_db / 20))
        mix = xa + xb
        pk = float(np.abs(mix).max())
        if pk > 0.99:
            sc = 0.99 / pk
            mix, xa, xb = mix * sc, xa * sc, xb * sc
        d = "%s/mp_%06d" % (OUT, made)
        os.makedirs(d, exist_ok=True)
        sf.write(d + "/mix.wav", mix.astype(np.float32), 16000, subtype="FLOAT")
        sf.write(d + "/s1.wav", xa.astype(np.float32), 16000, subtype="FLOAT")
        sf.write(d + "/s2.wav", xb.astype(np.float32), 16000, subtype="FLOAT")
        for src_t, dst in ((ta, d + "/s1.txt"), (tb, d + "/s2.txt")):
            with open(src_t, encoding="utf-8") as f:
                txt = f.read().strip()
            with open(dst, "w", encoding="utf-8") as f:
                f.write(txt + "\n")
        made += 1
        if made % 2000 == 0:
            print("PURE %d/%d" % (made, N_PAIRS), flush=True)
    except Exception as e:
        print("ERR", repr(e)[:40], flush=True)
print("PURE_SYNTH_DONE made=%d" % made, flush=True)
