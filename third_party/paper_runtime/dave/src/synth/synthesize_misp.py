#!/usr/bin/env python3
# Real-acoustic mixture synthesis: far-field single-speaker MISP segments from two
# different speakers are summed directly, without simulated RIR or added noise — the
# source material already carries real room reverberation and background noise.
# Segment filenames are expected to follow "<session>_<speaker>_*.wav" (the speaker key
# is taken from the first two underscore-separated fields).
# Transcripts: run transcribe.py over the output directories afterwards.
# Environment: N_PAIRS = number of pairs to produce (default 20000).
import os, glob, random, json
import numpy as np, soundfile as sf
SRC = "/path/to/data/misp/far_field_segments"  # single-speaker far-field utterance wavs cut from MISP-Meeting (under its agreement)
OUT = "/path/to/data/misp_mix"
N_PAIRS = int(os.environ.get("N_PAIRS", "20000"))
random.seed(42)
wavs = sorted(glob.glob(SRC + "/*.wav"))
pool = []
for p in wavs:
    try:
        i = sf.info(p)
        d = i.frames / i.samplerate
        if 1.0 <= d <= 8.0:
            name = os.path.basename(p)[:-4]
            parts = name.split("_")
            spk = "_".join(parts[:2])
            pool.append((p, spk, d))
    except Exception:
        pass
print("usable segments=%d" % len(pool), flush=True)
by_spk = {}
for p, s, d in pool:
    by_spk.setdefault(s, []).append((p, d))
spks = sorted(by_spk)
print("speakers=%d" % len(spks), flush=True)
os.makedirs(OUT, exist_ok=True)
def fit(x, L):
    if len(x) >= L:
        st = random.randint(0, len(x) - L)
        return x[st:st + L]
    rep = (L + len(x) - 1) // len(x)
    return np.tile(x, rep)[:L]
made = 0
manifest = []
tries = 0
while made < N_PAIRS and tries < N_PAIRS * 5:
    tries += 1
    sa, sb = random.sample(spks, 2)
    pa, da = random.choice(by_spk[sa])
    pb, db = random.choice(by_spk[sb])
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
        g_db = random.uniform(-3, 3)
        xb = xb * (10 ** (g_db / 20))
        mix = xa + xb
        pk = float(np.abs(mix).max())
        if pk > 0.99:
            sc = 0.99 / pk
            mix, xa, xb = mix * sc, xa * sc, xb * sc
        d = "%s/mm_%06d" % (OUT, made)
        os.makedirs(d, exist_ok=True)
        sf.write(d + "/mix.wav", mix, 16000, subtype="FLOAT")
        sf.write(d + "/s1.wav", xa.astype(np.float32), 16000, subtype="FLOAT")
        sf.write(d + "/s2.wav", xb.astype(np.float32), 16000, subtype="FLOAT")
        manifest.append((d, os.path.basename(pa)[:-4], os.path.basename(pb)[:-4]))
        made += 1
        if made % 2000 == 0:
            print("SYNTH %d/%d" % (made, N_PAIRS), flush=True)
    except Exception as e:
        print("ERR", repr(e)[:40], flush=True)
json.dump(manifest, open(OUT + "/manifest.json", "w"))
print("MISP_SYNTH_DONE made=%d" % made, flush=True)
