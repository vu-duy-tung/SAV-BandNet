#!/usr/bin/env python3
"""REAL-AVSE separation inference.

Separate a two-speaker mixture into two anonymous streams with the audio-only
TIGER-M backbone (2.56M). Self-contained: only torch, numpy, soundfile required.

Usage:
    python inference.py --mixture path/to/mix.wav --out output
    python inference.py --mixture path/to/dir_of_wavs --out output --device cuda
"""
import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from models import TIGER  # noqa: E402

TIGER_M_CFG = dict(
    sample_rate=16000, in_channels=512, out_channels=256,
    num_blocks=12, upsampling_depth=5, win=640, stride=160, num_sources=2,
)


def load_model(ckpt_path, device):
    model = TIGER(**TIGER_M_CFG)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def separate(model, wav_path, out_dir, device):
    mix, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        sys.exit("%s: expected 16 kHz audio, got %d Hz — resample first" % (wav_path, sr))
    if mix.ndim > 1:
        mix = mix.mean(1)
    with torch.no_grad():
        est = model(torch.from_numpy(mix[None]).to(device))
    name = os.path.splitext(os.path.basename(wav_path))[0]
    od = os.path.join(out_dir, name)
    os.makedirs(od, exist_ok=True)
    sf.write(os.path.join(od, "s1.wav"), est[0, 0].float().cpu().numpy(), sr, subtype="FLOAT")
    sf.write(os.path.join(od, "s2.wav"), est[0, 1].float().cpu().numpy(), sr, subtype="FLOAT")
    return od


def main():
    ap = argparse.ArgumentParser(
        description="REAL-AVSE: separate a two-speaker mixture into two streams "
                    "(audio-only TIGER-M, 2.56M)."
    )
    ap.add_argument("--mixture", required=True, help="a mixture .wav, or a directory of .wav files")
    ap.add_argument("--ckpt", default="model/real_avse_tiger_m.ckpt", help="model checkpoint")
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = load_model(args.ckpt, args.device)
    if os.path.isdir(args.mixture):
        wavs = sorted(glob.glob(os.path.join(args.mixture, "*.wav")))
    else:
        wavs = [args.mixture]
    if not wavs:
        sys.exit("No .wav found at %s" % args.mixture)
    for w in wavs:
        od = separate(model, w, args.out, args.device)
        print("%s -> %s/{s1,s2}.wav" % (w, od))


if __name__ == "__main__":
    main()
