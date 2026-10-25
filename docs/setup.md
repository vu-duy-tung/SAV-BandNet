# Environment

Use Python 3.11, a CUDA-enabled PyTorch build, and a compatible C++/CUDA compiler. From this repository:

```bash
pip install -e .
pip install mamba-ssm==2.2.5 --no-build-isolation
hf download playwithmino/SAV-BandNet --local-dir assets/default
hf download playwithmino/SAVE --repo-type dataset --local-dir "$SAVE"
```

The Mamba extension must be compiled for the installed PyTorch/CUDA versions. The tested server environment is recorded in the release verification receipt; portability to a different CUDA toolchain is not established by that check. Vendored model sources are in `third_party/paper_runtime`; their notices apply separately from trained weights and dataset assets.

Use `python -m sav_bandnet train --data "$SAVE" --output "$NAS/training/sav-bandnet" --check` to validate bundle hashes and split counts without initializing a GPU. Inference supports disjoint `--rank` / `--world` shards, each on a separately selected `CUDA_VISIBLE_DEVICES` device.
