# SAV-BandNet

SAV-BandNet extracts a visible target speaker from a six-channel audio mixture. It combines a frozen CMLR lip encoder, a visually conditioned IPDNet2 spatial encoder, TIGER band-split audio features, multimodal conditioning, and a learned mask-based multichannel Wiener filter.

This repository contains inference code and five demonstration examples. Model checkpoints and the full dataset are distributed separately and are **not included**. Checkpoint download links are not available in this repository yet.

## Installation

The verified environment is Linux, Python 3.11, PyTorch 2.7.1, CUDA 12.6, and Mamba 2.2.5. A compatible NVIDIA GPU is required for the spatial encoder. CPU-only execution and other CUDA versions have not been validated.

```bash
git clone https://github.com/vu-duy-tung/SAV-BandNet.git
cd SAV-BandNet
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install packaging ninja einops
python -m pip install mamba-ssm==2.2.5 --no-build-isolation
python -m pip install -e .
```

Mamba contains a compiled CUDA extension. If a compatible wheel is unavailable, installation requires a CUDA toolkit and C++ compiler compatible with the installed PyTorch version. This release was tested using the existing Mamba installation; a clean installation on other machines has not yet been verified.

## Inference

After obtaining the complete SAV-BandNet checkpoint, save it as `checkpoints/sav_bandnet.pt`. The checkpoint must contain `model` and `model_config`; the model state includes both pretrained encoders. Separate CMLR, IPDNet2, and TIGER checkpoints are not needed. Only load checkpoints from a trusted source.

Run the first example with:

```bash
python -m spatial_av_bandnet.infer \
  --checkpoint checkpoints/sav_bandnet.pt \
  --mixture demos/sample_01/mixture_6ch.wav \
  --mouths demos/sample_01/mouths.npz \
  --channel-ids 0 1 2 3 4 5 \
  --output outputs/sample_01.wav
```

Replace the two input paths to process downloaded data. WAV and FLAC audio are supported. The output is a mono, 16 kHz floating-point WAV with the same duration as the input. Inference preserves the model's output level; it does not apply peak normalization.

### Input requirements

- **Audio:** six synchronized channels sampled at 16 kHz. The training geometry is a horizontal circular array with a radius of 3.5 cm. Microphone 0 lies on the positive x-axis, and IDs increase counterclockwise. `--channel-ids` specifies the physical microphone ID of each stored audio channel. A different array geometry requires separate validation.
- **Video features:** aligned grayscale mouth crops, 96 × 96 pixels, at 25 frames per second. Supply an NPZ file containing `data` as `uint8 [frames,96,96]` and `valid` as a Boolean frame mask. An optional Boolean `present` mask is combined with `valid`. Frame centers are at 0.02, 0.06, 0.10 seconds, and so forth.
- **Alignment:** the audio and video must share the same scene time origin. Invalid or absent mouth frames must be marked in the masks. Full-face video must first be tracked and cropped around the selected speaker's mouth; automatic face tracking is outside this release. Do not use the compressed preview MP4 in place of the supplied NPZ when reproducing a demo.

For independently prepared crops:

```python
import numpy as np
np.savez_compressed("mouths.npz", data=mouth_crops, valid=valid_frames)
```

Inference uses four-second windows: 1.6 seconds of preceding context, 1.6 seconds of emitted output, and 0.8 seconds of following context. It pads scene boundaries and concatenates the central outputs. CMLR features are recomputed within each four-second context. This is offline inference with future context, not causal streaming. The saved checkpoint configuration determines the filtering and activity-gate behavior; the current model uses a Wiener filter and bypasses the activity gate.

## Demonstrations

These five selected examples illustrate successful extraction under different acoustic conditions. They are qualitative examples, not a representative performance summary. Each player shows the target mouth crops with either the mixture or the extracted speech. Mixture playback uses microphone 0; inference uses all six channels.

### Example 1: Reverberant room

**Input: target video and mixture audio**

https://github.com/user-attachments/assets/dd07b1ca-12e6-4694-ab66-b96853bb8e9f

**Output: target video and extracted speech**

https://github.com/user-attachments/assets/d6a64c27-e372-40fa-b699-d2bd2fe997b1

### Example 2: Office

**Input: target video and mixture audio**

https://github.com/user-attachments/assets/666e037f-4ad1-47de-9dd6-4057644bad79

**Output: target video and extracted speech**

https://github.com/user-attachments/assets/f9c1a942-5500-42c3-810a-861f282f8c0d

### Example 3: Outdoor scene

**Input: target video and mixture audio**

https://github.com/user-attachments/assets/0e7e2221-dcbb-41f4-a553-6f01407070b5

**Output: target video and extracted speech**

https://github.com/user-attachments/assets/70725e75-0bc5-4ebd-b8c2-26e9271ee294

### Example 4: Dry room

**Input: target video and mixture audio**

https://github.com/user-attachments/assets/d1e4ce63-0c5c-45d9-b644-7a5cfccacae0

**Output: target video and extracted speech**

https://github.com/user-attachments/assets/b607069d-aa09-4203-9808-7c9072eaeed4

### Example 5: Reverberant room

**Input: target video and mixture audio**

https://github.com/user-attachments/assets/41389dc0-df63-402c-830b-3e53a96c48bd

**Output: target video and extracted speech**

https://github.com/user-attachments/assets/c1cabfcf-eb8a-4476-9a7c-a1ecbca30b65

## Code and acknowledgments

`spatial_av_bandnet/` contains the model, alignment, conditioning, spatial filtering, and inference entry point. `third_party/` retains the architecture dependencies required by the complete checkpoint. The release does not contain training datasets, training checkpoints, optimizer states, ASR weights, or evaluation reports.

The implementation builds on CMLR/ESPnet, IPDNet2/Mamba, and TIGER. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained notices and release limitations. This is a private research snapshot; no blanket redistribution license is granted for the code or demo media.
