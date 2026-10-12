"""Offline inference with the same four-second context used in evaluation."""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .model import SpatialAVBandNet, ModelConfig


def crop(x, start, length):
    out = np.zeros((length, *x.shape[1:]), dtype=x.dtype)
    lo, hi = max(0, start), min(len(x), start + length)
    if hi > lo:
        out[lo-start:hi-start] = x[lo:hi]
    return out


def load_model(checkpoint, device='cuda'):
    # Only load checkpoints obtained from a trusted source.
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'model_config' not in ck or 'model' not in ck:
        raise ValueError('Expected a complete SAV-BandNet checkpoint with model/model_config')
    model = SpatialAVBandNet(ModelConfig(**ck['model_config']), tiger_checkpoint=None)
    model.load_state_dict(ck['model'], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def extract(model, audio, mouths, valid, channel_ids, device='cuda'):
    output = np.zeros(len(audio), np.float32)
    for central in range(0, len(audio), 25600):
        start = central - 25600
        wave = torch.from_numpy(crop(audio, start, 64000).T.copy())[None].to(device)
        lip = torch.from_numpy(crop(mouths, start//640, 100))[None].to(device)
        mask = torch.from_numpy(crop(valid, start//640, 100))[None].to(device)
        pred = model(wave, channel_ids=channel_ids, mouths=lip, mouth_valid=mask)
        end = min(central + 25600, len(audio))
        output[central:end] = pred[0, 25600:25600+end-central].cpu().numpy()
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mixture', required=True, help='Six-channel 16 kHz WAV or FLAC')
    p.add_argument('--mouths', required=True, help='NPZ: uint8 data [V,96,96], bool valid/present [V], 25 fps')
    p.add_argument('--channel-ids', required=True, type=int, nargs=6)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    audio, sr = sf.read(a.mixture, dtype='float32', always_2d=True)
    if sr != 16000 or audio.shape[1] != 6 or len(audio) == 0 or not np.isfinite(audio).all():
        p.error('Audio must be finite, nonempty, six-channel, 16 kHz')
    if sorted(a.channel_ids) != list(range(6)):
        p.error('Channel IDs must contain 0 through 5 exactly once')
    with np.load(a.mouths, allow_pickle=False) as z:
        mouths, valid = z['data'], z['valid']
        if 'present' in z:
            valid = valid & z['present']
    if mouths.dtype != np.uint8 or mouths.shape[1:] != (96,96) or valid.dtype != bool or valid.shape != (len(mouths),):
        p.error('Invalid mouth crop or validity shape/dtype')
    if abs(len(mouths)/25 - len(audio)/16000) > .041:
        p.error('Audio and mouth sequence durations differ by more than one video frame')
    model = load_model(a.checkpoint, a.device)
    result = extract(model, audio, mouths, valid, a.channel_ids, a.device)
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite extraction output')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.output, result, 16000, subtype='FLOAT')
    print(f'Saved {len(result)/16000:.2f} seconds to {a.output}')


if __name__ == '__main__':
    main()
