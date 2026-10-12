"""Frozen pretrained encoders. Original checkpoints stay outside this repository."""
from argparse import Namespace
from pathlib import Path
import importlib
import json
import sys
import torch
from torch import nn
from torch.nn import functional as F

from .alignment import align_visual

VENDOR = Path(__file__).resolve().parents[1]/"third_party"
TIGER_CHECKPOINT = None


class FrozenModule(nn.Module):
    def train(self, mode=True):
        # Parent model.train() must not reactivate dropout/batch statistics.
        return super().train(False)


class CMLREncoder(FrozenModule):
    """Mouth uint8 [B,V,96,96] -> normalized CMLR [B,V,256].

    Uses supplied clip context. This is NOT equivalent to full-donor cached context.
    Invalid pixels are excluded from clip luminance statistics and replaced by mean.
    """
    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(VENDOR/"cmlr"))
        from espnet.nets.pytorch_backend.transformer.encoder import Encoder
        a = Namespace(**json.loads((VENDOR/"cmlr/config.json").read_text()))
        attn = a.transformer_encoder_attn_layer_type
        if getattr(a, "rel_pos_type", None) is None and attn == "rel_mha":
            attn = "legacy_rel_mha"
        self.encoder = Encoder(
            idim=80, attention_dim=a.adim, attention_heads=a.aheads,
            linear_units=a.eunits, num_blocks=a.elayers,
            input_layer=a.transformer_input_layer, dropout_rate=a.dropout_rate,
            positional_dropout_rate=a.dropout_rate,
            attention_dropout_rate=a.dropout_rate if a.transformer_attn_dropout_rate is None else a.transformer_attn_dropout_rate,
            encoder_attn_layer_type=attn, macaron_style=a.macaron_style,
            use_cnn_module=a.use_cnn_module, cnn_module_kernel=a.cnn_module_kernel,
            zero_triu=getattr(a, "zero_triu", False), a_upsample_ratio=a.a_upsample_ratio,
            relu_type=getattr(a, "relu_type", "swish"))
        self.requires_grad_(False).eval()

    @staticmethod
    def preprocess(mouths, valid):
        if mouths.dtype != torch.uint8 or mouths.ndim != 4 or mouths.shape[-2:] != (96,96):
            raise ValueError("Expected grayscale uint8 mouth crops [B,V,96,96]")
        x = mouths.float()
        mask = valid[..., None, None]
        count = (mask.sum((1,2,3), keepdim=True)*96*96).clamp_min(1)
        mean = (x*mask).sum((1,2,3), keepdim=True)/count
        var = ((x-mean).square()*mask).sum((1,2,3), keepdim=True)/count
        calibrated = ((x-mean)*(255*.165/var.sqrt().clamp_min(1))+255*.421).clamp(0,255)
        # Match original numpy uint8 truncation before center crop.
        calibrated = calibrated.to(torch.uint8).float()
        calibrated = torch.where(mask, calibrated, torch.full_like(calibrated, 255*.421))
        return ((calibrated[...,4:92,4:92]/255-.421)/.165).unsqueeze(1)

    @torch.no_grad()
    def forward(self, mouths, valid):
        if valid.dtype != torch.bool or valid.shape != mouths.shape[:2]:
            raise ValueError("Mouth validity must be boolean [B,V]")
        features = self.encoder(self.preprocess(mouths, valid), None)[0]
        return F.layer_norm(features, (256,)).masked_fill(~valid[...,None], 0)


class SpatialEncoder(FrozenModule):
    """Strictly load selected MaskedAVNet and capture post-target features.

    Hooks expose tensors within the SAME forward; no extra inference or routing.
    """
    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(VENDOR/"ipdnet2_av_exp3"))
        from masked_model import MaskedAVNet
        self.net = MaskedAVNet(False)
        self._target = None
        self._visual = None
        self.net.target_inverse.register_forward_pre_hook(self._capture_target)
        self.net.film[0].register_forward_pre_hook(self._capture_visual)
        self.requires_grad_(False).eval()

    def _capture_target(self, module, args):
        self._target = args[0]

    def _capture_visual(self, module, args):
        self._visual = args[0]

    @torch.no_grad()
    def forward(self, wave, visual, valid, visual_times):
        # Complete the final 100ms cell rather than discarding the waveform tail.
        n = wave.shape[-1]
        frames = max(2, (n+1599)//1600)
        times = torch.arange(frames, device=wave.device, dtype=wave.dtype)*.1+.04
        visual, valid = align_visual(visual, valid, visual_times, times)
        valid = valid & (times < n/16000)[None]
        visual = visual.masked_fill(~valid[...,None], 0)
        padded = F.pad(wave, (0, frames*1600-n))
        phase, _ = self.net(padded, visual, valid)
        if phase.shape[1] != frames or self._target is None or self._visual is None:
            raise RuntimeError("Unexpected spatial timeline/feature interface")
        result = dict(
            phase=phase, hidden=self._target,
            visual=self._visual*valid[:,None], valid=valid, times=times,
            padded_samples=frames*1600-n)
        self._target = self._visual = None
        return result
