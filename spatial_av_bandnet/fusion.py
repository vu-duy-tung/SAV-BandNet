"""Local-context feature conditioning and a differentiable target-presence gate."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .alignment import interpolate


class ContextualFiLM(nn.Module):
    """486 modalities -> 256 latent -> local band/time context -> 512 scale/bias."""
    def __init__(self, inputs=486, channels=256, contextual=True):
        super().__init__()
        self.norm = nn.LayerNorm(inputs)
        self.input = nn.Linear(inputs, channels)
        self.context = (nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, padding_mode="replicate"),
            nn.SiLU(), nn.Conv2d(channels, channels, 1),
        ) if contextual else None)
        self.output = nn.Linear(channels, 2*channels)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs):
        # B,K,L,input_dim. Fine audio and frozen AV encoders already have context;
        # this small residual additionally relates the aligned modalities locally.
        h = F.silu(self.input(self.norm(inputs)))
        if self.context is not None:
            h = h+.1*self.context(h.permute(0,3,1,2)).permute(0,2,3,1)
        return self.output(h)


class ActivityGate(nn.Module):
    """Predict 10ms activity from target-conditioned fine features.

    No visual-validity multiplier: missing video is not target absence.
    The gate is smoothed in probability space, then interpolated to audio samples.
    """
    def __init__(self, channels=256, hidden=64, initial_open=.98):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.temporal = nn.Conv1d(channels, hidden, 3, padding=1, padding_mode="replicate")
        self.output = nn.Conv1d(hidden, 1, 1)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.constant_(self.output.bias, math.log(initial_open/(1-initial_open)))

    def forward(self, refined):
        # Refined B,K,D,T already mixes audio, spatial and visual information.
        h = self.norm(refined.mean(1).transpose(1,2)).transpose(1,2)
        logits = self.output(F.silu(self.temporal(h)))[:,0]
        probability = logits.sigmoid()
        smoothed = F.avg_pool1d(F.pad(probability[:,None],(2,2),mode="replicate"),5,stride=1)[:,0]
        return logits, smoothed


def apply_activity_gate(waveform, frame_gate, frame_times, sample_rate=16000):
    """Apply a bounded gain after iSTFT, outside MVDR's covariance normalization."""
    sample_times = torch.arange(waveform.shape[-1],device=waveform.device,dtype=waveform.dtype)/sample_rate
    sample_gate = interpolate(frame_gate, frame_times, sample_times)
    return waveform*sample_gate
