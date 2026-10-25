"""Timestamp alignment; all times are seconds from the same audio origin."""
import torch
from torch import nn


def interpolate(x, source, target):
    """Linear interpolation on the last axis, with endpoint extension.

    Feature interpolation only: never use this to resample microphone audio.
    """
    if source.ndim != 1 or target.ndim != 1 or x.shape[-1] != source.numel():
        raise ValueError("Expected one shared, ordered timestamp grid per batch")
    if not source.numel() or (source[1:] <= source[:-1]).any():
        raise ValueError("Source timestamps must be nonempty and strictly increasing")
    if source.numel() == 1:
        return x.expand(*x.shape[:-1], target.numel())
    right = torch.searchsorted(source.contiguous(), target.contiguous()).clamp(1, len(source)-1)
    left = right-1
    alpha = ((target-source[left])/(source[right]-source[left])).clamp(0, 1)
    return x[..., left]*(1-alpha)+x[..., right]*alpha


def align_visual(features, valid, source_times, target_times):
    """Require both interpolation endpoints to be valid; out-of-range is missing."""
    if features.ndim != 3 or valid.shape != features.shape[:2]:
        raise ValueError("Visual features [B,V,D], validity [B,V]")
    source_times = source_times.to(device=features.device, dtype=features.dtype)
    target_times = target_times.to(device=features.device, dtype=features.dtype)
    # Invalid NaNs must never poison the frozen encoder/normalization.
    features = torch.where(valid[..., None], features, torch.zeros_like(features))
    aligned = interpolate(features.transpose(1, 2), source_times, target_times).transpose(1, 2)
    right = torch.searchsorted(source_times, target_times).clamp(0, len(source_times)-1)
    exact = (source_times[right]-target_times).abs() < 1e-6
    left = torch.where(exact, right, (right-1).clamp_min(0))
    mask = valid[:, left] & valid[:, right]
    mask = mask & (target_times >= source_times[0]-1e-6) & (target_times <= source_times[-1]+1e-6)
    return aligned.masked_fill(~mask[..., None], 0), mask


class TemporalCompression(nn.Module):
    """Nonlinear temporal/channel mixing before timestamp-centered pooling."""
    def __init__(self, channels, mode="conv"):
        super().__init__()
        if mode not in ("conv", "pool"):
            raise ValueError("Compression mode must be conv or pool")
        self.temporal = (nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2, groups=channels, padding_mode="replicate"),
            nn.SiLU(), nn.Conv1d(channels, channels, 1),
        ) if mode == "conv" else None)
        self.offsets = nn.Parameter(torch.zeros(channels, 10))

    def forward(self, fine, fine_times, coarse_times):
        if self.temporal is not None:
            b,k,d,t = fine.shape
            fine = fine+.1*self.temporal(fine.reshape(b*k,d,t)).reshape(b,k,d,t)
        # [B,K,D,T] -> ten 10ms samples centered on each 100ms cell.
        offsets = (torch.arange(10, device=fine.device, dtype=fine.dtype)-4.5)*.01
        query = coarse_times[:, None]+offsets[None]
        sampled = interpolate(fine, fine_times, query.flatten())
        sampled = sampled.reshape(*fine.shape[:-1], len(coarse_times), 10)
        weights = self.offsets.softmax(-1)[None, None, :, None, :]
        return (sampled*weights).sum(-1)
