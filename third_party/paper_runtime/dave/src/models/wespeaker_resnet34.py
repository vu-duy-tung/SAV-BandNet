"""Standalone WeSpeaker ResNet34 speaker-embedding extractor.

The architecture is lifted verbatim from WeSpeaker
(``wespeaker/models/resnet.py`` + ``pooling_layers.py``) so the pretrained
``cnceleb-resnet34-LM`` checkpoint loads with matching keys, WITHOUT installing
the ``wespeaker`` package (its ``__init__`` pulls heavy deps like silero_vad).

Pretrained config (cnceleb v2 ``resnet_lm.yaml``):
    ResNet34, feat_dim=80, embed_dim=256, pooling_func='TSTP', two_emb_layer=False
    fbank: num_mel_bins=80, frame_length=25ms, frame_shift=10ms

This module exposes :class:`WeSpeakerResNet34`, which wraps the network with
the matching kaldi-fbank front end + CMN so callers can go straight from a
16 kHz waveform to a 256-d L2-normalized speaker embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi


# ---------------------------------------------------------------------------
# Pooling: Temporal Statistics Pooling (mean + std), as in WeSpeaker.
# ---------------------------------------------------------------------------

class TSTP(nn.Module):
    """Temporal statistics pooling: concatenate temporal mean and std.

    Output dim is ``2 * in_dim`` (the flattened (C*F) channel-frequency map).
    """

    def __init__(self, in_dim=0, **kwargs):
        super().__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # x: (B, C, F, T) -- last axis is time.
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        return torch.cat((pooling_mean, pooling_std), 1)

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim


# ---------------------------------------------------------------------------
# ResNet (WeSpeaker variant: 3x3 stem, no maxpool, m_channels=32).
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """WeSpeaker ResNet backbone -> TSTP -> linear embedding.

    Returns ``(dummy, embedding)``; with ``two_emb_layer=False`` the first
    element is a scalar placeholder, matching the upstream signature so the
    pretrained state_dict keys (``seg_1``, ``pool`` ...) line up exactly.
    """

    def __init__(self, block, num_blocks, m_channels=32, feat_dim=80,
                 embed_dim=256, pooling_func="TSTP", two_emb_layer=False):
        super().__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)

        self.pool = TSTP(in_dim=self.stats_dim * block.expansion)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, F) fbank -> (B, F, T) -> (B, 1, F, T)
        x = x.permute(0, 2, 1).unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        stats = self.pool(out)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        return torch.tensor(0.0), embed_a


def _resnet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], feat_dim=feat_dim,
                  embed_dim=embed_dim, pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


# ---------------------------------------------------------------------------
# High-level extractor: waveform -> L2-normalized speaker embedding.
# ---------------------------------------------------------------------------

class WeSpeakerResNet34(nn.Module):
    """Waveform-to-embedding wrapper around the WeSpeaker ResNet34 backbone.

    Front end matches the WeSpeaker recipe: 80-dim kaldi fbank (25/10 ms,
    no dither at inference), sentence-level cepstral mean normalization (CMN),
    then ResNet34 + TSTP -> 256-d embedding (L2-normalized for cosine scoring).

    Args:
        ckpt_path: path to ``avg_model.pt`` (raw torch state_dict). If None, the
            network is left randomly initialized (useful only for shape checks).
        feat_dim/embed_dim: must match the checkpoint (80 / 256 for cnceleb-LM).
        sample_rate: input waveform rate (must be 16 kHz for the fbank front end).
    """

    def __init__(self, ckpt_path=None, feat_dim=80, embed_dim=256,
                 sample_rate=16000):
        super().__init__()
        if sample_rate != 16000:
            raise ValueError(f"WeSpeaker front end expects 16 kHz, got {sample_rate}")
        self.sample_rate = sample_rate
        self.feat_dim = feat_dim
        self.resnet = _resnet34(feat_dim=feat_dim, embed_dim=embed_dim,
                                pooling_func="TSTP", two_emb_layer=False)
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, ckpt_path):
        """Load the WeSpeaker ``avg_model.pt`` state_dict into the backbone.

        Upstream checkpoints store the bare model state_dict (sometimes nested
        under ``state_dict`` / ``model``, and occasionally with a ``projection.``
        ArcMargin head that we drop). Keys are matched leniently.
        """
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        # Drop the ArcMargin projection head (training-only) and any prefix.
        clean = {}
        for k, v in sd.items():
            if k.startswith("projection") or "projection." in k:
                continue
            clean[k] = v
        missing, unexpected = self.resnet.load_state_dict(clean, strict=False)
        # The ArcMargin head is expected to be absent; warn only on real gaps.
        real_missing = [m for m in missing if "num_batches_tracked" not in m]
        if real_missing:
            print(f"[WeSpeakerResNet34] missing keys ({len(real_missing)}): "
                  f"{real_missing[:8]}")
        if unexpected:
            print(f"[WeSpeakerResNet34] unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:8]}")
        return self

    def compute_fbank(self, wav):
        """16 kHz waveform -> (T, 80) CMN-normalized fbank features.

        Args:
            wav: 1-D float tensor in [-1, 1] (or (1, T)). Kaldi expects 16-bit
                scale, so it is multiplied by 32768 internally here.
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # (1, T)
        feat = kaldi.fbank(
            wav * (1 << 15),
            num_mel_bins=self.feat_dim,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=self.sample_rate,
            window_type="hamming",
            use_energy=False,
        )
        # Sentence-level cepstral mean normalization.
        feat = feat - feat.mean(dim=0, keepdim=True)
        return feat

    @torch.no_grad()
    def forward(self, wav):
        """16 kHz waveform -> (256,) L2-normalized speaker embedding."""
        feat = self.compute_fbank(wav).unsqueeze(0)  # (1, T, 80)
        feat = feat.to(next(self.resnet.parameters()).device)
        _, emb = self.resnet(feat)
        emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0)
