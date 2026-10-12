"""Spatial-AV-BandNet: fine -> coarse multimodal fusion -> fine masks -> MVDR."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from third_party.tiger_models.tiger import TIGER
from .alignment import interpolate, TemporalCompression
from .beamforming import LocalMVDR, StableSpatialFilter
from .fusion import ContextualFiLM, ActivityGate, apply_activity_gate
from .encoders import CMLREncoder, SpatialEncoder, TIGER_CHECKPOINT


@dataclass
class ModelConfig:
    sample_rate: int = 16000
    channels: int = 6
    n_fft: int = 640
    hop: int = 160
    pre_iterations: int = 6
    post_iterations: int = 6
    covariance_frames: int = 101  # provisional 1s centered context; not tuned
    gradient_checkpointing: bool = True
    compression_mode: str = "conv"  # "pool" is the original compression ablation
    contextual_fusion: bool = True
    activity_gate: bool = True
    bypass_activity_gate: bool = False
    spatial_filter: str = "legacy_mvdr"

    def __post_init__(self):
        if (self.sample_rate,self.channels,self.n_fft,self.hop) != (16000,6,640,160):
            raise ValueError("Checkpoint-compatible prototype requires 16kHz/6ch/640FFT/160hop")
        if min(self.pre_iterations,self.post_iterations) < 1:
            raise ValueError("Fine processing is required before and after fusion")


class SpatialAVBandNet(nn.Module):
    def __init__(self, config=None, tiger_checkpoint=TIGER_CHECKPOINT,
                 spatial_encoder=None, lip_encoder=None):
        super().__init__()
        self.config = config or ModelConfig()
        tiger = TIGER(sample_rate=16000, in_channels=512, out_channels=256,
                      num_blocks=12, upsampling_depth=5, win=640, stride=160, num_sources=2)
        if tiger_checkpoint is not None:
            ck = torch.load(tiger_checkpoint, map_location="cpu", weights_only=True)
            tiger.load_state_dict(ck.get("state_dict", ck), strict=True)
        self.band_encoder = tiger.BN
        self.trunk = tiger.separator  # Shared pretrained recurrent blocks, not two separators.
        # The selected upstream UConvBlock computes but never consumes its deepest
        # local/global fusion output. Preserve checkpoint/forward parity; exclude
        # these dormant weights from the trainable set instead of silently fixing TIGER.
        for path in (self.trunk.freq_path, self.trunk.frame_path):
            path[0].loc_glo_fus[-1].requires_grad_(False)
        self.band_widths = list(map(int, tiger.band_width))
        self.spatial_encoder = spatial_encoder if spatial_encoder is not None else SpatialEncoder()
        self.lip_encoder = lip_encoder if lip_encoder is not None else CMLREncoder()
        self.compression = TemporalCompression(256, self.config.compression_mode)
        # Per-band input: audio256, spatial96, visual128, five pair scores + validity.
        self.conditioner = ContextualFiLM(contextual=self.config.contextual_fusion)
        self.activity_gate = ActivityGate() if self.config.activity_gate else None
        self.skip_merge = nn.Conv2d(512,256,1)
        # Preserve a direct fine-feature path at initialization; keep coarse-path
        # influence nonzero so every new connection receives first-step gradients.
        with torch.no_grad():
            self.skip_merge.weight.zero_()
            self.skip_merge.weight[:, :256, 0, 0].copy_(torch.eye(256))
            self.skip_merge.weight[:, 256:, 0, 0].copy_(.1*torch.eye(256))
            self.skip_merge.bias.zero_()
        self.mask_heads = nn.ModuleList([nn.Conv1d(256,2*w,1) for w in self.band_widths])
        self.beamformer = (LocalMVDR(self.config.covariance_frames) if self.config.spatial_filter == "legacy_mvdr"
                           else StableSpatialFilter(self.config.covariance_frames,self.config.spatial_filter))
        self.register_buffer("window", torch.hann_window(640))
        # Nominal geometry is the training contract; IDs must be explicitly supplied.
        angle = torch.arange(6)*torch.pi/3
        self.register_buffer("microphone_positions_m", torch.stack(
            [.035*angle.cos(), .035*angle.sin(), torch.zeros(6)],-1))
        fine_hz = torch.arange(321)*25.
        spatial_hz = torch.arange(1,257)*31.25
        self.register_buffer("fine_hz",fine_hz)
        self.register_buffer("spatial_hz",spatial_hz)
        # Map each fine frequency to a spatial hidden bin by its physical interval.
        hidden_index = ((fine_hz/31.25).ceil().long()-1).clamp(0,255)//16
        hidden_map = F.one_hot(hidden_index,16).float()
        hidden_map[0].zero_()  # spatial model has no DC bin
        band_map = torch.zeros(len(self.band_widths),321)
        offset = 0
        for k,width in enumerate(self.band_widths):
            band_map[k,offset:offset+width] = 1/width
            offset += width
        self.register_buffer("band_map",band_map)
        self.register_buffer("hidden_map", band_map@hidden_map)
        self.initialization = dict(tiger_checkpoint=str(tiger_checkpoint), config=asdict(self.config))

    def _trunk(self, x, iterations):
        b,k,d,t = x.shape
        x = x.permute(0,2,1,3).contiguous()
        mixture = x
        for i in range(iterations):
            if i:
                x = self.trunk.concat_block(mixture+x)
            if self.config.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(self.trunk.freq_time_process,x,b,k,d,t,use_reentrant=False)
            else:
                x = self.trunk.freq_time_process(x,b,k,d,t)
        return x.permute(0,2,1,3).contiguous()

    def _spatial_features(self, spec, fine_times, spatial):
        times = spatial["times"]
        # Predicted phase is [B,L,cos256+sin256,5], convention X0*conj(Xm).
        real,imag = spatial["phase"].split(256,dim=2)
        predicted = torch.complex(real,imag).permute(0,3,1,2)  # B,5,L,Fssl
        predicted = interpolate(predicted,self.spatial_hz,self.fine_hz)
        predicted = predicted.permute(0,1,3,2)  # B,5,F,L
        predicted = interpolate(predicted,times,fine_times)
        cross = spec[:,0:1]*spec[:,1:].conj()
        observed = cross/cross.abs().clamp_min(1e-8)
        # Keep prediction amplitude (uncertainty); never hard-decode an angle.
        compatibility = (observed*predicted.conj()).real
        compatibility = compatibility.clone()
        compatibility[:,:,0] = 0
        band_scores = torch.einsum("kf,bpft->bkpt",self.band_map,compatibility)
        scores = self.compression_scores(band_scores,fine_times,times)
        hidden = torch.einsum("kh,bdlh->bkdl",self.hidden_map,spatial["hidden"])
        visual = spatial["visual"][:,None].expand(-1,len(self.band_widths),-1,-1)
        valid = spatial["valid"][:,None,None].expand(-1,len(self.band_widths),1,-1)
        return hidden,visual,scores,valid

    @staticmethod
    def compression_scores(scores,fine_times,times):
        offsets = (torch.arange(10,device=scores.device,dtype=scores.dtype)-4.5)*.01
        values = interpolate(scores,fine_times,(times[:,None]+offsets).flatten())
        return values.reshape(*scores.shape[:-1],len(times),10).mean(-1)

    @torch.no_grad()
    def encode_scene_visual(self, mouths, valid, times=None):
        """Shared raw/cache entry point: encode precisely the supplied scene window.

        Cache this result for the SAME window, rather than slicing features
        encoded from a longer donor. The marker declares provenance; it cannot
        detect a dishonest or incorrectly labeled external cache.
        """
        if times is None:
            times = torch.arange(mouths.shape[1],device=mouths.device)*.04+.02
        return dict(visual_features=self.lip_encoder(mouths,valid),
                    visual_valid=valid, visual_times=times,
                    visual_context="scene_window_v1")

    def forward(self, wave, *, channel_ids, mouths=None, mouth_valid=None,
                visual_features=None, visual_valid=None, visual_times=None,
                visual_context=None, return_details=False):
        """One target query; unpadded equal-length batches, offline.

        Supply raw uint8 mouths OR cached normalized CMLR features with timestamps.
        channel_ids describes the canonical physical ID of each input channel.
        """
        if wave.ndim != 3 or wave.shape[1] != 6 or wave.shape[-1] < 640:
            raise ValueError("Expected audio [B,6,N], N>=640 at 16kHz")
        if wave.dtype != torch.float32 or not torch.isfinite(wave).all():
            raise ValueError("Audio must be finite float32")
        if sorted(channel_ids) != list(range(6)):
            raise ValueError("Explicit channel_ids must contain canonical IDs 0..5 once")
        order = torch.tensor(channel_ids,device=wave.device).argsort()
        wave = wave[:,order]
        if (mouths is None) == (visual_features is None):
            raise ValueError("Supply exactly one of raw mouths or CMLR features")
        if visual_context not in (None, "scene_window_v1"):
            raise ValueError("Unsupported visual context: encode the scene window, not the donor")
        if mouths is not None:
            if mouth_valid is None:
                raise ValueError("Supply mouth validity")
            encoded_visual = self.encode_scene_visual(mouths,mouth_valid,visual_times)
            visual_features = encoded_visual["visual_features"]
            visual_valid = encoded_visual["visual_valid"]
            visual_times = encoded_visual["visual_times"]
        elif visual_times is None or visual_context != "scene_window_v1":
            raise ValueError("Cached features require timestamps and visual_context='scene_window_v1'")
        if visual_valid is None or visual_valid.dtype != torch.bool:
            raise ValueError("Supply boolean visual validity")
        if visual_features.shape[0] != wave.shape[0] or visual_features.shape[-1] != 256:
            raise ValueError("CMLR features must be [B,V,256]")
        if not torch.isfinite(visual_features[visual_valid]).all():
            raise ValueError("Valid visual features must be finite")
        spatial = self.spatial_encoder(wave,visual_features,visual_valid,visual_times)
        b,_,n = wave.shape
        spec = torch.stft(wave.reshape(b*6,n),640,160,window=self.window,
                          center=True,return_complex=True).reshape(b,6,321,-1)
        times = torch.arange(spec.shape[-1],device=wave.device,dtype=wave.dtype)*.01
        ri = torch.stack([spec[:,0].real,spec[:,0].imag],1)
        encoded=[]
        for encoder,band in zip(self.band_encoder,ri.split(self.band_widths,dim=2)):
            encoded.append(encoder(band.flatten(1,2)))
        fine = self._trunk(torch.stack(encoded,1),self.config.pre_iterations)
        coarse = self.compression(fine,times,spatial["times"])
        hidden,visual,scores,valid = self._spatial_features(spec,times,spatial)
        inputs = torch.cat([coarse,hidden,visual,scores,valid.to(wave.dtype)],dim=2)
        gain,bias = self.conditioner(inputs.permute(0,1,3,2)).permute(0,1,3,2).chunk(2,dim=2)
        fused = coarse*(1+.25*gain.tanh())+.25*bias.tanh()
        restored = interpolate(fused,spatial["times"],times)
        merged = self.skip_merge(torch.cat([fine,restored],2).permute(0,2,1,3))
        refined = self._trunk(merged.permute(0,2,1,3),self.config.post_iterations)
        logits = torch.cat([head(refined[:,k]).reshape(b,2,w,-1)
                            for k,(head,w) in enumerate(zip(self.mask_heads,self.band_widths))],2)
        masks = logits.softmax(1)
        enhanced,weights = self.beamformer(spec,masks)
        ungated = torch.istft(enhanced,640,160,window=self.window,center=True,length=n)
        activity_logits = torch.zeros(b,len(times),device=wave.device,dtype=wave.dtype)
        gate = torch.ones(b,len(times),device=wave.device,dtype=wave.dtype)
        if self.activity_gate is not None and not self.config.bypass_activity_gate:
            activity_logits,gate = self.activity_gate(refined)
        output = ungated if self.config.bypass_activity_gate else apply_activity_gate(ungated,gate,times,self.config.sample_rate)
        if not return_details:
            return output
        return dict(waveform=output,ungated_waveform=ungated,gate_bypassed=self.config.bypass_activity_gate,
                    activity_logits=activity_logits,activity_gate=gate,
                    visual_context="scene_window_v1",
                    masks=masks,mask_logits=logits,weights=weights,spatial=spatial,
                    fine_times=times,fine_shape=tuple(fine.shape),
                    fusion_shape=tuple(fused.shape),refined_shape=tuple(refined.shape))
