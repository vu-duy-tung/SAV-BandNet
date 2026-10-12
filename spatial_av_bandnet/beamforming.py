"""Differentiable local Souden MVDR. No eigendecomposition or CPU routing."""
import torch
from torch import nn


class LocalMVDR(nn.Module):
    def __init__(self, window_frames=101, diagonal_loading=1e-3, absolute_loading=1e-8):
        super().__init__()
        if window_frames < 3 or window_frames % 2 != 1:
            raise ValueError("Covariance window must be odd and >= 3")
        self.window_frames = window_frames
        self.diagonal_loading = diagonal_loading
        self.absolute_loading = absolute_loading

    def forward(self, spectrum, masks):
        # spectrum [B,C,F,T], masks [B,2,F,T]; calculations in complex64/128.
        if masks.shape != (spectrum.shape[0], 2, *spectrum.shape[2:]):
            raise ValueError("Masks must match the original STFT frequency/time grid")
        x = spectrum.permute(0, 2, 3, 1)  # B,F,T,C
        outer = x[..., :, None]*x[..., None, :].conj()
        t = x.shape[2]
        index = torch.arange(t, device=x.device)
        lo = (index-self.window_frames//2).clamp_min(0)
        hi = (index+self.window_frames//2+1).clamp_max(t)

        def rolling(a):
            prefix = torch.cat([torch.zeros_like(a[:, :, :1]), a.cumsum(2)], 2)
            return prefix[:, :, hi]-prefix[:, :, lo]

        covariances = []
        for k in range(2):
            mask = masks[:, k]
            count = rolling(mask).clamp_min(1e-8)
            covariance = rolling(outer*mask[..., None, None])/count[..., None, None]
            # Limit cancellation asymmetry from prefix sums.
            covariances.append((covariance+covariance.transpose(-1, -2).conj())*.5)
        target, noise = covariances
        eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
        power = noise.diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(0)
        noise = noise+(self.diagonal_loading*power+self.absolute_loading)[..., None, None]*eye
        matrix = torch.linalg.solve(noise, target)
        trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1).real
        weights = matrix[..., :, 0]/trace.clamp_min(1e-8)[..., None]
        # A genuinely empty target covariance gives zero output, not a mic0 fallback.
        weights = torch.where((trace > 1e-8)[..., None], weights, torch.zeros_like(weights))
        enhanced = (weights.conj()*x).sum(-1)
        return enhanced, weights


def direct_sum(a, width):
    """Local summation without subtracting large cumulative totals."""
    from torch.nn import functional as F
    x = a.movedim(2, -1); shape = x.shape
    def pool(v):
        return (F.avg_pool1d(v.reshape(-1,1,shape[-1]), width, stride=1,
                padding=width//2, count_include_pad=True)*width).reshape(shape)
    out = torch.complex(pool(x.real),pool(x.imag)) if x.is_complex() else pool(x)
    return out.movedim(-1,2)


class StableSpatialFilter(LocalMVDR):
    """Double-precision direct covariance sums; Souden or physical-power Wiener."""
    def __init__(self, window_frames=21, kind="wiener"):
        super().__init__(window_frames)
        if kind not in ("wiener","stable_mvdr"): raise ValueError(kind)
        self.kind = kind

    def forward(self, spectrum, masks):
        if masks.shape != (spectrum.shape[0],2,*spectrum.shape[2:]):
            raise ValueError("Masks must match spectrum")
        # Frequency chunks bound covariance/autograd allocations. No CPU routing.
        outputs=[]; weights=[]
        for spec,m in zip(spectrum.split(32,2),masks.split(32,2)):
            x=spec.to(torch.complex128).permute(0,2,3,1)
            m=m.double(); width=self.window_frames
            outer=x[..., :,None]*x[...,None,:].conj()
            def hermitian(c): return (c+c.transpose(-1,-2).conj())*.5
            def load(c):
                power=c.diagonal(dim1=-2,dim2=-1).real.mean(-1).clamp_min(0)
                return c+(self.diagonal_loading*power+self.absolute_loading)[...,None,None]*torch.eye(6,device=x.device,dtype=x.dtype)
            if self.kind=="wiener":
                count=direct_sum(torch.ones_like(m[:,0]),width)
                rxx=hermitian(direct_sum(outer,width)/count[...,None,None])
                # Only the target cross-covariance column is needed. Rxx is
                # independent of masks because the two masks sum to one.
                cross=direct_sum(outer[..., :,0]*m[:,0,...,None],width)/count[...,None]
                w=torch.linalg.solve(load(rxx),cross[...,None])[...,0]
            else:
                cov=[]
                for k in range(2):
                    mass=direct_sum(m[:,k],width).clamp_min(1e-8)
                    cov.append(hermitian(direct_sum(outer*m[:,k,...,None,None],width)/mass[...,None,None]))
                matrix=torch.linalg.solve(load(cov[1]),cov[0])
                trace=matrix.diagonal(dim1=-2,dim2=-1).sum(-1).real
                w=matrix[..., :,0]/trace.clamp_min(1e-8)[...,None]
                w=torch.where((trace>1e-8)[...,None],w,torch.zeros_like(w))
            outputs.append((w.conj()*x).sum(-1).to(spectrum.dtype))
            weights.append(w.to(spectrum.dtype))
        return torch.cat(outputs,1),torch.cat(weights,1)
