"""Small SECOND diffusion model for H3-alpha latent generation.

This model is NOT an H3 33B DiT LoRA. It learns p(z_alpha | z_rgb) from
paired source videos, in the *normalized* original H3 VAE 24-channel space.
Inputs are 17-frame windows (~7 latent frames), not raw RGB alpha masks.
Decoder-LoRA checkpoint remains a separate frozen component.

Use a cosine VP noising schedule and DDIM sampling. Real diffusion samples
must begin from random alpha latent noise; copying z_rgb or using GT alpha
is explicitly rejected by the inference entrypoint.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv3d(channels, channels, (3,3,3), padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, (3,3,3), padding=1)
        self.time = nn.Linear(time_dim, channels*2)
        # Near-identity initialization stabilizes small datasets.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, t_embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(t_embedding).chunk(2, dim=1)
        h = h * (1. + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return (x + h) * (2. ** -.5)


class ConditionalAlphaDenoiser(nn.Module):
    """Low-cost 3D conditional diffusion denoiser; never modifies H3 weights."""

    def __init__(self, hidden=128, blocks=6, time_dim=256):
        super().__init__()
        if hidden < 8 or hidden % 8 or blocks < 1:
            raise ValueError("hidden must be divisible by 8; positive blocks")
        self.hidden, self.blocks, self.time_dim = hidden, blocks, time_dim
        # 24 noisy alpha + 24 RGB context channels.
        self.input = nn.Conv3d(48, hidden, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim))
        self.resblocks = nn.ModuleList(
            ResBlock3D(hidden, time_dim) for _ in range(blocks))
        self.output = nn.Sequential(
            nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv3d(hidden,24,3,padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def timestep_embedding(self, timesteps):
        half=self.time_dim//2
        freqs=torch.exp(torch.arange(half,device=timesteps.device,dtype=torch.float32) *
                        (-math.log(10000.)/max(half-1,1)))
        angles=timesteps.float()[:,None]*freqs[None]
        emb=torch.cat((angles.sin(),angles.cos()),dim=1)
        return self.time_mlp(emb)

    def forward(self, noisy_alpha, rgb_cond, timesteps):
        if noisy_alpha.ndim != 5 or noisy_alpha.shape != rgb_cond.shape or noisy_alpha.shape[1] != 24:
            raise ValueError("Need two equal [B,24,T,H,W] latent videos")
        if timesteps.shape != (noisy_alpha.shape[0],):
            raise ValueError("Noise timestep must be [batch]")
        emb=self.timestep_embedding(timesteps)
        h=self.input(torch.cat((noisy_alpha,rgb_cond),dim=1))
        for block in self.resblocks:
            h=block(h,emb)
        return self.output(h)


def cosine_alphas(steps=1000, offset=.008):
    if steps < 2:
        raise ValueError("Need at least two diffusion timesteps")
    grid=torch.linspace(0,steps,steps+1,dtype=torch.float64)/steps
    cumulative=torch.cos((grid+offset)/(1+offset)*math.pi*.5)**2
    cumulative=cumulative/cumulative[0]
    # Clamp ratio to beta <=.999, as in cosine DDPM schedules.
    betas=(1-(cumulative[1:]/cumulative[:-1])).clamp(.0001,.999)
    return torch.cumprod((1-betas).float(),dim=0)


def normalized_latents(raw, vae):
    mean=torch.as_tensor(vae.config.latents_mean,device=raw.device,
                         dtype=torch.float32).view(1,24,1,1,1)
    std=torch.as_tensor(vae.config.latents_std,device=raw.device,
                        dtype=torch.float32).view(1,24,1,1,1)
    if std.numel()!=24 or torch.any(std <= 0):
        raise ValueError("Invalid VAE latent normalization")
    return (raw.float()-mean)/std


def unnormalized_latents(normalized, vae):
    mean=torch.as_tensor(vae.config.latents_mean,device=normalized.device,
                         dtype=torch.float32).view(1,24,1,1,1)
    std=torch.as_tensor(vae.config.latents_std,device=normalized.device,
                        dtype=torch.float32).view(1,24,1,1,1)
    return normalized.float()*std+mean


def noise_loss(model, x0, rgb, alpha_bars, *, generator=None):
    if x0.shape != rgb.shape or x0.shape[1] != 24:
        raise ValueError("Training requires paired 24-channel normalized VAE latents")
    b=x0.shape[0]
    t=torch.randint(0,len(alpha_bars),(b,),device=x0.device,generator=generator)
    noise=torch.randn(x0.shape,device=x0.device,dtype=x0.dtype,generator=generator)
    a=alpha_bars[t].view(b,1,1,1,1)
    xt=a.sqrt()*x0+(1-a).sqrt()*noise
    prediction=model(xt,rgb,t)
    loss=F.mse_loss(prediction,noise)
    return loss, {"mse":float(loss.detach()),"t_mean":float(t.float().mean())}


@torch.no_grad()
def sample(model, rgb_cond, alpha_bars, *, steps=30, seed=42):
    """DDIM eta=0: actual conditional alpha latent diffusion from fresh noise."""
    if steps < 2 or steps > len(alpha_bars):
        raise ValueError("Sampling needs 2..training_diffusion_steps")
    gen=torch.Generator(device=rgb_cond.device).manual_seed(int(seed))
    z=torch.randn(rgb_cond.shape,device=rgb_cond.device,
                  dtype=rgb_cond.dtype,generator=gen)
    timeline=torch.linspace(len(alpha_bars)-1,0,steps,device=rgb_cond.device)
    timeline=timeline.round().long().unique_consecutive()
    for i,tensor_t in enumerate(timeline):
        t=int(tensor_t)
        prev=int(timeline[i+1]) if i+1 < len(timeline) else -1
        a=alpha_bars[t].float()
        ap=alpha_bars[prev].float() if prev >= 0 else a.new_tensor(1.)
        eps=model(z,rgb_cond,torch.full((z.shape[0],),t,device=z.device,dtype=torch.long))
        x0=(z-(1-a).sqrt()*eps)/a.sqrt().clamp_min(1e-6)
        # Prevent destructive excursions on a small data distribution.
        x0=x0.clamp(-8,8)
        z=ap.sqrt()*x0+(1-ap).sqrt()*eps
    if not torch.isfinite(z).all():
        raise FloatingPointError("Nonfinite generated alpha latents")
    return z
