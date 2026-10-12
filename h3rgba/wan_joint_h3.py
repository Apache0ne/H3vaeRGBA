"""Wan-Alpha's SHARED 24-channel latent idea adapted to original MiniMax-H3.

Wan-Alpha does NOT generate two independent latent streams at inference.
Its merger encodes RGB and alpha into ONE video latent, which its RGB and
alpha decoders both consume; its T2V LoRA then learns the joint distribution.

This H3 version reuses the proven DualAlphaCodec VAE LoRA, but learns:
  fused z = z_rgb + bounded fuser([z_rgb, z_alpha])
  alpha bridge = f(z) -> latent compatible with trained dual-alpha decoder
  RGB decoder = original frozen H3 RGB decoder on SAME fused z
  alpha decoder = pretrained+LoRA H3 alpha head on SAME fused z, via bridge.
NO GT alpha is needed at inference: predict_from_joint(z) consumes only z.
Stage 2 MUST train H3 DiT on normalized fused-z, not original z_rgb.
"""
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

from h3rgba.decoder_lora import temporarily_disable_decoder_lora
from h3rgba.wan_dual import denormalize_rgb
from h3rgba.alpha_latent_diffusion import normalized_latents
from h3rgba.dual_loss import alpha_losses, alpha_scores


class ResidualLatentNet(nn.Module):
    """3D spatiotemporal feature bridge; zero output initialization."""
    def __init__(self, channels_in, channels_out=24, hidden=128):
        super().__init__()
        if channels_in not in (24,48) or hidden%8 or hidden < 16:
            raise ValueError("Expected 24/48 input channels and hidden multiple of 8 >=16")
        self.in_proj=nn.Conv3d(channels_in,hidden,3,padding=1)
        self.norm1=nn.GroupNorm(8,hidden)
        self.mid=nn.Conv3d(hidden,hidden,3,padding=1)
        self.norm2=nn.GroupNorm(8,hidden)
        self.out=nn.Conv3d(hidden,channels_out,3,padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self,x):
        x=F.silu(self.norm1(self.in_proj(x.float())))
        x=F.silu(self.norm2(self.mid(x)))
        return self.out(x)


class WanJointH3Codec(nn.Module):
    """One diffuser-compatible latent, two VAE decode branches."""

    def __init__(self, pretrained_dual, *, hidden=128, shift_limit=.35,
                 bridge_scale=1., train_decoder_lora=True):
        super().__init__()
        if not 0 < shift_limit <= 2 or not 0 < bridge_scale <= 4:
            raise ValueError("Bad bounded shift/bridge strength")
        self.dual=pretrained_dual
        self.shift_limit=float(shift_limit)
        self.bridge_scale=float(bridge_scale)
        self.fuser=ResidualLatentNet(48,hidden=hidden)
        self.alpha_bridge=ResidualLatentNet(24,hidden=hidden)
        # Original RGB encoder and base weights remain strictly frozen.
        self.dual.vae.requires_grad_(False)
        self.dual.merger.requires_grad_(False)
        self.dual.alpha_head.requires_grad_(train_decoder_lora)
        from h3rgba.decoder_lora import decoder_lora_parameters
        for p in decoder_lora_parameters(self.dual.vae):
            p.requires_grad_(train_decoder_lora)
        # The fuser and alpha_bridge always receive updates.

    def train(self, mode=True):
        super().train(mode)
        self.dual.vae.encoder.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def fuse(self,z_rgb,z_alpha):
        if z_rgb.ndim!=5 or z_rgb.shape!=z_alpha.shape or z_rgb.shape[1]!=24:
            raise ValueError("Need RGB and alpha raw H3 latent tensors [B,24,T,H,W]")
        offset=self.shift_limit*torch.tanh(
            self.fuser(torch.cat([z_rgb,z_alpha],dim=1)))
        return z_rgb.float()+offset

    def alpha_latent_from_joint(self,z_joint):
        if z_joint.ndim!=5 or z_joint.shape[1]!=24:
            raise ValueError("Need single 24-channel fused H3 latent")
        correction=self.bridge_scale*self.alpha_bridge(z_joint.float())
        return z_joint.float()+correction

    def encode_joint(self,rgba,*,background=None):
        zrgb,zalpha=self.dual.encode_pair(rgba,background=background)
        return self.fuse(zrgb,zalpha),zrgb,zalpha

    def decode_joint_rgb(self,z_joint):
        """Gradient must flow back through FROZEN RGB decoder to joint fuser.

        Do NOT call DualAlphaCodec.decode_rgb: it is @torch.no_grad().
        """
        with temporarily_disable_decoder_lora(self.dual.vae):
            with (torch.autocast("cuda",dtype=torch.bfloat16)
                  if z_joint.device.type=="cuda" else nullcontext()):
                result=self.dual.vae._decode_clip(z_joint)
        result=result[:,:,self.dual.vae.frame_pre_padding:].float()
        return denormalize_rgb(result)

    def decode_joint_alpha(self,z_joint):
        z_alpha_hat=self.alpha_latent_from_joint(z_joint)
        return self.dual.decode_alpha(z_joint,z_alpha_hat)

    def decode_joint(self,z_joint):
        """Inference ONLY requires H3's native 24-channel latent output."""
        return self.decode_joint_rgb(z_joint),self.decode_joint_alpha(z_joint)

    def forward(self,rgba,*,background=None):
        z_joint,zrgb,zalpha=self.encode_joint(rgba,background=background)
        rgb,alpha=self.decode_joint(z_joint)
        return {"rgb":rgb,"alpha":alpha,"joint":z_joint,
                "rgb_teacher_latent":zrgb,"alpha_teacher_latent":zalpha}

    def normalized_diffusion_target(self,joint):
        return normalized_latents(joint,self.dual.vae)

    def bridge_state(self):
        return {
            "fuser."+k:v.detach().cpu().contiguous()
            for k,v in self.fuser.state_dict().items()
        } | {
            "alpha_bridge."+k:v.detach().cpu().contiguous()
            for k,v in self.alpha_bridge.state_dict().items()
        }

    def load_bridge(self,path):
        from safetensors.torch import load_file
        tensors=load_file(str(path),device="cpu")
        want=set(self.bridge_state())
        if set(tensors)!=want:
            raise ValueError(f"Bridge keys differ, missing={want-set(tensors)}")
        with torch.no_grad():
            for name,p in self.named_parameters():
                if name in tensors:
                    if p.shape!=tensors[name].shape:
                        raise ValueError(f"Bridge tensor {name} shape mismatch")
                    p.copy_(tensors[name].to(p.device,p.dtype))


def joint_rgba_loss(model,rgba,outputs,*,alpha_weight=2.,rgb_weight=1.,
                    latent_weight=.05,bridge_weight=.05,shift_weight=.01,
                    edge_weight=2.,composite_weight=1.,temporal_weight=.2,
                    hard_weight=.25):
    """Wan-style RGB+matte compositing with bounded latent-shift regularizer."""
    if rgba.ndim!=5 or rgba.shape[1]!=4:
        raise ValueError("Need [B,4,T,H,W] RGBA")
    rgb,alpha=outputs["rgb"],outputs["alpha"]
    if rgb.shape!=rgba[:,:3].shape or alpha.shape!=rgba[:,3:4].shape:
        raise ValueError("RGB/alpha decoder output shape mismatch")
    gt_rgb=rgba[:,:3].float()
    gt_alpha=rgba[:,3:4].float()
    alpha_loss,parts=alpha_losses(
        alpha,gt_alpha,gt_rgb,
        alpha_weight=alpha_weight,edge_weight=edge_weight,
        composite_weight=composite_weight,temporal_weight=temporal_weight,
        hard_weight=hard_weight,
    )
    # Foreground RGB preservation and composites enforce information in
    # the merged latent. Weight by matte, ignoring unseen FG outside alpha.
    rgb_l1=((rgb.float()-gt_rgb).abs()*gt_alpha).mean()
    composed=[]
    for color in (0.,1.,.5):
        comp_pred=rgb.float()*alpha.float()+color*(1-alpha.float())
        comp_gt=gt_rgb*gt_alpha+color*(1-gt_alpha)
        composed.append((comp_pred-comp_gt).abs().mean())
    comp_rgb=torch.stack(composed).mean()
    prior=F.mse_loss(
        model.normalized_diffusion_target(outputs["joint"]),
        model.normalized_diffusion_target(outputs["rgb_teacher_latent"])
    )
    # Teacher target is the successful dual codec's internal alpha latent.
    with torch.no_grad():
        teacher=model.dual.merge_latents(
            outputs["rgb_teacher_latent"],outputs["alpha_teacher_latent"])
    bridge=F.mse_loss(model.alpha_latent_from_joint(outputs["joint"]),teacher)
    shift=(outputs["joint"]-outputs["rgb_teacher_latent"]).square().mean()
    loss=(alpha_loss+rgb_weight*(rgb_l1+comp_rgb)+
          latent_weight*prior+bridge_weight*bridge+shift_weight*shift)
    metrics={
        "alpha_loss":float(alpha_loss.detach()),
        "alpha_mae":float((alpha.float()-gt_alpha).abs().mean().detach()),
        "rgb_foreground_l1":float(rgb_l1.detach()),
        "composite_rgb_l1":float(comp_rgb.detach()),
        "normalized_rgb_latent_prior":float(prior.detach()),
        "alpha_latent_bridge_mse":float(bridge.detach()),
        "raw_joint_shift_mse":float(shift.detach()),
        **{k:float(v.detach()) for k,v in parts.items()},
    }
    return loss,metrics
