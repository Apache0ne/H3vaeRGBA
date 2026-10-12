"""MiniMax-H3 data-ward rectified-flow LoRA loss on Wan shared RGB+A latents.

z_t = t*z_data+(1-t)*noise, velocity target = z_data-noise.
Uses actual H3 packed text/video rotary layout and original 24ch tokens.
Paired matting data have no audio, so this is VIDEO-only fine-tuning of H3's
joint video/audio backbone. Native joint audio/video fine-tuning is untested.
"""
import torch
from torch import nn


def rectified_flow_batch(clean, *, t=None, noise=None, generator=None):
    if clean.ndim!=5 or clean.shape[1]!=24 or clean.shape[0]!=1:
        raise ValueError("Expected one [1,24,F,H,W] H3 joint sample")
    if t is None:
        t=torch.rand((),device=clean.device,generator=generator)*.96+.02
    t=torch.as_tensor(t,device=clean.device,dtype=torch.float32).reshape(())
    if not 0<float(t)<1:
        raise ValueError("H3 clean-time t must be in (0,1)")
    if noise is None:
        noise=torch.randn(clean.shape,device=clean.device,
                          dtype=clean.dtype,generator=generator)
    if noise.shape!=clean.shape:
        raise ValueError("Noise and data shapes differ")
    return t*clean+(1-t)*noise,clean-noise,t


def h3_packed_flow_loss(model,clean,prompt_embeds,text_token_tags,*,t=None,noise=None):
    """Gradients for real H3 transformer low-rank adapters only."""
    if prompt_embeds.ndim!=3 or prompt_embeds.shape[0]!=1 or prompt_embeds.shape[2]!=5120:
        raise ValueError("Require actual Qwen3-VL H3 prompt embeddings [1,N,5120]")
    if text_token_tags.shape!=(prompt_embeds.shape[1],):
        raise ValueError("Wrong H3 text-token tag shape")
    if not torch.all((text_token_tags==1)|(text_token_tags==0)):
        raise ValueError("H3 tag values must be text=1 or vision=0")
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,patchify_video_latents)
    patch=tuple(model.config.patch_size)
    noisy,target,t=rectified_flow_batch(clean,t=t,noise=noise)
    _,_,frames,height,width=noisy.shape
    positions,tags,video_idx,audio_idx,text_idx,ncond,_=(
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags.cpu(),num_latent_frames=frames,
            latent_height=height,latent_width=width,
            num_audio_latents=0,patch_size=patch,audio_channels=32,
            audio_tag=2,video_tag=0,keyframe_anchors=(),
        )
    )
    if ncond or audio_idx.numel():
        raise RuntimeError("Unexpected conditioning/audio rows in video-only training")
    device=clean.device
    positions=positions.to(device)
    tags=tags.to(device)
    video_idx=video_idx.to(device)
    audio_idx=audio_idx.to(device)
    text_idx=text_idx.to(device)
    time_table=torch.stack((t,torch.ones_like(t)))
    time_idx=torch.zeros(tags.shape,device=device,dtype=torch.long)
    time_idx[text_idx]=1
    tokens=patchify_video_latents(noisy,patch)[None]
    target_tokens=patchify_video_latents(target,patch)[None]
    audio=torch.zeros((1,0,32),device=device,dtype=noisy.dtype)
    predicted,_=model(
        hidden_states=tokens,
        audio_hidden_states=audio,
        encoder_hidden_states=prompt_embeds.to(device),
        timestep=time_table,timestep_indices=time_idx,
        token_tags=tags,position_ids=positions,
        video_indices=video_idx,audio_indices=audio_idx,
        text_indices=text_idx,return_dict=False,
    )
    if predicted.shape!=target_tokens.shape:
        raise RuntimeError("H3 video velocity shape incompatible with packed latent")
    loss=(predicted.float()-target_tokens.float()).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("H3 DiT flow loss nonfinite")
    return loss,dict(
        h3_data_ward_flow_mse=float(loss.detach()),
        flow_t=float(t),
        video_rows=int(video_idx.numel()),
        text_rows=int(text_idx.numel()),
        audio_rows_unsupervised=0,
    )


def inject_h3_transformer_lora(model,rank=8):
    """Target H3 attention Q/K/V/out and SwiGLU FFN, preserving 24ch IO."""
    from h3rgba.decoder_lora import LoRALinear
    if rank<1:
        raise ValueError("Rank must be positive")
    model.requires_grad_(False)
    targets=[]
    for name,layer in list(model.named_modules()):
        if not name.startswith("transformer_blocks.") or not isinstance(layer,nn.Linear):
            continue
        if any(k in name for k in (
            ".attn.to_q",".attn.to_k",".attn.to_v",
            ".attn.to_out.0",".ff.net.")):
            targets.append((name,layer))
    if len(targets)<4:
        raise RuntimeError("H3 transformer QKV/FFN LoRA targets not found")
    for name,layer in targets:
        parent_name,leaf=name.rsplit(".",1)
        model.get_submodule(parent_name)._modules[leaf]=LoRALinear(layer,rank=rank)
    return len(targets)


def h3_dit_lora_state(model):
    from h3rgba.decoder_lora import LoRALinear
    state={}
    for name,layer in model.named_modules():
        if isinstance(layer,LoRALinear):
            state[name+".lora_down"]=layer.lora_down.detach().cpu().contiguous()
            state[name+".lora_up"]=layer.lora_up.detach().cpu().contiguous()
    if not state:
        raise RuntimeError("No LoRA matrices on H3 diffusion transformer")
    return state
