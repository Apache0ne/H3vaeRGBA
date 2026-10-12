#!/usr/bin/env python3
"""Controlled SAME-SAMPLE alpha decoder ablation for the 4000-step sidecar.

A: 4K conditional DDIM alpha latents + original H3 grayscale RGB decoder
   (LoRA DISABLED, no merger or trained head).
B: EXACT SAME generated alpha latents + trained dual-path merger,
   decoder LoRA, and alpha head (the existing H3 RGBA output pipeline).
C: REAL ground-truth alpha encoded through the dual codec ("oracle",
   only available on held-out paired source videos).

Same RGB encoding, same sidecar checkpoint, same 30 DDIM steps, same seed,
same windows. No H3 DiT rerun, no new training. Reports whether our dual
decoder is helping the sidecar at all, or the generated latent stream is bad.

Note: The ORIGINAL widened 4-channel VAE is a DIFFERENT architecture.
It cannot meaningfully decode the sidecar's separate z_alpha directly.
Comparing that checkpoint requires an independently trained shared-latent
sidecar and is deliberately NOT mislabeled as the original-VAE baseline here.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))

from h3rgba.alpha_latent_diffusion import (
    ConditionalAlphaDenoiser, cosine_alphas, normalized_latents,
    unnormalized_latents, sample,
)
from h3rgba.decoder_lora import temporarily_disable_decoder_lora
from h3rgba.wan_dual import denormalize_rgb
from h3rgba.benchmark_metrics import measure
from generate_h3_rgba_end_to_end import (
    read_mp4,letterbox_rgb,recover_mask,window_starts,blend_weights,sha256
)


@torch.inference_mode()
def decode_original_gray(codec,z_alpha):
    """Original RGB VAE, not the old 4-in/4-out widened VAE."""
    with temporarily_disable_decoder_lora(codec.vae):
        with torch.autocast("cuda",dtype=torch.bfloat16,
                            enabled=z_alpha.device.type=="cuda"):
            decoded=codec.vae._decode_clip(z_alpha)
    decoded=decoded[:,:,codec.vae.frame_pre_padding:].float()
    return denormalize_rgb(decoded).mean(dim=1,keepdim=True).clamp(0,1)


@torch.inference_mode()
def decode_both_same_latents(codec,z_rgb,z_alpha):
    """No second sampling call: both arms use EXACT same z_alpha."""
    a=decode_original_gray(codec,z_alpha)
    b=codec.decode_alpha(z_rgb,z_alpha).float().clamp(0,1)
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("Failed same-latent A/B decoder shape/finiteness")
    return {"original_gray":a,"dual_lora":b}


def assemble_overlaps(chunks,starts,frames):
    """Weighted average of independently predicted 17-frame windows."""
    if len(chunks)!=len(starts) or not chunks:
        raise ValueError("Each window must have paired decoder output")
    h,w=chunks[0].shape[1:3]
    total=np.zeros((frames,h,w),np.float32)
    weight=np.zeros((frames,1,1),np.float32)
    blend=blend_weights()
    for block,first in zip(chunks,starts):
        if block.shape!=(17,h,w):
            raise ValueError("Wrong 17-frame decoded alpha block shape")
        total[first:first+17]+=block*blend[:,None,None]
        weight[first:first+17]+=blend[:,None,None]
    if np.any(weight<=0):
        raise RuntimeError("Frames missing alpha estimates")
    return np.clip(total/weight,0,1)


def coverage_stats(mask):
    mask=np.asarray(mask,dtype=np.float32)
    if mask.ndim!=3 or not np.isfinite(mask).all():
        raise ValueError("Invalid alpha video")
    fraction=(mask>.5).mean(axis=(1,2))
    return {"coverage_mean":float(fraction.mean()),
            "coverage_min":float(fraction.min()),
            "coverage_max":float(fraction.max()),
            "coverage_frame_std":float(fraction.std()),
            "temporal_abs_difference":float(
                np.abs(np.diff(mask,axis=0)).mean()) if len(mask)>1 else 0.}


def preview_comparison(rgb,geometry,alphas,path,*,indices=None):
    """5 selected rows: source RGB / original VAE alpha / dual alpha /
    original checkerboard / dual checkerboard.
    """
    import cv2
    names=("H3 RGB", "Original-gray alpha", "Dual-path alpha",
           "Original on checker", "Dual on checker")
    n,h,w,_=rgb.shape
    selection=indices or sorted(set([0,n//4,n//2,3*n//4,n-1]))
    tile_w,tile_h=320,192
    canvas=Image.new("RGB",(5*tile_w,len(selection)*(tile_h+26)),(24,24,24))
    draw=ImageDraw.Draw(canvas)
    yy,xx=np.indices((h,w))
    checker=np.where((((xx//25)+(yy//25))%2)[...,None],
                     175,225).astype(np.float32)
    check=np.repeat(checker,3,axis=2)
    for row,index in enumerate(selection):
        alphaA=recover_mask(alphas["original_gray"][index],geometry,h,w)
        alphaB=recover_mask(alphas["dual_lora"][index],geometry,h,w)
        rgbfloat=rgb[index].astype(np.float32)
        images=[
            rgb[index],
            np.repeat((alphaA[...,None]*255).round().astype(np.uint8),3,axis=2),
            np.repeat((alphaB[...,None]*255).round().astype(np.uint8),3,axis=2),
            (alphaA[...,None]*rgbfloat+(1-alphaA[...,None])*check).round().clip(0,255).astype(np.uint8),
            (alphaB[...,None]*rgbfloat+(1-alphaB[...,None])*check).round().clip(0,255).astype(np.uint8),
        ]
        for col,(label,img) in enumerate(zip(names,images)):
            x=col*tile_w;y=row*(tile_h+26)
            draw.text((x+4,y+5),f"{label} | frame {index}",fill="white")
            canvas.paste(Image.fromarray(img).resize((tile_w,tile_h)),
                         (x,y+24))
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path)
    return str(path)


@torch.inference_mode()
def video_ab(rgb,codec,net,schedule,*,seed=42,steps=30,stride=12):
    """Run DDIM ONCE per window and decode with BOTH heads."""
    videos,geometry=letterbox_rgb(rgb,128)
    starts=window_starts(len(videos),stride=stride)
    blocks={"original_gray":[],"dual_lora":[]}
    before=time.monotonic()
    for i,start in enumerate(starts):
        window=videos[start:start+17]
        x=torch.from_numpy(window.copy()).permute(3,0,1,2)[None].to("cuda").float()/255.
        zr=codec.encode_rgb(x)
        zalpha=unnormalized_latents(
            sample(net,normalized_latents(zr,codec.vae),schedule,
                   steps=steps,seed=seed+i),
            codec.vae
        )
        both=decode_both_same_latents(codec,zr,zalpha)
        for label,result in both.items():
            blocks[label].append(result[0,0].cpu().numpy())
        print(f"SHARED_DDIM_WINDOW {i+1}/{len(starts)} ({start}-{start+16}) "
              f"same_z_alpha=True",flush=True)
    masks={label:assemble_overlaps(chunks,starts,len(rgb))
           for label,chunks in blocks.items()}
    return masks,geometry,round(time.monotonic()-before,2)


@torch.inference_mode()
def eval_unused_holdout(records,codec,net,schedule,*,limit=9,seed=42,steps=30):
    from train_vae import PairedVideoDataset
    from run_vae_22gb import training_split
    _,val=training_split(records,seed)
    unseen=val[3:]
    if not unseen:
        raise RuntimeError("No unused holdout clips in manifest")
    subset=unseen[:limit]
    data=PairedVideoDataset(subset,resolution=128,deterministic=True)
    rows=[]
    for i,rgba in enumerate(data):
        batch=rgba.unsqueeze(0).to("cuda",dtype=torch.float32)
        zr,gt_za=codec.encode_pair(batch)
        generated_za=unnormalized_latents(
            sample(net,normalized_latents(zr,codec.vae),schedule,
                   steps=steps,seed=seed+i),
            codec.vae
        )
        pred=decode_both_same_latents(codec,zr,generated_za)
        oracle=codec.decode_alpha(zr,gt_za).clamp(0,1)
        scores={k:measure(v,batch) for k,v in
                {**pred,"gt_alpha_oracle":oracle}.items()}
        row={
            "id":subset[i].get("id",str(i)),
            "original_gray_iou":scores["original_gray"]["alpha_iou"],
            "dual_lora_iou":scores["dual_lora"]["alpha_iou"],
            "gt_alpha_oracle_iou":scores["gt_alpha_oracle"]["alpha_iou"],
            "original_gray_mae":scores["original_gray"]["alpha_mae"],
            "dual_lora_mae":scores["dual_lora"]["alpha_mae"],
            "gt_alpha_oracle_mae":scores["gt_alpha_oracle"]["alpha_mae"],
            "original_gray_boundary_f1":scores["original_gray"]["alpha_boundary_f1_1px"],
            "dual_lora_boundary_f1":scores["dual_lora"]["alpha_boundary_f1_1px"],
            "gt_alpha_oracle_boundary_f1":scores["gt_alpha_oracle"]["alpha_boundary_f1_1px"],
        }
        rows.append(row)
        print("HOLDOUT_SAME_LATENTS",json.dumps(row),flush=True)
    return rows


def aggregate(rows):
    if not rows:
        return {}
    return {
        "samples":len(rows),
        "original_gray_iou":sum(x["original_gray_iou"] for x in rows)/len(rows),
        "dual_lora_iou":sum(x["dual_lora_iou"] for x in rows)/len(rows),
        "gt_alpha_oracle_iou":sum(x["gt_alpha_oracle_iou"] for x in rows)/len(rows),
        "original_gray_mae":sum(x["original_gray_mae"] for x in rows)/len(rows),
        "dual_lora_mae":sum(x["dual_lora_mae"] for x in rows)/len(rows),
        "gt_alpha_oracle_mae":sum(x["gt_alpha_oracle_mae"] for x in rows)/len(rows),
        "dual_minus_original_iou":sum(
            x["dual_lora_iou"]-x["original_gray_iou"] for x in rows)/len(rows),
        "dual_iou_wins":sum(
            x["dual_lora_iou"]>x["original_gray_iou"] for x in rows),
        "oracle_minus_dual_iou":sum(
            x["gt_alpha_oracle_iou"]-x["dual_lora_iou"] for x in rows)/len(rows),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-lora",required=True)
    parser.add_argument("--alpha-diffusion",required=True)
    parser.add_argument("--rgb-mp4",default=None)
    parser.add_argument("--manifest",default=None)
    parser.add_argument("--output-dir",default="/content/h3_alpha_ab")
    parser.add_argument("--heldout-limit",type=int,default=9)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--sampling-steps",type=int,default=30)
    parser.add_argument("--stride",type=int,default=12)
    args=parser.parse_args()
    if not args.rgb_mp4 and not args.manifest:
        parser.error("Supply --rgb-mp4 and/or --manifest")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    checkpoint=torch.load(args.alpha_diffusion,map_location="cpu",weights_only=True)
    if checkpoint.get("vae_lora_sha256")!=sha256(args.vae_lora):
        raise RuntimeError("Sidecar and dual VAE LoRA checkpoint hashes do not match")
    cfg=checkpoint["settings"]
    if cfg["resolution"]!=128:
        raise RuntimeError("This matched-decoder A/B expects the sidecar 128px setting")
    from train_wan_dual_lora import make_model
    codec,_=make_model(
        argparse.Namespace(rank=4,merger_scale=.15,resolution=128),
        torch.device("cuda")
    )
    codec.load_adapter(args.vae_lora)
    codec.requires_grad_(False)
    codec.eval()
    net=ConditionalAlphaDenoiser(hidden=cfg["hidden"],blocks=cfg["blocks"]).cuda()
    net.load_state_dict(checkpoint["model"],strict=True)
    net.eval()
    schedule=cosine_alphas(cfg["diffusion_steps"]).cuda()
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    report={
        "experiment":"same 4000-step diffusion sidecar z_alpha, two VAE decoding strategies",
        "sidecar_step":int(checkpoint["step"]),
        "same_rgb_encoder":True,
        "same_seed":args.seed,
        "same_z_alpha_for_both_arms":True,
        "original_gray_description":"Original pretrained H3 VAE RGB decode without LoRA; grayscale mean",
        "dual_lora_description":"Same z_alpha decoded through trained dual latent merger, LoRA, alpha head",
        "gt_alpha_oracle_description":"Only heldout dataset: encode GT alpha then dual LoRA; never used for sidecar prediction",
        "widened_4channel_vae_compared":False,
        "vae_lora_sha256":sha256(args.vae_lora),
        "sidecar_sha256":sha256(args.alpha_diffusion),
    }
    if args.rgb_mp4:
        rgb,fps=read_mp4(args.rgb_mp4)
        alphas,geometry,seconds=video_ab(
            rgb,codec,net,schedule,seed=args.seed,
            steps=args.sampling_steps,stride=args.stride
        )
        report["generated_h3_rgb"]={
            "source":args.rgb_mp4,"frames":len(rgb),"fps":fps,
            "latency_seconds":seconds,
            "original_gray":coverage_stats(alphas["original_gray"]),
            "dual_lora":coverage_stats(alphas["dual_lora"]),
        }
        for name,mask in alphas.items():
            np.savez_compressed(out/f"{name}_generated_alpha_128.npz",
                                alpha=mask.astype(np.float16))
        report["visual_comparison"]=preview_comparison(
            rgb,geometry,alphas,out/"same_sidecar_two_decoders.png"
        )
        print("GENERATED_H3_AB_PREVIEW",report["visual_comparison"],flush=True)
    if args.manifest:
        records=json.loads(Path(args.manifest).read_text())["clips"]
        rows=eval_unused_holdout(
            records,codec,net,schedule,
            limit=args.heldout_limit,seed=args.seed,steps=args.sampling_steps
        )
        with (out/"heldout_same_latents.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["unused_holdout"]=aggregate(rows)
        print("HOLDOUT_AB_SUMMARY",json.dumps(report["unused_holdout"]),flush=True)
    (out/"ab_summary.json").write_text(json.dumps(report,indent=2))
    print("SAME_SIDECAR_DUAL_VS_ORIGINAL_COMPLETE",out,flush=True)


if __name__=="__main__":
    main()
