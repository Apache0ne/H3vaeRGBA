#!/usr/bin/env python3
"""Generate a TRUE alpha channel from original H3-generated RGB+audio video.

Inputs: original real H3 T2VA MP4, trained VAE dual-alpha decoder LoRA,
trained *separate* conditional alpha latent diffusion model.
No GT alpha is read, no RGB frame is duplicated as alpha, and the alpha
latent sample ALWAYS starts from random noise. Runs IN A DIFFERENT PROCESS
from the original ~33B H3 generator, so both large models don't overlap
in GPU/host RAM. This is NOT an H3 diffusion-transformer LoRA.

Trained on 128x128 RGBA source matting; 960x544 output alpha gets upscaled.
The method does not magically recover colors already composited against an
unknown background. --assumed-background=black is an explicit APPROXIMATE
straight-alpha uncompositing option, suitable only if prompts produced a
near-black background. Real quantitative quality is unverified until tested.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from h3rgba.alpha_latent_diffusion import (
    ConditionalAlphaDenoiser, cosine_alphas,
    normalized_latents, unnormalized_latents, sample,
)


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):
            h.update(c)
    return h.hexdigest()


def read_mp4(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open H3 generated RGB video: {path}")
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 120:
        raise RuntimeError(f"Invalid video frame rate: {fps}")
    frames=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if len(frames)<17:
        raise RuntimeError("Need >=17 generated RGB frames; never read GT alpha")
    h,w=frames[0].shape[:2]
    if any(f.shape[:2]!=(h,w) for f in frames):
        raise RuntimeError("Input video has changing spatial size")
    return np.stack(frames),fps


def letterbox_rgb(video, resolution):
    """Same geometry and black padding as PairedVideoDataset training."""
    n,h,w,_=video.shape
    scale=resolution/max(h,w)
    rw=max(1,round(w*scale))
    rh=max(1,round(h*scale))
    left=(resolution-rw)//2
    top=(resolution-rh)//2
    result=np.zeros((n,resolution,resolution,3),np.uint8)
    for i,frame in enumerate(video):
        resized=cv2.resize(frame,(rw,rh),
            interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
        result[i,top:top+rh,left:left+rw]=resized
    return result, (top,left,rh,rw)


def recover_mask(alpha_square, geometry, h, w):
    top,left,rh,rw=geometry
    content=alpha_square[top:top+rh,left:left+rw]
    return cv2.resize(content,(w,h),interpolation=cv2.INTER_LINEAR).clip(0,1)


def window_starts(frames, size=17, stride=12):
    """Full 17-frame coverage of arbitrarily long generated video."""
    if frames<size or not 1<=stride<=size:
        raise ValueError("Need >=17 frames and 1..17 stride")
    starts=list(range(0,frames-size+1,stride))
    if starts[-1] != frames-size:
        starts.append(frames-size)
    return starts


def blend_weights(size=17):
    """Positive triangular overlap weight to smooth chunk transitions."""
    t=np.arange(size,dtype=np.float32)
    return np.maximum(.08,1.-np.abs(t-(size-1)/2)/((size+1)/2))


@torch.no_grad()
def infer_alpha(video_128, codec, model, schedule, *, sampling_steps=30,
                seed=42, stride=12):
    """Each window receives newly sampled z_alpha from conditional DDIM."""
    n=video_128.shape[0]
    starts=window_starts(n,stride=stride)
    accum=np.zeros((n,video_128.shape[1],video_128.shape[2]),np.float32)
    weights=np.zeros((n,1,1),np.float32)
    blend=blend_weights()
    for index,start in enumerate(starts):
        clip=video_128[start:start+17]
        rgb=torch.from_numpy(clip.copy()).permute(3,0,1,2)[None].float().to("cuda")/255
        zr=codec.encode_rgb(rgb)
        context=normalized_latents(zr,codec.vae)
        z_generated=sample(model,context,schedule,steps=sampling_steps,
                           seed=seed+index)
        z_alpha=unnormalized_latents(z_generated,codec.vae)
        predicted=codec.decode_alpha(zr,z_alpha).float().clamp(0,1)
        if tuple(predicted.shape)!=(1,1,17,clip.shape[1],clip.shape[2]):
            raise RuntimeError(f"Unexpected decoded RGBA alpha shape {tuple(predicted.shape)}")
        array=predicted[0,0].cpu().numpy()
        accum[start:start+17]+=array*blend[:,None,None]
        weights[start:start+17]+=blend[:,None,None]
        print(f"ALPHA_DIFFUSION_WINDOW {index+1}/{len(starts)} "
              f"frames={start}-{start+16}",flush=True)
    if (weights <= 0).any():
        raise RuntimeError("Uncovered output video frames")
    result=(accum/weights).clip(0,1)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite generated transparency mask")
    return result


def clean_foreground(rgb,alpha,mode):
    """Separate RGB color from alpha; black-background inversion is approximate."""
    image=rgb.astype(np.float32)/255.
    if mode=="retain":
        return rgb
    if mode!="black":
        raise ValueError("Unknown background mode")
    a=np.maximum(alpha[...,None],.08)
    return np.uint8(np.round((image/a).clip(0,1)*255))


def encode_rgba_pngs(frames, alpha, dest, geometry, *, background):
    """Write actual RGBA PNGs; no fake mask or 3-channel PNG substitution."""
    dest=Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    n,h,w,_=frames.shape
    fractions=[]
    for i in range(n):
        scaled=recover_mask(alpha[i],geometry,h,w)
        fractions.append(float((scaled>.5).mean()))
        a=(scaled*255).round().clip(0,255).astype(np.uint8)
        rgb=clean_foreground(frames[i],scaled,background)
        rgba=np.concatenate((rgb,a[:,:,None]),axis=-1)
        Image.fromarray(rgba,mode="RGBA").save(dest/f"frame_{i:05d}.png")
    return {"png_frames":n,
            "hard_alpha_coverage_mean":float(np.mean(fractions)),
            "hard_alpha_coverage_min":float(np.min(fractions)),
            "hard_alpha_coverage_max":float(np.max(fractions))}


def export_prores_pngs(folder,original_mp4,destination,fps):
    """ProRes 4444 + original H3 audio; verify alpha-capable pix_fmt."""
    if not shutil_which_ffmpeg():
        raise RuntimeError("ffmpeg and ffprobe required for verified alpha MOV export")
    command=[
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-framerate",f"{fps:.6f}","-i",str(Path(folder)/"frame_%05d.png"),
        "-i",str(original_mp4),
        "-map","0:v:0","-map","1:a:0?",
        "-c:v","prores_ks","-profile:v","4",
        "-pix_fmt","yuva444p10le",
        "-c:a","pcm_s16le",
        "-shortest",str(destination),
    ]
    subprocess.run(command,check=True)
    probe=subprocess.run(
        ["ffprobe","-v","error","-select_streams","v:0",
         "-show_entries","stream=pix_fmt,nb_frames,width,height",
         "-of","json",str(destination)],
        capture_output=True,text=True,check=True)
    streams=json.loads(probe.stdout)["streams"]
    if not streams or "yuva" not in streams[0].get("pix_fmt",""):
        raise RuntimeError("Export did not retain an alpha-capable pixel format")
    return streams[0]


def shutil_which_ffmpeg():
    import shutil
    return shutil.which("ffmpeg") and shutil.which("ffprobe")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb-mp4",required=True)
    p.add_argument("--vae-lora",required=True)
    p.add_argument("--alpha-diffusion",required=True)
    p.add_argument("--output-dir",default="/content/h3_end_to_end_rgba")
    p.add_argument("--sampling-steps",type=int,default=30)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--stride",type=int,default=12)
    p.add_argument("--assumed-background",choices=("black","retain"),default="black",
                   help="Black background uncompositing is approximate, not guaranteed")
    args=p.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required")
    if not Path(args.rgb_mp4).is_file() or not Path(args.vae_lora).is_file():
        raise FileNotFoundError("Need real generated H3 MP4 and VAE decoder LoRA")
    state=torch.load(args.alpha_diffusion,map_location="cpu",weights_only=True)
    if state["vae_lora_sha256"] != sha256(args.vae_lora):
        raise RuntimeError("Diffusion alpha model trained for DIFFERENT VAE LoRA; refusing")
    if not 2<=args.sampling_steps<=state["settings"]["diffusion_steps"]:
        raise ValueError("Invalid sampling steps")
    resolution=state["settings"]["resolution"]
    from train_wan_dual_lora import make_model
    codec,_=make_model(argparse.Namespace(rank=4,merger_scale=.15,
                                           resolution=resolution),torch.device("cuda"))
    codec.load_adapter(args.vae_lora)
    codec.requires_grad_(False)
    codec.eval()
    model=ConditionalAlphaDenoiser(
        hidden=state["settings"]["hidden"],
        blocks=state["settings"]["blocks"]).to("cuda")
    model.load_state_dict(state["model"],strict=True)
    model.eval()
    schedule=cosine_alphas(state["settings"]["diffusion_steps"]).to("cuda")
    images,fps=read_mp4(args.rgb_mp4)
    frames_128,geometry=letterbox_rgb(images,resolution)
    result=infer_alpha(frames_128,codec,model,schedule,seed=args.seed,
                       sampling_steps=args.sampling_steps,stride=args.stride)
    out=Path(args.output_dir)
    frames_path=out/"rgba_png"
    summary=encode_rgba_pngs(images,result,frames_path,geometry,
                             background=args.assumed_background)
    mov=out/"h3_generated_rgba_prores4444.mov"
    video_proof=export_prores_pngs(frames_path,args.rgb_mp4,mov,fps)
    report={
        "status":"H3_RGB_DIFFUSION_PLUS_TRAINED_ALPHA_SIDECAR_GENERATED_RGBA",
        "source_rgb_video":str(args.rgb_mp4),
        "source_video_frames":len(images),
        "vae_lora_sha256":sha256(args.vae_lora),
        "alpha_diffusion_sha256":sha256(args.alpha_diffusion),
        "alpha_diffusion_training_step":state["step"],
        "alpha_latents_source":"DDIM_sampled_noise_conditioned_on_generated_RGB",
        "ground_truth_alpha_used":False,
        "alpha_coverage":summary,
        "alpha_export":video_proof,
        "prores4444_mov":str(mov),
        "rgba_png_directory":str(frames_path),
        "limitations":"RGB is generated with an existing background. Black-background "
                      "color uncompositing is approximate. 128px alpha network is "
                      "upscaled. This is a separate conditional alpha diffusion "
                      "sidecar, NOT a LoRA on the H3 33B transformer. Measured "
                      "generated-alpha accuracy requires independent evaluation.",
    }
    (out/"end_to_end_report.json").write_text(json.dumps(report,indent=2))
    print("REAL_END_TO_END_GENERATED_RGBA",json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
