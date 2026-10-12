#!/usr/bin/env python3
"""Native Wan-Alpha-style MiniMax-H3 RGBA generation, single latent stream.

Two subprocess phases avoid 33B H3 denoiser + 2.6B VAE coexistence on A100:
    --generate-only : 33B H3 DiT + trained JOINT DiT LoRA; capture EXACT
                      denoised 24ch normalized latent before VAE decode.
    --decode-only   : original H3 VAE + trained Wan-JOINT bridge/dual LoRA;
                      decode SAME native 24ch latent as RGB AND alpha,
                      write RGBA PNGs and ProRes4444 MOV with original H3 audio.

No separately-generated alpha sidecar, no ground-truth alpha at inference,
and no artificial RGB masking. This only works with a DiT LoRA trained on
the joint latent distribution produced by WanJointH3Codec, not H3's original
RGB-only distribution. Tested against structure on CPU, A100 GPU execution
remains a separate milestone.
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Allow execution by absolute filename from /content Colab, not just repo cwd.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import numpy as np
import torch
from PIL import Image


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):
            h.update(c)
    return h.hexdigest()


def inject_trained_dit(model,checkpoint,rank):
    """Load *real* trained adapter into actual int8 H3 transformer."""
    from safetensors.torch import load_file
    from h3rgba.wan_h3_flow import inject_h3_transformer_lora,h3_dit_lora_state
    expected=load_file(str(checkpoint),device="cpu")
    if all(name in dict(model.named_parameters()) for name in expected):
        raise RuntimeError("H3 LoRA already injected; refusing duplicate application")
    count=inject_h3_transformer_lora(model,rank=rank)
    targets=dict(model.named_parameters())
    if set(expected)!=set(h3_dit_lora_state(model)):
        raise RuntimeError("Joint DiT LoRA checkpoint does not match H3 module layout")
    with torch.no_grad():
        for name,tensor in expected.items():
            param=targets[name]
            if param.shape!=tensor.shape:
                raise RuntimeError(f"H3 DiT LoRA parameter shape mismatch: {name}")
            param.copy_(tensor.to(param.device,param.dtype))
    print("WAN_JOINT_H3_DIT_ADAPTER_LOADED",json.dumps({
        "layers":count,"checkpoint":str(checkpoint),
        "checkpoint_sha256":sha256(checkpoint)}),flush=True)
    return count


def install_upstream_hooks(runner_module,adapter,rank,latent_file):
    """Capture final native H3 latents AND inject trained DiT LoRA.

    Patches this process only, not the upstream cloned repo on disk.
    """
    Runner=runner_module.MiniMaxH3Runner
    if getattr(Runner,"_wan_joint_hooked",False):
        raise RuntimeError("Wan joint hook already installed")
    load_impl=Runner._ensure_transformer

    def ensure_with_lora(self,*args,**kwargs):
        outcome=load_impl(self,*args,**kwargs)
        model=getattr(self._pipe,"transformer",None)
        if model is None:
            raise RuntimeError("H3 pretrained transformer is absent")
        if not getattr(model,"_wan_joint_lora_loaded",False):
            inject_trained_dit(model,adapter,rank)
            model._wan_joint_lora_loaded=True
        return outcome

    Runner._ensure_transformer=ensure_with_lora
    Runner._wan_joint_hooked=True

    native_decoder=runner_module._cpu_norm_video_decode_step
    def decoder_with_capture():
        step=native_decoder()
        original_call=step.__call__
        # Python instance __call__ override is NOT honored by special
        # method lookup. Return a dynamically-defined SUBCLASS instead.
        base=type(step)

        class CaptureJointLatentStep(base):
            @torch.inference_mode()
            def __call__(self,components,state):
                block_state=self.get_block_state(state)
                z=block_state.latents
                if z.ndim!=5 or z.shape[0]!=1 or z.shape[1]!=24:
                    raise RuntimeError(
                        "H3 native decoder received wrong normalized latent shape")
                latent_file.parent.mkdir(parents=True,exist_ok=True)
                pending=latent_file.with_suffix(".pt.partial")
                torch.save({
                    "latents":z.detach().cpu().float().contiguous(),
                    "source":"actual_h3_denoiser_with_trained_wan_joint_dit_lora",
                    "dit_lora_sha256":sha256(adapter),
                },pending)
                os.replace(pending,latent_file)
                print("WAN_JOINT_H3_NATIVE_LATENTS_CAPTURED",
                      json.dumps({"file":str(latent_file),
                                  "shape":list(z.shape)}),flush=True)
                return super().__call__(components,state)
        return CaptureJointLatentStep()

    runner_module._cpu_norm_video_decode_step=decoder_with_capture


def generate(args):
    if not Path(args.dit_lora).is_file():
        raise FileNotFoundError("TRAINED H3 joint-latent DiT LoRA is mandatory")
    for key,value in {
        "H3_TRANSFORMER_QUANT":"int8",
        "H3_TE_QUANT":"bnb-4bit",
        "H3_ATTN_BACKEND":"default",
        "H3_LOWVRAM":"0",
        "H3_TURBO_LORA":"0",
        "H3_CACHE":"none",
        "H3_PRUNED":"0",
        "H3_HYPERFLOW":"0",
    }.items():
        os.environ[key]=value
    upstream=Path(args.upstream_repo)
    if not (upstream/"core/runner.py").is_file():
        raise FileNotFoundError("Missing full H3 generation repo")
    sys.path.insert(0,str(upstream))
    import core.runner as runner_module
    root=Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=True)
    latent_file=root/"joint_h3_generated_normalized_latents.pt"
    # Old latents cannot be accepted as evidence of a newly generated sample.
    latent_file.unlink(missing_ok=True)
    install_upstream_hooks(runner_module,args.dit_lora,args.rank,latent_file)
    runner=runner_module.MiniMaxH3Runner(root/"upstream")
    result=runner.generate(
        prompt=args.prompt,height=args.height,width=args.width,
        seconds=args.seconds,num_inference_steps=args.steps,
        seed=args.seed,cache="none",attn="default",turbo=False,mute=False,
    )
    mp4=Path(result["mp4_path"])
    if not mp4.is_absolute():
        mp4=root/"upstream"/mp4
    if not mp4.is_file() or not latent_file.is_file():
        raise RuntimeError("Native H3 LoRA generation missing latent checkpoint/MP4")
    report={
        "status":"H3_NATIVE_JOINT_DIT_LORA_GENERATED_24CH",
        "rgb_audio_mp4":str(mp4),"generated_latents":str(latent_file),
        "dit_lora_hash":sha256(args.dit_lora),
        "generated_latents_sha256":sha256(latent_file),
        "alpha_stream_generated_separately":False,
        "requires_joint_vae_decode":True,
    }
    (root/"wan_h3_generation.json").write_text(json.dumps(report,indent=2))
    print("WAN_JOINT_DIT_GENERATION_COMPLETE",json.dumps(report),flush=True)


def decode(args):
    from h3rgba.alpha_latent_diffusion import unnormalized_latents
    from h3rgba.wan_joint_h3 import WanJointH3Codec
    from train_wan_dual_lora import make_model
    from safetensors.torch import load_file
    root=Path(args.output_dir)
    report_file=root/"wan_h3_generation.json"
    if not report_file.is_file():
        raise FileNotFoundError(
            "Joint DiT generation report missing. Run --generate-only first")
    report=json.loads(report_file.read_text())
    if sha256(args.dit_lora)!=report["dit_lora_hash"]:
        raise RuntimeError("DiT adapter differs from generation checkpoint")
    if sha256(report["generated_latents"])!=report.get("generated_latents_sha256"):
        raise RuntimeError("Captured native H3 joint latent file is stale or corrupted")
    src=torch.load(report["generated_latents"],map_location="cpu",weights_only=True)
    if src.get("dit_lora_sha256")!=sha256(args.dit_lora):
        raise RuntimeError("Captured latents originated from a different DiT LoRA")
    z=src["latents"]
    if z.ndim!=5 or z.shape[1]!=24 or not torch.isfinite(z).all():
        raise RuntimeError("Invalid generated H3 joint latent capture")
    dual,_=make_model(argparse.Namespace(rank=4,merger_scale=.15,
                                         resolution=128),torch.device("cuda"))
    dual.load_adapter(args.joint_dual)
    joint=WanJointH3Codec(dual,hidden=args.hidden,
                          shift_limit=args.shift_limit,
                          bridge_scale=args.bridge_scale,
                          train_decoder_lora=False).cuda()
    joint.load_bridge(args.joint_bridge)
    joint.eval()
    with torch.inference_mode():
        joint_raw=unnormalized_latents(z.cuda(),dual.vae)
        rgb,alpha=joint.decode_native_h3_latents(joint_raw)
        rgb=rgb[0].permute(1,2,3,0).cpu().numpy()
        alpha=alpha[0,0].cpu().numpy()
    if rgb.shape[:3]!=alpha.shape:
        raise RuntimeError("Native decoded RGBA shapes differ")
    frames_dir=root/"native_rgba_png"
    frames_dir.mkdir(parents=True,exist_ok=True)
    for i in range(rgb.shape[0]):
        pixels=np.uint8(np.rint(np.concatenate(
            [rgb[i].clip(0,1),alpha[i,:,:,None].clip(0,1)],axis=2)*255))
        Image.fromarray(pixels,mode="RGBA").save(frames_dir/f"frame_{i:05d}.png")
    mp4=Path(report["rgb_audio_mp4"])
    destination=root/"h3_wan_joint_rgba.mov"
    cmd=[
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-framerate","24","-i",str(frames_dir/"frame_%05d.png"),
        "-i",str(mp4),"-map","0:v:0","-map","1:a:0?",
        "-c:v","prores_ks","-profile:v","4","-pix_fmt","yuva444p10le",
        "-c:a","pcm_s16le","-shortest",str(destination)
    ]
    subprocess.run(cmd,check=True)
    probe=subprocess.run(
        ["ffprobe","-v","error","-select_streams","v:0",
         "-show_entries","stream=codec_name,pix_fmt,nb_frames",
         "-of","json",str(destination)],
        capture_output=True,text=True,check=True)
    stream=json.loads(probe.stdout)["streams"][0]
    if not stream["pix_fmt"].startswith("yuva"):
        raise RuntimeError("ProRes4444 failed to preserve alpha")
    final={
        **report,"status":"H3_WAN_JOINT_NATIVE_RGBA_EXPORTED_UNVALIDATED",
        "rgba_mov":str(destination),"png_dir":str(frames_dir),
        "video_frames":int(rgb.shape[0]),
        "alpha_hard_coverage":float((alpha>.5).mean()),
        "pixel_format":stream["pix_fmt"],
        "gt_alpha_used_during_generation":False,
        "warning":"Alpha fidelity and full 124-frame temporal consistency "
                  "NOT verified until real generated video is reviewed.",
    }
    (root/"wan_h3_rgba_report.json").write_text(json.dumps(final,indent=2))
    print("WAN_JOINT_NATIVE_RGBA_EXPORTED",json.dumps(final),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate-only",action="store_true")
    mode.add_argument("--decode-only",action="store_true")
    p.add_argument("--dit-lora",required=True)
    p.add_argument("--joint-bridge",default=None)
    p.add_argument("--joint-dual",default=None)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--shift-limit",type=float,default=.35)
    p.add_argument("--bridge-scale",type=float,default=1.)
    p.add_argument("--rank",type=int,default=8)
    p.add_argument("--upstream-repo",default="/content/Diffusers_minimax-h3")
    p.add_argument("--output-dir",default="/content/h3_wan_native_rgba")
    p.add_argument("--height",type=int,default=544)
    p.add_argument("--width",type=int,default=960)
    p.add_argument("--seconds",type=float,default=5.)
    p.add_argument("--steps",type=int,default=30)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--prompt",default="A red fox in front of a transparent background, "
                                    "fine hair, studio lighting, cinematic animation")
    args=p.parse_args()
    if args.generate_only:
        generate(args)
    else:
        if not args.joint_bridge or not args.joint_dual:
            p.error("--decode-only needs trained --joint-bridge and --joint-dual")
        decode(args)


if __name__=="__main__":
    main()
