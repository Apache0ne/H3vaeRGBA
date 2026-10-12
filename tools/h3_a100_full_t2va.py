#!/usr/bin/env python3
"""FULL MiniMax-H3 video+audio diffusion on 1x A100 80GB and ~50GB host RAM.

This uses an external H3 runner's validated phased residency implementation:
https://github.com/animede/Diffusers_minimax-h3
(LoRA VAE training from this repo remains entirely separate.)

Runtime method (NOT full-bf16 CPU-auto-offload):
 - Qwen3-VL-32B text encoder in 4-bit NF4 on GPU (~21GB)
 - real H3 T2VA 33B diffusion transformer in torchao INT8 (~34GB GPU)
 - original pretrained video+audio VAEs, phase-managed
 - SDPA attention for A100 (Ampere; not Hopper/Blackwell kernels)
Upstream reports ~59.7GB peak VRAM for its 80GB-tier configuration. It
does not report an A100/53GB-RAM specific completed run, so this script
records *actual* measurements and will fail rather than invent success.

Usage:
  python tools/h3_a100_full_t2va.py --preflight-only
  python tools/h3_a100_full_t2va.py --download-only
  python tools/h3_a100_full_t2va.py --generate --steps 30

The release checkpoint is >140GB and can require hours to fetch. No need
to download transformer_ref/ nor full 498GB repo for T2VA. This does not
generate alpha: alpha VAE LoRA must be coupled to a *trained* diffusion
alpha latent predictor to generate transparency.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


MODEL_ID = "MiniMaxAI/MiniMax-H3"
RUNNER_URL = "https://github.com/animede/Diffusers_minimax-h3"
# Only files needed for original text-to-video-and-audio workflow.
PATTERNS = (
    "modular_model_index.json", "transformer/*",
    "text_encoder/*", "vae/*", "audio_vae/*",
    "scheduler/*", "audio_scheduler/*",
    "tokenizer/*", "processor/*",
)


def inspect_host(*, gpu_gib=None, ram_available_gb=None,
                 ram_total_gb=None, disk_free_gb=None,
                 phase="download", disk_path=None):
    """Before download require storage for weights; after download just working space."""
    if phase not in ("download", "generate"):
        raise ValueError("phase must be download or generate")
    """A100-phase residency check; NOT the 85GB CPU-streaming guard."""
    import psutil
    if gpu_gib is None:
        import torch
        gpu_gib = (torch.cuda.get_device_properties(0).total_memory / 2**30
                   if torch.cuda.is_available() else 0.0)
    vm = psutil.virtual_memory()
    if ram_available_gb is None:
        ram_available_gb = vm.available / 1e9
    if ram_total_gb is None:
        ram_total_gb = vm.total / 1e9
    if disk_path is None:
        disk_path = os.environ.get("HF_HOME") or (
            "/content" if Path("/content").exists() else "/")
    disk_path = Path(disk_path).expanduser()
    disk_path.mkdir(parents=True, exist_ok=True)
    if disk_free_gb is None:
        disk_free_gb = shutil.disk_usage(disk_path).free / 1e9
    # Once weights are downloaded, generation requires only working space,
    # not an additional 155 GB of free storage.
    required_disk_gb = 155. if phase == "download" else 10.
    # Peak VRAM 59.7GB upstream; 70GiB minimum provides meaningful margin
    # without assuming full 80GiB free. Host RAM ~53GB is fine in principle,
    # but peak load-time RAM still must be measured, not inferred.
    checks = {
        "gpu_80gb_class": gpu_gib >= 70.0,
        "host_ram_at_least_40gb": ram_total_gb >= 40.0,
        "available_host_ram_25gb": ram_available_gb >= 25.0,
        ("free_disk_155gb" if phase=="download" else "free_disk_10gb"):
            disk_free_gb >= required_disk_gb,
    }
    return {
        "gpu_gib":round(gpu_gib,2),
        "host_ram_total_decimal_gb":round(ram_total_gb,2),
        "host_ram_available_decimal_gb":round(ram_available_gb,2),
        "disk_free_decimal_gb":round(disk_free_gb,2),
        "disk_check_path":str(disk_path.resolve()),
        "disk_phase":phase,
        "disk_required_decimal_gb":required_disk_gb,
        "checks": checks, "can_proceed":all(checks.values()),
        "configuration":"GPU NF4 Qwen text + INT8 H3 transformer, "
                        "SDPA on Ampere, phase-managed VAE",
        "upstream_peak_vram_decimal_gb":59.7,
        "note":"VRAM and host RAM are separate. Minimums guard against "
               "likely failure but cannot guarantee quantization/load fit.",
    }


def verify_mp4(path, *, min_frames=124):
    """Verify real muxed video/audio with PyAV; no fabricated success."""
    import av
    target=Path(path)
    if not target.is_file() or target.stat().st_size <= 4096:
        raise RuntimeError(f"No real MP4 output at {target}")
    with av.open(str(target)) as container:
        video_streams = list(container.streams.video)
        audio_streams = list(container.streams.audio)
        if not video_streams or not audio_streams:
            raise RuntimeError("H3 output MP4 must contain BOTH video and audio")
        stream=video_streams[0]
        n=sum(1 for _ in container.decode(video=0))
        if n < min_frames:
            raise RuntimeError(f"H3 generated {n} frames, expected >= {min_frames}")
        if stream.width < 256 or stream.height < 256:
            raise RuntimeError("Unexpected H3 output video resolution")
    return {"mp4_exists":True, "mp4_bytes":target.stat().st_size,
            "decoded_video_frames":n, "has_audio_stream":True,
            "width":stream.width,"height":stream.height}


def download_t2va(cache_dir=None):
    """Only T2VA partitions, not transformer_ref or whole ~498GB repository."""
    from huggingface_hub import snapshot_download
    kwargs={"repo_id":MODEL_ID,"allow_patterns":list(PATTERNS),
            "max_workers":3}
    if cache_dir:
        kwargs["cache_dir"]=str(cache_dir)
    print("DOWNLOADING_REAL_H3_T2VA_FILES",json.dumps(kwargs),flush=True)
    return snapshot_download(**kwargs)


def import_upstream(repo):
    repo=Path(repo).resolve()
    target=repo/"core"/"runner.py"
    if not target.is_file():
        raise FileNotFoundError(
            f"Missing external runner: {target}. Clone {RUNNER_URL} there."
        )
    # Avoid accidentally selecting incompatible community variants or the
    # VAE-only runner. These env vars are read at module import time.
    os.environ.update({
        "H3_TRANSFORMER_QUANT":"int8",
        "H3_TE_QUANT":"bnb-4bit",
        "H3_ATTN_BACKEND":"default",
        "H3_LOWVRAM":"0",
        "H3_TURBO_LORA":"0",
        "H3_CACHE":"none",
        "H3_PRUNED":"0",
        "H3_HYPERFLOW":"0",
    })
    sys.path.insert(0,str(repo))
    from core.runner import MiniMaxH3Runner
    return MiniMaxH3Runner


def generate(args):
    import gc
    import psutil
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A100 GPU not detected")
    major, minor=torch.cuda.get_device_capability(0)
    if (major,minor)!=(8,0):
        print(f"WARNING: GPU sm_{major}{minor} is not A100 sm_80; "
              "using SDPA anyway",flush=True)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU required")

    Runner=import_upstream(args.upstream_repo)
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    mem_before=psutil.virtual_memory().available/1e9
    print("ACTUAL_H3_DIFFUSION_START",
          json.dumps({"gpu":torch.cuda.get_device_name(0),
                      "prompt":args.prompt,"steps":args.steps,
                      "resolution":[args.width,args.height],
                      "seconds":args.seconds,
                      "ram_available_gb_before":mem_before}),flush=True)
    started=time.monotonic()
    runner=Runner(output)
    with torch.inference_mode():
        result=runner.generate(
            prompt=args.prompt,
            height=args.height,width=args.width,
            seconds=args.seconds,
            num_inference_steps=args.steps, seed=args.seed,
            cache="none",attn="default",turbo=False, mute=False,
        )
    target=Path(result["mp4_path"])
    if not target.is_absolute():
        target=output/target
    proof=verify_mp4(target)
    report={
        "mode":"REAL_H3_PRETRAINED_FULL_T2VA_A100",
        "runner":RUNNER_URL,
        "actual_gpu":torch.cuda.get_device_name(0),
        "gpu_memory_allocated_peak_gib":round(
            torch.cuda.max_memory_allocated()/2**30,3),
        "gpu_memory_reserved_peak_gib":round(
            torch.cuda.max_memory_reserved()/2**30,3),
        "host_ram_available_after_gb":round(
            psutil.virtual_memory().available/1e9,3),
        "elapsed_seconds":round(time.monotonic()-started,2),
        "generation":result,"verified_output":proof,
        "generated_rgb_audio_video":str(target),
        "alpha_generated":False,
        "alpha_note":"This real original H3 diffusion model outputs RGB "
                     "video+audio, not a 24-channel alpha latent stream. "
                     "Trained VAE alpha LoRA is NOT a diffusion alpha model.",
    }
    (output/"h3_a100_full_t2va_report.json").write_text(json.dumps(report,indent=2))
    print("REAL_H3_DIFFUSION_RGB_AUDIO_VERIFIED",str(target),flush=True)
    print("ALPHA_NOT_GENERATED",flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only",action="store_true")
    modes.add_argument("--download-only",action="store_true")
    modes.add_argument("--generate",action="store_true")
    p.add_argument("--upstream-repo",default="/content/Diffusers_minimax-h3")
    p.add_argument("--output-dir",default="/content/h3_a100_diffusion_output")
    p.add_argument("--hf-home",default=None,
                   help="Explicit writable Hugging Face cache home, e.g. scratch mount")
    p.add_argument("--disk-phase",choices=("download","generate"),default="download",
                   help="Storage required for preflight: before weights or after download")
    p.add_argument("--height",type=int,default=544)
    p.add_argument("--width",type=int,default=960)
    p.add_argument("--seconds",type=float,default=5.0)
    p.add_argument("--steps",type=int,default=30,
                   help="30 sigma grid points (~29 evaluations), not a fake 2-step sample")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--prompt",default=(
        "A pastel pink ringed planet slowly rotates against black space, "
        "its thin blue rings shimmering, cinematic lighting; soft ambient sound"
    ))
    p.add_argument("--skip-resource-guard",action="store_true",
                   help="Override safety check after inspecting system manually")
    args=p.parse_args()
    if args.hf_home:
        # Must take effect BEFORE huggingface_hub or diffusers are imported.
        home=Path(args.hf_home).expanduser().resolve()
        home.mkdir(parents=True,exist_ok=True)
        os.environ["HF_HOME"]=str(home)
    if args.height<256 or args.width<256 or args.height%32 or args.width%32:
        p.error("Height/width must be >=256 and divisible by 32")
    if args.seconds<5 or args.seconds>15 or args.steps<3:
        p.error("Seconds must be 5..15 and steps >=3")
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    proof=inspect_host(phase=("generate" if args.generate else args.disk_phase))
    (output/"h3_a100_preflight.json").write_text(json.dumps(proof,indent=2))
    print("A100_PHASED_RESOURCE_PREFLIGHT",json.dumps(proof,indent=2),flush=True)
    if args.preflight_only:
        return
    if not proof["can_proceed"] and not args.skip_resource_guard:
        raise SystemExit("RESOURCE_PREFLIGHT_BLOCKED: inspect h3_a100_preflight.json")
    if args.download_only:
        print("SNAPSHOT",download_t2va(),flush=True)
        return
    report=generate(args)
    print("DONE",json.dumps({"generated_mp4":report["generated_rgb_audio_video"],
                              "peak_vram_gib":report["gpu_memory_reserved_peak_gib"],
                              "status":"RGB_AUDIO_GENERATED_ALPHA_NOT_TRAINED"}),flush=True)


if __name__=="__main__":
    main()
