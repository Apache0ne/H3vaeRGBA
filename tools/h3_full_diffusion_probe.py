#!/usr/bin/env python3
"""Real MiniMax-H3 diffusion smoke test (t2va video+audio), not VAE-only.

Uses the OFFICIAL ModularPipeline, true pretrained diffusion transformer,
Qwen3-VL text encoder, two schedulers, audio VAE and video VAE.

IMPORTANT: this generates conventional RGB video. The standalone dual-path
alpha-decoder LoRA cannot generate a matte without independently generated
alpha latents. This script deliberately reports that as UNSUPPORTED, never
manufactures a mask or calls ordinary RGB output a transparency success.

The official int8 CPU-streamed recipe holds ~75 decimal GB of quantized
weights in system RAM. 22 GB GPU RAM alone cannot make a ~53 GB-RAM Colab
runtime fit. Run --preflight-only before any multi-gigabyte downloads.

Reference: https://huggingface.co/docs/diffusers/api/pipelines/minimax_h3
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


MODEL = "MiniMaxAI/MiniMax-H3"
# 61.7 GB transformer + 62.1 GB conditioner + 10.4 GB video VAE in
# released BF16 checkpoints. Disk check deliberately allows extra space.
MIN_AVAILABLE_RAM_GB = 85
MIN_FREE_DISK_GB = 155


def resources(output_dir, *, available_ram_gb=None, free_disk_gb=None,
              cuda_available=None, bf16_available=None):
    """Pure testable host-resource contract; no model import/download."""
    if available_ram_gb is None:
        try:
            # Prefer actual memory available (includes reclaimable cache).
            import psutil
            available_ram_gb = psutil.virtual_memory().available / 1e9
            total_ram_gb = psutil.virtual_memory().total / 1e9
        except ImportError:
            available_ram_gb = (os.sysconf("SC_AVPHYS_PAGES") *
                                os.sysconf("SC_PAGE_SIZE")) / 1e9
            total_ram_gb = (os.sysconf("SC_PHYS_PAGES") *
                            os.sysconf("SC_PAGE_SIZE")) / 1e9
    else:
        total_ram_gb = None
    if free_disk_gb is None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        free_disk_gb = shutil.disk_usage(path).free / 1e9
    if cuda_available is None or bf16_available is None:
        try:
            import torch
            if cuda_available is None:
                cuda_available = torch.cuda.is_available()
            if bf16_available is None:
                bf16_available = (torch.cuda.is_available() and
                                  torch.cuda.is_bf16_supported())
        except ImportError:
            cuda_available = bool(cuda_available)
            bf16_available = bool(bf16_available)
    checks = {
        "ram": available_ram_gb >= MIN_AVAILABLE_RAM_GB,
        "disk": free_disk_gb >= MIN_FREE_DISK_GB,
        "cuda": bool(cuda_available),
        "bf16": bool(bf16_available),
    }
    return {
        "available_system_ram_decimal_GB":round(available_ram_gb,2),
        "total_system_ram_decimal_GB":round(total_ram_gb,2)
                                      if total_ram_gb is not None else None,
        "free_output_disk_decimal_GB":round(free_disk_gb,2),
        "min_available_ram_decimal_GB": MIN_AVAILABLE_RAM_GB,
        "min_free_disk_decimal_GB": MIN_FREE_DISK_GB,
        "gpu_cuda":bool(cuda_available),
        "gpu_bf16":bool(bf16_available),
        "checks": checks,
        "can_attempt_official_int8_streamed_inference": all(checks.values()),
        "memory_note":"The official guide estimates about 75 GB host RAM "
                      "for INT8 streaming. This conservative guard includes "
                      "headroom. GPU VRAM is NOT host RAM.",
    }


def inspect_result(result, *, requested_frames, requested_width, requested_height):
    """Sanity-check real decoded video+audio from the official H3 pipeline."""
    if not isinstance(result, dict):
        try:
            result = dict(result)
        except Exception as exc:
            raise RuntimeError("Expected H3 modular pipeline output mapping") from exc
    if "videos" not in result or "audio" not in result or "sampling_rate" not in result:
        raise RuntimeError("Missing official H3 videos/audio/sampling_rate outputs")
    if "latents" not in result:
        raise RuntimeError("Missing original H3 denoised video latents: "
                           "cannot verify diffusion/VAE latent interface")
    latents = result["latents"]
    latent_shape = tuple(latents.shape)
    expected_latent_frames = (requested_frames - 5) // 17 * 5 + 2
    expected_latent_shape = (
        1, 24, expected_latent_frames,
        requested_height // 16, requested_width // 16
    )
    if latent_shape != expected_latent_shape:
        raise RuntimeError(f"H3 output latents shape {latent_shape}, "
                           f"expected {expected_latent_shape}")
    videos = result["videos"]
    if not videos or not videos[0]:
        raise RuntimeError("H3 video result is empty")
    frames = videos[0]
    if len(frames) != requested_frames:
        raise RuntimeError(
            f"H3 generated {len(frames)} frames, expected {requested_frames}"
        )
    first = frames[0]
    if hasattr(first, "size") and isinstance(first.size, tuple):
        w,h = first.size
    elif hasattr(first, "shape"):
        shape = tuple(first.shape)
        if len(shape) < 3:
            raise RuntimeError(f"Unexpected generated frame shape: {shape}")
        h,w = shape[0],shape[1]
    else:
        raise RuntimeError("Unrecognized frame representation")
    if (w,h) != (requested_width,requested_height):
        raise RuntimeError(f"Generated {(w,h)} instead of {(requested_width,requested_height)}")
    sr = result["sampling_rate"]
    if int(sr) <= 0:
        raise RuntimeError("Invalid H3 audio sample rate")
    audio = result["audio"]
    if audio is None or len(audio) == 0:
        raise RuntimeError("Generated audio is empty")
    return {
        "generated_video_frames":len(frames), "width":w, "height":h,
        "denoised_video_latents_shape":list(latent_shape),
        "denoised_video_latent_channels":latent_shape[1],
        "audio_sample_rate":int(sr),
        "returned_alpha_field":any(
            k in result for k in ("alpha", "alphas", "alpha_latents", "masks")
        ),
        "alpha_is_generated_by_trained_diffusion_model":False,
        "supports_true_rgba_generation":False,
        "status":"REAL_H3_RGB_AUDIO_DIFFUSION_ONLY",
    }


def alpha_generation_contract(*, has_diffusion_alpha_latents=False,
                              has_supplied_ground_truth_alpha=False):
    """Correctness gate: VAE LoRA alone is NOT an alpha diffusion model."""
    return {
        "original_h3_generates_rgb_video":True,
        "original_h3_generates_alpha_latents":False,
        "has_diffusion_alpha_latents":bool(has_diffusion_alpha_latents),
        "has_supplied_ground_truth_alpha":bool(has_supplied_ground_truth_alpha),
        "can_claim_generated_alpha":bool(has_diffusion_alpha_latents),
        "can_evaluate_oracle_alpha_reconstruction":bool(has_supplied_ground_truth_alpha),
        "note":"Generated 24-channel RGB video latents are NOT the missing "
               "24-channel alpha latent stream. An external GT alpha supplied "
               "for reconstruction is not generated transparency.",
    }


def generate(args, report):
    """Use the official supported int8+streamed CPU-offload recipe."""
    import torch
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    from diffusers.utils.export_utils import encode_video

    print("LOADING REAL H3 t2va TRANSFORMER + QWEN3-VL CONDITIONER", flush=True)
    print("DO NOT RUN tools/colab_compat.py --repair IN THIS ENVIRONMENT: "
          "it may uninstall torchao required for int8", flush=True)
    pipe = ModularPipeline.from_pretrained(MODEL, workflow="t2va")
    pipe.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            MODEL, subfolder="transformer", dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "proj_in", "audio_proj_in", "context_embedder",
                    "time_embedder", "time_proj", "token_refiner",
                    "norm_out", "proj_out", "audio_proj_out",
                ],
            ),
            low_cpu_mem_usage=False,
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, subfolder="text_encoder", dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "model.visual", "model.language_model.embed_tokens",
                    "model.language_model.norm", "lm_head",
                ],
            ),
        ),
    )
    pipe.load_components(dtype=torch.bfloat16)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    offload=dict(onload_device=torch.device("cuda"),
                 offload_device=torch.device("cpu"),use_stream=True)
    pipe.transformer.enable_group_offload(
        offload_type="block_level",num_blocks_per_group=1,**offload
    )
    apply_group_offloading(
        pipe.text_encoder.model,offload_type="leaf_level",**offload
    )
    pipe.vae.to("cuda")
    pipe.audio_vae.to("cuda")
    report["loaded_real_diffusion_components"] = True
    torch.cuda.reset_peak_memory_stats()

    print("RUNNING ACTUAL H3 DIFFUSION",flush=True)
    began=time.monotonic()
    with torch.inference_mode():
        outputs=pipe(
            prompt=args.prompt, height=args.height, width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
            output=["videos","audio","sampling_rate","latents"],
        )
    torch.cuda.synchronize()
    report["diffusion_runtime_seconds"]=round(time.monotonic()-began,2)
    report["peak_cuda_reserved_GiB"]=round(torch.cuda.max_memory_reserved()/2**30,3)
    report["output_validation"]=inspect_result(
        outputs,requested_frames=args.num_frames,
        requested_width=args.width,requested_height=args.height
    )
    output=Path(args.output_dir)
    result_path=output/"h3_real_diffusion_rgb_audio.mp4"
    encode_video(outputs["videos"][0],fps=24,output_path=str(result_path),
                 audio=outputs["audio"][0],audio_sample_rate=outputs["sampling_rate"])
    report["generated_rgb_audio_video"]=str(result_path)
    report["status"]="H3_REAL_T2VA_RGB_AUDIO_GENERATION_COMPLETE_ALPHA_NOT_GENERATED"
    print("REAL_H3_DIFFUSION_RGB_AUDIO_COMPLETE",result_path,flush=True)
    print("ALPHA_DIFFUSION_NOT_TRAINED: no generated matte available",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--force-resource-override",action="store_true",
                   help="Dangerous: bypass host-RAM/disk guard; may be OOM killed")
    p.add_argument("--output-dir",default="/content/h3_real_diffusion_probe")
    p.add_argument("--prompt",default="A colorful planet with thin blue rings, drifting slowly through space")
    p.add_argument("--height",type=int,default=192)
    p.add_argument("--width",type=int,default=320)
    p.add_argument("--num-frames",type=int,default=124,
                   help="H3 requires 17*n+5; minimum 5sec at 24fps")
    p.add_argument("--num-inference-steps",type=int,default=3,
                   help="Sigma grid points (3 -> 2 forward denoising evaluations)")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if args.width%32 or args.height%32 or min(args.width,args.height)<32:
        p.error("Height and width must be positive multiples of 32")
    if args.num_frames != 124 and not (
        args.num_frames>=124 and (args.num_frames-5)%17==0
        and args.num_frames<=362
    ):
        p.error("H3 24fps 5-15s requires 17*n+5 frames, minimum 124")
    if args.num_inference_steps<2:
        p.error("At least 2 sigma grid points are required")
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    report={
        "model":MODEL,"workflow":"t2va",
        "prompt":args.prompt,"seed":args.seed,
        "requested_width":args.width,"requested_height":args.height,
        "requested_frames":args.num_frames,"inference_steps":args.num_inference_steps,
        "loaded_real_diffusion_components":False,
        "alpha_generation_contract":alpha_generation_contract(),
    }
    proof=resources(output)
    report["resources"]=proof
    print("H3_FULL_DIFFUSION_RESOURCE_PREFLIGHT",
          json.dumps(proof,indent=2),flush=True)
    if args.preflight_only:
        report["status"]="PREFLIGHT_ONLY_NO_WEIGHTS_LOADED"
        (output/"h3_full_diffusion_report.json").write_text(json.dumps(report,indent=2))
        print("PREFLIGHT_COMPLETE_NO_MODEL_DOWNLOAD",flush=True)
        return
    if not proof["can_attempt_official_int8_streamed_inference"]:
        if not args.force_resource_override:
            report["status"]="BLOCKED_INSUFFICIENT_RESOURCES"
            (output/"h3_full_diffusion_report.json").write_text(json.dumps(report,indent=2))
            raise SystemExit("Full H3 requires substantially more HOST RAM than the "
                             "VAE-only experiment. Guard stopped downloads. "
                             "Use a high-RAM runtime; no fake RGBA result produced.")
        print("RESOURCE OVERRIDE: risk of memory kill or disk exhaustion",flush=True)
    try:
        generate(args,report)
    finally:
        (output/"h3_full_diffusion_report.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
