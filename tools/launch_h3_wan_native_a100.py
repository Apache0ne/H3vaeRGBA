#!/usr/bin/env python3
"""Complete experimental Wan-Alpha TWO-STAGE port for MiniMax-H3 on A100 80GB.

Prerequisites: Existing official H3 weights and 500-step dual VAE checkpoint
in Drive. Reuse the already prepared 403 paired Wan-Alpha videos.

Stages:
  0. Match ACTUAL captioned Wan-Alpha T2V CSV clips to manifest.
  1. Train a ONE-STREAM 24ch H3 joint RGB+alpha VAE. 2-step safety smoke,
     then continue training from same optimizer state. Backups to Drive.
  2. Cache normalized joint video latents (not separate sidecar stream).
  3. Encode authentic prompts with frozen Qwen3-VL conditioner in separate
     Python 3.12/A100 subprocess; release its GPU memory afterwards.
  4. Fine-tune genuine H3 33B DiT LoRA on these merged 24ch latents,
     with correct packed rotary layout & H3 data-ward velocity.
  5. Generate with this H3 LoRA and save the ACTUAL 24ch denoised latents.
  6. Decode SAME latents to RGB and alpha via the joint VAE, save transparent
     ProRes4444 + H3 audio. No standalone alpha sidecar, no GT-alpha inference.

The A100 GPU stages have NOT yet been validated end-to-end; stage 2 will
stop rather than fake success if the 33B int8 backward is unsupported.
Run --preflight-only before allocating GPU.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Allow execution by absolute filename from /content Colab, not just repo cwd.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from tools.launch_h3_rgba_a100 import launch
from tools.validate_h3_wan_cache import (
    validate_joint_cache, validate_text_cache, sha256 as file_sha256,
)


REPO=Path(__file__).resolve().parent.parent


def generation_is_current(report_path, trained_dit_lora):
    """A JSON file alone never proves that the CURRENT LoRA generated the video."""
    report_path=Path(report_path)
    if not report_path.is_file():
        return False
    try:
        prior=json.loads(report_path.read_text())
        return (
            prior.get("dit_lora_hash")==file_sha256(trained_dit_lora)
            and Path(prior["rgb_audio_mp4"]).is_file()
            and Path(prior["generated_latents"]).is_file()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(args):
    out=Path(args.run_root)
    out.mkdir(parents=True,exist_ok=True)
    os.environ["H3_E2E_LOG_PATH"]=str(out/"wan_joint_launcher.log")
    old=Path(args.old_vae_lora)
    manifest=Path(args.manifest)
    if not old.is_file():
        raise FileNotFoundError("Missing proven 500-step dual VAE LoRA: "+str(old))
    if not manifest.is_file():
        raise FileNotFoundError("Missing paired Wan training manifest: "+str(manifest))
    phases={
        "captions":out/"official_wan_captions.json",
        "vae":out/"joint_vae",
        "latents":out/"joint_latents",
        "text":out/"qwen_text",
        "dit":out/"joint_dit",
        "generation":out/"native_generation",
    }
    print("WAN_H3_NATIVE_PLAN",json.dumps({
        "old_v1_dual_checkpoint":str(old),
        "manifest":str(manifest),
        "run_root":str(out),
        "new_stage1_vae_steps":args.vae_steps,
        "new_stage2_dit_steps":args.dit_steps,
        "method":"Wan-Alpha single joint video latent, H3 DiT LoRA, native RGB and alpha VAE decode",
        "not_used":"old 4000-step independent alpha diffusion sidecar",
        "warnings":"Native H3 transformer+VAE training remains UNVERIFIED on A100",
    },indent=2),flush=True)
    if args.preflight_only:
        launch([sys.executable,REPO/"tools/import_wan_alpha_captions.py",
                "--manifest",manifest,"--output",phases["captions"]])
        print("WAN_H3_PREFLIGHT_COMPLETE: source captions matched, no pretrained models loaded",flush=True)
        return
    if not os.path.isdir("/content/drive"):
        raise RuntimeError("Google Drive must be mounted for persistent checkpoint backups")
    backup=Path(args.backup_root)
    backup.mkdir(parents=True,exist_ok=True)
    if not phases["captions"].is_file():
        launch([sys.executable,REPO/"tools/import_wan_alpha_captions.py",
                "--manifest",manifest,"--output",phases["captions"]])

    def vae_args():
        return [sys.executable,"-u",REPO/"train_h3_wan_joint.py",
                "--manifest",manifest,"--vae-lora",old,
                "--output-dir",phases["vae"],
                "--backup-dir",backup/"joint_vae",
                "--resolution","128","--hidden","128",
                "--shift-limit",".35","--bridge-scale","1",
                "--train-decoder-lora","--seed",str(args.seed),
                "--val-every","50","--save-every","50"]
    latest=phases["vae"]/"checkpoints/joint_latest.pt"
    bridge=phases["vae"]/"checkpoints/joint_bridge_best.safetensors"
    adapted=phases["vae"]/"checkpoints/joint_dual_best.safetensors"
    if not latest.is_file():
        print("[WAN STAGE 1] TWO-STEP actual VAE training smoke",flush=True)
        launch(vae_args()+["--steps","2"])
    print("[WAN STAGE 1] Train/continue single-latent RGB+A VAE",flush=True)
    launch(vae_args()+["--resume","--steps",str(args.vae_steps)])
    if not bridge.is_file() or not adapted.is_file():
        raise RuntimeError("Shared-latent VAE adaptation not saved")
    stage1_metrics=phases["vae"]/"checkpoints/joint_best_metrics.json"
    if not stage1_metrics.is_file():
        raise RuntimeError("No stage-1 joint-latent validation metrics available")
    metrics=json.loads(stage1_metrics.read_text())
    print("[WAN STAGE 1 QUALITY GATE]",json.dumps({
        "best_alpha_iou":metrics.get("alpha_iou"),
        "best_rgb_visible_l1":metrics.get("rgb_visible_l1"),
        "minimum_alpha_iou":args.min_stage1_iou,
        "maximum_rgb_l1":args.max_stage1_rgb_l1,
    }),flush=True)
    if (metrics.get("alpha_iou",0)<args.min_stage1_iou or
        metrics.get("rgb_visible_l1",100)>args.max_stage1_rgb_l1):
        raise RuntimeError(
            "WAN JOINT STAGE 1 DID NOT ACHIEVE USEFUL SINGLE-STREAM RGBA "
            "RECONSTRUCTION. STOPPING BEFORE COSTLY H3 33B DiT TRAINING. "
            "Inspect joint VAE training/visuals and improve merger first. "
            f"Best metrics: {metrics}"
        )
    print("[WAN STAGE 1 COMPLETE] Joint RGB and alpha VAE quality gate passed",flush=True)

    print("[WAN STAGE 2] Encode joint 24-channel latent video targets",flush=True)
    index=phases["latents"]/"index.json"
    cache_inputs = dict(
        bridge=bridge, dual=adapted, base_lora=old,
        manifest=manifest, captions=phases["captions"],
        resolution=args.dit_cache_resolution, seed=args.seed,
    )
    if index.is_file():
        try:
            validate_joint_cache(index, **cache_inputs)
        except (RuntimeError, FileNotFoundError, KeyError, ValueError) as exc:
            print("[WAN STAGE 2] Invalid cached latents, rebuilding:", exc, flush=True)
            index.unlink()
    if not index.is_file():
        launch([sys.executable,"-u",REPO/"tools/cache_h3_wan_joint_targets.py",
                "--manifest",manifest,
                "--captions-json",phases["captions"],
                "--joint-bridge",bridge,"--joint-dual",adapted,
                "--base-vae-lora",old,
                "--output-dir",phases["latents"],
                "--resolution",str(args.dit_cache_resolution),
                "--hidden","128","--seed",str(args.seed)])
    cache_counts=validate_joint_cache(index, **cache_inputs)
    print("[WAN STAGE 2 COMPLETE] Verified fused video latents",
          json.dumps(cache_counts),flush=True)

    py=Path(args.a100_python)
    if not py.is_file():
        raise FileNotFoundError("Need existing isolated A100 H3 Python 3.12: "+str(py))
    print("[WAN STAGE 3] Cache real caption Qwen3-VL-32B embeddings",flush=True)
    prompts=phases["text"]/"prompts.json"
    # Missing manifest cannot establish integrity of existing per-prompt files.
    force_qwen_reencode=not prompts.is_file()
    if prompts.is_file():
        try:
            validate_text_cache(index, prompts)
        except (RuntimeError, FileNotFoundError, KeyError, ValueError) as exc:
            print("[WAN STAGE 3] Invalid cached text prompts, rebuilding:", exc, flush=True)
            prompts.unlink()
            force_qwen_reencode=True
    if not prompts.is_file():
        launch([py,"-u",REPO/"tools/cache_h3_wan_text_embeddings.py",
                "--joint-index",index,
                "--output-dir",phases["text"],
                "--upstream-repo",args.upstream_repo,
                *(["--force-reencode"] if force_qwen_reencode else [])])
    count=validate_text_cache(index,prompts)
    print("[WAN STAGE 3 COMPLETE] Verified REAL Qwen embeds:",
          count,"unique prompts; GPU TE process released",flush=True)

    print("[WAN STAGE 4] Train H3 33B DiT LoRA on JOINT latents",flush=True)
    config=[py,"-u",REPO/"train_h3_wan_dit_lora.py",
            "--joint-index",index,
            "--prompts-json",prompts,
            "--output-dir",phases["dit"],
            "--backup-dir",backup/"joint_dit",
            "--rank",str(args.dit_rank),
            "--weight-mode",args.dit_weight_mode,
            "--lr","2e-5","--seed",str(args.seed),
            "--val-every","50","--save-every","50"]
    dit_latest=phases["dit"]/"checkpoints/state_latest.pt"
    if not dit_latest.is_file():
        launch(config+["--steps","2"])
    launch(config+["--resume","--steps",str(args.dit_steps)])
    dit_best=phases["dit"]/"checkpoints/dit_lora_best.safetensors"
    if not dit_best.is_file():
        raise RuntimeError("Joint H3 DiT LoRA was not saved")
    print("[WAN STAGE 4 COMPLETE] Real H3 DiT adapter",str(dit_best),flush=True)

    print("[WAN STAGE 5] Generate H3 with JOINT distribution DiT LoRA",flush=True)
    native=[py,"-u",REPO/"tools/generate_h3_wan_joint.py",
            "--dit-lora",dit_best,
            "--output-dir",phases["generation"],
            "--upstream-repo",args.upstream_repo,
            "--rank",str(args.dit_rank),
            "--height",str(args.height),"--width",str(args.width),
            "--seconds",str(args.seconds),
            "--steps",str(args.generate_steps),
            "--seed",str(args.seed),"--prompt",args.prompt]
    generation_report=phases["generation"]/"wan_h3_generation.json"
    if generation_report.is_file() and not generation_is_current(generation_report,dit_best):
        print("[WAN STAGE 5] Previous generation is stale or incomplete; "
              "rerunning with CURRENT trained DiT adapter",flush=True)
        generation_report.unlink()
    if not generation_report.is_file():
        launch(native+["--generate-only"])
    print("[WAN STAGE 5 COMPLETE] Native denoised joint H3 latent captured",flush=True)

    print("[WAN STAGE 6] Decode native joint latent as RGB+alpha",flush=True)
    decode=[
        sys.executable,"-u",REPO/"tools/generate_h3_wan_joint.py",
        "--decode-only","--dit-lora",dit_best,
        "--joint-bridge",bridge,"--joint-dual",adapted,
        "--output-dir",phases["generation"],"--hidden","128",
        "--shift-limit",".35","--bridge-scale","1",
    ]
    launch(decode)
    report=phases["generation"]/"wan_h3_rgba_report.json"
    if not report.is_file():
        raise RuntimeError("No alpha-video provenance report produced")
    video=Path(json.loads(report.read_text())["rgba_mov"])
    dest=backup/"generated"
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(video,dest/"wan_joint_rgba.mov")
    shutil.copy2(report,dest/"wan_joint_rgba_report.json")

    print("[WAN STAGE 7] Final reserved GT holdout evaluation (no training)",flush=True)
    final_holdout=out/"final_heldout"
    launch([sys.executable,"-u",REPO/"tools/validate_h3_wan_holdout.py",
            "--manifest",manifest,
            "--base-vae-lora",old,
            "--joint-dual",adapted,"--joint-bridge",bridge,
            "--seed",str(args.seed),"--resolution","128",
            "--output-dir",final_holdout])
    holdout_report=final_holdout/"joint_vae_holdout_report.json"
    holdout=json.loads(holdout_report.read_text())
    actual_video=json.loads(report.read_text())
    heldout_scores=holdout["metrics"]
    # A held-out VAE reconstruction measures the codec, NOT novel DiT outputs.
    gate_checks={
        "heldout_joint_vae_alpha_iou_at_least_0_60":
            heldout_scores["alpha_iou"]>=0.60,
        "heldout_joint_vae_rgb_l1_at_most_0_12":
            heldout_scores["rgb_visible_l1"]<=0.12,
        "generated_alpha_not_all_opaque_or_empty":
            0.005<actual_video["alpha_hard_coverage"]<0.995,
    }
    final={
        "status": ("EXPORTED_WITH_BASIC_QUALITY_GATES"
                   if all(gate_checks.values()) else "EXPORTED_QUALITY_GATES_FAILED"),
        "generated_video_alpha_ground_truth_verified":False,
        "one_joint_h3_latent":True,
        "no_independent_alpha_sidecar":True,
        "stage1_reserved_holdout":holdout["metrics"],
        "stage1_holdout_report":str(holdout_report),
        "native_rgba_mov":str(video),
        "checks":gate_checks,
        "important":"Final generated alpha still requires visual inspection and "
                    "independent ground-truth test before production claims",
    }
    final_path=out/"end_to_end_status.json"
    final_path.write_text(json.dumps(final,indent=2))
    shutil.copy2(final_path,dest/"end_to_end_status.json")
    shutil.copy2(holdout_report,dest/"joint_vae_holdout_report.json")
    print("WAN_NATIVE_TWO_STAGE_COMPLETE_UNVALIDATED_GENERATION_QUALITY",
          json.dumps(final,indent=2),flush=True)
    if not all(gate_checks.values()):
        raise RuntimeError("Native H3 RGBA file exported but quality gates failed. "
                           "Inspect end_to_end_status.json and holdout previews.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-vae-lora",default="/content/drive/MyDrive/H3_DualLoRA_Results/dual_lora_best.safetensors")
    p.add_argument("--manifest",default="/content/h3_rgba_end_to_end/paired_training/manifest.json")
    p.add_argument("--run-root",default="/content/h3_wan_joint_native")
    p.add_argument("--backup-root",default="/content/drive/MyDrive/H3_WanJoint_Native")
    p.add_argument("--a100-python",default="/content/h3_a100_venv/bin/python")
    p.add_argument("--upstream-repo",default="/content/Diffusers_minimax-h3")
    p.add_argument("--vae-steps",type=int,default=1000)
    p.add_argument("--dit-steps",type=int,default=1000)
    p.add_argument("--dit-rank",type=int,default=8)
    p.add_argument("--dit-cache-resolution",type=int,default=256,
                   help="Resolution of joint latents cached for H3 DiT training; "
                        "independent of 128px VAE training. Requires enough A100 VRAM.")
    p.add_argument("--dit-weight-mode",choices=("bf16","int8"),default="bf16",
                   help="BF16 training fits 80GB-class GPUs in principle; int8 "
                        "training may lack autograd support")
    p.add_argument("--min-stage1-iou",type=float,default=.70,
                   help="Require shared 24ch VAE IoU before spending 33B training compute")
    p.add_argument("--max-stage1-rgb-l1",type=float,default=.12,
                   help="Preserve original H3 RGB reconstruction before DiT training")
    p.add_argument("--height",type=int,default=544)
    p.add_argument("--width",type=int,default=960)
    p.add_argument("--seconds",type=float,default=5.)
    p.add_argument("--generate-steps",type=int,default=30)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--prompt",default=(
        "A furry red fox in front of a completely transparent background, "
        "isolated subject, accurate fine fur edges, full body, cinematic lighting."
    ))
    p.add_argument("--preflight-only",action="store_true")
    args=p.parse_args()
    if min(args.vae_steps,args.dit_steps,args.dit_rank)<2:
        p.error("Two optimizer steps minimum per training stage")
    if args.dit_cache_resolution<128 or args.dit_cache_resolution%32:
        p.error("--dit-cache-resolution must be >=128 and divisible by 32")
    run(args)


if __name__=="__main__":
    main()
