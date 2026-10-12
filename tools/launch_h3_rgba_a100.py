#!/usr/bin/env python3
"""One-command A100 80GB MiniMax-H3 text -> RGB+audio -> generated-alpha -> RGBA.

Sequential phases deliberately use separate Python environments/processes:
  1. Install official T2VA 33B H3 runner with 4-bit Qwen + INT8 H3 DiT.
  2. Download ONLY real H3 T2VA partitions; generate a real RGB+audio MP4.
  3. Download paired Wan-Alpha foreground/alpha clips and official H3 VAE.
  4. Restore the successful 500-step VAE LoRA from Drive; if absent train it.
  5. Train a SECOND alpha latent diffusion sidecar (conditioned on RGB latents).
  6. Apply generated-alpha sampling to real H3 RGB frames in 17-frame windows.
  7. Save RGBA PNGs and ProRes4444 MOV WITH H3 audio, plus provenance report.

IMPORTANT:
  - This does NOT fine-tune H3's 33B diffusion model; alpha generation is a
    separate small conditional diffusion model, trained from paired video.
  - The 128px alpha network is upscaled to the H3 output resolution.
  - Alpha inference may be poor on backgrounds unlike Wan-Alpha matting data.
  - Single RGB video cannot uniquely recover straight foreground colors if
    the scene's background is unknown; solid-black prompts only approximate.
  - Current code has CPU structural tests, NOT end-to-end A100 GPU validation.
  - Requires mounted /content/drive if --drive-root is supplied.
"""
import argparse
import json
import os
from collections import deque
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
UPSTREAM_URL = "https://github.com/animede/Diffusers_minimax-h3.git"


def launch(command, *, cwd=None, env=None):
    """Stream child stdout+stderr to Colab AND persist the real failure.

    Colab's outer CalledProcessError hides the meaningful child traceback.
    The rolling failure excerpt and launcher.log identify the failing stage.
    """
    cmd=[str(x) for x in command]
    label=" ".join(cmd)
    log_path=Path(os.environ.get(
        "H3_E2E_LOG_PATH", "/content/h3_rgba_end_to_end/launcher.log"))
    log_path.parent.mkdir(parents=True,exist_ok=True)
    print("\n>>>",label,flush=True)
    tail=deque(maxlen=65)
    with log_path.open("a",encoding="utf-8") as log:
        log.write("\n>>> "+label+"\n")
        log.flush()
        process=subprocess.Popen(
            cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,errors="replace",bufsize=1
        )
        for line in process.stdout:
            print(line,end="",flush=True)
            log.write(line)
            log.flush()
            tail.append(line.rstrip())
        rc=process.wait()
        log.write(f"EXIT_CODE={rc}\n")
    if rc:
        detail="\n".join(tail)
        raise RuntimeError(
            f"Internal H3 stage failed with exit code {rc}: {label}\n"
            f"Saved log: {log_path}\n"
            f"CHILD_PROCESS_ERROR_TAIL:\n{detail}"
        )


def best_vae_lora(drive, local):
    """Never pretend GitHub contains the trained 500-step adapter."""
    if local.is_file():
        return local
    if drive is None:
        return None
    root=Path(drive)
    standard=root/"H3_DualLoRA_Results/dual_lora_best.safetensors"
    if standard.is_file():
        return standard
    backups=root/"H3_DualLoRA_Backups"
    snapshots=sorted(
        p for p in backups.glob("step_*")
        if (p/"COMPLETE.json").is_file()
    ) if backups.exists() else []
    if snapshots:
        for snap in reversed(snapshots):
            source=snap/"checkpoints/dual_lora_best.safetensors"
            if source.is_file():
                return source
    return None


def best_sidecar(drive, local):
    if local.is_file():
        return local
    if drive:
        p=Path(drive)/"H3_AlphaDiffusion/checkpoints/sidecar_best.pt"
        if p.is_file():
            return p
    return None


def prepare_data(workdir, drive):
    """Paired MP4 dataset on local disk, ZIP download optionally cached on Drive."""
    downloads=workdir/"downloads"
    downloads.mkdir(parents=True,exist_ok=True)
    if drive:
        cache=Path(drive)/"H3vaeRGBA_cache/WanAlphaZips"
        cache.mkdir(parents=True,exist_ok=True)
    else:
        cache=downloads
    zipnames=["data_fgr.zip","data_pha.zip"]
    missing=[name for name in zipnames
             if not (cache/name).is_file() or (cache/name).stat().st_size==0]
    if missing:
        launch([sys.executable,"-c",
            "from huggingface_hub import snapshot_download; "
            "snapshot_download('htdong/Wan-Alpha-dataset',repo_type='dataset',"
            "allow_patterns=['data_fgr.zip','data_pha.zip'],"
            f"local_dir={str(cache)!r},max_workers=2)"],cwd=REPO)
    for name in zipnames:
        source=cache/name
        target=downloads/name
        if not source.exists():
            raise FileNotFoundError(source)
        if source.resolve()!=target.resolve() and not target.exists():
            target.symlink_to(source)
    if not (workdir/"manifest.json").exists():
        launch([sys.executable,REPO/"tools/prepare_data.py",
                "--workdir",workdir,"--skip-download"],cwd=REPO)
    report=json.loads((workdir/"manifest.json").read_text())
    if len(report["clips"])<20:
        raise RuntimeError("Unexpectedly small paired alpha training dataset")
    print("PAIRED_VIDEO_DATA_READY",len(report["clips"]),flush=True)
    return workdir/"manifest.json"


def ensure_vae_pretrained():
    """Download only 10.416GB official H3 VAE, no 33B DiT duplicate."""
    launch([sys.executable,"-c",
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('MiniMaxAI/MiniMax-H3',"
        "allow_patterns=['vae/config.json','vae/diffusion_pytorch_model-*.safetensors',"
        "'vae/diffusion_pytorch_model.safetensors.index.json'],max_workers=3)"],
        cwd=REPO)
    launch([sys.executable,REPO/"tools/verify_h3_cache.py"],cwd=REPO)


def prepare_h3_environment(venv):
    py=venv/"bin/python"
    launch([sys.executable,"-m","pip","install","-q","uv"])
    launch(["uv","python","install","3.12"])
    if not py.exists():
        launch(["uv","venv","--python","3.12",venv])
    launch(["uv","pip","install","--python",py,"torch==2.9.0",
            "--index-url","https://download.pytorch.org/whl/cu128"])
    launch(["uv","pip","install","--python",py,
            "-r",REPO/"requirements-a100-h3.txt"])
    return py


def ensure_full_h3_generation(args):
    output=Path(args.run_root)/"full_h3"
    output.mkdir(parents=True,exist_ok=True)
    existing=output/"h3_a100_full_t2va_report.json"
    if existing.exists():
        status=json.loads(existing.read_text())
        candidate=Path(status.get("generated_rgb_audio_video",""))
        if candidate.is_file():
            print("REUSING_VERIFIED_H3_FULL_DIFFUSION",candidate,flush=True)
            return candidate
    upstream=Path(args.upstream_repo)
    if not (upstream/"core/runner.py").is_file():
        launch(["git","clone",UPSTREAM_URL,upstream])
    py=prepare_h3_environment(Path(args.h3_venv))
    cmd=[py,REPO/"tools/h3_a100_full_t2va.py",
         "--upstream-repo",upstream,"--output-dir",output]
    if args.hf_home:
        cmd+=["--hf-home",args.hf_home]
    launch(cmd+["--preflight-only"])
    proof=json.loads((output/"h3_a100_preflight.json").read_text())
    if not proof["can_proceed"]:
        raise RuntimeError("A100/host/disk preflight failed: "
                           +json.dumps(proof["checks"]))
    launch(cmd+["--download-only"])
    launch(cmd+["--generate","--height",args.height,
                "--width",args.width,"--seconds",args.seconds,
                "--steps",args.h3_steps,"--seed",args.seed,
                "--prompt",args.prompt])
    report=json.loads(existing.read_text())
    path=Path(report["generated_rgb_audio_video"])
    if not path.is_file():
        raise RuntimeError("33B H3 inference produced no MP4")
    return path


def ensure_vae_lora(args,workdir,drive):
    output=Path(args.run_root)/"dual_vae"
    best=output/"checkpoints/dual_lora_best.safetensors"
    existing=best_vae_lora(drive,best)
    if existing:
        print("RESTORED_PROVEN_500_STEP_VAE_LORA",existing,flush=True)
        return Path(existing)
    print("PREVIOUS_VAE_LORA_UNAVAILABLE: training original rank-4 dual "
          "decoder LoRA for 500 steps; requires paired dataset.",flush=True)
    config=[
        sys.executable,"-u",REPO/"train_wan_dual_lora.py",
        "--workdir",workdir,"--output-dir",output,
        "--skip-download","--resolution","128",
        "--rank","4","--merger-scale",".15",
        "--batch-size","1","--grad-accum","1",
        "--lr","5e-5","--seed",args.seed,
        "--alpha-weight","2","--edge-weight","2",
        "--composite-weight","1","--hard-weight",".25",
        "--temporal-weight",".2",
    ]
    if drive:
        config+=["--backup-dir",Path(drive)/"H3_DualLoRA_Backups"]
    if (output/"checkpoints/dual_lora_last.safetensors").exists():
        launch(config+["--resume","--steps","500",
                       "--val-every","25","--save-every","25"],cwd=REPO)
    else:
        launch(config+["--steps","2","--val-every","1","--save-every","1"],cwd=REPO)
        launch(config+["--resume","--steps","500","--val-every","25",
                       "--save-every","25"],cwd=REPO)
    if not best.is_file():
        raise RuntimeError("No 500-step VAE LoRA was produced")
    if drive:
        dest=Path(drive)/"H3_DualLoRA_Results"
        dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(best,dest/"dual_lora_best.safetensors")
    return best


def ensure_alpha_sidecar(args,vae_lora,manifest,drive):
    """Never skip unfinished diffusion training merely because best.pt exists."""
    import torch

    output=Path(args.run_root)/"alpha_diffusion"
    folder=output/"checkpoints"
    folder.mkdir(parents=True,exist_ok=True)
    best=folder/"sidecar_best.pt"
    latest=folder/"sidecar_latest.pt"

    persistent=(Path(drive)/"H3_AlphaDiffusion/checkpoints" if drive else None)
    if persistent is not None:
        for name in ("sidecar_latest.pt","sidecar_best.pt"):
            source=persistent/name
            destination=folder/name
            if source.is_file() and not destination.is_file():
                shutil.copy2(source,destination)

    if not latest.exists() and best.exists():
        # Model selection checkpoint contains full optimizer state, but may
        # precede the latest training step. Use as a valid fallback resume.
        shutil.copy2(best,latest)

    def read_step(path):
        if not path.exists():
            return 0
        value=torch.load(path,map_location="cpu",weights_only=True)
        return int(value["step"])

    completed=read_step(latest)
    if best.exists() and completed>=args.alpha_steps:
        print("ALPHA_DIFFUSION_TARGET_TRAINING_COMPLETED",
              json.dumps({"latest_step":completed,"best_step":read_step(best)}),
              flush=True)
        return best

    if completed:
        print("RESUMING_ALPHA_SIDECAR",
              json.dumps({"step":completed,"target":args.alpha_steps}),
              flush=True)

    cmd=[sys.executable,"-u",REPO/"train_h3_alpha_diffusion.py",
         "--manifest",manifest,
         "--vae-lora",vae_lora,
         "--output-dir",output,
         "--cache-dir",Path(args.run_root)/"alpha_latent_cache",
         "--resolution","128","--clips-per-video","2",
         "--hidden","128","--blocks","6",
         "--batch-size","4","--steps",args.alpha_steps,
         "--val-every","250","--save-every","250",
         "--sampling-steps","30","--seed",args.seed]
    if drive:
        cmd+=["--backup-dir",persistent]
    if latest.exists():
        cmd+=["--resume"]
    launch(cmd,cwd=REPO)
    if not best.exists() or read_step(latest)<args.alpha_steps:
        raise RuntimeError("Alpha diffusion training did not complete target steps")
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",default="/content/h3_rgba_end_to_end")
    p.add_argument("--upstream-repo",default="/content/Diffusers_minimax-h3")
    p.add_argument("--h3-venv",default="/content/h3_a100_venv")
    p.add_argument("--drive-root",default="/content/drive/MyDrive")
    p.add_argument("--hf-home",default=None,
                   help="Optional HF cache on large scratch drive; do not "
                        "change after partially downloading weights")
    p.add_argument("--width",type=int,default=960)
    p.add_argument("--height",type=int,default=544)
    p.add_argument("--seconds",type=float,default=5.)
    p.add_argument("--h3-steps",type=int,default=30)
    p.add_argument("--alpha-steps",type=int,default=4000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--prompt",default=(
        "A single furry red fox running in profile, isolated against a "
        "perfectly uniform black background, fine hair edges, steady camera, "
        "studio lighting, soft footsteps and wind sound."
    ))
    p.add_argument("--dry-run",action="store_true",
                   help="A100 preflight only; no model weights or training")
    args=p.parse_args()
    root=Path(args.run_root)
    root.mkdir(parents=True,exist_ok=True)
    os.environ["H3_E2E_LOG_PATH"]=str(root/"launcher.log")
    if args.hf_home:
        cache_root=Path(args.hf_home).expanduser().resolve()
        cache_root.mkdir(parents=True,exist_ok=True)
        os.environ["HF_HOME"]=str(cache_root)
        print("H3_HUGGINGFACE_CACHE_HOME",cache_root,flush=True)
    drive=Path(args.drive_root) if args.drive_root else None
    if drive and not drive.is_dir():
        if args.dry_run:
            drive=None
        else:
            raise RuntimeError("Mount Google Drive in Colab first, or --drive-root ''")
    print("END_TO_END_CONFIG",json.dumps({
        "mode":"ORIGINAL_H3_T2VA_PLUS_SEPARATELY_TRAINED_ALPHA_DDPM",
        "h3_pretrained":True,"vae_lora_rank":4,
        "alpha_diffusion_steps":args.alpha_steps,
        "rgb_prompt":args.prompt,
        "output_root":str(root),
        "alpha_generated_from_noise":True,
        "is_H3_transformer_alpha_lora":False,
        "limitations":"VAE LoRA + alpha diffusion sidecar, not H3 joint RGB-alpha DiT finetuning"
    },indent=2),flush=True)
    if args.dry_run:
        py=Path(args.h3_venv)/"bin/python"
        if py.exists():
            launch([py,REPO/"tools/h3_a100_full_t2va.py",
                    "--preflight-only","--output-dir",root/"full_h3"])
        else:
            print("SKIPPED_H3_PREFLIGHT: isolated Python 3.12 env not yet installed")
        return

    # Phase A: real 33B DiT generation in isolated venv. After child process
    # exits, its GPU allocations are released before VAE sidecar training.
    rgb_video=ensure_full_h3_generation(args)

    # Phase B: VAE-only environment with pinned release, never mix its
    # diffusers/TorchAO versions with the external 33B inference environment.
    launch([sys.executable,"-m","pip","install","-q",
            "-r",REPO/"requirements.txt"],cwd=REPO)
    launch([sys.executable,REPO/"tools/colab_compat.py","--repair"],cwd=REPO)
    ensure_vae_pretrained()
    workdir=root/"paired_training"
    manifest=prepare_data(workdir,drive)
    adapter=ensure_vae_lora(args,workdir,drive)
    sidecar=ensure_alpha_sidecar(args,adapter,manifest,drive)

    # Phase C: no GT alpha; sample alpha latents from noise conditioned on
    # real H3 generated RGB, decode LoRA in overlapped 17-frame windows.
    output=root/"generated_transparent_video"
    launch([sys.executable,"-u",REPO/"generate_h3_rgba_end_to_end.py",
            "--rgb-mp4",rgb_video,
            "--vae-lora",adapter,
            "--alpha-diffusion",sidecar,
            "--output-dir",output,
            "--sampling-steps","30",
            "--seed",args.seed,
            "--assumed-background","black"],cwd=REPO)
    source=output/"end_to_end_report.json"
    report=json.loads(source.read_text())
    report["original_h3_generation_report"]=str(
        root/"full_h3/h3_a100_full_t2va_report.json")
    report["alpha_sidecar_best_checkpoint"]=str(sidecar)
    report["dual_vae_adapter"]=str(adapter)
    report["final_alpha_quality_tested_on_real_h3_output"]=False
    source.write_text(json.dumps(report,indent=2))
    if drive:
        # Sync *final video and provenance*, not hundreds of large PNGs.
        backup=drive/"H3_RGBA_EndToEnd_Results"
        backup.mkdir(parents=True,exist_ok=True)
        shutil.copy2(report["prores4444_mov"],backup/"generated_rgba.mov")
        shutil.copy2(source,backup/"end_to_end_report.json")
        print("PERSISTENT_GENERATED_RGBA",backup/"generated_rgba.mov",flush=True)
    print("END_TO_END_RGBA_GENERATION_COMPLETED",report["prores4444_mov"],flush=True)
    print("NOTE: alpha sidecar quality must still be judged against "
          "independent video matte ground truth.",flush=True)


if __name__=="__main__":
    main()
