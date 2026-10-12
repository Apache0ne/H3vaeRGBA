#!/usr/bin/env python3
"""Stage W: Wan-Alpha-inspired dual-path MiniMax-H3 VAE ALPHA DECODER LoRA.

The pretrained H3 RGB VAE is frozen. Separately encode foreground RGB and
grayscale alpha with the SAME original RGB encoder, fuse their latents with
a tiny zero-initialized 48->24 residual mapper, and train a LoRA-equipped
decoder plus an alpha-only output head. Original RGB reconstruction always
bypasses LoRA, so its weights and reconstruction are protected by design.

Only trains the H3 VAE alpha path, NOT its 33B video diffusion transformer.
A subsequent compatible DiT LoRA is required for alpha-aware video generation.
"""
import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--workdir", default="/content/h3rgba_data")
    p.add_argument("--manifest", default=None)
    p.add_argument("--skip-download", action="store_true",
                   help="Use an already prepared manifest; fail rather than downloading data")
    p.add_argument("--output-dir", default="/content/h3rgba_runs/wan_dual_h3")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--rank", type=int, default=4, help="Decoder transformer LoRA rank")
    p.add_argument("--merger-scale", type=float, default=.15)
    p.add_argument("--steps", type=int, default=2, help="Total optimizer steps, including resume")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-samples", type=int, default=3)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--alpha-weight", type=float, default=2.)
    p.add_argument("--edge-weight", type=float, default=2.)
    p.add_argument("--composite-weight", type=float, default=1.)
    p.add_argument("--hard-weight", type=float, default=.25)
    p.add_argument("--temporal-weight", type=float, default=.2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--preflight-only", action="store_true", help="Audit and decode dataset, no H3 VAE or CUDA")
    return p.parse_args()


def say(*x):
    print(*x, flush=True)


def settings(args):
    fields = ("rank", "resolution", "merger_scale", "seed", "lr", "grad_accum",
              "alpha_weight", "edge_weight", "composite_weight", "hard_weight",
              "temporal_weight")
    return {name:getattr(args,name) for name in fields}


def check_args(args):
    if args.batch_size != 1 or args.resolution < 64 or args.resolution % 32:
        raise ValueError("Use batch 1 and resolution divisible by 32, minimum 64")
    if args.rank < 1 or args.rank > 32 or not 0 < args.merger_scale <= 1:
        raise ValueError("Rank 1..32 and merger scale (0,1] required")
    if min(args.steps, args.lr, args.grad_accum, args.val_every, args.save_every, args.val_samples) <= 0:
        raise ValueError("Steps/LR/accumulation/intervals/validation must be positive")
    if min(args.alpha_weight, args.edge_weight, args.composite_weight,
           args.hard_weight, args.temporal_weight) < 0:
        raise ValueError("Loss weights must be nonnegative")
    if args.resume and args.eval_only:
        raise ValueError("--resume and --eval-only are mutually exclusive")


def get_loaders(args):
    import torch
    from torch.utils.data import DataLoader
    from train_vae import PairedVideoDataset
    from run_vae_22gb import prepare_manifest, training_split

    source = prepare_manifest(args)
    report = json.loads(source.read_text())
    train, val = training_split(report["clips"], args.seed)
    trainset = PairedVideoDataset(train, resolution=args.resolution)
    valset = PairedVideoDataset(val[:args.val_samples], resolution=args.resolution,
                                deterministic=True)
    say("DATA",json.dumps({"manifest":str(source),"valid_pairs":len(report["clips"]),
                             "train":len(trainset),"val":len(valset),
                             "resolution":args.resolution,"frames":17}))
    return (DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True),
            DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True))


def make_model(args, device):
    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    from huggingface_hub import hf_hub_download
    from tools.verify_loaded_vae import verify_loaded_vae
    from h3rgba.wan_dual import DualAlphaCodec

    say("LOADING OFFICIAL PRETRAINED H3 RGB VAE (cached; 3 shards)")
    t0 = time.time()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        "MiniMaxAI/MiniMax-H3", subfolder="vae", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_safetensors=True, local_files_only=True,
    )
    conf = Path(hf_hub_download("MiniMaxAI/MiniMax-H3", "vae/config.json",
                               local_files_only=True))
    proof = verify_loaded_vae(vae, conf.parent.parent)
    say("PRETRAINED_WEIGHTS_VERIFIED time_s",round(time.time()-t0,2))
    vae.enable_gradient_checkpointing()
    if args.resolution <= 128:
        vae.disable_tiling()
    vae.to(device)
    vae.encoder.float()
    vae.quant_conv.float()
    vae.decoder.proj_out.float()
    codec = DualAlphaCodec(vae, rank=args.rank, merger_scale=args.merger_scale)
    codec.to(device)
    codec.train()
    original = sum(p.numel() for p in codec.vae.parameters())
    trainable = sum(p.numel() for p in codec.trainable_parameters())
    assert all(not p.requires_grad for name,p in codec.vae.named_parameters()
               if "lora_down" not in name and "lora_up" not in name)
    say(f"MODEL decoder_lora_layers={codec.lora_layers} pretrained_VAE_params={original:,} trainable={trainable:,}")
    return codec, proof


def random_background(rgba):
    """One random solid contrast color shared across all frames of each clip."""
    import torch
    choice = torch.randint(0, 8, (rgba.shape[0], 3, 1, 1, 1), device=rgba.device)
    return (choice % 2).float()


def calculate_loss(codec, rgba, args, *, train=False):
    import torch
    from h3rgba.dual_loss import alpha_losses, alpha_scores

    background = random_background(rgba) if train else None
    # Both frozen VAE encoder passes are no-grad, decoder LoRA is trainable.
    alpha_pred, _ = codec.rgba_forward(rgba, background=background, decode_rgb=False)
    loss, components = alpha_losses(
        alpha_pred, rgba[:, 3:4], rgba[:, :3],
        alpha_weight=args.alpha_weight, edge_weight=args.edge_weight,
        composite_weight=args.composite_weight,
        temporal_weight=args.temporal_weight, hard_weight=args.hard_weight,
    )
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Nonfinite dual-decoder loss")
    return loss, components, alpha_pred


def evaluate(codec, loader, args, *, step, output_dir):
    import torch
    from h3rgba.dual_loss import alpha_scores
    from h3rgba.preservation import visible_rgb_mae
    from run_vae_22gb import save_frames

    codec.eval()
    records = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            rgba = batch.to("cuda", dtype=torch.float32, non_blocking=True)
            alpha, rgb = codec.rgba_forward(rgba, background=None, decode_rgb=True)
            if rgb.shape != rgba[:, :3].shape or alpha.shape != rgba[:, 3:4].shape:
                raise RuntimeError("Dual RGB/alpha decoder output shape mismatch")
            loss, components = alpha_losses_from_result(alpha, rgba, args)
            scores = alpha_scores(alpha, rgba[:, 3:4])
            scores["rgb_visible_mae_original_h3"] = float(
                visible_rgb_mae(rgb, rgba[:, :3], rgba[:, 3:4]))
            scores["loss"] = float(loss)
            records.append(scores)
            if i == 0:
                predicted = torch.cat((rgb.clamp(0,1),alpha.clamp(0,1)),dim=1)
                sheet = save_frames(
                    rgba[0],predicted[0],output_dir / "previews" / f"step_{step:06d}",
                    max_frames=5
                )
                say("PREVIEW",sheet)
    codec.train()
    result = {key:round(sum(d[key] for d in records)/len(records),6)
              for key in records[0]}
    result["step"] = step
    say("VALIDATION",json.dumps(result))
    with (output_dir / "validation.jsonl").open("a") as file:
        file.write(json.dumps(result)+"\n")
    return result


def alpha_losses_from_result(alpha, rgba, args):
    from h3rgba.dual_loss import alpha_losses
    return alpha_losses(alpha,rgba[:,3:4],rgba[:,:3],
        alpha_weight=args.alpha_weight, edge_weight=args.edge_weight,
        composite_weight=args.composite_weight, hard_weight=args.hard_weight,
        temporal_weight=args.temporal_weight)


def checkpoint(codec, optimizer, args, output, step, metrics):
    import torch
    from safetensors.torch import save_file
    folder = output / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"dual_lora_step_{step:06d}.safetensors"
    save_file(codec.adapter_state(),str(path))
    shutil.copyfile(path,folder/"dual_lora_last.safetensors")
    temporary = folder/"optimizer_last.pt.tmp"
    torch.save({"step":step,"optimizer":optimizer.state_dict(),
                "settings":settings(args),"metrics":metrics},temporary)
    os.replace(temporary,folder/"optimizer_last.pt")
    say("CHECKPOINT",str(path),round(path.stat().st_size/2**20,2),"MiB")


def main():
    import torch
    from h3rgba.dual_loss import alpha_scores
    args = arguments()
    check_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_loader, val_loader = get_loaders(args)
    sample = next(iter(val_loader))
    say("DATA_SAMPLE",tuple(sample.shape),"alpha_minmax",
        (float(sample[:,3:4].min()),float(sample[:,3:4].max())))
    if args.preflight_only:
        say("PREFLIGHT_OK: no H3 model or CUDA loaded")
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required for H3 VAE alpha-decoder LoRA")
    device = torch.device("cuda")
    say("GPU",torch.cuda.get_device_name(0),
        round(torch.cuda.get_device_properties(0).total_memory/2**30,2),"GiB")
    torch.set_num_threads(min(8,os.cpu_count() or 4))
    output = Path(args.output_dir).resolve()
    folder = output / "checkpoints"
    if not (args.resume or args.eval_only) and (
            (output/"metrics.jsonl").exists() or
            (folder/"dual_lora_last.safetensors").exists()):
        raise FileExistsError("Existing run: choose a new --output-dir or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    (output/"settings.json").write_text(json.dumps(settings(args),indent=2))
    codec, proof = make_model(args,device)
    (output/"pretrained_weight_verification.json").write_text(json.dumps(proof,indent=2))
    parameters = codec.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters,lr=args.lr,weight_decay=0.)
    step = 0
    if args.resume or args.eval_only:
        last = folder/"dual_lora_last.safetensors"
        if not last.is_file():
            raise FileNotFoundError(f"Dual-decoder checkpoint not found: {last}")
        codec.load_adapter(last)
        if args.resume:
            state = torch.load(folder/"optimizer_last.pt",map_location="cpu",weights_only=True)
            if state["settings"] != settings(args):
                raise ValueError("Resume config mismatch; use original exact hyperparameters")
            optimizer.load_state_dict(state["optimizer"])
            step = int(state["step"])
            say("RESUMED_AT_STEP",step)

    torch.cuda.reset_peak_memory_stats()
    if args.eval_only:
        evaluate(codec,val_loader,args,step=step,output_dir=output)
        return

    # Step-zero evaluation is critical: checks if original pretrained H3 VAE
    # already reconstructs alpha masks before any LoRA training.
    if step == 0:
        evaluate(codec,val_loader,args,step=0,output_dir=output)

    say("TRAIN_START stage=dual_decoder_lora rank",args.rank)
    codec.train()
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    epoch = 0
    while step < args.steps:
        epoch += 1
        for rgba in train_loader:
            rgba = rgba.to(device,dtype=torch.float32,non_blocking=True)
            loss, components, pred = calculate_loss(codec,rgba,args,train=True)
            (loss/args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            comp = {k:round(float(v.detach()),6) for k,v in components.items()}
            scores = alpha_scores(pred.detach(),rgba[:,3:4])
            result={"step":step,"epoch":epoch,"loss":round(float(loss.detach()),6),
                    "grad_norm":round(float(norm),6),
                    "peak_alloc_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
                    "peak_reserved_gib":round(torch.cuda.max_memory_reserved()/2**30,3),
                    **comp,**{k:round(v,6) for k,v in scores.items()}}
            say("STEP",json.dumps(result))
            with (output/"metrics.jsonl").open("a") as file:
                file.write(json.dumps(result)+"\n")
            del loss,pred,rgba
            if step % args.val_every == 0 or step == 1 or step == args.steps:
                evaluate(codec,val_loader,args,step=step,output_dir=output)
            if step % args.save_every == 0 or step == args.steps:
                checkpoint(codec,optimizer,args,output,step,result)
            if step >= args.steps:
                break
    say("FINISHED",json.dumps({"steps":step,"checkpoint":str(folder/"dual_lora_last.safetensors"),
                                "validation":str(output/"validation.jsonl")}))
    say("NOTE: only VAE decoder alpha LoRA trained; full H3 33B DiT was never loaded.")


if __name__=="__main__":
    main()
