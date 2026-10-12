#!/usr/bin/env python3
"""Train a REAL, small RGB-conditioned ALPHA LATENT DIFFUSION sidecar for H3.

Pretrained H3 video VAE is frozen; train diffusion p(z_alpha|z_rgb) on
paired foreground+matte videos. Its generated alpha latents are decoded by
the existing 500-step dual-path VAE LoRA at inference. This is a genuinely
newly trained alpha diffusion model, NOT a 33B H3-transformer LoRA.

Execution: load VAE ONCE -> cache frozen, normalized paired RGB/alpha
latents -> train 3D DDPM epsilon predictor from cache -> periodically sample
alpha via DDIM and decode with frozen VAE LoRA -> checkpoint.
No ground-truth alpha latents are supplied during generation/sampling.
"""
import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from h3rgba.alpha_latent_diffusion import (
    ConditionalAlphaDenoiser, cosine_alphas, normalized_latents,
    unnormalized_latents, noise_loss, sample,
)


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output-dir",default="/content/h3_alpha_sidecar_run")
    p.add_argument("--cache-dir",default="/content/h3_alpha_latent_cache")
    p.add_argument("--vae-lora",required=True)
    p.add_argument("--resolution",type=int,default=128)
    p.add_argument("--clips-per-video",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--blocks",type=int,default=6)
    p.add_argument("--steps",type=int,default=4000)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--val-every",type=int,default=250)
    p.add_argument("--save-every",type=int,default=250)
    p.add_argument("--diffusion-steps",type=int,default=1000)
    p.add_argument("--sampling-steps",type=int,default=30)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--cache-only",action="store_true")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--backup-dir",default=None)
    return p.parse_args()


def settings(args):
    return {k:getattr(args,k) for k in (
        "resolution","clips_per_video","seed","hidden","blocks",
        "lr","batch_size","diffusion_steps","sampling_steps"
    )}


def checksum(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def load_source(args):
    from run_vae_22gb import training_split
    obj=json.loads(Path(args.manifest).read_text())
    train,val=training_split(obj["clips"],args.seed)
    return train,val


class LatentPairs(Dataset):
    def __init__(self, root, group):
        self.files=sorted((Path(root)/group).glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No cached latent pairs: {root}/{group}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        # Only data tensors, never arbitrary objects from untrusted checkpoints.
        z=torch.load(self.files[index],map_location="cpu",weights_only=True)
        if z.shape[0]!=48 or z.ndim!=4 or not torch.isfinite(z).all():
            raise RuntimeError(f"Bad cached H3 latent pair: {self.files[index]}")
        return z.float()


def prepare_cache(args, device):
    from train_vae import PairedVideoDataset
    from train_wan_dual_lora import make_model

    root=Path(args.cache_dir).resolve()
    train,valid=load_source(args)
    manifest_digest=checksum(args.manifest)
    adapter_digest=checksum(args.vae_lora)
    proof=root/"cache_index.json"
    expected={
        "manifest_sha256":manifest_digest,
        "decoder_lora_sha256":adapter_digest,
        "resolution":args.resolution,"seed":args.seed,
        "clips_per_video":args.clips_per_video,
        "train_items":len(train),"validation_items":min(3,len(valid))
    }
    if proof.exists():
        record=json.loads(proof.read_text())
        if {key:record.get(key) for key in expected} == expected and (
            len(list((root/"train").glob("*.pt"))) == expected["train_items"]*args.clips_per_video and
            len(list((root/"val").glob("*.pt"))) == expected["validation_items"]
        ):
            print("PAIRED_LATENT_CACHE_REUSED",json.dumps(expected),flush=True)
            return
        raise ValueError("Existing latent cache belongs to different manifest/settings. "
                         "Choose a new --cache-dir, don't mix training data.")
    if root.exists() and list(root.glob("**/*.pt")):
        raise RuntimeError("Unindexed partial cache found; use a clean --cache-dir")

    torch.cuda.empty_cache()
    codec,_ = make_model(
        argparse.Namespace(rank=4,merger_scale=.15,resolution=args.resolution),device
    )
    codec.load_adapter(args.vae_lora)
    codec.eval()
    for group,records,repeats in (
        ("train",train,args.clips_per_video),("val",valid[:3],1)
    ):
        folder=root/group
        folder.mkdir(parents=True,exist_ok=True)
        ds=PairedVideoDataset(records,resolution=args.resolution,deterministic=(group=="val"))
        count=0
        for j in range(len(ds)):
            for k in range(repeats):
                # Reproducible temporal cropping; repeated windows add variety.
                random.seed(args.seed + j*10007+k*71)
                rgba=ds[j][None].to(device,dtype=torch.float32)
                with torch.no_grad():
                    zr,za=codec.encode_pair(rgba)
                    stacked=torch.cat([
                        normalized_latents(zr,codec.vae),
                        normalized_latents(za,codec.vae)
                    ],dim=1)[0].half().cpu().contiguous()
                if stacked.shape[0]!=48 or not torch.isfinite(stacked).all():
                    raise RuntimeError("Cached alpha/RGB latent pair invalid")
                torch.save(stacked,folder/f"{j:06d}_{k:02d}.pt")
                count+=1
                if count%50==0:
                    print("CACHE_PROGRESS",group,count,flush=True)
        print("CACHE_COMPLETE",group,count,flush=True)
    proof.write_text(json.dumps(expected,indent=2))
    del codec
    gc.collect()
    torch.cuda.empty_cache()
    print("PRETRAINED_H3_LATENT_CACHE_VERIFIED",str(proof),flush=True)


def save_snapshot(net,opt,args,step,best_iou,*,tag):
    root=Path(args.output_dir)/"checkpoints"
    root.mkdir(parents=True,exist_ok=True)
    path=root/f"sidecar_{tag}.pt"
    temp=path.with_suffix(".tmp")
    payload={
        "step":int(step),"model":net.state_dict(),"optimizer":opt.state_dict(),
        "best_alpha_iou":float(best_iou),
        "settings":settings(args),
        "vae_lora_sha256":checksum(args.vae_lora),
        "manifest_sha256":checksum(args.manifest),
    }
    torch.save(payload,temp)
    os.replace(temp,path)
    if args.backup_dir:
        backup=Path(args.backup_dir).expanduser()
        backup.mkdir(parents=True,exist_ok=True)
        partial=backup/(path.name+".partial")
        shutil.copy2(path,partial)
        os.replace(partial,backup/path.name)
    print("SAVED_ALPHA_DIFFUSION",path,flush=True)


@torch.no_grad()
def validate(net, codec, valid, schedule, args,device,step):
    """Select best on the original THREE validation clips, not the unused 9."""
    from h3rgba.dual_loss import alpha_scores
    net.eval()
    ious=[]
    for ix in range(len(valid)):
        raw=valid[ix].unsqueeze(0).to(device)
        zr=raw[:,:24].float()
        za=raw[:,24:].float()  # ONLY for optional latent error diagnostic
        generated=sample(net,zr,schedule,steps=args.sampling_steps,
                         seed=args.seed+ix)
        # Target pixel matte must come from decoding its alpha latent; no
        # cached image tensor is required, but this is a surrogate VAE target.
        estimated=codec.decode_alpha(
            unnormalized_latents(zr,codec.vae),
            unnormalized_latents(generated,codec.vae)
        ).clamp(0,1)
        reference=codec.decode_alpha(
            unnormalized_latents(zr,codec.vae),
            unnormalized_latents(za,codec.vae)
        ).clamp(0,1)
        if estimated.shape!=reference.shape:
            raise RuntimeError("Alpha diffusion and teacher decoded mask shape mismatch")
        scores=alpha_scores(estimated,reference)
        ious.append(float(scores["alpha_iou"]))
    score=sum(ious)/len(ious)
    print("ALPHA_SIDECAR_VALIDATION",json.dumps(
        {"step":step,"alpha_iou_vs_frozen_vae_oracle":score,
         "samples":len(ious),"individual_iou":ious}),flush=True)
    net.train()
    return score


def main():
    import argparse as _argparse
    args=args_parser()
    if min(args.resolution,args.clips_per_video,args.batch_size,args.steps,
           args.sampling_steps,args.save_every,args.val_every)<=0:
        raise ValueError("Invalid nonpositive training setting")
    if args.resolution<64 or args.resolution%32:
        raise ValueError("Resolution must be >=64, divisible by 32")
    if not Path(args.vae_lora).is_file():
        raise FileNotFoundError("The trained 500-step VAE decoder LoRA is required: "+args.vae_lora)
    train,val=load_source(args)
    print("ALPHA_SIDECAR_SPLIT",json.dumps(
        {"train_clips":len(train),"val_clips_for_model_selection":min(3,len(val)),
         "unused_validation_clips":max(len(val)-3,0),
         "alpha_checkpoint":args.vae_lora}),flush=True)
    if args.preflight_only:
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required for real pretrained H3 encoding")
    device=torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prepare_cache(args,device)
    if args.cache_only:
        return
    train_ds=LatentPairs(args.cache_dir,"train")
    val_ds=LatentPairs(args.cache_dir,"val")
    loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    # Re-load same VAE only for validation, with frozen weights and adapter.
    from train_wan_dual_lora import make_model
    codec,_=make_model(_argparse.Namespace(
        rank=4,merger_scale=.15,resolution=args.resolution),device)
    codec.load_adapter(args.vae_lora)
    codec.requires_grad_(False)
    codec.eval()
    net=ConditionalAlphaDenoiser(hidden=args.hidden,blocks=args.blocks).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.01)
    schedule=cosine_alphas(args.diffusion_steps).to(device)
    begin=0
    best=-1.
    root=Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=True)
    last=root/"checkpoints/sidecar_latest.pt"
    if args.resume:
        checkpoint=torch.load(last,map_location="cpu",weights_only=True)
        if checkpoint["settings"]!=settings(args) or (
            checkpoint["vae_lora_sha256"]!=checksum(args.vae_lora) or
            checkpoint["manifest_sha256"]!=checksum(args.manifest)):
            raise RuntimeError("Incompatible resume checkpoint / pretrained VAE LoRA / dataset")
        net.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        begin=int(checkpoint["step"])
        best=float(checkpoint["best_alpha_iou"])
    elif last.exists():
        raise FileExistsError(f"Will not overwrite existing run: {last}; use --resume")
    torch.cuda.reset_peak_memory_stats()
    step=begin
    loader_iter=iter(loader)
    while step<args.steps:
        try:
            batch=next(loader_iter)
        except StopIteration:
            loader_iter=iter(loader)
            batch=next(loader_iter)
        batch=batch.to(device)
        loss,_=noise_loss(net,batch[:,24:],batch[:,:24],schedule)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step+=1
        if step==1 or step%25==0 or step==args.steps:
            print("ALPHA_DIFFUSION_STEP",json.dumps({
                "step":step,"noise_mse":float(loss.detach()),
                "cuda_peak_reserved_gib":round(torch.cuda.max_memory_reserved()/2**30,3)
            }),flush=True)
        if step%args.val_every==0 or step==args.steps:
            score=validate(net,codec,val_ds,schedule,args,device,step)
            with (root/"validation.jsonl").open("a") as f:
                f.write(json.dumps({"step":step,"alpha_iou_vs_vae_oracle":score})+"\n")
            if score>best:
                best=score
                save_snapshot(net,opt,args,step,best,tag="best")
        if step%args.save_every==0 or step==args.steps:
            save_snapshot(net,opt,args,step,best,tag="latest")
    print("ALPHA_DIFFUSION_TRAINING_COMPLETE",json.dumps(
        {"trained_steps":step,"best_iou_vs_oracle":best,
         "best_checkpoint":str(root/"checkpoints/sidecar_best.pt"),
         "alpha_generated_from_noise":True}),flush=True)


if __name__=="__main__":
    main()
