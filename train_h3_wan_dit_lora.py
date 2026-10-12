#!/usr/bin/env python3
"""Wan-Alpha stage TWO port: train MiniMax-H3 33B DiT LoRA on fused 24ch RGB+A.

ACTUAL H3 data-ward rectified-flow loss, original packed temporal/video/text
layout, REAL Qwen3-VL prompt embeddings, actual H3 DiT weights quantized INT8.
All original 33B weights frozen. No 4000-step standalone sidecar.

This is an experimental single-A100 implementation. Full 124-frame inference,
audio consistency, and GPU memory headroom are NOT established until an A100
run succeeds. Must NOT call this a finished RGBA generator from CPU tests.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from h3rgba.wan_h3_flow import (
    h3_packed_flow_loss,inject_h3_transformer_lora,h3_dit_lora_state
)


def filehash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):
            h.update(c)
    return h.hexdigest()


def read_inputs(joint_index,prompts_json):
    index=json.loads(Path(joint_index).read_text())["samples"]
    text=json.loads(Path(prompts_json).read_text())
    assert text["conditioner"]=="REAL_Qwen3VL_32B_H3_hidden_state_50"
    records=[]
    for row in index:
        latent=Path(row["latent"])
        path=Path(text["prompt_embeds_by_text"].get(row["caption"],""))
        if not latent.is_file() or not path.is_file():
            raise FileNotFoundError("Missing native fused latent or REAL Qwen3-VL embedding for "+row["id"])
        if row["group"] not in ("train","validation"):
            raise ValueError("Unknown group; unseen holdouts must not enter training")
        records.append(dict(row,prompt_path=str(path)))
    if not any(x["group"]=="train" for x in records) or not any(
        x["group"]=="validation" for x in records):
        raise ValueError("Need train and selection validation samples")
    return records


def batch_from_record(row,device):
    raw=torch.load(row["latent"],map_location="cpu",weights_only=True)
    text=torch.load(row["prompt_path"],map_location="cpu",weights_only=True)
    x=raw["latent"]
    emb=text["embeds"]
    tags=text["tags"]
    if (x.ndim!=4 or x.shape[0]!=24 or not torch.isfinite(x).all()
        or emb.ndim!=3 or emb.shape[0]!=1 or emb.shape[-1]!=5120
        or tags.shape!=(emb.shape[1],)):
        raise RuntimeError("Invalid cached H3 fused video or REAL Qwen embeddings")
    return (x[None].to(device,dtype=torch.float32),
            emb.to(device,dtype=torch.bfloat16),
            tags.to(device,dtype=torch.long))


def load_transformer(rank,weight_mode="bf16"):
    from diffusers import MiniMaxH3Transformer3DModel
    print("LOADING_OFFICIAL_H3_33B_WAN_JOINT_DIFFUSION_LORA",json.dumps({
        "weight_mode":weight_mode,"notes":"Full BF16 preferred on A100 80GB for verified "
        "autograd through frozen base. Int8 backward remains experimental."}),flush=True)
    kwargs=dict(
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if weight_mode=="int8":
        from diffusers import TorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig
        kwargs["quantization_config"]=TorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=[
                "proj_in","audio_proj_in","context_embedder",
                "time_embedder","time_proj","token_refiner",
                "norm_out","proj_out","audio_proj_out",
            ])
    elif weight_mode!="bf16":
        raise ValueError("weight_mode must be bf16 or int8")
    model=MiniMaxH3Transformer3DModel.from_pretrained(
        "MiniMaxAI/MiniMax-H3",subfolder="transformer",**kwargs
    )
    model.to("cuda")
    model.enable_gradient_checkpointing()
    n=inject_h3_transformer_lora(model,rank=rank)
    model.train()
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not 0<count<200_000_000:
        raise RuntimeError("LoRA expected <200M trainable parameters; incorrect freeze")
    print("H3_DIT_LORA_INJECTED",json.dumps({
        "matched_linear_layers":n,"trainable":count,
        "pretrained_h3_frozen":True
    }),flush=True)
    return model


@torch.no_grad()
def eval_set(model,records,seed):
    model.eval()
    losses=[]
    for j,row in enumerate(records):
        z,emb,tags=batch_from_record(row,"cuda")
        gen=torch.Generator(device="cuda").manual_seed(seed+40000+j)
        noise=torch.randn(z.shape,device=z.device,generator=gen)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss,_=h3_packed_flow_loss(
                model,z,emb,tags,t=torch.tensor(.5,device=z.device),
                noise=noise)
        losses.append(float(loss))
    model.train()
    result=sum(losses)/len(losses)
    print("H3_WAN_JOINT_DIT_VAL_FLOW_MSE",round(result,6),flush=True)
    return result


def save(model,optimizer,args,step,best,*,new_best=False):
    root=Path(args.output_dir)/"checkpoints"
    root.mkdir(parents=True,exist_ok=True)
    tensors=h3_dit_lora_state(model)
    save_file(tensors,str(root/"dit_lora_latest.safetensors"))
    tmp=root/"state_latest.tmp"
    torch.save({
        "step":step,"best":best,
        "optimizer":optimizer.state_dict(),
        "settings":{
            "rank":args.rank,"lr":args.lr,"seed":args.seed,
            "weight_mode":args.weight_mode,
            "joint_index_hash":filehash(args.joint_index),
            "prompts_hash":filehash(args.prompts_json),
        }
    },tmp)
    os.replace(tmp,root/"state_latest.pt")
    if new_best:
        shutil.copy2(root/"dit_lora_latest.safetensors",
                     root/"dit_lora_best.safetensors")
        (root/"best_validation.json").write_text(json.dumps(best,indent=2))
    if args.backup_dir:
        dest=Path(args.backup_dir)
        dest.mkdir(parents=True,exist_ok=True)
        filenames=["dit_lora_latest.safetensors","state_latest.pt"]
        if new_best:
            filenames+=["dit_lora_best.safetensors","best_validation.json"]
        for filename in filenames:
            tmp=dest/(filename+".partial")
            shutil.copy2(root/filename,tmp)
            os.replace(tmp,dest/filename)
    print("H3_WAN_DIT_LORA_SAVED",json.dumps({
        "step":step,"best_updated":new_best,
        "checkpoint":str(root/"dit_lora_latest.safetensors")
    }),flush=True)


def load_lora_file(model,path):
    from safetensors.torch import load_file
    s=load_file(str(path),device="cpu")
    modules=dict(model.named_parameters())
    if set(s)!={k for k in modules if k.endswith((".lora_down",".lora_up"))}:
        raise RuntimeError("Saved DiT LoRA adapter key mismatch")
    with torch.no_grad():
        for name,value in s.items():
            p=modules[name]
            if p.shape!=value.shape:
                raise RuntimeError("LoRA parameter shape mismatch")
            p.copy_(value.to(p.device,p.dtype))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-index",required=True)
    p.add_argument("--prompts-json",required=True)
    p.add_argument("--output-dir",default="/content/h3_wan_dit_lora")
    p.add_argument("--backup-dir",default=None)
    p.add_argument("--steps",type=int,default=2)
    p.add_argument("--rank",type=int,default=8)
    p.add_argument("--weight-mode",choices=("bf16","int8"),default="bf16",
                   help="BF16 base on A100 80GB preferred for LoRA backward; "
                        "int8 training is experimental")
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--save-every",type=int,default=50)
    p.add_argument("--val-every",type=int,default=50)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--preflight-only",action="store_true")
    args=p.parse_args()
    if min(args.steps,args.rank,args.lr,args.save_every,args.val_every)<=0:
        raise ValueError("Positive configuration required")
    records=read_inputs(args.joint_index,args.prompts_json)
    train=[x for x in records if x["group"]=="train"]
    val=[x for x in records if x["group"]=="validation"]
    print("H3_WAN_DIT_DATA",json.dumps({
        "train":len(train),"validation":len(val),
        "method":"real H3 rectified-flow LoRA on ONE fused RGB+A latent",
        "audio_supervision":"NONE",
        "H3_generation_verified":False
    }),flush=True)
    if args.preflight_only:
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU with BF16 required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model=load_transformer(args.rank,weight_mode=args.weight_mode)
    optimizer=torch.optim.AdamW(
        (q for q in model.parameters() if q.requires_grad),
        lr=args.lr,weight_decay=0.01)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    resume_file=out/"checkpoints/state_latest.pt"
    step=0
    best=None
    if args.resume:
        ckpt=torch.load(resume_file,map_location="cpu",weights_only=True)
        want={
            "rank":args.rank,"lr":args.lr,"seed":args.seed,
            "joint_index_hash":filehash(args.joint_index),
            "prompts_hash":filehash(args.prompts_json),
        }
        if ckpt["settings"]!=want:
            raise RuntimeError("Resume model data or adapter settings mismatch")
        load_lora_file(model,out/"checkpoints/dit_lora_latest.safetensors")
        optimizer.load_state_dict(ckpt["optimizer"])
        step=int(ckpt["step"])
        best=ckpt["best"]
    elif resume_file.exists():
        raise RuntimeError("Previous run exists; use --resume")
    torch.cuda.reset_peak_memory_stats()
    while step<args.steps:
        row=train[step%len(train)]
        z,emb,tags=batch_from_record(row,"cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss,parts=h3_packed_flow_loss(model,z,emb,tags)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (q for q in model.parameters() if q.requires_grad),1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step+=1
        if step==1 or step%10==0:
            print("H3_WAN_JOINT_DIT_STEP",json.dumps({
                "step":step,**parts,
                "peak_gpu_gib":round(torch.cuda.max_memory_reserved()/2**30,2),
            }),flush=True)
        new_best=False
        if step%args.val_every==0 or step==args.steps:
            score=eval_set(model,val,args.seed)
            new_best=best is None or score<best["flow_mse"]
            if new_best:
                best={"step":step,"flow_mse":score}
        if step%args.save_every==0 or step==args.steps or new_best:
            save(model,optimizer,args,step,best,new_best=new_best)
    print("H3_WAN_DIT_STAGE2_TRAIN_COMPLETE",json.dumps({
        "step":step,"best":best,
        "note":"Requires full H3 generation with this adapter loaded and "
               "the WAN-JOINT VAE decoder; original H3 inference is RGB-only."
    }),flush=True)


if __name__=="__main__":
    main()
