#!/usr/bin/env python3
"""Wan-Alpha stage ONE for MiniMax-H3: train a SHARED 24ch RGB+alpha VAE.

Original decoder LoRA from 500-step H3 experiment is INITIALIZATION, not
the entire method. Train 48->24 merger + one-stream alpha bridge, optionally
continue decoder LoRA/head, with RGB and alpha reconstruction from the exact
SAME fused latent. No 4000-step sidecar is used.

Afterwards train an H3 33B DiT LoRA on normalized fused latents with paired
text captions. Original RGB-only H3 DiT output cannot be assumed joint-RGBA.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from h3rgba.wan_joint_h3 import WanJointH3Codec,joint_rgba_loss
from h3rgba.dual_loss import alpha_scores


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def make_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--vae-lora",required=True)
    p.add_argument("--output-dir",default="/content/h3_wan_joint")
    p.add_argument("--backup-dir",default=None)
    p.add_argument("--resolution",type=int,default=128)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--shift-limit",type=float,default=.35)
    p.add_argument("--bridge-scale",type=float,default=1.)
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--lr",type=float,default=3e-5)
    p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--val-every",type=int,default=100)
    p.add_argument("--train-decoder-lora",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--preflight-only",action="store_true")
    return p


def settings(a):
    return {k:getattr(a,k) for k in
        ("resolution","seed","hidden","shift_limit","bridge_scale",
         "lr","train_decoder_lora")}


def load_splits(args):
    from run_vae_22gb import training_split
    from train_vae import PairedVideoDataset
    records=json.loads(Path(args.manifest).read_text())["clips"]
    train,val=training_split(records,args.seed)
    print("WAN_JOINT_SPLIT",json.dumps({"total":len(records),
        "train":len(train),"model_selection_val":min(3,len(val)),
        "unused_holdout":max(0,len(val)-3),
        "note":"Only first 3 validation clips may select a checkpoint"}),flush=True)
    return (
        DataLoader(PairedVideoDataset(train,resolution=args.resolution),
                   batch_size=1,shuffle=True,num_workers=0),
        DataLoader(PairedVideoDataset(val[:3],resolution=args.resolution,
                                      deterministic=True),
                   batch_size=1,shuffle=False,num_workers=0),
    )


def build(args,device):
    from train_wan_dual_lora import make_model
    dual,_=make_model(argparse.Namespace(
        rank=4,merger_scale=.15,resolution=args.resolution),device)
    dual.load_adapter(args.vae_lora)
    codec=WanJointH3Codec(dual,hidden=args.hidden,
                          shift_limit=args.shift_limit,
                          bridge_scale=args.bridge_scale,
                          train_decoder_lora=args.train_decoder_lora).to(device)
    count=sum(p.numel() for p in codec.trainable_parameters())
    print("WAN_JOINT_TRAINABLE_PARAMS",count,flush=True)
    return codec


def save_validation_preview(rgba,result,output,step):
    """Visible ground-truth vs single-latent RGB/alpha and checkerboard."""
    import numpy as np
    from PIL import Image, ImageDraw
    base=Path(output)/"previews"
    base.mkdir(parents=True,exist_ok=True)
    gt=rgba[0].detach().cpu().float().clamp(0,1)
    pred_rgb=result["rgb"][0].detach().cpu().float().clamp(0,1)
    pred_a=result["alpha"][0].detach().cpu().float().clamp(0,1)
    selected=[0,8,16]
    _,frames,h,w=gt.shape
    sheet=Image.new("RGB",(w*5,len(selected)*(h+24)),(25,25,25))
    draw=ImageDraw.Draw(sheet)
    labels=["GT RGB","Joint RGB","GT alpha","Joint alpha","Joint over checker"]
    ygrid,xgrid=np.indices((h,w))
    checker=(np.where((xgrid//8+ygrid//8)%2,175,235)
             .astype(np.float32)[:,:,None])
    for row,index in enumerate(selected):
        gt_rgb=gt[:3,index].permute(1,2,0).numpy()
        reconstructed=pred_rgb[:,index].permute(1,2,0).numpy()
        gt_alpha=gt[3,index].numpy()[:,:,None]
        estimated=pred_a[0,index].numpy()[:,:,None]
        check_rgb=np.repeat(checker,3,axis=2)/255
        imgs=[gt_rgb,reconstructed,np.repeat(gt_alpha,3,axis=2),
              np.repeat(estimated,3,axis=2),
              reconstructed*estimated+check_rgb*(1-estimated)]
        for col,(label,img) in enumerate(zip(labels,imgs)):
            x=col*w;y=row*(h+24)
            draw.text((x+4,y+4),label+" f"+str(index),fill="white")
            sheet.paste(Image.fromarray(
                np.uint8(np.rint(np.clip(img,0,1)*255))), (x,y+24))
    path=base/f"step_{step:06d}.png"
    sheet.save(path)
    print("WAN_JOINT_RGB_ALPHA_PREVIEW",path,flush=True)


@torch.no_grad()
def validate(model,loader,step,output=None):
    model.eval()
    totals=[]
    for i,batch in enumerate(loader):
        rgba=batch.to("cuda",dtype=torch.float32)
        result=model(rgba)
        scores=alpha_scores(result["alpha"],rgba[:,3:4])
        scores["rgb_visible_l1"]=float(
            ((result["rgb"]-rgba[:,:3]).abs()*rgba[:,3:4]).mean())
        scores["joint_shift_mse"]=float(
            (result["joint"]-result["rgb_teacher_latent"]).square().mean())
        totals.append(scores)
        if i==0 and output is not None:
            save_validation_preview(rgba,result,output,step)
    model.train()
    aggregate={key:round(sum(x[key] for x in totals)/len(totals),6)
               for key in totals[0]}
    aggregate["step"]=step
    print("WAN_JOINT_VALIDATION",json.dumps(aggregate),flush=True)
    return aggregate


def save(args,model,optimizer,step,best,*,is_best=False):
    root=Path(args.output_dir)/"checkpoints"
    root.mkdir(parents=True,exist_ok=True)
    tmp=root/"joint_latest.tmp"
    payload={"step":step,"best":best,"settings":settings(args),
             "manifest_hash":sha256(args.manifest),
             "base_adapter_hash":sha256(args.vae_lora),
             "fusion":model.fuser.state_dict(),
             "alpha_bridge":model.alpha_bridge.state_dict(),
             "dual_lora":model.dual.adapter_state(),
             "optimizer":optimizer.state_dict()}
    torch.save(payload,tmp)
    os.replace(tmp,root/"joint_latest.pt")
    if is_best:
        save_file(model.bridge_state(),str(root/"joint_bridge_best.safetensors"))
        save_file(model.dual.adapter_state(),str(root/"joint_dual_best.safetensors"))
        (root/"joint_best_metrics.json").write_text(json.dumps(best,indent=2))
    if args.backup_dir:
        backup=Path(args.backup_dir)
        backup.mkdir(parents=True,exist_ok=True)
        files=["joint_latest.pt"]
        if is_best:
            files+=["joint_bridge_best.safetensors",
                    "joint_dual_best.safetensors","joint_best_metrics.json"]
        for name in files:
            source=root/name
            temp=backup/(name+".partial")
            shutil.copy2(source,temp)
            os.replace(temp,backup/name)
    print("WAN_JOINT_CHECKPOINT",json.dumps({
        "step":step,"best":is_best,"path":str(root/"joint_latest.pt")}),flush=True)


def run(args):
    if args.resolution<64 or args.resolution%32:
        raise ValueError("Resolution must be divisible by 32 and >=64")
    if min(args.hidden,args.steps,args.save_every,args.val_every,args.lr)<=0:
        raise ValueError("Positive hyperparameters required")
    if not Path(args.vae_lora).is_file():
        raise FileNotFoundError("Missing 500-step dual VAE LoRA checkpoint")
    train,val=load_splits(args)
    if args.preflight_only:
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 capable CUDA required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model=build(args,"cuda")
    optimizer=torch.optim.AdamW(model.trainable_parameters(),
                                lr=args.lr,weight_decay=.01)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    record=out/"settings.json"
    if record.exists() and json.loads(record.read_text())!=settings(args):
        raise RuntimeError("Run directory settings mismatch")
    record.write_text(json.dumps(settings(args),indent=2))
    begin=0
    best=None
    latest=out/"checkpoints/joint_latest.pt"
    if args.resume:
        state=torch.load(latest,map_location="cpu",weights_only=True)
        if (state["settings"]!=settings(args) or
            state["manifest_hash"]!=sha256(args.manifest) or
            state["base_adapter_hash"]!=sha256(args.vae_lora)):
            raise RuntimeError("Resume provenance mismatch")
        model.fuser.load_state_dict(state["fusion"])
        model.alpha_bridge.load_state_dict(state["alpha_bridge"])
        with torch.no_grad():
            for name,v in state["dual_lora"].items():
                real="dual.vae."+name if name.startswith("decoder.") else "dual."+name
                param=dict(model.named_parameters())[real]
                param.copy_(v.to(device=param.device,dtype=param.dtype))
        optimizer.load_state_dict(state["optimizer"])
        begin=state["step"]
        best=state["best"]
    elif latest.exists():
        raise RuntimeError("Existing joint training run; use --resume")
    iterator=iter(train)
    model.train()
    for step in range(begin+1,args.steps+1):
        try:
            batch=next(iterator)
        except StopIteration:
            iterator=iter(train)
            batch=next(iterator)
        rgba=batch.to("cuda",dtype=torch.float32)
        outputs=model(rgba)
        loss,parts=joint_rgba_loss(model,rgba,outputs)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite joint RGB alpha loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step==1 or step%25==0:
            print("WAN_JOINT_STEP",json.dumps({
                "step":step,"loss":float(loss.detach()),**parts,
                "gpu_peak_reserved_gib":round(torch.cuda.max_memory_reserved()/2**30,2),
            }),flush=True)
        selected=False
        if step%args.val_every==0 or step==args.steps:
            scored=validate(model,val,step,output=out)
            with (out/"validation.jsonl").open("a") as f:
                f.write(json.dumps(scored)+"\n")
            if (best is None or
                (scored["alpha_iou"],-scored["alpha_mae"]) >
                (best["alpha_iou"],-best["alpha_mae"])):
                best=scored
                selected=True
        if step%args.save_every==0 or selected or step==args.steps:
            save(args,model,optimizer,step,best,is_best=selected)
    print("WAN_JOINT_STAGE1_COMPLETE",json.dumps({
        "steps":args.steps,"best":best,
        "bridge":str(out/"checkpoints/joint_bridge_best.safetensors"),
        "dual":str(out/"checkpoints/joint_dual_best.safetensors"),
        "next":"Train H3 33B DiT LoRA on normalized joint latent targets with real captions. "
               "Do NOT use original RGB-only DiT as proven alpha generator."}),flush=True)


if __name__=="__main__":
    run(make_parser().parse_args())
