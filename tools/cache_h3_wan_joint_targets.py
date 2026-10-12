#!/usr/bin/env python3
"""Encode Wan-joint 24ch H3 RGBA training targets with a trained shared VAE.

Needs stage-1 joint_bridge_best.safetensors plus joint_dual_best.safetensors.
Exports only *train* and original 3 *selection validation* videos. Nine unused
holdouts are NOT used for any adapter learning. Requires real descriptive
per-video captions JSON mapping exact manifest ID -> text. Never fabricates
captions from filenames. Text embeddings are cached separately by Qwen3-VL.
"""
import argparse
import hashlib
import json
from pathlib import Path

# Allow execution by absolute filename from /content Colab, not just repo cwd.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from h3rgba.wan_joint_h3 import WanJointH3Codec


def filehash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):
            h.update(c)
    return h.hexdigest()


def captions_for(items,captions):
    missing=[row["id"] for row in items if not isinstance(
        captions.get(row["id"]),str) or not captions[row["id"]].strip()]
    if missing:
        raise ValueError(
            f"Missing REAL captions for {len(missing)} clips; examples: "
            f"{missing[:8]}. No fake generic captions will be substituted.")
    return [captions[r["id"]].strip() for r in items]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--captions-json",required=True)
    parser.add_argument("--joint-bridge",required=True)
    parser.add_argument("--joint-dual",required=True)
    parser.add_argument("--base-vae-lora",required=True)
    parser.add_argument("--output-dir",default="/content/h3_wan_joint_latents")
    parser.add_argument("--resolution",type=int,default=128)
    parser.add_argument("--hidden",type=int,default=128)
    parser.add_argument("--shift-limit",type=float,default=.35)
    parser.add_argument("--bridge-scale",type=float,default=1.)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    records=json.loads(Path(args.manifest).read_text())["clips"]
    captions=json.loads(Path(args.captions_json).read_text())
    from run_vae_22gb import training_split
    train,val=training_split(records,args.seed)
    groups={"train":train,"validation":val[:3]}
    missing={}
    for group,items in groups.items():
        try:
            captions_for(items,captions)
        except ValueError as e:
            missing[group]=str(e)
    print("H3_WAN_JOINT_DATA_PROOF",json.dumps({
        "total":len(records),"train":len(train),"val":len(val[:3]),
        "unseen_reserved":len(val[3:]),
        "captions_missing":missing,
        "manifest_hash":filehash(args.manifest),
    }),flush=True)
    if missing:
        raise ValueError("Need actual prompt descriptions for paired videos")
    if args.preflight_only:
        print("CAPTIONS_AND_SPLIT_VERIFIED_NO_VAE_LOAD",flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required to encode H3 pretrained VAE")
    from train_wan_dual_lora import make_model
    from train_vae import PairedVideoDataset
    import argparse as ap
    dual,_=make_model(ap.Namespace(rank=4,merger_scale=.15,
                                    resolution=args.resolution),"cuda")
    dual.load_adapter(args.joint_dual)
    joint=WanJointH3Codec(dual,hidden=args.hidden,
                          shift_limit=args.shift_limit,
                          bridge_scale=args.bridge_scale,
                          train_decoder_lora=False).cuda()
    joint.load_bridge(args.joint_bridge)
    joint.eval()
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    index=[]
    with torch.inference_mode():
        for group,items in groups.items():
            data=PairedVideoDataset(items,resolution=args.resolution,
                                    deterministic=True)
            group_dir=output/group
            group_dir.mkdir(parents=True,exist_ok=True)
            for k,(sample,row) in enumerate(zip(data,items)):
                latent,_,_=joint.encode_joint(sample[None].to("cuda").float())
                norm=joint.normalized_diffusion_target(latent)[0]
                if norm.ndim!=4 or norm.shape[0]!=24 or not torch.isfinite(norm).all():
                    raise RuntimeError("Bad H3 shared diffusion training target")
                dest=group_dir/f"{k:05d}.pt"
                torch.save({"latent":norm.cpu().half().contiguous()},dest)
                index.append({"id":row["id"],"caption":captions[row["id"]],
                              "group":group,"latent":str(dest)})
                if (k+1)%25==0:
                    print("WAN_JOINT_TARGET_CACHE",group,k+1,len(items),flush=True)
    record={
        "format":"MiniMaxH3 single normalized 24ch joint RGB-alpha latent",
        "stage1_bridge_hash":filehash(args.joint_bridge),
        "stage1_dual_hash":filehash(args.joint_dual),
        "source_adapter_hash":filehash(args.base_vae_lora),
        "manifest_hash":filehash(args.manifest),
        "captions_hash":filehash(args.captions_json),
        "resolution":args.resolution,
        "seed":args.seed,
        "hidden":args.hidden,
        "shift_limit":args.shift_limit,
        "bridge_scale":args.bridge_scale,
        "samples":index,
    }
    (output/"index.json").write_text(json.dumps(record,indent=2))
    print("WAN_H3_JOINT_TARGET_CACHE_COMPLETE",json.dumps({
        "train":len(train),"validation":len(val[:3]),
        "reserved_unused":len(val[3:]),"index":str(output/"index.json")
    }),flush=True)


if __name__=="__main__":
    main()
