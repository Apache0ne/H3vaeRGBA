#!/usr/bin/env python3
"""Cache real Qwen3-VL-32B embeddings for Wan-joint H3 DiT LoRA training.

Run with the ISOLATED A100 Python 3.12 environment in a separate process,
after the fused latent cache has been built, without H3 33B loaded.
Uses exact upstream H3 prompt presentation builder, no random embeddings,
no fake captions, no retraining Qwen.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Allow execution by absolute filename from /content Colab, not just repo cwd.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def run(args):
    index=json.loads(Path(args.joint_index).read_text())["samples"]
    prompts=sorted({entry["caption"] for entry in index})
    if not prompts or any(not s.strip() for s in prompts):
        raise ValueError("No actual captions")
    directory=Path(args.output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    print("H3_QWEN_PROMPT_CACHE_PLAN",json.dumps({
        "unique_prompts":len(prompts),"samples":len(index),
        "output_dir":str(directory)
    }),flush=True)
    if args.preflight_only:
        return
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 A100 required for official 32B Qwen encoding")

    # Environment flags are consumed at import time by the upstream runner.
    os.environ["H3_TRANSFORMER_QUANT"]="int8"
    os.environ["H3_TE_QUANT"]="bnb-4bit"
    os.environ["H3_ATTN_BACKEND"]="default"
    os.environ["H3_TURBO_LORA"]="0"
    os.environ["H3_LOWVRAM"]="0"
    root=Path(args.upstream_repo)
    if not (root/"core/runner.py").is_file():
        raise FileNotFoundError(f"Real H3 upstream runner missing at {root}")
    sys.path.insert(0,str(root))
    from core.runner import MiniMaxH3Runner,_encode_h3_prompt

    runner=MiniMaxH3Runner(directory/"upstream_runner")
    runner._ensure_pipe_shell()
    runner._load_text_encoder()
    computed={}
    try:
        with torch.inference_mode(),runner._te_attached():
            for number,prompt in enumerate(prompts,1):
                key=hashlib.sha256(prompt.encode()).hexdigest()[:32]
                path=directory/f"{key}.pt"
                if not path.is_file():
                    emb,tags=_encode_h3_prompt(
                        runner._pipe,prompt,None,
                        device=runner._encode_device,dtype=torch.bfloat16)
                    if (emb.ndim!=3 or emb.shape[0]!=1 or emb.shape[2]!=5120
                        or tags.shape!=(emb.shape[1],)):
                        raise RuntimeError("Qwen output does not match native H3 conditioning")
                    torch.save({"embeds":emb.detach().cpu().half(),
                                "tags":tags.cpu().long()},path)
                computed[prompt]=str(path)
                print(f"QWEN_PROMPT_EMBEDS {number}/{len(prompts)} "
                      f"{key} cached=True",flush=True)
    finally:
        runner._free_text_encoder(force=True)
    index_hash=hashlib.sha256(Path(args.joint_index).read_bytes()).hexdigest()
    result={
        "source_index":str(Path(args.joint_index).resolve()),
        "joint_index_sha256":index_hash,
        "conditioner":"REAL_Qwen3VL_32B_H3_hidden_state_50",
        "prompt_embeds_by_text":computed,
    }
    (directory/"prompts.json").write_text(json.dumps(result,indent=2))
    print("H3_QWEN_TEXT_EMBEDDING_CACHE_COMPLETE",
          str(directory/"prompts.json"),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-index",required=True)
    p.add_argument("--output-dir",default="/content/h3_wan_joint_text")
    p.add_argument("--upstream-repo",default="/content/Diffusers_minimax-h3")
    p.add_argument("--preflight-only",action="store_true")
    run(p.parse_args())


if __name__=="__main__":
    main()
