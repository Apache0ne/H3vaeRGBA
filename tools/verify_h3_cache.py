"""Check whether the *official MiniMax-H3 video VAE* is really cached.

This does not download files or load model tensors. It verifies all three
published VAE safetensors shards and their weight index in an HF cache snapshot.

Usage: python tools/verify_h3_cache.py
"""
import argparse
import json
import os
from pathlib import Path

SHARDS = [f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors" for i in range(1, 4)]
INDEX = "diffusion_pytorch_model.safetensors.index.json"
EXPECTED = set(SHARDS + [INDEX, "config.json"])


def check_snapshot(snapshot):
    snapshot = Path(snapshot)
    vae = snapshot / "vae"
    print(f"SNAPSHOT: {snapshot}", flush=True)
    print(f"VAE DIRECTORY: {vae} | exists={vae.is_dir()}", flush=True)
    if not vae.is_dir():
        return False
    total = 0
    for name in sorted(EXPECTED):
        file = vae / name
        present = file.is_file()
        size = file.stat().st_size if present else 0
        print(f"{'OK' if present else 'MISSING':7} vae/{name}  {size / 1e9:.3f} GB", flush=True)
        if not present:
            return False
        if name.endswith(".safetensors"):
            if size < 50_000_000:
                print(f"UNEXPECTEDLY SMALL: {name} ({size} bytes)", flush=True)
                return False
            total += size

    index = json.loads((vae / INDEX).read_text())
    weights = index.get("weight_map", {})
    if not weights or not set(weights.values()).issubset(set(SHARDS)):
        print("Invalid VAE shard index: unexpected shard names or empty weight map", flush=True)
        return False
    for key in ("encoder.conv_in.weight", "decoder.proj_out.weight"):
        print(f"WEIGHT_MAP {key} -> {weights.get(key, '<missing>')}", flush=True)
        if key not in weights:
            print("Required VAE checkpoint tensor absent from index", flush=True)
            return False
    print(f"OFFICIAL_VAE_SHARDS_PRESENT=3 | total={total / 1e9:.3f} GB | indexed_tensors={len(weights)}",
          flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=None,
                        help="Optional explicit HF snapshot directory")
    args = parser.parse_args()
    if args.snapshot:
        roots = [Path(args.snapshot)]
    else:
        from huggingface_hub import scan_cache_dir
        result = scan_cache_dir()
        roots = [rev.snapshot_path for repo in result.repos
                 if repo.repo_type == "model" and repo.repo_id == "MiniMaxAI/MiniMax-H3"
                 for rev in repo.revisions]
    if not roots:
        raise SystemExit("No MiniMaxAI/MiniMax-H3 snapshot in this cache. Check HF_HOME/HF_HUB_CACHE.")
    success = False
    for root in roots:
        success = check_snapshot(root) or success
    if not success:
        raise SystemExit("OFFICIAL_VAE_SHARDS_MISSING: The visual VAE shards cannot all be verified locally.")
    print("H3_VAE_CACHE_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
