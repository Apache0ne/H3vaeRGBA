#!/usr/bin/env python3
"""Fail-closed provenance verification for H3 Wan joint latent and Qwen caches.

Runs on CPU; never downloads H3 weights. An existing index is NOT evidence
that it belongs to the current VAE bridge, adapter, manifest, or captions.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def validate_joint_cache(index_path, *, bridge, dual, base_lora, manifest, captions,
                         resolution=None, seed=None):
    index = _read(index_path)
    expected = {
        "stage1_bridge_hash": bridge,
        "stage1_dual_hash": dual,
        "source_adapter_hash": base_lora,
        "manifest_hash": manifest,
        "captions_hash": captions,
    }
    for field, path in expected.items():
        current = sha256(path)
        if index.get(field) != current:
            raise RuntimeError(
                f"STALE_JOINT_CACHE: {field} differs from {path}; "
                "recache joint latents with the current trained VAE."
            )
    if resolution is not None and index.get("resolution") != resolution:
        raise RuntimeError(
            f"STALE_JOINT_CACHE: resolution is {index.get('resolution')}; "
            f"requested {resolution}. Re-encode joint H3 latents."
        )
    if seed is not None and index.get("seed") != seed:
        raise RuntimeError(
            f"STALE_JOINT_CACHE: split seed is {index.get('seed')}; "
            f"requested {seed}. Recreate deterministic data split."
        )
    samples = index.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("Joint latent cache has no samples")
    seen = set()
    counts = {"train": 0, "validation": 0}
    for row in samples:
        clip_id = row.get("id")
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen:
            raise RuntimeError(f"Missing/duplicate joint cache sample ID: {clip_id}")
        seen.add(clip_id)
        group = row.get("group")
        if group not in counts:
            raise RuntimeError("Joint cache may only contain train/selection-validation data")
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise RuntimeError(f"Missing real caption for {clip_id}")
        if not Path(row["latent"]).is_file():
            raise FileNotFoundError(f"Missing cached H3 latent for {clip_id}: {row['latent']}")
        counts[group] += 1
    if min(counts.values()) < 1:
        raise RuntimeError("Joint cache lacks train or selection-validation samples")
    return counts


def validate_text_cache(index_path, prompts_json):
    index = _read(index_path)
    prompts = _read(prompts_json)
    current = sha256(index_path)
    if prompts.get("joint_index_sha256") != current:
        raise RuntimeError("STALE_TEXT_CACHE: Qwen cache was built for a different joint index")
    if prompts.get("conditioner") != "REAL_Qwen3VL_32B_H3_hidden_state_50":
        raise RuntimeError("Wrong H3 text-conditioning source")
    mapping = prompts.get("prompt_embeds_by_text", {})
    for row in index.get("samples", []):
        path = mapping.get(row["caption"])
        if not path or not Path(path).is_file():
            raise RuntimeError(f"Missing REAL Qwen prompt embedding for {row['id']}")
    return len(mapping)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-index", required=True)
    p.add_argument("--bridge", required=True)
    p.add_argument("--dual", required=True)
    p.add_argument("--base-lora", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--captions", required=True)
    p.add_argument("--prompts-json")
    p.add_argument("--resolution", type=int)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    counts = validate_joint_cache(
        args.joint_index, bridge=args.bridge, dual=args.dual,
        base_lora=args.base_lora, manifest=args.manifest, captions=args.captions,
        resolution=args.resolution, seed=args.seed,
    )
    result = {"status": "VALID_H3_WAN_JOINT_CACHE", **counts}
    if args.prompts_json:
        result["cached_qwen_prompts"] = validate_text_cache(
            args.joint_index, args.prompts_json
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
