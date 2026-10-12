"""Regression coverage for native H3 Wan two-stage training and inference gates.

No mock metric is counted as GPU training quality: these are CPU contract tests.
"""
import json
from types import SimpleNamespace

import pytest

from train_h3_wan_dit_lora import run_signature
from tools.validate_h3_wan_cache import (
    sha256, validate_joint_cache, validate_text_cache,
)
from tools.launch_h3_wan_native_a100 import generation_is_current


def fixtures(tmp_path):
    paths = {}
    for name in ("bridge", "dual", "base_lora", "manifest", "captions"):
        p = tmp_path / (name + ".safetensors")
        p.write_bytes(("real " + name).encode())
        paths[name] = p
    latent = tmp_path / "latent.pt"
    latent.write_bytes(b"nonempty-test-latent")
    other = tmp_path / "val.pt"
    other.write_bytes(b"nonempty-test-validation")
    payload = {
        "stage1_bridge_hash": sha256(paths["bridge"]),
        "stage1_dual_hash": sha256(paths["dual"]),
        "source_adapter_hash": sha256(paths["base_lora"]),
        "manifest_hash": sha256(paths["manifest"]),
        "captions_hash": sha256(paths["captions"]),
        "resolution": 256,
        "seed": 42,
        "samples": [
            {"id": "clip-1", "caption": "a cat", "group": "train", "latent": str(latent),
             "latent_sha256": sha256(latent)},
            {"id": "clip-2", "caption": "a dog", "group": "validation", "latent": str(other),
             "latent_sha256": sha256(other)},
        ],
    }
    index = tmp_path / "index.json"
    index.write_text(json.dumps(payload))
    paths["index"] = index
    return paths


def test_dit_resume_signature_includes_same_weight_mode_and_real_cache_hashes(tmp_path):
    paths = fixtures(tmp_path)
    args = SimpleNamespace(
        rank=8, lr=2e-5, seed=42, weight_mode="bf16",
        joint_index=str(paths["index"]), prompts_json=str(paths["captions"]),
    )
    saved = run_signature(args)
    assert saved == run_signature(args)
    assert saved["weight_mode"] == "bf16"
    args.weight_mode = "int8"
    assert saved != run_signature(args)
    args.weight_mode = "bf16"
    paths["captions"].write_bytes(b"different prompts")
    assert saved != run_signature(args)


def test_cache_provenance_refuses_changed_bridge_or_lost_latent(tmp_path):
    paths = fixtures(tmp_path)
    kwargs = {k: paths[k] for k in
              ("bridge", "dual", "base_lora", "manifest", "captions")}
    assert validate_joint_cache(paths["index"], **kwargs) == {
        "train": 1, "validation": 1
    }
    paths["bridge"].write_bytes(b"updated VAE after more training")
    with pytest.raises(RuntimeError, match="STALE_JOINT_CACHE"):
        validate_joint_cache(paths["index"], **kwargs)
    paths["bridge"].write_bytes(b"real bridge")
    (tmp_path / "val.pt").unlink()
    with pytest.raises(FileNotFoundError, match="clip-2"):
        validate_joint_cache(paths["index"], **kwargs)


def test_text_embeddings_are_bound_to_exact_latent_index(tmp_path):
    paths = fixtures(tmp_path)
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"real embeddings A")
    b.write_bytes(b"real embeddings B")
    text = tmp_path / "prompts.json"
    text.write_text(json.dumps({
        "conditioner": "REAL_Qwen3VL_32B_H3_hidden_state_50",
        "joint_index_sha256": sha256(paths["index"]),
        "prompt_embeds_by_text": {"a cat": str(a), "a dog": str(b)},
        "prompt_embeds_sha256": {"a cat": sha256(a), "a dog": sha256(b)},
    }))
    assert validate_text_cache(paths["index"], text) == 2
    data = json.loads(paths["index"].read_text())
    data["samples"][0]["caption"] = "another cat"
    paths["index"].write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="STALE_TEXT_CACHE"):
        validate_text_cache(paths["index"], text)


def test_old_generation_report_never_suppresses_new_trained_lora(tmp_path):
    old = tmp_path / "adapter.safetensors"
    old.write_bytes(b"first weight checkpoint")
    movie = tmp_path / "video.mp4"
    latents = tmp_path / "joint.pt"
    movie.write_bytes(b"video")
    latents.write_bytes(b"latents")
    report = tmp_path / "wan_h3_generation.json"
    report.write_text(json.dumps({
        "dit_lora_hash": sha256(old),
        "rgb_audio_mp4": str(movie),
        "generated_latents": str(latents),
        "generated_latents_sha256": sha256(latents),
    }))
    assert generation_is_current(report,old)
    old.write_bytes(b"new weight checkpoint")
    assert not generation_is_current(report,old)
    report.unlink()
    assert not generation_is_current(report,old)


def test_final_holdout_scoring_requires_actual_alpha_match():
    import torch
    from tools.validate_h3_wan_holdout import score_clip, aggregate_holdout

    rgba = torch.zeros(1,4,3,8,8)
    rgba[:, :3] = 0.8
    rgba[:, 3:, :, 2:6, 2:6] = 1.0
    perfect = {"rgb": rgba[:, :3].clone(), "alpha": rgba[:, 3:].clone()}
    metrics = score_clip(rgba, perfect)
    assert metrics["alpha_iou"] == pytest.approx(1.0)
    assert metrics["composite_rgb_l1"] == pytest.approx(0.0,abs=1e-6)
    assert metrics["alpha_temporal_error"] == pytest.approx(0.0)
    wrong = {"rgb": rgba[:, :3].clone(), "alpha": torch.ones_like(rgba[:, 3:])}
    bad = score_clip(rgba, wrong)
    assert bad["alpha_iou"] < 0.50
    assert bad["composite_rgb_l1"] > 0.0
    summary = aggregate_holdout([metrics, bad])
    assert summary["alpha_iou"] < 1.0
    with pytest.raises(ValueError, match="No reserved"):
        aggregate_holdout([])


def test_reject_wrong_video_latent_resolution_or_split_seed(tmp_path):
    paths=fixtures(tmp_path)
    kwargs={k:paths[k] for k in ("bridge","dual","base_lora","manifest","captions")}
    assert validate_joint_cache(paths["index"], resolution=256,seed=42,**kwargs)
    with pytest.raises(RuntimeError,match="resolution"):
        validate_joint_cache(paths["index"],resolution=128,seed=42,**kwargs)
    with pytest.raises(RuntimeError,match="split seed"):
        validate_joint_cache(paths["index"],resolution=256,seed=999,**kwargs)


def test_corrupted_cached_latent_is_detected_before_33b_training(tmp_path):
    paths=fixtures(tmp_path)
    kwargs={k:paths[k] for k in ("bridge","dual","base_lora","manifest","captions")}
    (tmp_path/"latent.pt").write_bytes(b"modified latent tensors")
    with pytest.raises(RuntimeError,match="tensor for clip-1 changed"):
        validate_joint_cache(paths["index"],**kwargs)


def test_corrupt_real_qwen_embeddings_fail_before_training(tmp_path):
    paths=fixtures(tmp_path)
    e=tmp_path/"qwen.pt"
    e.write_bytes(b"Qwen tensor 5120D")
    text=tmp_path/"prompts.json"
    text.write_text(json.dumps({
        "conditioner":"REAL_Qwen3VL_32B_H3_hidden_state_50",
        "joint_index_sha256":sha256(paths["index"]),
        "prompt_embeds_by_text":{"a cat":str(e),"a dog":str(e)},
        "prompt_embeds_sha256":{"a cat":sha256(e),"a dog":sha256(e)},
    }))
    assert validate_text_cache(paths["index"],text)==2
    e.write_bytes(b"BAD QWEN TENSOR")
    with pytest.raises(RuntimeError,match="STALE_TEXT_CACHE"):
        validate_text_cache(paths["index"],text)


def test_tampered_captured_h3_native_latents_fail_generation_guard(tmp_path):
    adapter=tmp_path/"dit_lora.safetensors"
    adapter.write_bytes(b"trained real adapter")
    latents=tmp_path/"latents.pt"
    latents.write_bytes(b"original generated H3 latents")
    movie=tmp_path/"rgb_audio.mp4"
    movie.write_bytes(b"video")
    report=tmp_path/"wan_h3_generation.json"
    report.write_text(json.dumps({
        "dit_lora_hash":sha256(adapter),
        "generated_latents_sha256":sha256(latents),
        "generated_latents":str(latents),
        "rgb_audio_mp4":str(movie),
    }))
    assert generation_is_current(report,adapter)
    latents.write_bytes(b"truncated result")
    assert not generation_is_current(report,adapter)
