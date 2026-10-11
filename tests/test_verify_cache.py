"""Offline tests for official H3 VAE shard verification. Sparse dummy files only."""
import json
from pathlib import Path

from tools.verify_h3_cache import check_snapshot, SHARDS, INDEX


def test_official_vae_shard_layout(tmp_path):
    vae = tmp_path / "vae"
    vae.mkdir()
    (vae / "config.json").write_text('{"_class_name":"AutoencoderKLMiniMaxH3"}')
    for shard in SHARDS:
        # Sparse file, uses negligible actual disk blocks.
        with (vae / shard).open("wb") as f:
            f.truncate(55_000_000)
    (vae / INDEX).write_text(json.dumps({
        "metadata": {"total_size": 165_000_000},
        "weight_map": {
            "encoder.conv_in.weight": SHARDS[1],
            "decoder.proj_out.weight": SHARDS[0],
        },
    }))
    assert check_snapshot(tmp_path)
    (vae / SHARDS[0]).unlink()
    assert not check_snapshot(tmp_path)
