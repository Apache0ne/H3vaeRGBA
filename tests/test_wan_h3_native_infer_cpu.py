"""CPU regressions for native one-latent H3 Wan RGBA decoder and DiT hooks."""
from types import SimpleNamespace

import pytest
import torch

from benchmark_wan_dual import tiny_model
from h3rgba.wan_joint_h3 import WanJointH3Codec
from tools.import_wan_alpha_captions import match_captions
import tools.generate_h3_wan_joint as gen


def test_full_h3_native_vae_decode_both_rgb_and_alpha_from_one_tensor():
    torch.manual_seed(23)
    codec=WanJointH3Codec(tiny_model(),hidden=16,
                          train_decoder_lora=False)
    codec.eval()
    rgba=torch.rand(1,4,17,32,32)
    with torch.no_grad():
        z,_,_=codec.encode_joint(rgba)
        rgb,alpha=codec.decode_native_h3_latents(z)
    assert rgb.shape[1]==3 and alpha.shape[1]==1
    assert rgb.shape[0]==alpha.shape[0]==1
    assert rgb.shape[2:]==alpha.shape[2:]
    assert rgb.shape[-2:]==(32,32)
    assert torch.isfinite(alpha).all()
    with pytest.raises(ValueError):
        codec.decode_native_h3_latents(torch.zeros(1,48,7,2,2))


def test_import_official_caption_csv_only_exact_filename_stems():
    rows=[{"id":"videomatte240k_train_0099_3",
           "foreground":"/tmp/fgr/VideoMatte240K_train_0099_3.mp4"},
          {"id":"unknown_clip","foreground":"/tmp/x/no_match.mp4"}]
    csv="video_fgr,video_pha,prompt\n" + (
        "fgr/VideoMatte240K_train_0099_3.mp4,"
        "pha/VideoMatte240K_train_0099_3.mp4,"
        "A girl on a transparent background\n"
    )
    mapped,missing,duplicates=match_captions(rows,csv)
    assert mapped=={"videomatte240k_train_0099_3":
                   "A girl on a transparent background"}
    assert missing==["unknown_clip"]
    assert duplicates==0


def test_dit_lora_injection_hook_requires_real_trained_adapter(monkeypatch,tmp_path):
    saved=tmp_path/"adapter.safetensors"
    saved.write_bytes(b"mock_checkpoint")
    fake_model=SimpleNamespace()
    called=[]
    def mock_inject(model,adapter,rank):
        assert model is fake_model
        assert adapter==str(saved)
        called.append(rank)
    monkeypatch.setattr(gen,"inject_trained_dit",mock_inject)
    class FakeRunner:
        def __init__(self):
            self._pipe=SimpleNamespace(transformer=fake_model)
        def _ensure_transformer(self,*a,**k):
            return None
    class FakeDecoder:
        def get_block_state(self,state):
            return SimpleNamespace(latents=torch.ones(1,24,7,2,2))
        def __call__(self,components,state):
            return components,state
    fake_module=SimpleNamespace(
        MiniMaxH3Runner=FakeRunner,
        _cpu_norm_video_decode_step=lambda:FakeDecoder(),
    )
    dest=tmp_path/"captured.pt"
    gen.install_upstream_hooks(fake_module,str(saved),8,dest)
    inst=FakeRunner()
    inst._ensure_transformer()
    inst._ensure_transformer()
    assert called==[8]
    step=fake_module._cpu_norm_video_decode_step()
    step(None,None)
    tensors=torch.load(dest,map_location="cpu",weights_only=True)
    assert tensors["latents"].shape==(1,24,7,2,2)
    assert tensors["source"]=="actual_h3_denoiser_with_trained_wan_joint_dit_lora"
