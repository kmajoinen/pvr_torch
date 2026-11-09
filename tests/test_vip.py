"""VIP objective (src/losses.py): frame sampling, the loss, the preset, the fc
head and VIP's final crop, and parity with the official vip/trainer.py
(skipped unless the vip package is installed -- without its dependencies,
whose pins (hydra-core 1.1.1, omegaconf 2.1.1, pillow 9.0.1, gdown 4.4.0)
would break this repo's environment:
    pip install gdown
    pip install --no-deps git+https://github.com/facebookresearch/vip.git)."""

import copy
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.losses import VIP, AuxBatch, VIPFrameSampler, aux_loss_classes

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
D = 16


class _Enc(nn.Module):
    out_size = D

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 16, stride=16), nn.Flatten(), nn.Linear(4 * 14 * 14, D))

    def forward(self, x):
        return self.net(x / 255.0)


def _official_draws(length, rng):
    """VIPBuffer._sample's index formulas (frames 0..length-1), on the same
    kind of generator."""
    start = rng.integers(0, length - 2)
    end = rng.integers(start + 1, length)
    s0 = rng.integers(start, end)
    return [start, end, s0, min(s0 + 1, end)]


def test_frame_sampler_is_the_official_draws_on_frames_numbered_from_1():
    s = VIPFrameSampler()
    for length in range(s.min_length, 40):
        a, b = np.random.default_rng(length), np.random.default_rng(length)
        for _ in range(50):
            frames = s(length, a)
            assert frames == [i + 1 for i in _official_draws(length, b)]
            start, goal, s0, s1 = frames
            assert 1 <= start <= length - 2 and start < goal <= length
            assert start <= s0 < goal and s1 == s0 + 1


def test_loss_without_negatives_is_the_formula():
    torch.manual_seed(0)
    feats = torch.randn(5, 4, D)
    loss, m = VIP(_Enc(), gamma=0.9, l1_weight=0.1, l2_weight=0.2)(AuxBatch(encoder=None, clip_feats=feats))
    e0, eg, s0, s1 = feats.unbind(1)
    v = lambda a: -torch.linalg.norm(a - eg, dim=-1)
    vip = 0.1 * -v(e0).mean() + torch.log(1e-8 + torch.exp(-(-1 + 0.9 * v(s1) - v(s0))).mean())
    flat = feats.reshape(20, D)
    expected = 0.2 * flat.norm(dim=-1).mean() + 0.1 * flat.abs().sum(-1).mean() + vip
    assert torch.allclose(loss, expected) and torch.allclose(m["vip_loss"], vip)
    assert set(m) == {"vip_loss", "l1loss", "l2loss"}


def test_negatives_add_a_term_and_draw_one_permutation_each():
    torch.manual_seed(0)
    feats = torch.randn(6, 4, D)
    base, _ = VIP(_Enc())(AuxBatch(encoder=None, clip_feats=feats))
    calls = []
    real = torch.randperm
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch, "randperm", lambda n, **k: calls.append(n) or real(n, **k))
        neg, _ = VIP(_Enc(), num_negatives=3)(AuxBatch(encoder=None, clip_feats=feats))
    assert calls == [6, 6, 6] and not torch.allclose(base, neg)


def test_bad_params_fail_at_validation():
    for bad in ({"gamma": 1.0}, {"l1_weight": -1}, {"num_negatives": 1.5}):
        with pytest.raises(ValueError):
            aux_loss_classes([{"name": "vip", "weight": 1.0, **bad}])


@pytest.mark.parametrize("overrides", [["+experiment=vip"], ["+experiment=vip", "embedding=resnet18"],
                                       ["embedding=resnet18", "+experiment=vip"]])
def test_vip_preset(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_pretrain", overrides=overrides + ["pretrain_data.root=/x"])
    assert cfg.embedding.pretrained is False and cfg.embedding.head == "fc" and cfg.embedding.head_dim == 1024
    assert cfg.embedding.name == ("resnet18" if "embedding=resnet18" in overrides else "resnet50")
    assert cfg.objectives[0].name == "vip" and cfg.objectives[0].num_negatives == 3
    assert cfg.batch_size == 32 and cfg.lr == 1e-4 and cfg.train_steps == 2_000_000
    assert cfg.augmentation.name == "random_resized_crop" and cfg.clip_augmentation == "shared"
    assert OmegaConf.to_container(cfg.final_crop) == {"resize": 256, "crop": 224}


def test_final_crop_is_the_official_resize_and_center_crop():
    from torchvision import transforms
    from pretrain import final_crop
    x = torch.rand(3, 3, 224, 224)
    official = nn.Sequential(transforms.Resize(256), transforms.CenterCrop(224))(x)
    ours = final_crop(x, OmegaConf.create({"resize": 256, "crop": 224}))
    assert ours.shape == (3, 3, 224, 224) and torch.allclose(ours, official, atol=1e-5)


def test_fc_head_is_a_plain_linear_that_saves_and_reloads(tmp_path):
    from src.embeddings import EmbeddingNet
    from src.rl.common import encoder_record
    enc = EmbeddingNet("resnet18", pretrained=False, train=True, disable_cuda=True, head="fc", head_dim=32)
    assert enc.out_size == 32 and [type(m) for m in enc.head.modules()][1:] == [nn.Linear]
    cfg = OmegaConf.create({"embedding": {"name": "resnet18", "head": "fc", "head_dim": 32}})
    torch.save(encoder_record(cfg, enc), tmp_path / "enc.pt")
    back = EmbeddingNet("resnet18", disable_cuda=True, head="fc", head_dim=32,
                        checkpoint=str(tmp_path / "enc.pt"))
    x = torch.rand(2, 3, 224, 224)
    enc.eval(), back.eval()
    assert torch.allclose(enc.encode_preprocessed(x), back.encode_preprocessed(x)) and not back.build_warnings


def test_a_single_frame_fc_checkpoint_encodes_each_stacked_frame(tmp_path):
    from src.embeddings import EmbeddingNet
    from src.rl.common import encoder_record
    enc = EmbeddingNet("resnet18", pretrained=False, train=True, disable_cuda=True, head="fc", head_dim=32)
    cfg = OmegaConf.create({"embedding": {"name": "resnet18", "head": "fc", "head_dim": 32}})
    torch.save(encoder_record(cfg, enc), tmp_path / "enc.pt")
    stacked = EmbeddingNet("resnet18", disable_cuda=True, head="fc", head_dim=32, frame_stack=3,
                           checkpoint=str(tmp_path / "enc.pt"))
    assert stacked.out_size == 3 * 32 and not stacked.build_warnings
    x = torch.rand(2, 9, 224, 224)
    enc.eval(), stacked.eval()
    with torch.no_grad():
        per_frame = torch.cat([enc.encode_preprocessed(x[:, 3 * i:3 * i + 3]) for i in range(3)], 1)
        assert torch.allclose(stacked.encode_preprocessed(x), per_frame, atol=1e-5)


@pytest.mark.parametrize("checkpoint, ok", [(None, False), ("/x/enc.pt", True)])
def test_a_frozen_fc_head_needs_trained_weights(checkpoint, ok):
    from src.encoder_config import encoder_settings
    over = ["embedding=resnet50", "+embedding.head=fc"]
    if checkpoint:
        over.append(f"+embedding.checkpoint={checkpoint}")
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_sac", overrides=over)
    if ok:
        encoder_settings(cfg, valid_grads=("none", "critic", "actor", "both"))
    else:
        with pytest.raises(ValueError, match="trainable"):
            encoder_settings(cfg, valid_grads=("none", "critic", "actor", "both"))


def test_rl_scripts_reject_vip():
    from src.encoder_config import encoder_settings
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_sac", overrides=["encoder.aux=[{name:vip,weight:1.0}]"])
    with pytest.raises(ValueError, match="pretrain.py"):
        encoder_settings(cfg, valid_grads=("none", "critic", "actor", "both"))


@pytest.mark.parametrize("num_negatives", [0, 3])
def test_parity_with_official_trainer(num_negatives, monkeypatch):
    trainer = pytest.importorskip("vip.trainer")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self, raising=False)
    B = 6
    torch.manual_seed(0)
    enc_a = _Enc()
    enc_b = copy.deepcopy(enc_a)
    ours = VIP(enc_b, gamma=0.98, l1_weight=1e-3, l2_weight=1e-3, num_negatives=num_negatives).train()
    mod = types.SimpleNamespace(l2weight=1e-3, l1weight=1e-3, gamma=0.98, num_negatives=num_negatives)
    mod.sim = lambda a, b: -torch.linalg.norm(a - b, dim=-1)
    mod.encoder_opt = torch.optim.Adam(enc_a.parameters(), lr=1e-3)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.module = enc_a, mod

        def forward(self, x):
            return self.enc(x)

    torch.manual_seed(1)
    ims = torch.rand(B, 4, 3, 224, 224) * 255
    # float32: the official loader collates float64 rewards, which promote its
    # TD term to double precision. The losses still agree to ~1e-7, but Adam's
    # first step (about lr * sign(grad)) flips on near-zero gradients, so the
    # weights after it would differ by up to ~lr. In float32 both match exactly.
    rewards = torch.full((B,), -1.0)
    torch.manual_seed(2)
    official, _ = trainer.Trainer(1).update(Model(), (ims, rewards), 0)

    opt = torch.optim.Adam(enc_b.parameters(), lr=1e-3)
    torch.manual_seed(2)
    feats = enc_b(ims.reshape(B * 4, 3, 224, 224)).reshape(B, 4, D)
    loss, m = ours(AuxBatch(encoder=enc_b, clip_feats=feats))
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert official["full_loss"] == pytest.approx(loss.item(), rel=1e-6, abs=1e-6)
    for k in ("vip_loss", "l1loss", "l2loss"):
        assert official[k] == pytest.approx(float(m[k]), rel=1e-6, abs=1e-6), k
    for a, b in zip(enc_a.parameters(), enc_b.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_a_frozen_trainable_head_must_be_loaded_from_the_checkpoint(tmp_path):
    from src.embeddings import EmbeddingNet
    from src.encoder_config import check_head_is_trained
    from src.rl.common import encoder_record
    enc = EmbeddingNet("resnet18", pretrained=False, train=True, disable_cuda=True, head="fc", head_dim=32)
    cfg = OmegaConf.create({"embedding": {"name": "resnet18", "head": "fc", "head_dim": 32}})
    torch.save(encoder_record(cfg, enc), tmp_path / "with_head.pt")
    torch.save(enc.embedding.state_dict(), tmp_path / "backbone.pt")       # e.g. an official snapshot
    kw = dict(disable_cuda=True, head="fc", head_dim=32)
    with_head = EmbeddingNet("resnet18", checkpoint=str(tmp_path / "with_head.pt"), **kw)
    backbone = EmbeddingNet("resnet18", checkpoint=str(tmp_path / "backbone.pt"), **kw)
    assert with_head.head_loaded and not backbone.head_loaded
    check_head_is_trained(with_head, trains=False)
    check_head_is_trained(backbone, trains=True)            # it trains: fine
    with pytest.raises(ValueError, match="random"):
        check_head_is_trained(backbone, trains=False)
    plain = EmbeddingNet("resnet18", checkpoint=str(tmp_path / "backbone.pt"), disable_cuda=True)
    check_head_is_trained(plain, trains=False)              # no trainable head: nothing to load
