"""Layer truncation and heads (src/truncation.py) through EmbeddingNet."""

import os

import pytest
import torch

from src.embeddings import EmbeddingNet
from src.truncation import resolve_layer
from src.vision_models.rl3d import rl3d_resnet18_backbone


def _u8(n=2, size=84):
    torch.manual_seed(0)
    return torch.randint(0, 256, (n, 3, size, size), dtype=torch.uint8)


def _net(**kw):
    torch.manual_seed(0)
    return EmbeddingNet("resnet18", pretrained=False, disable_cuda=True, **kw)


def test_layer4_avgpool_equals_the_full_model():
    full, cut = _net(), _net(layer="layer4", head="avgpool")
    x = _u8()
    assert torch.allclose(full.encode(x), cut.encode(x), atol=1e-6)


@pytest.mark.parametrize("layer, dim", [("layer1", 64), ("layer2", 128), ("layer3", 256)])
def test_stage_dims_and_early_exit(layer, dim):
    e = _net(layer=layer)
    ran = []
    e.embedding.layer4.register_forward_hook(lambda *a: ran.append(1))
    assert e.encode(_u8()).shape == (2, dim)
    assert not ran   # later stages never execute


def test_linear_head_trains_on_a_frozen_backbone():
    e = _net(layer="layer2", head="linear", head_dim=32)
    e.set_trainable("head")
    e.encode(_u8()).sum().backward()
    assert e.head.fc.weight.grad is not None
    assert all(p.grad is None for p in e.embedding.parameters())


def test_rl3d_numbered_sequential_alias():
    assert resolve_layer(rl3d_resnet18_backbone(), "layer2")[0] == "5"


@pytest.mark.parametrize("kw, match", [
    ({"layer": "layer9"}, "doesn't match"),
    ({"layer": 3}, "block list"),
    ({"layer": "layer2", "head": "cls"}, "tokens"),
    ({"head": "avgpool"}, "already a vector"),
])
def test_errors(kw, match):
    with pytest.raises(ValueError, match=match):
        _net(**kw)


# ---- frame stacking ---------------------------------------------------------

def _stacked(k=3, n=2, size=84):
    torch.manual_seed(1)
    return torch.randint(0, 256, (n, 3 * k, size, size), dtype=torch.uint8)


@pytest.mark.parametrize("fusion", ["concat", "diff"])
def test_frame_fusion_matches_per_frame_encoding(fusion):
    common = dict(layer="layer2", head="flatten", input_size=84,
                  preprocess={"mean": [0.5] * 3, "std": [1.0] * 3})
    e = _net(frame_stack=3, frame_fusion=fusion, **common)
    single = _net(**common)
    x = _stacked()
    with torch.no_grad():
        maps = torch.stack([single._raw_output(single.preprocess(x[:, 3 * i:3 * i + 3]))
                            for i in range(3)], 1)                    # (N, 3, 128, 11, 11)
        if fusion == "diff":                                          # PIE-G
            current = maps[:, 1:]
            maps = torch.cat([current, current - maps[:, :2]], 1)
        assert torch.allclose(e.encode(x), maps.reshape(2, -1), atol=1e-5)
    # PIE-G's ResEncoder out_dim at 84px with 3 frames: 4 * 128 * 11 * 11
    assert _net(frame_stack=3, frame_fusion="diff", **common).out_size == 61952


def test_stacked_augmentation_moves_every_frame_alike():
    from src.augmentations import make_augmentation
    e = _net(frame_stack=3)
    frame = _stacked(k=1)
    x = frame.repeat(1, 3, 1, 1)                                      # three identical frames
    torch.manual_seed(0)
    y = make_augmentation("random_shift")(e.preprocess(x))
    assert torch.equal(y[:, 0:3], y[:, 3:6]) and torch.equal(y[:, 0:3], y[:, 6:9])


def test_random_keeps_channel_stacking_by_default():
    e = EmbeddingNet("random", pretrained=False, disable_cuda=True, frame_stack=3)
    assert e.frame_fusion == "channels" and e.encode(_stacked()).shape == (2, e.out_size)


def test_dinov2_style_tokens_fuse_along_features():
    from src.truncation import fuse_frames
    tokens = torch.randn(2 * 3, 5, 4)                                 # (N*k, T, D)
    assert fuse_frames(tokens, 2, 3, "concat").shape == (2, 5, 12)
    assert fuse_frames(tokens, 2, 3, "diff").shape == (2, 5, 16)


# ---- BatchNorm adaptation ----------------------------------------------------

def test_bn_train_mode_updates_running_stats_only_in_updates():
    e = _net()
    e.eval()
    bn = e.embedding.bn1
    before = bn.running_mean.clone()
    e.set_bn_train(True)
    assert bn.training and not e.embedding.training     # only BatchNorm switches
    with torch.no_grad():
        e.encode(_u8())
    assert not torch.equal(bn.running_mean, before)
    e.set_bn_train(False)
    after = bn.running_mean.clone()
    with torch.no_grad():
        e.encode(_u8())
    assert torch.equal(bn.running_mean, after)
    assert e.has_batchnorm()


# ---- ViTs (download weights: set PVR_TEST_NETWORK=1 to run) -------------------

network = pytest.mark.skipif(os.environ.get("PVR_TEST_NETWORK") != "1",
                             reason="needs network (weight downloads)")


def _raw_tokens(e, x):
    """The truncated encoder's raw output for uint8 frames x."""
    with torch.no_grad():
        return e._forward(e._normalize(e.preprocess(x)))


@network
def test_dinov2_last_block_matches_model_output():
    e = EmbeddingNet("dinov2_vits14", disable_cuda=True, layer=12, head="cls")
    full = EmbeddingNet("dinov2_vits14", disable_cuda=True)
    x = _u8()
    tokens = _raw_tokens(e, x)                       # (N, 1 + patches, 384)
    with torch.no_grad():
        assert torch.allclose(e.embedding.norm(tokens)[:, 0], full.encode(x), atol=1e-5)
    assert e.encode(x).shape == (2, 384)
    assert EmbeddingNet("dinov2_vits14", disable_cuda=True, layer=6).encode(x).shape == (2, 384)


@network
def test_clip_vit_uses_visual_blocks_and_sequence_first_tokens():
    e = EmbeddingNet("clip_vit", disable_cuda=True, layer=12, head="cls")
    assert e.layer == "visual.transformer.resblocks.11"   # not the text tower's
    full = EmbeddingNet("clip_vit", disable_cuda=True)
    x = _u8()
    tokens = _raw_tokens(e, x)
    assert tokens.shape[1] == 2                      # (T, N, D): sequence first
    v = e.embedding.visual
    with torch.no_grad():
        assert torch.allclose(v.ln_post(tokens[0]) @ v.proj, full.encode(x), atol=1e-4)


@network
def test_clip_rn50_stage():
    e = EmbeddingNet("clip_rn50", disable_cuda=True, layer="layer4")
    assert e.layer == "visual.layer4"
    assert e.encode(_u8()).shape == (2, 2048)
