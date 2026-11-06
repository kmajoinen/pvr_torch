"""MAE / MVP pretraining: the encoder/decoder split of the official MAE model,
the `mae` objective, the vit_* architectures and their checkpoints, video
frames as an image dataset, AdamW and MAE's learning-rate schedule, and the
presets."""

import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.losses import MAE, AuxBatch, aux_loss_classes
from src.vision_models.mae import MAEDecoder, MAEEncoder, MaskedAutoencoderViT

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
LN = partial(nn.LayerNorm, eps=1e-6)


def _small_pair():
    """A small official MAE model and the two halves with its weights."""
    torch.manual_seed(0)
    full = MaskedAutoencoderViT(patch_size=16, embed_dim=96, depth=2, num_heads=3, decoder_embed_dim=64,
                                decoder_depth=2, decoder_num_heads=4, mlp_ratio=4, norm_layer=LN,
                                norm_pix_loss=True)
    enc = MAEEncoder(patch_size=16, embed_dim=96, depth=2, num_heads=3, mlp_ratio=4, norm_layer=LN)
    dec = MAEDecoder(196, 16, embed_dim=96, decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
                     norm_layer=LN, norm_pix_loss=True)
    sd = full.state_dict()
    assert set(enc.state_dict()) | set(dec.state_dict()) == set(sd)
    assert not set(enc.state_dict()) & set(dec.state_dict())
    enc.load_state_dict({k: sd[k] for k in enc.state_dict()})
    dec.load_state_dict({k: sd[k] for k in dec.state_dict()})
    return full, enc, dec


def test_split_halves_compute_the_official_model():
    full, enc, dec = _small_pair()
    x = torch.randn(3, 3, 224, 224)
    torch.manual_seed(5)
    loss_a, pred_a, mask_a = full(x, mask_ratio=0.75)
    torch.manual_seed(5)
    latent, mask_b, ids = enc.forward_encoder(x, 0.75)
    pred_b = dec.forward_decoder(latent, ids)
    loss_b = dec.forward_loss(x, pred_b, mask_b)
    assert torch.equal(loss_a, loss_b) and torch.equal(mask_a, mask_b) and torch.equal(pred_a, pred_b)
    loss_a.backward()
    loss_b.backward()
    for name, p in list(enc.named_parameters()) + list(dec.named_parameters()):
        if p.grad is not None:
            assert torch.equal(p.grad, full.get_parameter(name).grad), name


def test_unmasked_forward_is_the_cls_token_in_patch_order():
    from src.embeddings import _forward_mae
    full, enc, _ = _small_pair()
    x = torch.randn(2, 3, 224, 224)
    state = torch.random.get_rng_state()
    out = enc(x)
    assert torch.equal(torch.random.get_rng_state(), state)        # no masking noise drawn
    assert out.shape == (2, 96) and torch.allclose(out, _forward_mae(full, x), atol=1e-6)


@pytest.mark.parametrize("name, width, heads", [("vit_s16", 384, 12), ("vit_b16", 768, 12)])
def test_vit_archs(name, width, heads):
    from src.embeddings import CHECKPOINT_ARCHS, EmbeddingNet
    enc = EmbeddingNet(name, pretrained=False, train=True, disable_cuda=True)
    assert name in CHECKPOINT_ARCHS and enc.out_size == width
    assert enc.embedding.blocks[0].attn.num_heads == heads
    with pytest.raises(ValueError, match="no default weights"):
        EmbeddingNet(name, pretrained=True, disable_cuda=True)


def test_vit_truncation_takes_block_counts():
    from src.embeddings import EmbeddingNet
    enc = EmbeddingNet("vit_s16", pretrained=False, disable_cuda=True, layer=6, head="cls")
    assert enc.layer == "blocks.5" and enc.encode_preprocessed(torch.rand(1, 3, 224, 224)).shape == (1, 384)


def test_mae_objective_is_the_official_loss_on_the_normalised_view():
    from src.embeddings import EmbeddingNet
    enc = EmbeddingNet("vit_s16", pretrained=False, train=True, disable_cuda=True)
    obj = MAE(enc, decoder_dim=64, decoder_depth=1, decoder_heads=4)
    full = MaskedAutoencoderViT(patch_size=16, embed_dim=384, depth=12, num_heads=12, decoder_embed_dim=64,
                                decoder_depth=1, decoder_num_heads=4, mlp_ratio=4, norm_layer=LN,
                                norm_pix_loss=True)
    full.load_state_dict({**enc.embedding.state_dict(), **obj.decoder.state_dict()})
    view = torch.rand(2, 3, 224, 224)
    torch.manual_seed(3)
    ours = obj(AuxBatch(encoder=enc, views=[view]))
    torch.manual_seed(3)
    official, _, _ = full(enc.normalize(view), mask_ratio=0.75)
    assert torch.allclose(ours, official)
    assert {n for n, _ in obj.named_parameters()} == {"decoder." + n for n, _ in obj.decoder.named_parameters()}


def test_mae_needs_a_whole_vit():
    from src.embeddings import EmbeddingNet
    with pytest.raises(ValueError, match="vit_s16"):
        MAE(EmbeddingNet("resnet18", pretrained=False, disable_cuda=True))
    with pytest.raises(ValueError, match="layer"):
        MAE(EmbeddingNet("vit_s16", pretrained=False, disable_cuda=True, layer=6, head="cls"))
    with pytest.raises(ValueError, match="mask_ratio"):
        aux_loss_classes([{"name": "mae", "weight": 1.0, "mask_ratio": 1.0}])


def test_vit_checkpoints_load_from_pretrain_snapshots_and_mae_files(tmp_path):
    from src.embeddings import EmbeddingNet
    from src.rl.common import encoder_record
    enc = EmbeddingNet("vit_s16", pretrained=False, train=True, disable_cuda=True)
    cfg = OmegaConf.create({"embedding": {"name": "vit_s16"}})
    torch.save(encoder_record(cfg, enc), tmp_path / "snapshot.pt")
    # A whole MAE model in the official format: the decoder keys are dropped.
    full = MaskedAutoencoderViT(patch_size=16, embed_dim=384, depth=12, num_heads=12, decoder_embed_dim=64,
                                decoder_depth=1, decoder_num_heads=4, mlp_ratio=4, norm_layer=LN)
    full.load_state_dict({**enc.embedding.state_dict(),
                          **MAEDecoder(196, 16, embed_dim=384, decoder_embed_dim=64, decoder_depth=1,
                                       decoder_num_heads=4, norm_layer=LN).state_dict()})
    torch.save({"model": full.state_dict()}, tmp_path / "mae.pth")
    x = torch.rand(2, 3, 224, 224)
    want = enc.eval().encode_preprocessed(x)
    for f in ("snapshot.pt", "mae.pth"):
        back = EmbeddingNet("custom", arch="vit_s16", disable_cuda=True, checkpoint=str(tmp_path / f))
        assert torch.allclose(back.eval().encode_preprocessed(x), want, atol=1e-6), f


def _frame_shards(root, videos=3, frames=13, keys="{v}/{i:06d}.jpg"):
    """Frame i of each video is a constant image of value 10 * i."""
    from torchvision.io import encode_jpeg
    from src.data.shards import ShardWriter
    with ShardWriter(os.path.join(root, "train")) as w:
        for v in range(videos):
            for i in range(1, frames + 1):
                img = torch.full((3, 32, 32), 10 * i, dtype=torch.uint8)
                w.add(keys.format(v=v, i=i), encode_jpeg(img, quality=100).numpy().tobytes())


def _values(ds):
    return [int(img.float().mean().round()) for img, _ in ds]


def test_video_frames_load_as_unlabelled_images_thinned_per_video(tmp_path):
    # 13 frames per video, every 4th: frames 1, 5, 9, 13 of each video. A
    # stride over all frames back to back would drift across videos instead.
    from src.data.images import ImageDataset
    _frame_shards(str(tmp_path))
    all_frames = ImageDataset(str(tmp_path), "train", train=False, size=32, resize=32)
    every4 = ImageDataset(str(tmp_path), "train", train=False, size=32, resize=32, every=4)
    assert len(all_frames) == 39 and len(every4) == 12 and not every4.labelled
    assert _values(every4) == [10, 50, 90, 130] * 3
    assert {label for _, label in every4} == {-1} and "key" not in every4.index
    assert every4.thinning == "every=4: per video" and all_frames.thinning is None
    # Every video keeps its first frame, however short.
    assert _values(ImageDataset(str(tmp_path), "train", train=False, size=32, resize=32, every=40)) == [10] * 3
    with pytest.raises(ValueError, match="bicubic"):
        ImageDataset(str(tmp_path), "train", train=True, interpolation="lanczos")


def test_other_images_thin_in_stored_order(tmp_path):
    from src.data.images import ImageDataset
    _frame_shards(str(tmp_path), keys="img_{v}_{i}.jpg")         # not <video>/<number>
    every4 = ImageDataset(str(tmp_path), "train", train=False, size=32, resize=32, every=4)
    assert len(every4) == 10 and _values(every4)[:5] == [10, 50, 90, 130, 40]
    assert every4.thinning == "every=4: stored order"


def test_adamw_skips_weight_decay_on_biases_and_norms():
    from pretrain import build_optimizer
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    cfg = OmegaConf.create({"optimizer": "adamw", "betas": [0.9, 0.95], "weight_decay": 0.05,
                            "no_weight_decay_1d": True})
    opt = build_optimizer(cfg, model.named_parameters(), 1e-3)
    assert isinstance(opt, torch.optim.AdamW) and opt.defaults["betas"] == (0.9, 0.95)
    no_decay, decay = opt.param_groups
    assert no_decay["weight_decay"] == 0.0 and len(no_decay["params"]) == 3      # bias, LN weight, LN bias
    assert decay["weight_decay"] == 0.05 and decay["params"] == [model[0].weight]
    cfg.no_weight_decay_1d = False
    assert [g["weight_decay"] for g in build_optimizer(cfg, model.named_parameters(), 1e-3).param_groups] == [0.05]


def test_warmup_cosine_is_maes_schedule():
    import math
    from pretrain import lr_factor
    s = OmegaConf.create({"name": "warmup_cosine", "warmup": 40, "total": 800})
    mae = lambda e: e / 40 if e < 40 else 0.5 * (1 + math.cos(math.pi * (e - 40) / 760))
    for step in (0, 50, 4000, 4050, 40_000, 79_999, 80_000):
        assert lr_factor(s, step, steps_per_epoch=100) == pytest.approx(mae(step / 100))
    assert lr_factor(s, 0, 100) == 0.0 and lr_factor(s, 4000, 100) == 1.0


@pytest.mark.parametrize("preset, arch, data", [("mae", "vit_b16", "imagenet"), ("mvp", "vit_s16", "ssv2_frames")])
def test_presets(preset, arch, data):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_pretrain", overrides=[f"+experiment={preset}", "pretrain_data.root=/x"])
    assert cfg.embedding.name == arch and cfg.embedding.pretrained is False and cfg.pretrain_data.name == data
    assert cfg.pretrain_data.type == "images" and cfg.pretrain_data.interpolation == "bicubic"
    assert list(cfg.pretrain_data.train_crop.scale) == ([0.2, 1.0] if preset == "mae" else [0.1, 0.75])
    assert cfg.augmentation.name == "horizontal_flip"
    assert cfg.optimizer == "adamw" and list(cfg.betas) == [0.9, 0.95] and cfg.weight_decay == 0.05
    assert cfg.no_weight_decay_1d and cfg.lr == 1.5e-4 and cfg.lr_reference_batch == 256
    assert cfg.lr_schedule.name == "warmup_cosine" and cfg.objectives[0].name == "mae"
    assert cfg.objectives[0].mask_ratio == 0.75 and cfg.objectives[0].norm_pix_loss
    assert cfg.lr_schedule.warmup == 40 and cfg.epochs == (800 if preset == "mae" else 1600)
    if preset == "mvp":
        assert cfg.pretrain_data.every == 40


def test_train_images_stretches_epochs_and_schedule():
    from pretrain import scale_to_images
    sched = {"name": "warmup_cosine", "warmup": 40, "total": None, "milestones": [30, 60]}
    # MVP's budget over a 200K-frame set: 1600 x 700K images.
    epochs, s = scale_to_images(1600, sched, 200_000, 1600 * 700_000)
    assert epochs == pytest.approx(5600) and s["warmup"] == pytest.approx(140)     # still 2.5%
    assert s["milestones"] == pytest.approx([105, 210]) and s["total"] is None
    assert sched["warmup"] == 40                                                   # input untouched


def test_mvp_budget_toggle_composes():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_pretrain",
                      overrides=["+experiment=mvp", "pretrain_data.root=/x", "train_images=1.12e9"])
    assert cfg.epochs == 1600 and cfg.train_images == pytest.approx(1600 * 700_000)
