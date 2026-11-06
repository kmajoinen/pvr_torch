"""EmbeddingNet's preprocess/encode split: preprocess() must return [0, 1]
floats at the encoder's input resolution for every pipeline shape in
src/embeddings.py, including R3M/VIP's [0, 255] convention."""

import os

import pytest
import torch
import torch.nn as nn
import torchvision.transforms as T
from omegaconf import OmegaConf

from src.embeddings import EmbeddingNet, _ScaleTo255, _split_transforms, preprocess_transforms
from src.encoder_config import embedding_kwargs

IMAGENET = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _u8(n=2, size=84):
    torch.manual_seed(0)
    return torch.randint(0, 256, (n, 3, size, size), dtype=torch.uint8)


@pytest.mark.parametrize("name, steps, post_types", [
    ("imagenet", [T.Resize(64), T.CenterCrop(56), T.ConvertImageDtype(torch.float), IMAGENET], [T.Normalize]),
    ("r3m_vip", [T.Resize(64), T.CenterCrop(56), T.ConvertImageDtype(torch.float), _ScaleTo255()], [_ScaleTo255]),
    ("plain", [T.Resize(64), T.CenterCrop(56), T.ConvertImageDtype(torch.float)], []),
])
def test_pre_aug_output_is_unit_range_and_composition_is_unchanged(name, steps, post_types):
    full = nn.Sequential(*steps)
    pre, post = _split_transforms(full)
    x = _u8()
    mid = pre(x)
    assert mid.is_floating_point() and mid.shape[-1] == 56
    assert 0.0 <= mid.min() and mid.max() <= 1.0
    assert torch.allclose(post(mid), full(x))
    assert [type(t) for t in (post if isinstance(post, nn.Sequential) else [])] == post_types


def test_no_dtype_conversion_falls_back_to_trailing_normalize():
    # maskrcnn_l3 shape: stays in [0, 255]; augmentation's range check rejects it.
    pre, post = _split_transforms(nn.Sequential(T.Resize(64), T.CenterCrop(56), IMAGENET))
    assert [type(t) for t in post] == [T.Normalize]
    assert len(list(pre.children())) == 2


def test_spatial_step_after_conversion_is_rejected():
    bad = nn.Sequential(T.ConvertImageDtype(torch.float), T.Resize(64), IMAGENET)
    with pytest.raises(RuntimeError, match="spatial step"):
        _split_transforms(bad)


def test_embeddingnet_split_on_a_real_encoder():
    e = EmbeddingNet("random", pretrained=False, disable_cuda=True)
    x = _u8()
    p = e.preprocess(x)
    assert 0.0 <= p.min() and p.max() <= 1.0
    assert torch.allclose(e.encode(x), e.encode_preprocessed(p))


# ---- preprocess override ------------------------------------------------------

PIEG = {"resize": None, "crop": None, "mean": [0.5, 0.5, 0.5], "std": [1.0, 1.0, 1.0]}


def test_preprocess_spec_pieg_style():
    pre, post = _split_transforms(preprocess_transforms(PIEG))
    x = _u8(size=84)
    mid = pre(x)
    assert mid.shape[-1] == 84 and 0.0 <= mid.min() and mid.max() <= 1.0
    assert torch.allclose(post(mid), x.float() / 255 - 0.5)


def test_preprocess_spec_resize_crop():
    t = preprocess_transforms({"resize": 128, "crop": 112, "interpolation": "bicubic"})
    assert t(_u8(size=84)).shape[-2:] == (112, 112)


@pytest.mark.parametrize("spec, match", [
    ({"resize": 128, "foo": 1}, "unknown preprocess keys"),
    ({"mean": [0.5, 0.5, 0.5]}, "both mean and std"),
    ({"mean": [0.5], "std": [1.0, 1.0]}, "lengths differ"),
    ({"interpolation": "bicubic"}, "no effect without"),
    ({"resize": 128, "interpolation": "lanczos3"}, "interpolation"),
])
def test_preprocess_spec_errors(spec, match):
    with pytest.raises(ValueError, match=match):
        preprocess_transforms(spec)


def test_embeddingnet_preprocess_override_changes_input_resolution():
    e = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True, preprocess=PIEG)
    assert e.preprocess(_u8(size=84)).shape[-1] == 84
    assert e.encode(_u8(size=84)).shape == (2, 512)   # pooled: size-independent


def test_input_size_matters_when_output_depends_on_it():
    # random's conv stack has no pooling, so without a resize its output size
    # follows the frame size.
    no_resize = {"resize": None, "crop": None}
    ok = EmbeddingNet("random", pretrained=False, disable_cuda=True,
                      preprocess=no_resize, input_size=84)
    assert ok.encode(_u8(size=84)).shape == (2, ok.out_size)
    wrong = EmbeddingNet("random", pretrained=False, disable_cuda=True, preprocess=no_resize)
    with pytest.raises(RuntimeError, match="input_size"):
        wrong.encode(_u8(size=84))


# ---- arch + checkpoint ---------------------------------------------------------

def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    src = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True)
    path = tmp_path / "weights.pt"
    torch.save({"state_dict": src.embedding.state_dict()}, path)   # wrapper is unwrapped
    loaded = EmbeddingNet("my_run", arch="resnet18", checkpoint=str(path), disable_cuda=True)
    assert loaded.embedding_name == "my_run" and loaded.arch == "resnet18"
    x = _u8()
    assert torch.allclose(src.encode(x), loaded.encode(x))


def test_official_r3m_snapshot_loads_its_image_encoder(tmp_path):
    # The official R3M code saves {"r3m": DataParallel state_dict}: the image
    # encoder under module.convnet.*, plus language parts that are dropped.
    torch.manual_seed(0)
    src = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True)
    sd = {f"module.convnet.{k}": v for k, v in src.embedding.state_dict().items()}
    sd["module.lang_rew.pred.0.weight"] = torch.zeros(3, 3)
    sd["module.lang_enc.model.embeddings.word_embeddings.weight"] = torch.zeros(3, 3)
    torch.save({"r3m": sd, "global_step": 10}, tmp_path / "snapshot.pt")
    loaded = EmbeddingNet("r3m_run", arch="resnet18", checkpoint=str(tmp_path / "snapshot.pt"),
                          disable_cuda=True)
    x = _u8()
    assert torch.allclose(src.encode(x), loaded.encode(x))


def test_checkpoint_of_the_wrong_architecture_is_rejected(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save(EmbeddingNet("random", pretrained=False, disable_cuda=True).embedding.state_dict(), path)
    with pytest.raises(RuntimeError, match="zero parameters matched"):
        EmbeddingNet("my_run", arch="resnet18", checkpoint=str(path), disable_cuda=True)


def test_partial_checkpoint_is_rejected(tmp_path):
    state = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True).embedding.state_dict()
    state.pop("layer4.1.conv2.weight")
    path = tmp_path / "weights.pt"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="missing keys"):
        EmbeddingNet("my_run", arch="resnet18", checkpoint=str(path), disable_cuda=True)


def _trained_net():
    torch.manual_seed(0)
    e = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True,
                     layer="layer2", head="linear", head_dim=16)
    with torch.no_grad():   # move off the init, so a fresh head would differ
        e.head.fc.weight.add_(0.1)
        e.embedding.conv1.weight.add_(0.1)
    return e


def test_repo_saved_encoder_round_trip_keeps_the_head(tmp_path):
    src = _trained_net()
    torch.save(src.state_dict(), tmp_path / "best_encoder.pt")
    torch.save({"actor": {}, "encoder": src.state_dict()}, tmp_path / "final_checkpoint.pt")
    x = _u8()
    for f in ("best_encoder.pt", "final_checkpoint.pt"):
        loaded = EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / f), disable_cuda=True,
                              layer="layer2", head="linear", head_dim=16)
        assert torch.allclose(src.encode(x), loaded.encode(x))


def test_repo_saved_encoder_with_a_different_head_is_rejected(tmp_path):
    torch.save(_trained_net().state_dict(), tmp_path / "w.pt")
    with pytest.raises(RuntimeError, match="head mismatch"):
        EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / "w.pt"), disable_cuda=True,
                     layer="layer2", head="avgpool")


def test_repo_saved_backbone_gets_a_fresh_head(tmp_path):
    src = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True)   # no head params
    torch.save(src.state_dict(), tmp_path / "w.pt")
    loaded = EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / "w.pt"), disable_cuda=True,
                          layer="layer2", head="linear", head_dim=16)
    assert torch.equal(loaded.embedding.conv1.weight, src.embedding.conv1.weight)


def test_stock_torchvision_state_dict_loads(tmp_path):
    import torchvision.models as models
    torch.manual_seed(0)
    tv = models.resnet18(weights=None)            # includes fc.*, which EmbeddingNet drops
    torch.save(tv.state_dict(), tmp_path / "w.pt")
    loaded = EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / "w.pt"), disable_cuda=True)
    assert torch.equal(loaded.embedding.conv1.weight, tv.conv1.weight)


def test_saved_build_record_flags_a_different_layer_or_preprocess(tmp_path, capsys):
    src = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True, layer="layer2")
    torch.save({"encoder": src.state_dict(),
                "embedding": {"name": "resnet18", "layer": "layer2", "head": None}},
               tmp_path / "best_encoder.pt")
    same = EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / "best_encoder.pt"),
                        disable_cuda=True, layer="layer2")
    assert "built differently" not in capsys.readouterr().out
    EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / "best_encoder.pt"),
                 disable_cuda=True, layer="layer3", preprocess=PIEG)
    out = capsys.readouterr().out
    assert "built differently" in out and "layer: saved 'layer2', now 'layer3'" in out
    assert "preprocess" in out
    # final_checkpoint.pt records the whole config; its embedding entry is used.
    torch.save({"actor": {}, "encoder": src.state_dict(),
                "config": {"embedding": {"name": "resnet18", "layer": "layer1"}}},
               tmp_path / "final_checkpoint.pt")
    EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / "final_checkpoint.pt"),
                 disable_cuda=True, layer="layer2")
    assert "layer: saved 'layer1'" in capsys.readouterr().out
    assert same.build_spec["head"] == "avgpool"


def test_equivalent_preprocess_specs_are_not_flagged(tmp_path, capsys):
    src = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True,
                       preprocess={"mean": [0.5] * 3, "std": [1] * 3})
    torch.save({"encoder": src.state_dict(),
                "embedding": {"name": "resnet18", "preprocess": {"mean": [0.5] * 3, "std": [1] * 3}},
                "input_size": 64}, tmp_path / "w.pt")
    spelled_out = {"resize": None, "crop": None, "interpolation": None,
                   "mean": [0.5, 0.5, 0.5], "std": [1.0, 1.0, 1.0]}
    e = EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / "w.pt"),
                     disable_cuda=True, preprocess=spelled_out, input_size=64)
    assert e.build_warnings == [] and "built differently" not in capsys.readouterr().out


def test_frame_size_is_compared_only_without_a_resize(tmp_path):
    src = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True, preprocess=PIEG, input_size=84)
    torch.save({"encoder": src.state_dict(), "embedding": {"name": "resnet18", "preprocess": PIEG},
                "input_size": 84}, tmp_path / "w.pt")
    e = EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / "w.pt"),
                     disable_cuda=True, preprocess=PIEG, input_size=64)
    assert any("frame size" in w for w in e.build_warnings)
    # With a resize, the frame size doesn't change what the encoder sees.
    torch.save({"encoder": _net_state(), "embedding": {"name": "resnet18"}, "input_size": 84},
               tmp_path / "w2.pt")
    e2 = EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / "w2.pt"),
                      disable_cuda=True, input_size=64)
    assert e2.build_warnings == []


def _net_state():
    return EmbeddingNet("resnet18", pretrained=False, disable_cuda=True).state_dict()


def test_frozen_runs_final_checkpoint_is_rejected_clearly(tmp_path):
    torch.save({"actor": {}, "qf1": {}, "config": {"embedding": {"name": "resnet18"}}},
               tmp_path / "final_checkpoint.pt")
    with pytest.raises(ValueError, match="encoder was frozen"):
        EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / "final_checkpoint.pt"),
                     disable_cuda=True)


def test_weights_next_to_a_foreign_config_load(tmp_path):
    import torchvision.models as models
    tv = models.resnet18(weights=None)
    # A config with an `embedding` key that isn't this repo's: no build record.
    torch.save({"model": tv.state_dict(), "config": {"lr": 1e-3, "embedding": {"layer": "layer1"}}},
               tmp_path / "w.pt")
    e = EmbeddingNet("r", arch="resnet18", checkpoint=str(tmp_path / "w.pt"), disable_cuda=True)
    assert torch.equal(e.embedding.conv1.weight, tv.conv1.weight)
    assert e.build_warnings == []


def test_uber_model_registers_its_sub_models():
    from src.embeddings import UberModel
    parts = [nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4)) for _ in range(2)]
    m = UberModel(parts)
    assert len(list(m.parameters())) == 8                  # conv w/b + bn w/b, twice
    assert sum(isinstance(x, nn.BatchNorm2d) for x in m.modules()) == 2
    assert any(k.startswith("models.1.") for k in m.state_dict())


def test_bn_stat_drift():
    e = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True)
    ref = e.bn_stats()
    assert ref and e.bn_stat_drift(ref) == 0.0
    e.set_bn_train(True)
    with torch.no_grad():
        e.encode(_u8())
    assert e.bn_stat_drift(ref) > 0.0


def test_mae_tokens_keep_their_order():
    from src.embeddings import _forward_mae
    from src.truncation import truncate
    from src.vision_models.mae import mae_vit_base_patch16
    torch.manual_seed(0)
    m = mae_vit_base_patch16().eval()
    fwd, _ = truncate(m, _forward_mae, 3)
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        torch.manual_seed(0); a = fwd(m, x)
        torch.manual_seed(1); b = fwd(m, x)
    assert torch.equal(a, b)


def test_flatten_head():
    e = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True, preprocess=PIEG,
                     input_size=84, layer="layer2", head="flatten")
    assert e.encode(_u8(size=84)).shape == (2, 128 * 11 * 11)
    assert not list(e.head.parameters())


def test_l2sp_and_its_anchor_cover_only_the_backbone():
    from src.losses import AuxBatch, build_aux_losses
    e = _trained_net()
    (term,) = build_aux_losses([{"name": "l2sp", "weight": 1.0}], e)
    with torch.no_grad():
        e.head.fc.weight.add_(1.0)                  # head drift: not penalised
    assert term.loss(AuxBatch(encoder=e)).item() == 0.0


def _emb_cfg(**fields):
    return OmegaConf.create({"embedding": {"name": "my_run", **fields}})


def test_embedding_kwargs_resolves_and_validates(tmp_path, monkeypatch):
    (tmp_path / "w.pt").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    kw = embedding_kwargs(_emb_cfg(arch="resnet18", checkpoint="w.pt", preprocess=PIEG))
    assert kw["checkpoint"] == str(tmp_path / "w.pt") and kw["arch"] == "resnet18"
    assert kw["preprocess"]["mean"] == [0.5, 0.5, 0.5]
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        embedding_kwargs(_emb_cfg(arch="resnet18", checkpoint="missing.pt"))
    with pytest.raises(ValueError, match="can't be loaded into arch='vip_resnet50'"):
        embedding_kwargs(_emb_cfg(arch="vip_resnet50", checkpoint="w.pt"))
    with pytest.raises(ValueError, match="unknown preprocess keys"):
        embedding_kwargs(_emb_cfg(preprocess={"rezise": 128}))
    assert embedding_kwargs(OmegaConf.create({"embedding": {"name": None}})) == {}


def test_custom_checkpoint_config_requires_a_checkpoint():
    # OmegaConf's .get() would quietly return None for `???`, falling back to
    # the architecture's default weights.
    from hydra import compose, initialize_config_dir
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    with initialize_config_dir(config_dir=root, version_base=None):
        cfg = compose("config_sac", overrides=["embedding=custom_checkpoint"])
    with pytest.raises(ValueError, match="embedding.checkpoint must be set"):
        embedding_kwargs(cfg)


# ---- DINOv2 (downloads weights: set PVR_TEST_NETWORK=1 to run) ----------------

@pytest.mark.skipif(os.environ.get("PVR_TEST_NETWORK") != "1", reason="needs network (torch.hub)")
def test_dinov2_vits14():
    e = EmbeddingNet("dinov2_vits14", disable_cuda=True)
    x = _u8(size=84)
    p = e.preprocess(x)
    assert p.shape[-1] == 224 and 0.0 <= p.min() and p.max() <= 1.0
    assert e.encode(x).shape == (2, 384)
