"""Training selected encoder modules only (encoder.trainable=[...],
EmbeddingNet.set_trainable): what trains, where gradient flows, and the
config checks."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.embeddings import EmbeddingNet
from src.encoder_config import check_head_is_trained, encoder_settings

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
SAC_GRADS = ("none", "critic", "actor", "both")


def _trained(net):
    return {n for n, p in net.named_parameters() if p.requires_grad}


@pytest.fixture(scope="module")
def resnet():
    net = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True)
    return net.eval()


def test_one_stage_trains_alone_and_only_it_gets_gradient(resnet):
    resnet.set_trainable(["layer3"])
    assert _trained(resnet) and all(n.startswith("embedding.layer3.") for n in _trained(resnet))
    assert resnet.trainable_modules == {"layer3": sum(p.numel() for p in resnet.embedding.layer3.parameters())}
    resnet.encode_preprocessed(torch.rand(2, 3, 224, 224)).sum().backward()
    got = {n for n, p in resnet.named_parameters() if p.grad is not None}
    assert got == _trained(resnet)
    resnet.zero_grad(set_to_none=True)


def test_exact_module_paths_and_several_modules(resnet):
    resnet.set_trainable(["layer3.1.conv2"])
    assert _trained(resnet) == {"embedding.layer3.1.conv2.weight"}
    resnet.set_trainable(["layer3", "layer4"])
    assert {n.split(".")[1] for n in _trained(resnet)} == {"layer3", "layer4"}


def test_string_modes_still_work(resnet):
    resnet.set_trainable("all")
    assert len(_trained(resnet)) == len(list(resnet.parameters())) and resnet.trainable_modules is None
    resnet.set_trainable("none")
    assert not _trained(resnet)


def test_a_vit_block():
    vit = EmbeddingNet("vit_s16", pretrained=False, disable_cuda=True).eval()
    vit.set_trainable(["blocks.6"])
    assert _trained(vit) and all(n.startswith("embedding.blocks.6.") for n in _trained(vit))


def test_the_head_can_be_listed():
    net = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True, head="fc", head_dim=16).eval()
    net.set_trainable(["layer4", "head"])
    assert {n.split(".")[0] if n.startswith("head") else n.split(".")[1] for n in _trained(net)} == {"layer4", "head"}


@pytest.mark.parametrize("mode, match", [
    (["avgpool"], "no parameters"),
    (["layer9"], "encoder.trainable entry 'layer9'"),
    ([], "module names"),
    ([6], "module names"),
    (["head"], "head has no"),
])
def test_bad_selections(resnet, mode, match):
    with pytest.raises(ValueError, match=match):
        resnet.set_trainable(mode)


def test_a_module_after_the_layer_cut_gets_no_gradient():
    net = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True, layer="layer2").eval()
    with pytest.raises(ValueError, match="no gradient.*embedding.layer=layer2"):
        net.set_trainable(["layer4"])


def test_the_gradient_check_changes_no_state():
    net = EmbeddingNet("resnet18", pretrained=False, train=True, disable_cuda=True)   # train mode (pretrain.py)
    bn = net.embedding.layer3[0].bn1
    stats = (bn.running_mean.clone(), bn.num_batches_tracked.clone())
    net.set_trainable(["layer3"])
    assert torch.equal(bn.running_mean, stats[0]) and torch.equal(bn.num_batches_tracked, stats[1])
    assert net.embedding.training and bn.training                       # modes restored
    assert all(p.grad is None for p in net.parameters())


def _settings(*overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_sac", overrides=list(overrides))
    return encoder_settings(cfg, valid_grads=SAC_GRADS)


def test_config_accepts_a_list():
    s = _settings("embedding=resnet18", "encoder.grads=critic", "encoder.trainable=[layer3,layer4]")
    assert s.trains and s.trainable == ("layer3", "layer4")


@pytest.mark.parametrize("overrides, match", [
    (["encoder.trainable=layer3"], "expected all, head, or a list"),
    (["encoder.trainable=[]"], "module names"),
    (["encoder.trainable=[head]"], "has no parameters"),
    (["+embedding.head=fc", "encoder.trainable=[layer4]"], "leaves it out"),
])
def test_config_rejects(overrides, match):
    with pytest.raises(ValueError, match=match):
        _settings("embedding=resnet18", "encoder.grads=critic", *overrides)


def test_a_head_left_out_of_the_list_needs_trained_weights(tmp_path):
    from omegaconf import OmegaConf
    from src.rl.common import encoder_record
    src = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True, head="fc", head_dim=16)
    torch.save(src.embedding.state_dict(), tmp_path / "backbone.pt")
    cfg = OmegaConf.create({"embedding": {"name": "resnet18", "head": "fc", "head_dim": 16}})
    torch.save(encoder_record(cfg, src), tmp_path / "with_head.pt")
    kw = dict(disable_cuda=True, head="fc", head_dim=16)
    with pytest.raises(ValueError, match="leaves it out"):
        check_head_is_trained(EmbeddingNet("resnet18", checkpoint=str(tmp_path / "backbone.pt"), **kw),
                              trains=True, trainable=("layer4",))
    check_head_is_trained(EmbeddingNet("resnet18", checkpoint=str(tmp_path / "with_head.pt"), **kw),
                          trains=True, trainable=("layer4",))
    check_head_is_trained(EmbeddingNet("resnet18", checkpoint=str(tmp_path / "backbone.pt"), **kw),
                          trains=True, trainable=("layer4", "head"))


def test_bn_adapt_moves_only_the_selected_modules_statistics():
    net = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True).eval()
    net.set_trainable(["layer4"])
    frozen, adapting = net.embedding.layer2[0].bn1, net.embedding.layer4[0].bn1
    before = frozen.running_mean.clone(), adapting.running_mean.clone()
    net.set_bn_train(True)
    assert adapting.training and not frozen.training
    with torch.no_grad():
        net.encode_preprocessed(torch.rand(4, 3, 224, 224))
    net.set_bn_train(False)
    assert torch.equal(frozen.running_mean, before[0])
    assert not torch.equal(adapting.running_mean, before[1])


def test_bn_adapt_without_a_selection_still_covers_the_whole_backbone():
    net = EmbeddingNet("resnet18", pretrained=False, disable_cuda=True).eval()
    net.set_trainable("head")              # PIE-G: frozen backbone, adapting statistics
    assert len(net._batchnorms()) == sum(isinstance(m, torch.nn.BatchNorm2d) for m in net.embedding.modules())


def test_a_selection_without_batchnorm_has_none(resnet):
    resnet.set_trainable(["layer3.1.conv2"])
    assert not resnet.has_batchnorm()
    resnet.set_trainable(["layer3"])
    assert resnet.has_batchnorm()


@pytest.mark.parametrize("mode", [["layer3", "layer3.1.conv2"], ["layer3.1.conv2", "layer3"], ["layer3", "layer3"]])
def test_overlapping_entries_are_rejected(resnet, mode):
    with pytest.raises(ValueError, match="overlaps"):
        resnet.set_trainable(mode)


def test_a_dotted_name_from_the_command_line(resnet):
    s = _settings("embedding=resnet18", "encoder.grads=critic", "encoder.trainable=[layer3.1.conv2,layer4]")
    assert s.trainable == ("layer3.1.conv2", "layer4")
    resnet.set_trainable(s.trainable)
    assert set(resnet.trainable_modules) == {"layer3.1.conv2", "layer4"}


def test_a_failed_selection_leaves_nothing_trainable(resnet):
    with pytest.raises(ValueError, match="overlaps"):
        resnet.set_trainable(["layer4", "layer3", "layer3.1.conv2"])
    assert not _trained(resnet) and resnet.trainable_modules is None and resnet._trainable_roots is None
