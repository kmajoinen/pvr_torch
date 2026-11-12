"""Voltron's released encoders (src/vision_models/voltron.py): checkpoints in
Voltron's format convert to this repo's architectures and compute the same
features as Voltron's own forward pass.

The reference below ports the parts of siddk/voltron-robotics (MIT license)
that compute an image's features: voltron/models/util/transformer.py's
PatchEmbed/Attention/Block, VMVP.encode (with the CLS token concatenated on
dim=1, as in MVP's code -- the released line omits dim and fails), and
VR3M.get_representations. The weights are random, saved in Voltron's
(state_dict, optimizer) format with its module names."""

import os

import pytest
import torch
import torch.nn as nn
import torchvision
from hydra import compose, initialize_config_dir

from src.embeddings import EmbeddingNet
from src.vision_models.mae import get_2d_sincos_pos_embed
from src.vision_models.voltron import convert_voltron, is_voltron_checkpoint

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
D, HEADS, DEPTH = 384, 6, 12


# ---- reference: Voltron's modules -------------------------------------------------

class VPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, D, kernel_size=16, stride=16)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)        # bsz (h w) embed


class VAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads, self.scale = HEADS, (D // HEADS) ** -0.5
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ (k.transpose(-2, -1) * self.scale)
        return self.proj((scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C))


class VBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm_attn = nn.LayerNorm(D, eps=1e-6)
        self.attn = VAttention()
        self.pre_norm_mlp = nn.LayerNorm(D, eps=1e-6)
        self.mlp = nn.Sequential(nn.Sequential(nn.Linear(D, 4 * D), nn.GELU()), nn.Dropout(0.0), nn.Linear(4 * D, D))

    def forward(self, x):
        x = x + self.attn(self.pre_norm_attn(x))
        return x + self.mlp(self.pre_norm_mlp(x))


class VoltronViT(nn.Module):
    """VMVP's encoder (cls=True) or VR3M's (cls=False)."""

    def __init__(self, cls: bool):
        super().__init__()
        self.cls = cls
        self.patch2embed = VPatchEmbed()
        pe = torch.from_numpy(get_2d_sincos_pos_embed(D, 14, cls_token=cls)).float()[None]
        if cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            self.encoder_pe = nn.Parameter(pe, requires_grad=False)
            self.encoder_blocks = nn.ModuleList([VBlock() for _ in range(DEPTH)])
            self.encoder_norm = nn.LayerNorm(D, eps=1e-6)
        else:
            self.pe = nn.Parameter(pe, requires_grad=False)
            self.blocks = nn.ModuleList([VBlock() for _ in range(DEPTH)])
            self.norm = nn.LayerNorm(D, eps=1e-6)

    def features(self, img):
        patches = self.patch2embed(img)
        if self.cls:      # VMVP.encode, then get_representations(mode="cls")
            x = torch.cat([self.cls_token.expand(img.shape[0], -1, -1), patches], dim=1) + self.encoder_pe
            for block in self.encoder_blocks:
                x = block(x)
            return self.encoder_norm(x)[:, 0]
        x = patches + self.pe   # VR3M.get_representations
        for block in self.blocks:
            x = block(x)
        return self.norm(x).mean(1)


def _randomise(module):
    torch.manual_seed(0)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "pe" not in name.split(".")[-1]:
                p.add_(torch.randn_like(p) * 0.05)    # move norms/biases off their init too
    return module


def _save(tmp_path, name, state_dict, extra=None):
    sd = dict(state_dict, **(extra or {}))
    path = tmp_path / f"{name}.pt"
    torch.save([sd, {"optimizer": "state"}], path)
    return str(path)


# ---- conversion -----------------------------------------------------------------

@pytest.mark.parametrize("cls, arch", [(True, "vit_s16_h6"), (False, "vit_s16_h6_gap")])
def test_voltron_vits_load_and_match_voltrons_forward(tmp_path, cls, arch):
    ref = _randomise(VoltronViT(cls)).eval()
    extra = ({"decoder_norm.weight": torch.ones(192), "mask_token": torch.zeros(1, 1, 192)} if cls
             else {"language_reward.0.weight": torch.zeros(1, 1), "lm.embeddings.word_embeddings.weight": torch.zeros(2, 2)})
    path = _save(tmp_path, arch, ref.state_dict(), extra)
    enc = EmbeddingNet("voltron", arch=arch, checkpoint=path, disable_cuda=True, input_size=224,
                       preprocess={"resize": None, "crop": None, "mean": None, "std": None}).eval()
    x = torch.rand(3, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(enc.encode_preprocessed(x), ref.features(x), atol=1e-5)


def test_r3m_resnet50_loads_and_takes_0_255_pixels(tmp_path):
    torch.manual_seed(0)
    ref = torchvision.models.resnet50(weights=None)
    ref.fc = nn.Identity()
    ref.eval()
    sd = {"resnet." + k: v for k, v in ref.state_dict().items()}
    path = _save(tmp_path, "r-r3m-rn50", sd, {"language_reward.0.weight": torch.zeros(1)})
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_sac", overrides=["embedding=voltron_r_r3m_rn50",
                                                          f"embedding.checkpoint={path}"])
    enc = EmbeddingNet("voltron_r_r3m_rn50", arch="resnet50", checkpoint=path, disable_cuda=True, input_size=224,
                       preprocess=dict(cfg.embedding.preprocess)).eval()
    frames = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
    with torch.no_grad():
        assert torch.allclose(enc.encode(frames), ref(frames.float()), rtol=1e-4, atol=1e-3)


def test_six_heads_are_not_twelve(tmp_path):
    ref = _randomise(VoltronViT(True)).eval()
    path = _save(tmp_path, "r-mvp", ref.state_dict())
    pre = {"resize": None, "crop": None, "mean": None, "std": None}
    six = EmbeddingNet("v", arch="vit_s16_h6", checkpoint=path, disable_cuda=True, input_size=224, preprocess=pre).eval()
    twelve = EmbeddingNet("v", arch="vit_s16", checkpoint=path, disable_cuda=True, input_size=224, preprocess=pre).eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():   # same shapes, so both load -- but only one is R-MVP's function
        assert not torch.allclose(six.encode_preprocessed(x), twelve.encode_preprocessed(x), atol=1e-3)


def test_only_voltron_files_are_recognised():
    assert is_voltron_checkpoint([{"patch2embed.proj.weight": 0}, {}])
    assert is_voltron_checkpoint(({"resnet.conv1.weight": 0}, {}))
    assert not is_voltron_checkpoint({"patch_embed.proj.weight": 0})
    assert not is_voltron_checkpoint([{"conv1.weight": 0}, {}])
    with pytest.raises(KeyError, match="unexpected key"):
        convert_voltron({"patch2embed.proj.weight": 0, "something_else": 0})


# ---- the CLS-free ViT --------------------------------------------------------------

def test_a_vit_without_cls_token_pools_all_tokens_and_refuses_cls():
    torch.manual_seed(0)
    full = EmbeddingNet("v", arch="vit_s16_h6_gap", pretrained=False, disable_cuda=True).eval()
    cut = EmbeddingNet("v", arch="vit_s16_h6_gap", pretrained=False, disable_cuda=True,
                       layer=12, head="avgpool").eval()
    cut.load_state_dict(full.state_dict())
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        tokens = full.embedding.forward_features(x)
    assert tokens.shape[1] == 196                                   # patches only
    with pytest.raises(ValueError, match="no CLS token"):
        EmbeddingNet("v", arch="vit_s16_h6_gap", pretrained=False, disable_cuda=True, layer=6, head="cls")
    with pytest.raises(ValueError, match="needs a CLS token"):
        full.embedding.forward_encoder(x, 0.75)                     # MAE needs one
    assert cut.out_size == 384


def test_voltron_configs_compose():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        for name, arch in [("voltron_r_mvp", "vit_s16_h6"), ("voltron_r_r3m_vit", "vit_s16_h6_gap"),
                           ("voltron_r_r3m_rn50", "resnet50")]:
            cfg = compose(config_name="config_sac", overrides=[f"embedding={name}"])
            assert cfg.embedding.arch == arch and cfg.embedding.checkpoint.endswith(".pt")


# ---- loading untrusted files ---------------------------------------------------------

class _Payload:
    """Unpickling this calls open(marker, "w"): stands in for code a planted
    checkpoint would run."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def _resnet18_weights():
    torch.manual_seed(0)
    return torchvision.models.resnet18(weights=None).state_dict()


def test_a_checkpoint_with_code_in_it_is_refused_without_running_it(tmp_path):
    marker = tmp_path / "ran"
    path = tmp_path / "planted.pt"
    torch.save({**_resnet18_weights(), "extra": _Payload(str(marker))}, path)
    with pytest.raises(ValueError, match="trust_checkpoint"):
        EmbeddingNet("resnet18", checkpoint=str(path), disable_cuda=True)
    assert not marker.exists()


def test_a_trusted_checkpoint_may_pickle_more_than_tensors(tmp_path):
    import argparse
    path = tmp_path / "with_args.pt"
    torch.save({"model": _resnet18_weights(), "args": argparse.Namespace(lr=1e-3)}, path)
    with pytest.raises(ValueError, match="trust_checkpoint"):
        EmbeddingNet("resnet18", checkpoint=str(path), disable_cuda=True)
    EmbeddingNet("resnet18", checkpoint=str(path), disable_cuda=True, trust_checkpoint=True)


def test_trust_checkpoint_reaches_the_encoder_from_the_config(tmp_path):
    from src.encoder_config import embedding_kwargs
    path = tmp_path / "w.pt"
    torch.save(_resnet18_weights(), path)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        default = compose(config_name="config_sac", overrides=[
            "embedding=custom_checkpoint", "embedding.arch=resnet18", f"embedding.checkpoint={path}"])
        trusted = compose(config_name="config_sac", overrides=[
            "embedding=custom_checkpoint", "embedding.arch=resnet18", f"embedding.checkpoint={path}",
            "embedding.trust_checkpoint=true"])
    assert embedding_kwargs(default)["trust_checkpoint"] is False
    assert embedding_kwargs(trusted)["trust_checkpoint"] is True
