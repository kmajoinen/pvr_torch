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
    def __init__(self, dim=D, heads=HEADS):
        super().__init__()
        self.n_heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ (k.transpose(-2, -1) * self.scale)
        return self.proj((scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C))


class VBlock(nn.Module):
    def __init__(self, dim=D, heads=HEADS):
        super().__init__()
        self.pre_norm_attn = nn.LayerNorm(dim, eps=1e-6)
        self.attn = VAttention(dim, heads)
        self.pre_norm_mlp = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU()), nn.Dropout(0.0),
                                 nn.Linear(4 * dim, dim))

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
            assert cfg.embedding.pretrained is False    # no silent default weights without the file


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


# ---- R-MVP's input pipeline and MAE decoder ---------------------------------------------

def test_r_mvp_config_uses_voltrons_pipeline_not_maes_eval_crop():
    import torchvision.transforms as T
    from src.embeddings import preprocess_transforms
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        spec = dict(compose(config_name="config_sac", overrides=["embedding=voltron_r_mvp"]).embedding.preprocess)
    steps = list(preprocess_transforms(spec))
    assert [type(t) for t in steps] == [T.Resize, T.CenterCrop, T.ConvertImageDtype, T.Normalize]
    assert steps[0].size == 224 and steps[0].interpolation == T.InterpolationMode.BILINEAR
    assert list(steps[1].size) == [224, 224]
    assert steps[3].mean == [0.485, 0.456, 0.406] and steps[3].std == [0.229, 0.224, 0.225]
    # On a square 224px frame Voltron's pipeline keeps the whole frame; MAE's
    # evaluation transform (vit_* default: resize 256, crop 224) zooms in.
    frame = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)
    ours = EmbeddingNet("v", arch="vit_s16_h6", pretrained=False, disable_cuda=True, preprocess=spec)
    mae_default = EmbeddingNet("v", arch="vit_s16_h6", pretrained=False, disable_cuda=True)
    assert torch.allclose(ours.preprocess(frame), frame.float() / 255, atol=1e-6)
    assert not torch.allclose(mae_default.preprocess(frame), frame.float() / 255, atol=1e-2)


class VoltronMVP(VoltronViT):
    """VMVP's masked pass: forward_encoder, forward_decoder, compute_loss."""

    def __init__(self):
        super().__init__(cls=True)
        self.encoder2decoder = nn.Linear(D, 192)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 192) * 0.02)
        pe = torch.from_numpy(get_2d_sincos_pos_embed(192, 14, cls_token=True)).float()[None]
        self.decoder_pe = nn.Parameter(pe, requires_grad=False)
        self.decoder_blocks = nn.ModuleList([VBlock(192, 6) for _ in range(6)])
        self.decoder_norm = nn.LayerNorm(192, eps=1e-6)
        self.decoder_prediction = nn.Linear(192, 16 * 16 * 3)

    def loss(self, imgs, mask_ratio=0.75):
        patches = self.patch2embed(imgs) + self.encoder_pe[:, 1:, :]
        bsz, n, dim = patches.shape
        n_keep = int(n * (1 - mask_ratio))
        shuffle = torch.argsort(torch.rand(bsz, n), dim=1)
        restore = torch.argsort(shuffle, dim=1)
        visible = torch.gather(patches, 1, shuffle[:, :n_keep, None].repeat(1, 1, dim))
        mask = torch.ones(bsz, n)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, restore)
        x = torch.cat([(self.cls_token + self.encoder_pe[:, :1, :]).expand(bsz, -1, -1), visible], dim=1)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder2decoder(self.encoder_norm(x))
        tokens = torch.cat([x[:, 1:, :], self.mask_token.repeat(bsz, n - n_keep, 1)], dim=1)
        x = torch.cat([x[:, :1, :], torch.gather(tokens, 1, restore[..., None].repeat(1, 1, 192))], dim=1)
        x = x + self.decoder_pe
        for block in self.decoder_blocks:
            x = block(x)
        pred = self.decoder_prediction(self.decoder_norm(x))[:, 1:, :]
        t = imgs.reshape(bsz, 3, 14, 16, 14, 16).permute(0, 2, 4, 3, 5, 1).reshape(bsz, 196, 768)
        t = (t - t.mean(-1, keepdim=True)) / (t.var(-1, unbiased=True, keepdim=True) + 1e-6) ** 0.5
        return (((pred - t) ** 2).mean(-1) * mask).sum() / mask.sum()


def test_r_mvp_masked_loss_matches_voltrons_with_its_decoder_settings(tmp_path):
    from src.vision_models.mae import MAEDecoder
    from src.vision_models.voltron import convert_voltron_decoder, rmvp_decoder
    ref = _randomise(VoltronMVP()).eval()
    path = _save(tmp_path, "r-mvp", ref.state_dict())
    enc = EmbeddingNet("v", arch="vit_s16_h6", checkpoint=path, disable_cuda=True, input_size=224,
                       preprocess={"resize": None, "crop": None, "mean": None, "std": None}).embedding.eval()
    sd = torch.load(path, weights_only=True)[0]
    dec = rmvp_decoder().eval()
    dec.load_state_dict(convert_voltron_decoder(sd))
    imgs = torch.rand(2, 3, 224, 224)

    def ours(decoder):
        torch.manual_seed(5)                            # the same masking draws for both
        latent, mask, restore = enc.forward_encoder(imgs, 0.75)
        return decoder.forward_loss(imgs, decoder.forward_decoder(latent, restore), mask)

    with torch.no_grad():
        torch.manual_seed(5)
        theirs = ref.loss(imgs)
        assert torch.allclose(ours(dec), theirs, rtol=1e-5, atol=1e-6)
        # MAEDecoder's defaults (16 heads, eps 1e-5) take the same weights and
        # compute something else: the trap rmvp_decoder() avoids.
        default = MAEDecoder(num_patches=196, patch_size=16, embed_dim=384, decoder_embed_dim=192,
                             decoder_depth=6, norm_pix_loss=True).eval()
        default.load_state_dict(convert_voltron_decoder(sd))
        assert not torch.allclose(ours(default), theirs, rtol=1e-3)
