"""R3M objective (src/losses.py): frame sampling, the loss terms, and parity
with the official r3m/trainer.py (skipped unless the r3m package is
installed: pip install git+https://github.com/facebookresearch/r3m.git)."""

import copy
import os
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from src.losses import R3M, AuxBatch, R3MFrameSampler, aux_loss_classes, objective_terms

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
D = 16


class _Enc(nn.Module):
    out_size = D

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 16, stride=16), nn.Flatten(), nn.Linear(4 * 14 * 14, D))

    def forward(self, x):
        return self.net(x / 255.0)


class _Tok:
    """Stand-in for DistilBERT's tokenizer (the real one needs transformers
    and a download): one id per word, padded to the longest sentence."""

    def __call__(self, langs, return_tensors="pt", padding=True):
        ids = [[1 + sum(map(ord, w)) % 50 for w in t.split()] for t in langs]
        width = max(1, max(map(len, ids)))
        input_ids = torch.zeros(len(ids), width, dtype=torch.long)
        mask = torch.zeros_like(input_ids)
        for i, row in enumerate(ids):
            input_ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, :len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


class _TextModel(nn.Module):
    """Stand-in for DistilBERT: embeddings + dropout, so train/eval mode and
    per-call dropout masks matter as they do for the real model."""

    def __init__(self):
        super().__init__()
        self.emb, self.drop = nn.Embedding(51, 768), nn.Dropout(0.1)
        self.calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.calls += 1
        return types.SimpleNamespace(last_hidden_state=self.drop(self.emb(input_ids)))


def _r3m(enc, **kw):
    loss = R3M(enc, hidden_dim=32, **kw)
    loss.encode_text.tokenizer, loss.encode_text.model = _Tok(), _TextModel()
    loss.train()
    return loss


@pytest.mark.parametrize("alpha", [0.2, 0.0, 0.4])
def test_frame_sampler_matches_the_official_windows(alpha):
    s, rng = R3MFrameSampler(alpha), np.random.default_rng(0)
    for length in range(s.min_length, 60):
        for _ in range(50):
            start, goal, s0, s1, s2 = s(length, rng)
            assert 1 <= start <= 1 + int(alpha * length)
            assert int((1 - alpha) * length) - 1 <= goal <= length - 1
            assert 1 <= s0 < s1 <= s2 <= length and s1 >= 2


def test_loss_terms_and_metrics():
    torch.manual_seed(0)
    enc = _Enc()
    feats = torch.randn(6, 5, D, requires_grad=True)
    text = ["opens drawer", "", "picks cup", "walks", "cuts onion", "x"]
    loss, m = _r3m(enc)(AuxBatch(encoder=enc, clip_feats=feats, text=text))
    assert set(m) == {"l0loss", "l1loss", "l2loss", "rewloss", "rewacc1", "rewacc2", "rewacc3",
                      "tcnloss", "aligned"}
    expected = 1e-5 * m["l2loss"] + 1e-5 * m["l1loss"] + m["rewloss"] + m["tcnloss"]
    assert torch.allclose(loss.detach(), expected)
    loss.backward()
    assert feats.grad.abs().sum() > 0


@pytest.mark.parametrize("text_dropout, train, calls", [(True, True, 15), (True, False, 1), (False, True, 1)])
def test_text_is_encoded_per_reward_call_only_with_dropout_in_training(text_dropout, train, calls):
    enc = _Enc()
    loss = _r3m(enc, text_dropout=text_dropout).train(train)
    model = loss.encode_text.model
    assert model.training == (train and text_dropout)
    loss(AuxBatch(encoder=enc, clip_feats=torch.randn(4, 5, D), text=["a b", "c", "d e f", "g"]))
    assert model.calls == calls          # official: 3 positives + 3 + 3 x 3 negatives


@pytest.mark.parametrize("text_dropout", [True, False])
def test_a_new_r3m_has_its_text_encoder_in_the_same_mode(text_dropout):
    loss = R3M(_Enc(), text_dropout=text_dropout)     # no train()/eval() call yet
    assert loss.training and loss.encode_text.training == text_dropout
    assert not loss.eval().encode_text.training


def test_language_term_off_needs_no_head_or_text():
    enc = _Enc()
    loss = R3M(enc, lang_weight=0.0, num_negatives=0)
    assert not list(loss.parameters())
    out, m = loss(AuxBatch(encoder=enc, clip_feats=torch.randn(4, 5, D), text=None))
    assert "rewloss" not in m and torch.isfinite(out)


def test_objective_terms_reports_metrics_under_the_objective_name():
    from src.losses import AuxTerm
    enc = _Enc()
    term = AuxTerm("r3m", 2.0, _r3m(enc))
    term.loss.eval()                     # deterministic text, so the total is comparable
    total, values = objective_terms([term], AuxBatch(encoder=enc, clip_feats=torch.randn(4, 5, D),
                                                     text=["a", "b", "c", "d"]))
    assert torch.allclose(total.detach(), 2.0 * values["r3m"])
    assert "r3m/tcnloss" in values and "r3m/rewacc1" in values


def test_rl_scripts_reject_clip_objectives():
    from src.encoder_config import encoder_settings
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_sac", overrides=["encoder.aux=[{name:r3m,weight:1.0}]"])
    with pytest.raises(ValueError, match="pretrain.py"):
        encoder_settings(cfg, valid_grads=("none", "critic", "actor", "both"))


def test_bad_params_fail_at_validation():
    with pytest.raises(ValueError, match="alpha"):
        aux_loss_classes([{"name": "r3m", "weight": 1.0, "alpha": 0.6}])


@pytest.mark.parametrize("overrides", [["+experiment=r3m"], ["+experiment=r3m", "embedding=resnet18"],
                                       ["embedding=resnet18", "+experiment=r3m"]])
def test_r3m_preset(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_pretrain", overrides=overrides + ["pretrain_data.root=/x"])
    # Frames as stored for pretraining; the saved encoder keeps the default
    # input pipeline (the one the released R3M weights get here).
    assert cfg.embedding.pretrained is False and cfg.embedding.get("preprocess") is None
    assert cfg.pretrain_data.frames_as_stored is True
    assert cfg.augmentation.name == "random_resized_crop" and list(cfg.augmentation.params.scale) == [0.2, 1.0]
    assert cfg.objectives[0].name == "r3m" and cfg.batch_size == 16 and cfg.lr == 1e-4
    assert cfg.embedding.name == ("resnet18" if "embedding=resnet18" in overrides else "resnet50")


@pytest.mark.parametrize("lang_w, tcn_w", [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0)])
def test_parity_with_official_trainer(lang_w, tcn_w, monkeypatch):
    _parity(lang_w, tcn_w, monkeypatch, real_text=False)


@pytest.mark.skipif(os.environ.get("PVR_TEST_NETWORK") != "1",
                    reason="downloads DistilBERT: set PVR_TEST_NETWORK=1 to run")
def test_parity_with_official_trainer_real_distilbert(monkeypatch):
    # Both sides load distilbert-base-uncased: the official encoder when it's
    # built, ours at its first forward pass. If this fails only here, check the
    # transformers version first (loading must not draw from torch's RNG, or
    # every later randperm shifts) and pin it in requirements.txt.
    pytest.importorskip("transformers")
    _parity(1.0, 1.0, monkeypatch, real_text=True)


def _parity(lang_w, tcn_w, monkeypatch, real_text):
    trainer = pytest.importorskip("r3m.trainer")
    from r3m.models.models_language import LangEncoder, LanguageReward
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self, raising=False)
    B = 6
    torch.manual_seed(0)
    enc_a = _Enc()
    enc_b = copy.deepcopy(enc_a)
    rew = LanguageReward(None, D, 32, 768)
    kw = dict(tcn_weight=tcn_w, lang_weight=lang_w, l1_weight=1e-3, l2_weight=1e-3)
    ours = R3M(enc_b, hidden_dim=32, **kw).train() if real_text else _r3m(enc_b, **kw)
    if lang_w > 0:
        ours.lang_rew.load_state_dict(rew.state_dict())
    # The official text encoder, as a submodule of the model: Trainer.update's
    # model.train() turns its dropout on, as in the real code. Stand-in
    # tokenizer and model unless real_text.
    if real_text:
        lang_enc = LangEncoder("cpu", 0, 0)
    else:
        lang_enc = LangEncoder.__new__(LangEncoder)
        nn.Module.__init__(lang_enc)
        lang_enc.tokenizer, lang_enc.device, lang_enc.lang_size = _Tok(), "cpu", 768
        lang_enc.model = copy.deepcopy(ours.encode_text.model)
    mod = types.SimpleNamespace(l2weight=1e-3, l1weight=1e-3, langweight=lang_w, tcnweight=tcn_w,
                                num_negatives=3)
    mod.sim = lambda a, b: -torch.linalg.norm(a - b, dim=-1)
    mod.get_reward = lambda e0, es, s: rew(e0, es, lang_enc(s))
    mod.encoder_opt = torch.optim.Adam(list(enc_a.parameters()) + list(rew.parameters()), lr=1e-3)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.lang, self.module = enc_a, lang_enc, mod

        def forward(self, x):
            return self.enc(x)

    torch.manual_seed(1)
    ims = torch.rand(B, 5, 3, 224, 224) * 255
    text = ["c opens drawer", "", "c picks cup", "c walks", "c cuts onion", "c x"]
    torch.manual_seed(2)
    official, _ = trainer.Trainer(1).update(Model(), (ims, text), 0)

    opt = torch.optim.Adam(list(enc_b.parameters()) + list(ours.parameters()), lr=1e-3)
    torch.manual_seed(2)
    feats = enc_b(ims.reshape(B * 5, 3, 224, 224)).reshape(B, 5, D)
    loss, m = ours(AuxBatch(encoder=enc_b, clip_feats=feats, text=text))
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert official["full_loss"] == pytest.approx(loss.item(), rel=1e-6, abs=1e-6)
    for k, v in official.items():
        if k != "full_loss":
            assert v == pytest.approx(float(m[k]), rel=1e-6, abs=1e-6), k
    for a, b in zip(enc_a.parameters(), enc_b.parameters()):
        assert torch.allclose(a, b, atol=1e-7)
