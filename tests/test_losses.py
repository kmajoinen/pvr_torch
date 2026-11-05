"""Tests for the aux-loss registry in src/losses.py."""

import pytest
import torch
import torch.nn as nn

from src.losses import (
    AuxBatch,
    AuxLoss,
    aux_loss_classes,
    build_aux_losses,
    clip_encoder_grads,
    encoder_param_groups,
    register_aux_loss,
    total_aux_loss,
)


def _encoder():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_l2sp_zero_at_init_and_grows_with_drift():
    enc = _encoder()
    (term,) = build_aux_losses([{"name": "l2sp", "weight": 1.0}], enc)
    assert term.loss(AuxBatch(encoder=enc)).item() == 0.0
    with torch.no_grad():
        enc[0].weight.add_(0.1)
    value = term.loss(AuxBatch(encoder=enc))
    assert value.item() == pytest.approx(0.01 * enc[0].weight.numel(), rel=1e-5)
    value.backward()
    assert enc[0].weight.grad is not None and enc[0].weight.grad.abs().sum() > 0


def test_total_is_weighted_and_values_are_unweighted():
    enc = _encoder()
    terms = build_aux_losses([{"name": "l2sp", "weight": 0.5}], enc)
    with torch.no_grad():
        enc[0].weight.add_(0.1)
    total, values = total_aux_loss(terms, AuxBatch(encoder=enc))
    assert total.item() == pytest.approx(0.5 * values["l2sp"])


def test_l2sp_does_not_register_the_encoder_as_a_submodule():
    enc = _encoder()
    (term,) = build_aux_losses([{"name": "l2sp", "weight": 1.0}], enc)
    assert list(term.loss.parameters()) == []


@pytest.mark.parametrize("specs, match", [
    ([{"name": "nope", "weight": 1.0}], "Unknown aux loss"),
    ([{"name": "l2sp"}], "weight"),
    ([{"name": "l2sp", "weight": -1.0}], "weight"),
    (["l2sp"], "look like"),
    ([{"name": "l2sp", "weight": 1.0}, {"name": "l2sp", "weight": 2.0}], "more than once"),
])
def test_bad_specs_raise(specs, match):
    with pytest.raises(ValueError, match=match):
        aux_loss_classes(specs)


def test_unknown_param_raises():
    with pytest.raises(TypeError, match="l2sp"):
        build_aux_losses([{"name": "l2sp", "weight": 1.0, "bogus": 3}], _encoder())


@pytest.mark.parametrize("target, gets_grad, detached", [
    ("clean", "feats_aug", "feats"),
    ("augmented", "feats", "feats_aug"),
])
def test_aug_consistency_detaches_the_target(target, gets_grad, detached):
    (term,) = build_aux_losses(
        [{"name": "aug_consistency", "weight": 1.0, "target": target}], _encoder()
    )
    feats = {k: torch.randn(4, 2, requires_grad=True) for k in ("feats", "feats_aug")}
    value = term.loss(AuxBatch(encoder=_encoder(), **feats))
    assert value.item() == pytest.approx(
        torch.nn.functional.mse_loss(feats["feats_aug"], feats["feats"]).item()
    )
    value.backward()
    assert feats[gets_grad].grad is not None
    assert feats[detached].grad is None


def test_aug_consistency_declares_its_needs_and_validates_target():
    from src.losses import AugConsistency
    assert AugConsistency.needs_features and AugConsistency.needs_augmented
    with pytest.raises(ValueError, match="target"):
        build_aux_losses([{"name": "aug_consistency", "weight": 1.0, "target": "both"}], _encoder())


@pytest.mark.parametrize("target, grads", [("clean", (False, True)), ("augmented", (True, False))])
def test_aug_consistency_grad_views_follow_target(target, grads):
    from src.losses import aux_grad_views
    terms = build_aux_losses([{"name": "aug_consistency", "weight": 1.0, "target": target},
                              {"name": "l2sp", "weight": 1.0}], _encoder())
    assert aux_grad_views(terms) == grads
    assert aux_grad_views([]) == (False, False)


@pytest.mark.parametrize("spec, exc, match", [
    ({"name": "aug_consistency", "weight": 1.0, "targt": "clean"}, TypeError, "targt"),
    ({"name": "aug_consistency", "weight": 1.0, "target": "clen"}, ValueError, "target"),
    ({"name": "l2sp", "weight": 1.0, "scale": 2}, TypeError, "scale"),
])
def test_param_names_and_values_checked_without_building(spec, exc, match):
    with pytest.raises(exc, match=match):
        aux_loss_classes([spec])


def test_grad_through_unread_view_rejected_at_build():
    @register_aux_loss("_test_bad_grad_decl")
    class BadGrad(AuxLoss):
        needs_features = False

        @property
        def grad_features(self):
            return True

        def forward(self, batch):
            return torch.zeros(())

    with pytest.raises(ValueError, match="doesn't read"):
        build_aux_losses([{"name": "_test_bad_grad_decl", "weight": 1.0}], _encoder())


def test_custom_loss_with_learnable_head():
    @register_aux_loss("_test_head_norm")
    class HeadNorm(AuxLoss):
        def __init__(self, encoder, width: int = 3):
            super().__init__(encoder)
            self.head = nn.Linear(2, width)

        def forward(self, batch):
            x = torch.ones(1, 4)
            return self.head(batch.encoder(x)).pow(2).mean()

    enc = _encoder()
    (term,) = build_aux_losses([{"name": "_test_head_norm", "weight": 1.0, "width": 5}], enc)
    assert term.loss.head.out_features == 5
    # Only the head's own params belong to the loss; the encoder isn't captured.
    assert {id(p) for p in term.loss.parameters()} == {id(p) for p in term.loss.head.parameters()}
    total, _ = total_aux_loss([term], AuxBatch(encoder=enc))
    total.backward()
    assert term.loss.head.weight.grad is not None
    assert enc[0].weight.grad is not None

    with pytest.raises(TypeError, match="AuxLoss"):
        register_aux_loss("_test_not_a_loss")(nn.Linear)


class _Encoder(nn.Module):
    """EmbeddingNet's interface, on a small conv net."""

    def __init__(self, out_size=8, frame_stack=1):
        super().__init__()
        torch.manual_seed(0)
        self.out_size = out_size
        self.build_spec = {"frame_stack": frame_stack}
        self.net = nn.Sequential(nn.Conv2d(3 * frame_stack, 4, 3, stride=2), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(16, out_size))

    def encode_preprocessed(self, x):
        return self.net(x)


def _obs(n=6, c=3, side=16):
    torch.manual_seed(1)
    return torch.rand(n, c, side, side)


# ---- ema_consistency --------------------------------------------------------

def _ema(enc, **params):
    (term,) = build_aux_losses([{"name": "ema_consistency", "weight": 1.0, **params}], enc)
    return term.loss


def test_ema_consistency_starts_at_zero_and_the_target_trains_nothing():
    enc = _Encoder()
    loss_fn = _ema(enc)
    x = _obs()
    value, metrics = loss_fn(AuxBatch(encoder=enc, obs=x, feats=enc.encode_preprocessed(x)))
    assert value.item() == pytest.approx(0.0, abs=1e-12) and metrics["cos"].item() == pytest.approx(1.0)
    assert not [p for p in loss_fn.parameters() if p.requires_grad]


def test_ema_consistency_target_follows_by_momentum():
    enc = _Encoder()
    loss_fn = _ema(enc, momentum=0.9)
    start = enc.net[0].weight.detach().clone()
    with torch.no_grad():
        enc.net[0].weight.add_(1.0)
    x = _obs()
    loss_fn(AuxBatch(encoder=enc, obs=x, feats=enc.encode_preprocessed(x)))
    # target <- 0.9 * target + 0.1 * encoder
    assert torch.allclose(loss_fn.target.net[0].weight, start + 0.1, atol=1e-6)
    assert enc.net[0].weight.requires_grad and not loss_fn.target.net[0].weight.requires_grad


@pytest.mark.parametrize("distance", ["mse", "cosine"])
def test_ema_consistency_pulls_the_encoder_toward_the_target(distance):
    enc = _Encoder()
    loss_fn = _ema(enc, distance=distance, momentum=0.9)
    with torch.no_grad():
        enc.net[-1].weight.add_(0.5)
    x = _obs()
    value, _ = loss_fn(AuxBatch(encoder=enc, obs=x, feats=enc.encode_preprocessed(x)))
    assert value.item() > 0
    value.backward()
    assert enc.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in loss_fn.target.parameters())


def test_ema_consistency_target_stays_in_eval_mode():
    enc = _Encoder().train()
    loss_fn = _ema(enc)
    loss_fn.train()
    assert loss_fn.training and not loss_fn.target.training


def test_ema_consistency_uses_the_callers_precision():
    enc = _Encoder()
    loss_fn = _ema(enc)
    used = []

    class Ctx:
        def __enter__(self):
            used.append(True)

        def __exit__(self, *_):
            return False

    x = _obs()
    loss_fn(AuxBatch(encoder=enc, obs=x, feats=enc.encode_preprocessed(x), autocast=Ctx))
    assert used == [True]


@pytest.mark.parametrize("spec", [{"momentum": 1.0}, {"momentum": -0.1}, {"distance": "l1"}])
def test_ema_consistency_bad_params(spec):
    with pytest.raises(ValueError, match="ema_consistency"):
        aux_loss_classes([{"name": "ema_consistency", "weight": 1.0, **spec}])


# ---- reconstruction ---------------------------------------------------------

def _recon(enc, **params):
    (term,) = build_aux_losses([{"name": "reconstruction", "weight": 1.0, **params}], enc)
    return term.loss


@pytest.mark.parametrize("size", [8, 16, 64])
@pytest.mark.parametrize("stack", [1, 3])
def test_reconstruction_output_matches_the_downsampled_frames(size, stack):
    enc = _Encoder(frame_stack=stack)
    loss_fn = _recon(enc, size=size)
    out = loss_fn.decode(torch.zeros(4, enc.out_size))
    assert out.shape == (4, 3 * stack, size, size)
    x = _obs(c=3 * stack)
    value, metrics = loss_fn(AuxBatch(encoder=enc, obs=x, feats=enc.encode_preprocessed(x)))
    assert value.ndim == 0 and metrics["psnr"].item() > 0


@pytest.mark.parametrize("grad", [True, False])
def test_reconstruction_grad_decides_whether_the_encoder_learns(grad):
    enc = _Encoder()
    loss_fn = _recon(enc, size=8, grad=grad)
    assert loss_fn.grad_features is grad
    x = _obs()
    value, _ = loss_fn(AuxBatch(encoder=enc, obs=x, feats=enc.encode_preprocessed(x)))
    value.backward()
    assert loss_fn.deconv[-1].weight.grad.abs().sum() > 0        # the decoder always learns
    has_grad = enc.net[-1].weight.grad is not None and enc.net[-1].weight.grad.abs().sum() > 0
    assert bool(has_grad) == grad


def test_reconstruction_latent_penalty_only_with_grad():
    enc = _Encoder()
    x = _obs()
    feats = enc.encode_preprocessed(x)
    torch.manual_seed(2)
    off = _recon(enc, size=8, grad=True, latent_l2=0.0)
    torch.manual_seed(2)
    on = _recon(enc, size=8, grad=True, latent_l2=1.0)
    torch.manual_seed(2)
    probe = _recon(enc, size=8, grad=False, latent_l2=1.0)
    base = off(AuxBatch(encoder=enc, obs=x, feats=feats))[0].item()
    expected = 0.5 * feats.pow(2).sum(1).mean().item()
    assert on(AuxBatch(encoder=enc, obs=x, feats=feats))[0].item() == pytest.approx(base + expected, rel=1e-5)
    assert probe(AuxBatch(encoder=enc, obs=x, feats=feats))[0].item() == pytest.approx(base, rel=1e-5)


def test_reconstruction_learns_to_decode():
    enc = _Encoder()
    loss_fn = _recon(enc, size=8)
    x = _obs()
    opt = torch.optim.Adam(list(enc.parameters()) + list(loss_fn.parameters()), lr=1e-3)
    first = None
    for _ in range(100):
        value, _ = loss_fn(AuxBatch(encoder=enc, obs=x, feats=enc.encode_preprocessed(x)))
        first = value.item() if first is None else first
        opt.zero_grad()
        value.backward()
        opt.step()
    assert value.item() < 0.5 * first


@pytest.mark.parametrize("spec", [{"size": 6}, {"size": 48}, {"size": 4}, {"grad": 1},
                                  {"latent_l2": -1.0}, {"channels": 0}])
def test_reconstruction_bad_params(spec):
    with pytest.raises(ValueError, match="reconstruction"):
        aux_loss_classes([{"name": "reconstruction", "weight": 1.0, **spec}])


# ---- optimizer groups ---------------------------------------------------------

def test_aux_params_get_their_own_learning_rate():
    enc = _Encoder()
    terms = build_aux_losses([{"name": "reconstruction", "weight": 1.0, "size": 8},
                              {"name": "ema_consistency", "weight": 1.0}], enc)
    groups = encoder_param_groups(enc, terms, 1e-3)
    opt = torch.optim.Adam(groups, lr=1e-5)
    assert [g["lr"] for g in opt.param_groups] == [1e-5, 1e-3]
    decoder = {id(p) for p in terms[0].loss.parameters()}
    assert {id(p) for p in groups[1]["params"]} == decoder       # not the EMA target's copies
    assert {id(p) for p in groups[0]["params"]} == {id(p) for p in enc.parameters()}


def test_aux_lr_defaults_to_the_encoder_rate_and_empty_aux_has_one_group():
    enc = _Encoder()
    terms = build_aux_losses([{"name": "reconstruction", "weight": 1.0, "size": 8}], enc)
    assert [g["lr"] for g in torch.optim.Adam(encoder_param_groups(enc, terms), lr=1e-5).param_groups] == [1e-5, 1e-5]
    assert len(encoder_param_groups(enc, [])) == 1


def _grads(groups, *scales):
    # Random, not constant: float32 norms of large constant tensors drift by
    # ~1e-4, in clip_grad_norm_ too, which would swamp what these tests check.
    torch.manual_seed(3)
    for g, s in zip(groups, scales):
        for p in g["params"]:
            p.grad = torch.randn_like(p) * s


def _norm(group):
    return torch.cat([p.grad.flatten() for p in group["params"]]).double().norm().item()


@pytest.mark.parametrize("limit", [1.0, 2.5])
def test_clipping_is_per_group_and_reports_the_encoders_norm(limit):
    enc = _Encoder()
    terms = build_aux_losses([{"name": "reconstruction", "weight": 1.0, "size": 8}], enc)
    groups = encoder_param_groups(enc, terms)
    _grads(groups, 1e-3, 1e3)                 # a huge decoder gradient, a small encoder one
    expected = _norm(groups[0])
    assert expected < limit
    assert clip_encoder_grads(groups, limit) == pytest.approx(expected, rel=1e-4)
    assert _norm(groups[0]) == pytest.approx(expected, rel=1e-6)       # the encoder's is untouched
    assert _norm(groups[1]) == pytest.approx(limit, rel=1e-4)


def test_clipping_applies_to_the_encoder_and_null_means_none():
    enc = _Encoder()
    groups = encoder_param_groups(enc, [])
    _grads(groups, 10.0)
    before = clip_encoder_grads(groups, None)
    assert _norm(groups[0]) == pytest.approx(before, rel=1e-5)
    clip_encoder_grads(groups, 1.0)
    assert _norm(groups[0]) == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("train, passes", [(True, 4), (False, 1)])
def test_moco_splits_the_batch_only_in_train_mode(train, passes):
    enc = _Encoder().train(train)
    (term,) = build_aux_losses([{"name": "moco", "weight": 1.0, "dim": 4, "queue_size": 16, "bn_splits": 4}], enc)
    calls = []
    orig = enc.encode_preprocessed
    enc.encode_preprocessed = lambda x: (calls.append(len(x)), orig(x))[1]
    x = _obs(n=8)
    term.loss(AuxBatch(encoder=enc, views=[x, x]))
    assert len(calls) == passes and sum(calls) == 8
