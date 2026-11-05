"""Routing tests for src/views.py: which view each consumer gets per stage,
gradient isolation, encode reuse (at most one extra pass), and the collapse
monitor."""

import pytest
import torch
import torch.nn as nn

from src.views import CollapseMonitor, encode_views


class Counter:
    """featurize stand-in: a linear encoder that records each call."""

    def __init__(self):
        torch.manual_seed(0)
        self.net = nn.Linear(12, 4)
        self.calls = []

    def __call__(self, x, grad):
        self.calls.append(grad)
        with torch.enable_grad() if grad else torch.no_grad():
            return self.net(x.flatten(1))


def _augment(x):
    return x + 1.0   # deterministic, so views are easy to tell apart


def _run(task_aug, task_grad, needs_clean, needs_aug, augment=_augment):
    f = Counter()
    x = torch.rand(5, 3, 2, 2)
    v = encode_views(f, f.net, x, augment, task_aug=task_aug, task_grad=task_grad,
                     aux_needs_clean=needs_clean, aux_needs_aug=needs_aug)
    return f, x, v


def test_stage_rl_task_sees_augmented_aux_untouched():
    f, x, v = _run(task_aug=True, task_grad=True, needs_clean=False, needs_aug=False)
    assert torch.equal(v.x_task, x + 1.0)
    assert v.aux.feats is None and v.aux.feats_aug is None
    assert len(f.calls) == 1


def test_stage_aux_task_sees_clean_one_extra_pass():
    f, x, v = _run(task_aug=False, task_grad=True, needs_clean=True, needs_aug=True)
    assert torch.equal(v.x_task, x)
    assert v.aux.feats is v.feats_task              # clean view shared with the task
    assert torch.equal(v.aux.obs_aug, x + 1.0)
    assert len(f.calls) == 2                        # task (shared) + augmented


def test_stage_both_shares_augmented_one_extra_pass():
    f, x, v = _run(task_aug=True, task_grad=True, needs_clean=True, needs_aug=True)
    assert v.aux.feats_aug is v.feats_task          # augmented view shared with the task
    assert torch.equal(v.aux.obs, x)
    assert len(f.calls) == 2


def test_shared_encode_keeps_task_gradient_out_of_the_encoder():
    # grads=none + aux reading the task's view: the encode carries gradient for
    # the aux loss, but the task must get a detached copy.
    f, _, v = _run(task_aug=False, task_grad=False, needs_clean=True, needs_aug=True)
    assert f.calls[0] is True
    assert not v.feats_task.requires_grad
    assert v.aux.feats.requires_grad and v.aux.feats_aug.requires_grad


def test_no_aux_needs_and_no_task_grad_builds_no_graph():
    f, _, v = _run(task_aug=True, task_grad=False, needs_clean=False, needs_aug=False)
    assert f.calls == [False]
    assert not v.feats_task.requires_grad


def _run_grads(task_aug, task_grad, grad_clean, grad_aug):
    f = Counter()
    x = torch.rand(5, 3, 2, 2)
    v = encode_views(f, f.net, x, _augment, task_aug=task_aug, task_grad=task_grad,
                     aux_needs_clean=True, aux_needs_aug=True,
                     aux_grad_clean=grad_clean, aux_grad_aug=grad_aug)
    return f, v


def test_target_view_builds_no_graph():
    # stage=both, aug_consistency target=clean: the clean view is only a target.
    f, v = _run_grads(task_aug=True, task_grad=True, grad_clean=False, grad_aug=True)
    assert f.calls == [True, False]
    assert not v.aux.feats.requires_grad and v.aux.feats_aug.requires_grad


def test_shared_view_reaches_aux_detached_when_it_is_the_target():
    # stage=aux (task sees clean, with gradient); target=clean: the aux loss
    # must get the shared clean encode detached, the task keeps its graph.
    f, v = _run_grads(task_aug=False, task_grad=True, grad_clean=False, grad_aug=True)
    assert v.feats_task.requires_grad
    assert not v.aux.feats.requires_grad and v.aux.feats_aug.requires_grad


def test_grad_through_unread_view_raises():
    f = Counter()
    with pytest.raises(ValueError, match="no aux loss reads"):
        encode_views(f, f.net, torch.rand(5, 3, 2, 2), _augment, task_aug=True, task_grad=False,
                     aux_needs_clean=False, aux_needs_aug=False, aux_grad_clean=True)


def test_augmented_view_without_augmentation_raises():
    with pytest.raises(ValueError, match="no augmentation"):
        _run(task_aug=True, task_grad=False, needs_clean=False, needs_aug=False, augment=None)


def test_collapse_monitor_warns_once(capsys):
    m = CollapseMonitor(warn_frac=0.1, warmup=0)
    m.update(torch.randn(64, 8), step=0)
    assert m.ratio == pytest.approx(1.0)
    m.update(torch.randn(64, 8) * 0.01, step=10, where="step 10")
    m.update(torch.randn(64, 8) * 0.01, step=20, where="step 20")
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1 and "step 10" in out
    assert m.ratio < 0.1


def test_collapse_monitor_skips_single_row_batches(capsys):
    m = CollapseMonitor(warn_frac=0.1, warmup=0)
    m.update(torch.randn(1, 8), step=0)                 # std of one row is NaN
    assert m.initial is None
    m.update(torch.randn(64, 8), step=1)
    m.update(torch.zeros(64, 8), step=2, where="step 2")
    assert "WARNING" in capsys.readouterr().out         # still armed


def test_collapse_baseline_waits_for_warmup(capsys):
    # A fresh head's first readings don't become the reference.
    m = CollapseMonitor(warn_frac=0.1, warmup=100)
    m.update(torch.randn(64, 8) * 100, step=0)          # pre-warmup: large, ignored
    assert m.initial is None and m.ratio != m.ratio     # NaN until the baseline exists
    m.update(torch.randn(64, 8), step=100)
    m.update(torch.randn(64, 8), step=200)
    assert m.ratio == pytest.approx(1.0, rel=0.3)
    assert "WARNING" not in capsys.readouterr().out


def _run_views(n, views_aug, task_aug=False, needs_aug=False, augment=None):
    f = Counter()
    x = torch.rand(5, 3, 2, 2)
    draws = []

    def counting(y):
        draws.append(1)
        return y + float(len(draws))       # each draw differs

    v = encode_views(f, f.net, x, counting if augment is None else augment, task_aug=task_aug,
                     task_grad=False, aux_needs_clean=False, aux_needs_aug=needs_aug,
                     aux_views=n, views_aug=views_aug)
    return x, v, draws


def test_views_are_clean_frames_without_an_aux_stage():
    x, v, draws = _run_views(1, views_aug=False)
    assert len(v.aux.views) == 1 and v.aux.views[0] is x and not draws


def test_views_are_independent_draws_with_an_aux_stage():
    x, v, draws = _run_views(2, views_aug=True)
    assert len(draws) == 2
    assert torch.equal(v.aux.views[0], x + 1.0) and torch.equal(v.aux.views[1], x + 2.0)


def test_first_view_reuses_the_updates_augmented_view():
    # stage=both: the task's draw is view 0, so only one more is drawn.
    x, v, draws = _run_views(2, views_aug=True, task_aug=True)
    assert len(draws) == 2 and v.aux.views[0] is v.x_task


def test_views_need_an_augmentation_when_drawn():
    f = Counter()
    with pytest.raises(ValueError, match="no augmentation"):
        encode_views(f, f.net, torch.rand(5, 3, 2, 2), None, task_aug=False, task_grad=False,
                     aux_needs_clean=False, aux_needs_aug=False, aux_views=2, views_aug=True)


def test_autocast_reaches_the_aux_batch():
    f = Counter()
    ctx = lambda: None    # noqa: E731
    v = encode_views(f, f.net, torch.rand(5, 3, 2, 2), None, task_aug=False, task_grad=False,
                     aux_needs_clean=False, aux_needs_aug=False, autocast=ctx)
    assert v.aux.autocast is ctx
