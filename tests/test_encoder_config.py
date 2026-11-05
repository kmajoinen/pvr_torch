"""Config-resolution tests: compose the real Hydra configs with overrides and
check what src/encoder_config.py decides, without building any env or
encoder. Includes the regression for augmentation silently doing nothing on
the frozen fast path."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException

from src.encoder_config import encoder_settings, resolve_buffer_obs

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
SAC_GRADS = ("none", "critic", "actor", "both")
BC_GRADS = ("none", "policy")


def _cfg(name, overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def _sac(*overrides):
    cfg = _cfg("config_sac", overrides)
    s = encoder_settings(cfg, SAC_GRADS)
    return s, resolve_buffer_obs(cfg, s)


def test_default_is_frozen_fast_path():
    s, (pixels, reason) = _sac()
    assert not s.trains and s.aug_name is None
    assert not pixels and reason.startswith("features (auto")


def test_augmentation_alone_switches_to_pixels():
    # Regression: augmentation used to be accepted on the features path and
    # never applied, while the log said it was on.
    s, (pixels, reason) = _sac("augmentation=random_shift")
    assert not s.trains
    assert pixels and "random_shift" in reason


def test_augmentation_with_forced_features_raises():
    with pytest.raises(ValueError, match="buffer_obs=features"):
        _sac("augmentation=color_jitter", "buffer_obs=features")


@pytest.mark.parametrize("grads", ["critic", "actor", "both"])
def test_rl_grads_train_the_encoder(grads):
    s, (pixels, _) = _sac(f"encoder.grads={grads}")
    assert s.trains and pixels


def test_aux_only_trains_the_encoder():
    s, (pixels, reason) = _sac("augmentation=random_shift", "augmentation.stage=aux",
                               "encoder.aux=[{name:aug_consistency,weight:0.1},{name:l2sp,weight:1.0}]")
    assert s.trains and s.grads == "none" and s.aux_names == ["aug_consistency", "l2sp"]
    assert pixels and "l2sp" in reason


@pytest.mark.parametrize("extra", [
    [],                                                        # trainable=all: a costly frozen run
    ["+embedding.layer=layer2", "+embedding.head=linear", "encoder.trainable=head"],  # random head
])
def test_aux_that_reads_no_observations_cannot_train_alone(extra):
    with pytest.raises(ValueError, match="nothing moves the encoder"):
        _sac("embedding=resnet18", "encoder.aux=[{name:l2sp,weight:1.0}]", *extra)


@pytest.mark.parametrize("aux", ["[{name:ema_consistency,weight:1.0}]",
                                 "[{name:ema_consistency,weight:1.0},{name:l2sp,weight:1.0}]"])
def test_anchor_losses_cannot_train_alone(aux):
    # ema_consistency reads observations, but only pulls the encoder toward
    # its own moving average: zero gradient until something else moves it.
    with pytest.raises(ValueError, match="nothing moves the encoder"):
        _sac("embedding=resnet18", f"encoder.aux={aux}")


@pytest.mark.parametrize("aux, extra", [
    ("[{name:ema_consistency,weight:1.0}]", ["encoder.grads=critic"]),
    ("[{name:ema_consistency,weight:1.0},{name:reconstruction,weight:1.0}]", []),
    ("[{name:reconstruction,weight:1.0,grad:false}]", []),     # a probe on a frozen encoder
    ("[{name:mae,weight:1.0}]", ["embedding=vit_s16"]),
])
def test_rl_aux_losses_that_train_or_probe(aux, extra):
    s, (pixels, _) = _sac("embedding=resnet18", f"encoder.aux={aux}", *extra)
    assert s.trains and pixels
    assert s.aux_views == (1 if "mae" in aux else 0) and not s.views_aug


def test_mae_views_follow_the_aux_stage():
    s, _ = _sac("embedding=vit_s16", "encoder.aux=[{name:mae,weight:1.0}]",
                "augmentation=random_shift", "augmentation.stage=aux")
    assert s.aux_views == 1 and s.views_aug and not s.task_aug


def test_l2sp_with_rl_gradients_is_fine():
    s, _ = _sac("encoder.grads=critic", "encoder.aux=[{name:l2sp,weight:1.0}]")
    assert s.trains


def test_slow_frozen_control():
    s, (pixels, reason) = _sac("buffer_obs=pixels")
    assert not s.trains and pixels and "slow-frozen" in reason


def test_old_finetune_key_fails_to_compose():
    with pytest.raises(ConfigCompositionException):
        _cfg("config_sac", ["finetune.enabled=true"])


def test_appended_finetune_key_raises_with_migration_message():
    cfg = _cfg("config_sac", ["+finetune.enabled=true"])
    with pytest.raises(ValueError, match="encoder.grads"):
        encoder_settings(cfg, SAC_GRADS)


@pytest.mark.parametrize("override", [
    "encoder.grads=critic",
    "augmentation=random_shift",
    "encoder.aux=[{name:l2sp,weight:1.0}]",
])
def test_encoder_settings_need_an_embedding(override):
    cfg = _cfg("config_sac", ["embedding=none", override])
    with pytest.raises(ValueError, match="no encoder"):
        encoder_settings(cfg, SAC_GRADS)


def test_state_mode_stores_state():
    s, (pixels, reason) = _sac("embedding=none")
    assert not pixels and reason.startswith("state")
    with pytest.raises(ValueError, match="needs an embedding"):
        _sac("embedding=none", "buffer_obs=pixels")


CONSISTENCY = "encoder.aux=[{name:aug_consistency,weight:0.1}]"


@pytest.mark.parametrize("stage, task_aug", [("aux", False), ("both", True)])
def test_aux_stages_with_a_reader(stage, task_aug):
    s, (pixels, _) = _sac("augmentation=random_shift", f"augmentation.stage={stage}", CONSISTENCY)
    assert s.trains and pixels
    assert s.task_aug is task_aug
    assert s.aux_needs_clean and s.aux_needs_aug


def test_stage_rl_routes_to_task_only():
    s, _ = _sac("augmentation=random_shift")
    assert s.task_aug and not s.aux_needs_aug


@pytest.mark.parametrize("stage", ["aux", "both"])
def test_aux_stage_without_a_reader_raises(stage):
    # l2sp doesn't read observations, so nobody would consume the augmented view.
    with pytest.raises(ValueError, match="none of encoder.aux"):
        _sac("augmentation=random_shift", f"augmentation.stage={stage}",
             "encoder.aux=[{name:l2sp,weight:1.0}]")


def test_reader_needs_aux_or_both_stage():
    with pytest.raises(ValueError, match="stage=aux or both"):
        _sac("augmentation=random_shift", CONSISTENCY)          # stage=rl
    with pytest.raises(ValueError, match="stage=aux or both"):
        _sac(CONSISTENCY)                                       # no augmentation


def test_stage_without_augmentation_raises():
    with pytest.raises(ValueError, match="no effect"):
        _sac("augmentation.stage=aux")


def test_invalid_values_raise():
    with pytest.raises(ValueError, match="encoder.grads"):
        _sac("encoder.grads=policy")
    with pytest.raises(ValueError, match="buffer_obs"):
        _sac("buffer_obs=sometimes")
    with pytest.raises(ValueError, match="Unknown aux loss"):
        _sac("encoder.aux=[{name:nope,weight:1.0}]")


@pytest.mark.parametrize("aux, exc", [
    ("{name:aug_consistency,weight:0.1,targt:clean}", TypeError),
    ("{name:aug_consistency,weight:0.1,target:clen}", ValueError),
    ("{name:l2sp,weight:1.0,scale:2}", TypeError),
])
def test_bad_aux_params_fail_before_the_encoder_loads(aux, exc):
    with pytest.raises(exc):
        _sac("augmentation=random_shift", "augmentation.stage=both", f"encoder.aux=[{aux}]")


def test_linear_head_on_frozen_backbone():
    s, (pixels, _) = _sac("embedding=resnet18", "+embedding.layer=layer2", "+embedding.head=linear",
                          "encoder.grads=critic", "encoder.trainable=head")
    assert s.trains and s.trainable == "head" and pixels


def test_linear_head_needs_a_training_encoder():
    with pytest.raises(ValueError, match="random"):
        _sac("embedding=resnet18", "+embedding.layer=layer2", "+embedding.head=linear")


def test_trainable_head_needs_a_linear_head():
    with pytest.raises(ValueError, match="trainable=head"):
        _sac("embedding=resnet18", "+embedding.layer=layer2", "encoder.grads=critic",
             "encoder.trainable=head")


@pytest.mark.parametrize("overrides, match", [
    (["+embedding.layer=0"], "embedding.layer"),
    (["+embedding.head=maxpool"], "embedding.head"),
    (["+embedding.head=avgpool", "+embedding.head_dim=64"], "head_dim"),
])
def test_layer_head_field_validation(overrides, match):
    from src.encoder_config import embedding_kwargs
    with pytest.raises(ValueError, match=match):
        embedding_kwargs(_cfg("config_sac", ["embedding=resnet18", *overrides]))


def test_bn_adapt_forces_pixels_without_training():
    s, (pixels, reason) = _sac("encoder.bn=adapt")
    assert s.bn_adapt and not s.trains
    assert pixels and "BatchNorm" in reason
    with pytest.raises(ValueError, match="encoder.bn"):
        _sac("encoder.bn=sometimes")
    with pytest.raises(ValueError, match="no encoder"):
        _sac("embedding=none", "encoder.bn=adapt")


def test_strong_view_settings(tmp_path):
    s, (pixels, reason) = _sac("augmentation=pieg", f"augmentation.strong.params.image_dir={tmp_path}")
    assert s.strong_name == "random_overlay" and s.strong_weight == 0.5
    assert s.aug_name == "random_shift" and pixels and "strong view" in reason
    with pytest.raises(ValueError, match="strong.weight"):
        _sac("augmentation=pieg", f"augmentation.strong.params.image_dir={tmp_path}",
             "augmentation.strong.weight=0")


@pytest.mark.parametrize("aug, key", [
    ("augmentation=random_overlay", "augmentation.params.image_dir"),
    ("augmentation=pieg", "augmentation.strong.params.image_dir"),
])
def test_unset_augmentation_settings_fail_at_startup(aug, key):
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        _sac(aug)


@pytest.mark.parametrize("overrides, match", [
    (["+embedding.frame_stack=0"], "frame_stack"),
    (["+embedding.frame_fusion=diff"], "needs embedding.frame_stack"),
    (["+embedding.frame_stack=3", "+embedding.frame_fusion=channels"], "only works with the `random`"),
    (["+embedding.frame_stack=3", "+embedding.frame_fusion=avg"], "frame_fusion"),
])
def test_frame_field_validation(overrides, match):
    from src.encoder_config import embedding_kwargs
    with pytest.raises(ValueError, match=match):
        embedding_kwargs(_cfg("config_sac", ["embedding=resnet18", *overrides]))


def test_bc_grads_vocabulary():
    cfg = _cfg("config_bc", ["embedding=resnet18", "encoder.grads=policy"])
    assert encoder_settings(cfg, BC_GRADS).trains
    cfg = _cfg("config_bc", ["embedding=resnet18", "encoder.grads=critic"])
    with pytest.raises(ValueError, match="encoder.grads"):
        encoder_settings(cfg, BC_GRADS)


def test_bc_default_is_frozen():
    s = encoder_settings(_cfg("config_bc"), BC_GRADS)
    assert not s.trains and s.aug_name is None


def test_moco_rejects_adapting_batchnorm():
    with pytest.raises(ValueError, match="key encoder"):
        _sac("embedding=resnet18", "encoder.aux=[{name:moco,weight:1.0}]", "encoder.bn=adapt",
             "augmentation=random_shift", "augmentation.stage=aux")


def test_bc_takes_the_new_aux_losses():
    cfg = _cfg("config_bc", ["embedding=resnet18", "encoder.aux=[{name:reconstruction,weight:1.0}]"])
    s = encoder_settings(cfg, BC_GRADS)
    assert s.trains and s.aux_needs_clean


@pytest.mark.parametrize("overrides, match", [
    (["encoder.grads=critic", "encoder.aux_lr=1e-3"], "encoder.aux is empty"),
    (["encoder.aux=[{name:reconstruction,weight:1.0}]", "encoder.aux_lr=-1.0"], "positive number"),
])
def test_aux_lr_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        _sac("embedding=resnet18", *overrides)


def test_aux_lr_accepted_with_aux_losses():
    s, _ = _sac("embedding=resnet18", "encoder.aux=[{name:reconstruction,weight:1.0}]", "encoder.aux_lr=1e-3")
    assert s.trains


def test_ema_consistency_rejects_adapting_batchnorm():
    with pytest.raises(ValueError, match="running ones"):
        _sac("embedding=resnet18", "encoder.grads=critic", "encoder.bn=adapt",
             "encoder.aux=[{name:ema_consistency,weight:1.0}]")


def test_drqv2_inherits_aux_lr():
    cfg = _cfg("config_drqv2", ["embedding=resnet18", "encoder.aux_lr=1e-3",
                                "encoder.aux=[{name:reconstruction,weight:1.0}]"])
    assert cfg.encoder.aux_lr == 1e-3 and encoder_settings(cfg, SAC_GRADS).trains
