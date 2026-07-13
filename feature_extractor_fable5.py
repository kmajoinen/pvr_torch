"""
Pretrained-visual-representation (PVR) encoders for SB3.

Two ways to plug a pretrained encoder into SAC, depending on whether it's
frozen:

1. FROZEN (standard PVR-eval protocol) -> `PVREncoderWrapper`
   A VecEnvWrapper that encodes each frame ONCE, at env-step time, and hands
   SAC a flat feature vector. SAC then runs `MlpPolicy` on features: the
   replay buffer stores features instead of pixels, and gradient steps never
   touch the encoder. This is the fast path -- with a frozen encoder the
   features are a fixed function of the frame, so re-encoding replayed
   pixels on every gradient step (what a features_extractor does) is pure
   waste: ~5 encoder forwards of a 256-batch per gradient step, i.e. >1000
   encoder image-forwards per env step, vs exactly 1 here.
   Pair it with `LayerNormExtractor` on the policy (see below).

2. FINETUNED -> `PVRFeaturesExtractor`
   The usual SB3 features-extractor route, for when the encoder trains
   jointly and features genuinely change every gradient step.

   NOTE: the extractor deep-copies the encoder it is given. SB3 constructs
   one extractor instance per network (actor/critic + critic_target) from
   the same `features_extractor_kwargs`; if they shared one nn.Module, the
   critic and target-critic parameter lists would alias the SAME tensors,
   and SB3's in-place polyak update then corrupts them (tau=1.0 on the
   BatchNorm running stats zeroes them outright -> inf/NaN activations).
"""
import copy
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnvWrapper

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RESNET_EMBED_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}
AVAILABLE_BACKBONES = list(RESNET_EMBED_DIMS.keys()) + ["mae_base"]


class _MAEEncoderAdapter(nn.Module):
    """MAE forward_encoder(x, mask_ratio) -> plain forward(x) -> (B, D) CLS."""

    def __init__(self, mae_model: nn.Module):
        super().__init__()
        self.mae_model = mae_model

    def forward(self, x: th.Tensor) -> th.Tensor:
        latent, *_ = self.mae_model.forward_encoder(x, mask_ratio=0.0)
        return latent[:, 0, :]


def _check_weights_finite(encoder: nn.Module, backbone: str):
    """Cheap one-off load-time check: fail loudly on a corrupted/truncated
    checkpoint cache instead of producing garbage mid-run."""
    bad = [
        name for name, t in
        list(encoder.named_parameters()) + list(encoder.named_buffers())
        if not th.isfinite(t).all()
    ]
    if bad:
        raise RuntimeError(
            f"Non-finite value(s) in {backbone} weights after loading: "
            f"{bad[:5]} -- checkpoint/cache likely corrupted, delete and re-download."
        )


def build_encoder(backbone: str, models_dir: str = "models"):
    """Returns (encoder, embed_dim, input_size) for a backbone name."""
    if backbone == "mae_base":
        from src.vision_models.mae import mae_vit_base_patch16

        mae_model = mae_vit_base_patch16()
        ckpt = th.load(
            os.path.join(models_dir, "mae_pretrain_vit_base.pth"),
            map_location="cpu",
        )
        mae_model.load_state_dict(ckpt["model"], strict=False)
        _check_weights_finite(mae_model, backbone)
        return _MAEEncoderAdapter(mae_model), 768, 224

    if backbone in RESNET_EMBED_DIMS:
        import torchvision.models as tv_models

        encoder = getattr(tv_models, backbone)(weights="IMAGENET1K_V1")
        encoder.fc = nn.Identity()
        _check_weights_finite(encoder, backbone)
        return encoder, RESNET_EMBED_DIMS[backbone], 224

    raise ValueError(f"Unknown backbone {backbone!r}, expected one of {AVAILABLE_BACKBONES}")


def _imagenet_preprocess(x: th.Tensor, mean: th.Tensor, std: th.Tensor,
                         input_size: int) -> th.Tensor:
    """x: float (B, 3, H, W) in [0, 255] -> normalized (B, 3, S, S)."""
    x = x / 255.0
    if x.shape[-1] != input_size or x.shape[-2] != input_size:
        x = F.interpolate(x, size=(input_size, input_size),
                          mode="bilinear", align_corners=False)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Fast path: frozen encoder as a VecEnv wrapper
# ---------------------------------------------------------------------------

class PVREncoderWrapper(VecEnvWrapper):
    """
    Encodes uint8 HWC pixel observations from a VecEnv into (embed_dim,)
    float32 features with a frozen encoder, once per env step, on GPU.

    The encoder lives entirely outside the SB3 policy: no aliasing with
    critic_target, nothing for polyak_update to touch, no per-gradient-step
    re-encoding, and the replay buffer stores compact feature vectors.
    """

    def __init__(
        self,
        venv,
        encoder: nn.Module,
        embed_dim: int,
        input_size: int = 224,
        device: str = "cuda",
        amp_bf16: bool = True,
    ):
        obs_space = gym.spaces.Box(-np.inf, np.inf, (embed_dim,), np.float32)
        super().__init__(venv, observation_space=obs_space)
        self.device = th.device(device)
        self.input_size = input_size
        self.amp_bf16 = amp_bf16 and self.device.type == "cuda"
        self.encoder = encoder.to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._mean = th.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = th.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @th.inference_mode()
    def _encode(self, obs: np.ndarray) -> np.ndarray:
        # VecEnv delivers (N, H, W, C) uint8
        x = th.as_tensor(obs, device=self.device).permute(0, 3, 1, 2).float()
        x = _imagenet_preprocess(x, self._mean, self._std, self.input_size)
        with th.autocast("cuda", dtype=th.bfloat16, enabled=self.amp_bf16):
            feats = self.encoder(x)
        return feats.float().cpu().numpy()

    def reset(self) -> np.ndarray:
        return self._encode(self.venv.reset())

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # SB3 reads the true final obs of an episode from here; encode it too
        # so the replay buffer stays consistent (one extra frame per episode).
        for info in infos:
            if "terminal_observation" in info:
                term = np.asarray(info["terminal_observation"])[None]
                info["terminal_observation"] = self._encode(term)[0]
        return self._encode(obs), rewards, dones, infos


class LayerNormExtractor(BaseFeaturesExtractor):
    """
    Drop-in features_extractor for MlpPolicy on top of PVREncoderWrapper.
    Raw frozen-encoder features have large, arbitrary scale/mean; feeding
    them straight into a fresh SAC MLP blows up mean/log_std within the
    first gradient steps. A learnable LayerNorm fixes the scale. (LayerNorm,
    not BatchNorm: rollouts call forward() with batch size n_envs, which
    BatchNorm1d rejects in train mode for n_envs=1.)
    """

    def __init__(self, observation_space: gym.spaces.Box):
        dim = int(np.prod(observation_space.shape))
        super().__init__(observation_space, features_dim=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.norm(observations)


# ---------------------------------------------------------------------------
# Finetune path: encoder inside the policy
# ---------------------------------------------------------------------------

class PVRFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3 features extractor wrapping an arbitrary image encoder, for the
    finetuning case (encoder trains with the RL head). For a frozen encoder
    prefer PVREncoderWrapper -- it is orders of magnitude faster.

    Expects float CHW observations in [0, 255] (set normalize_images=False
    on the policy); does /255 + ImageNet normalization + resize itself.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        encoder: nn.Module,
        embed_dim: int,
        input_size: int = 224,
        freeze: bool = False,
    ):
        super().__init__(observation_space, features_dim=embed_dim)
        # deepcopy: SB3 builds one extractor per network (incl. critic_target)
        # from the same kwargs; sharing one module aliases critic and target
        # parameters, and the in-place polyak update then corrupts them.
        self.encoder = copy.deepcopy(encoder)
        self.input_size = input_size
        self.freeze = freeze

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        self.register_buffer("mean", th.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("std", th.tensor(IMAGENET_STD).view(1, 3, 1, 1),
                             persistent=False)
        # See LayerNormExtractor docstring for why this is needed.
        self.feature_norm = nn.LayerNorm(embed_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = _imagenet_preprocess(observations, self.mean, self.std, self.input_size)
        if self.freeze:
            with th.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        return self.feature_norm(feats)

    def train(self, mode: bool = True):
        # keep a frozen encoder in eval() even when SB3 calls policy.train()
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self
