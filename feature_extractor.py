"""
PVRFeaturesExtractor: plug any of src/embeddings.py's pretrained visual
encoders into Stable-Baselines3's SAC/PPO/etc. as a drop-in replacement for
the default CNN. Backed by the same EmbeddingNet used by train_bc.py, so a
new PVR checkpoint needs a new configs/embedding/<name>.yaml, not new code
here.

Usage
-----
    from feature_extractor import PVRFeaturesExtractor

    policy_kwargs = dict(
        features_extractor_class=PVRFeaturesExtractor,
        features_extractor_kwargs=dict(embedding_name="resnet50", freeze=True),
        net_arch=[256, 256],          # actor/critic MLP heads on top
        normalize_images=False,        # extractor does its own normalization
    )
    model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, ...)
"""

import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.embeddings import EmbeddingNet


class PVRFeaturesExtractor(BaseFeaturesExtractor):
    """
    Wraps EmbeddingNet so it can be used as SB3's `features_extractor` for
    both the actor and the critic.

    Constructs its own EmbeddingNet from `embedding_name` (a string) rather
    than accepting a pre-built nn.Module -- SB3 calls its features-extractor
    class + kwargs multiple times (once for the shared actor/critic
    extractor, again for critic_target's separate one), reusing the same
    kwargs dict each time. Passing a live module through kwargs would make
    every one of those instances alias the SAME underlying weights, which
    corrupts them the moment SB3 hard-syncs BatchNorm running stats between
    critic and critic_target (self-referential polyak_update on aliased
    tensors). Building fresh from a string here means every instance gets
    independent weights automatically, with no aliasing possible.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        Pixel observation space from the (possibly VecTransposeImage-wrapped) env.
    embedding_name : str
        Name passed to EmbeddingNet / src.embeddings._get_embedding, e.g.
        "resnet18", "resnet50", "moco_aug", "mae_base". See
        configs/embedding/*.yaml for the names in active use.
    freeze : bool
        If True (the standard PVR-eval protocol), encoder params are frozen
        and kept in eval() mode -- you're only training the RL head on top
        of a fixed representation. Set False to finetune the encoder jointly.
    disable_cuda : bool
        Forces EmbeddingNet onto CPU regardless of GPU availability.
    model_dir : str, optional
        If given, look here for a checkpoint file named
        <embedding_name>.<any extension> instead of the default location
        (e.g. to load from cluster scratch storage, separate from where
        the code itself lives). See src.embeddings._resolve_model_dir_override.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        embedding_name: str,
        freeze: bool = True,
        disable_cuda: bool = False,
        model_dir: str = None,
    ):
        embedding_net = EmbeddingNet(
            embedding_name, pretrained=True, train=not freeze, disable_cuda=disable_cuda,
            model_dir=model_dir,
        )
        super().__init__(observation_space, features_dim=int(embedding_net.out_size))
        self.embedding_net = embedding_net
        self.freeze = freeze

        # Raw frozen-encoder features (e.g. ResNet avgpool output) have
        # large, arbitrary scale/mean -- feeding them straight into a
        # freshly-initialized SAC actor/critic MLP blows up mean/log_std
        # within the first few gradient steps post-learning_starts (NaNs).
        # LayerNorm (not BatchNorm) because rollout collection calls
        # forward() with batch size 1 (single env, one step at a time),
        # which BatchNorm1d rejects in train mode. Its affine params train
        # normally with the rest of the RL head even though encoder is frozen.
        self.feature_norm = nn.LayerNorm(self.features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # SB3 hands this a float32 tensor with values in [0, 255] (since
        # normalize_images=False) -- EmbeddingNet's transform pipeline needs
        # an actual integer dtype to correctly rescale to [0, 1] internally.
        obs_uint8 = observations.to(th.uint8)
        feats = self.embedding_net.encode(obs_uint8)

        if not th.isfinite(feats).all():
            raise RuntimeError(
                f"Non-finite embedding output from {self.embedding_net.embedding_name!r} "
                "-- check the checkpoint/cache for that backbone isn't corrupted."
            )

        return self.feature_norm(feats)

    def train(self, mode: bool = True):
        # keep a frozen encoder in eval() even when SB3 calls policy.train()
        super().train(mode)
        if self.freeze:
            self.embedding_net.eval()
        return self
