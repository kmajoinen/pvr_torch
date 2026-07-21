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

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnvWrapper

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


class FrozenPVRVecWrapper(VecEnvWrapper):
    """
    Encodes frames from a VecEnv into feature vectors with a frozen
    EmbeddingNet, once per env step, outside the SB3 policy entirely -- so
    the replay buffer stores compact features and gradient steps never
    touch the encoder. This is what makes mode=pvr_fast fast: mode=pvr puts
    the same frozen encoder inside PVRFeaturesExtractor, which still gets
    re-run on every gradient step's sampled minibatch even though its
    weights never change -- ~5 encoder forwards of a 256-image batch per
    gradient step (actor, critic, critic_target, ...), i.e. >1000 encoder
    image-forwards per env step. Here, encoding happens exactly once per
    env step and the cost of a gradient step is just a tiny MLP update.
    Pair with LayerNormExtractor on an MlpPolicy.

    Only valid for the frozen case -- finetuning fundamentally requires
    re-encoding replayed pixels with the *current* weights on every
    gradient step, since a cached feature vector describes an encoder that
    may no longer exist the moment weights change. Use PVRFeaturesExtractor
    (freeze=False) for finetuning instead; that cost is inherent to
    finetuning, not an implementation gap here.

    Structurally immune to the aliasing/polyak-corruption bug fixed
    elsewhere in this file: SB3's hard BatchNorm-stats sync only walks
    modules registered under model.policy/critic/critic_target, and this
    encoder is never passed as a features_extractor -- SB3 never
    constructs a second instance of it, so there's nothing to alias.

    model.save() does not capture this encoder (it was never part of
    model.policy) -- it's a deterministic function of embedding_name,
    already reproducible via the Hydra config. Re-wrap eval/deployment
    envs with the same embedding_name.

    IMPORTANT: wrap the *outer* VecEnv here (after make_vec_env returns),
    not per-sub-env inside the env-building lambda -- one EmbeddingNet
    instance must serve every sub-env via a single batched forward call.
    Wrapping per-sub-env would reconstruct the encoder once per env, and
    with SubprocVecEnv (n_envs>1) once per subprocess, unable to share GPU
    memory across process boundaries.
    """

    def __init__(
        self,
        venv,
        embedding_name: str,
        disable_cuda: bool = False,
        model_dir: str = None,
        amp_bf16: bool = True,
    ):
        embedding_net = EmbeddingNet(
            embedding_name, pretrained=True, train=False, disable_cuda=disable_cuda,
            model_dir=model_dir,
        )
        embedding_net.eval()
        obs_space = gym.spaces.Box(-np.inf, np.inf, (int(embedding_net.out_size),), np.float32)
        super().__init__(venv, observation_space=obs_space)
        self.embedding_net = embedding_net
        self.amp_bf16 = amp_bf16 and embedding_net.device.type == "cuda"

    @th.inference_mode()
    def _encode(self, obs: np.ndarray) -> np.ndarray:
        # obs is already (N, C, H, W) uint8: make_pixel_env's make_gym_env
        # pipeline applies ChannelsFirstWrapper per sub-env before
        # DummyVecEnv/SubprocVecEnv stacks them -- same contract
        # PVRFeaturesExtractor.forward() consumes. No permute here.
        x = th.as_tensor(obs)  # EmbeddingNet.encode() moves it to its own device
        with th.autocast("cuda", dtype=th.bfloat16, enabled=self.amp_bf16):
            feats = self.embedding_net.encode(x)
        feats = feats.float().cpu().numpy()
        if not np.isfinite(feats).all():
            raise RuntimeError(
                f"Non-finite embedding output from {self.embedding_net.embedding_name!r} "
                "during frozen-fast encoding -- checkpoint/cache likely corrupted."
            )
        return feats

    def reset(self) -> np.ndarray:
        return self._encode(self.venv.reset())

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # SB3 reads the true final obs of an episode from
        # infos[i]["terminal_observation"]; must be re-encoded too, or the
        # replay buffer gets a raw-pixel array where a feature vector is
        # expected.
        for info in infos:
            if "terminal_observation" in info:
                term = np.asarray(info["terminal_observation"])[None]
                info["terminal_observation"] = self._encode(term)[0]
        return self._encode(obs), rewards, dones, infos


class LayerNormExtractor(BaseFeaturesExtractor):
    """
    Drop-in features_extractor for MlpPolicy on top of FrozenPVRVecWrapper.
    Raw frozen-encoder features have large, arbitrary scale/mean -- same
    reasoning as PVRFeaturesExtractor.feature_norm.
    """

    def __init__(self, observation_space: gym.spaces.Box):
        dim = int(np.prod(observation_space.shape))
        super().__init__(observation_space, features_dim=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.norm(observations)
