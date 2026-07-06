"""
PVRFeaturesExtractor: plug any pretrained visual encoder (ResNet, R3M, VC-1,
a pvr_torch model, ...) into Stable-Baselines3's SAC/PPO/etc. as a drop-in
replacement for the default CNN.

Usage
-----
    from feature_extractor import PVRFeaturesExtractor
    import torchvision.models as models

    encoder = models.resnet18(weights="IMAGENET1K_V1")
    encoder.fc = torch.nn.Identity()  # strip classifier -> 512-d features

    policy_kwargs = dict(
        features_extractor_class=PVRFeaturesExtractor,
        features_extractor_kwargs=dict(encoder=encoder, embed_dim=512,
                                        input_size=224, freeze=True),
        net_arch=[256, 256],          # actor/critic MLP heads on top
        normalize_images=False,        # extractor does its own normalization
    )
    model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, ...)

TODO: to use your own pvr_torch model instead of torchvision ResNet, just
pass any nn.Module with a forward(x: (B,3,H,W)) -> (B, embed_dim) signature.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PVRFeaturesExtractor(BaseFeaturesExtractor):
    """
    Wraps an arbitrary image encoder so it can be used as SB3's
    `features_extractor` for both the actor and the critic.

    SB3 hands this module the observation as a float tensor, channel-first,
    shape (B, C, H, W). Whether it's already divided by 255 depends on the
    `normalize_images` flag you set on the policy (see usage above) -- this
    class assumes it is NOT pre-normalized unless you pass
    `already_normalized=True`.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        Pixel observation space from the (possibly VecTransposeImage-wrapped) env.
    encoder : nn.Module
        Backbone mapping (B, 3, input_size, input_size) -> (B, embed_dim).
        For torchvision ResNets, replace `.fc` with `nn.Identity()` first.
    embed_dim : int
        Output dim of `encoder`. SB3 uses this to size the actor/critic heads.
    input_size : int
        Spatial resolution the encoder expects (224 for ImageNet-pretrained
        ResNets and most PVRs -- R3M, VC-1, VIP, LIV, etc).
    freeze : bool
        If True (the standard PVR-eval protocol), encoder params are frozen
        and kept in eval() mode -- you're only training the RL head on top
        of a fixed representation. Set False to finetune the encoder jointly.
    already_normalized : bool
        Set True if an upstream wrapper already delivers [0,1] ImageNet-
        normalized pixels. Otherwise this class does /255 + normalize itself.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        encoder: nn.Module,
        embed_dim: int,
        input_size: int = 224,
        freeze: bool = True,
        already_normalized: bool = False,
    ):
        super().__init__(observation_space, features_dim=embed_dim)
        self.encoder = encoder
        self.input_size = input_size
        self.freeze = freeze
        self.already_normalized = already_normalized

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.register_buffer(
            "mean", th.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", th.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def _preprocess(self, obs: th.Tensor) -> th.Tensor:
        x = obs
        if not self.already_normalized:
            x = x / 255.0
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )
        if not self.already_normalized:
            x = (x - self.mean) / self.std
        return x

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self._preprocess(observations)
        if self.freeze:
            with th.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        return feats

    def train(self, mode: bool = True):
        # keep a frozen encoder in eval() even when SB3 calls policy.train()
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self
