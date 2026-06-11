import os
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision.models as models
import torchvision.transforms as T

# Checkpoint directory — all model files are expected here
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _ckpt(filename):
    return os.path.join(MODELS_DIR, filename)

try:
    import clip
except:
    print('clip not found, launch pip install git+https://github.com/openai/CLIP.git')

from src.vision_models.moco import (
    moco_conv3_compressed,
    moco_conv4_compressed,
    moco_conv5,
)
try:
    from src.vision_models.maskrcnn import mask_rcnn_model
    _HAS_DETECTRON2 = True
except ImportError:
    _HAS_DETECTRON2 = False
from src.vision_models.resnet import (
    resnet_conv3_compressed,
    resnet_conv4_compressed,
    resnet_conv5,
)
from src.vision_models.mae import (
        mae_vit_base_patch16,
        mae_vit_large_patch16,
        mae_vit_huge_patch14,
)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# ==============================================================================
# GET EMBEDDING
# ==============================================================================

class UberModel(nn.Module):
    def __init__(self, models):
        super(UberModel, self).__init__()
        self.models = models
        assert all(models[0].training == m.training for m in models)
        self.training = models[0].training

    def to(self, device):
        self.models = [m.to(device=device) for m in self.models]
        return self

    def forward(self, x):
        return torch.cat([m(x) for m in self.models],
            dim=1 if x.ndim > 1 else 0)


def _get_embedding(embedding_name='random', in_channels=3, pretrained=True, train=False):
    """
    See https://pytorch.org/vision/stable/models.html

    Args:
        embedding_name (str, 'random'): the name of the convolution model,
        in_channels (int, 3): number of channels of the input image,
        pretrained (bool, True): if True, the model's weights will be downloaded
            from torchvision (if possible),
        train (bool, False): if True the model will be trained during learning,
            if False its parameters will not change.

    """

    # Default transforms: https://pytorch.org/vision/stable/models.html
    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    # where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then
    # normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    transforms = nn.Sequential(
        T.Resize(256, interpolation=3) if 'mae' in embedding_name else T.Resize(256),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )

    assert in_channels == 3, 'Current models accept 3-channel inputs only.'

    # FIXED 5-LAYER CONV
    if embedding_name == 'random':
        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        model = nn.Sequential(
            init_(nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
        )

    # Make FC layers to be Identity
    # This works for the models below but may not work for any network

    # VANILLA RESNET
    elif embedding_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = Identity()
    elif embedding_name == 'resnet34':
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = Identity()
    elif embedding_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = Identity()
    elif embedding_name == 'resnet50_places':
        model = resnet_conv5(checkpoint_path=_ckpt('resnet50_places.pth.tar'))
    elif embedding_name == 'resnet50_l4':
        model = resnet_conv4_compressed(checkpoint_path=_ckpt('resnet50_l4.pth.tar'))
    elif embedding_name == 'resnet50_l3':
        model = resnet_conv3_compressed(checkpoint_path=_ckpt('resnet50_l3.tar'))
    elif embedding_name == 'resnet50_places_l4':
        model = resnet_conv4_compressed(checkpoint_path=_ckpt('resnet50_places_l4.tar'))
    elif embedding_name == 'resnet50_places_l3':
        model = resnet_conv3_compressed(checkpoint_path=_ckpt('resnet50_places_l3.tar'))

    # DEMYSTIFY
    elif embedding_name == 'demy':
        model = moco_conv5(checkpoint_path=_ckpt('demy.pth'))

    # MAE
    elif embedding_name == 'mae_base':
        model = mae_vit_base_patch16()
        checkpoint = torch.load(_ckpt('mae_pretrain_vit_base.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    elif embedding_name == 'mae_large':
        model = mae_vit_large_patch16()
        checkpoint = torch.load(_ckpt('mae_pretrain_vit_large.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    elif embedding_name == 'mae_huge':
        model = mae_vit_huge_patch14()
        checkpoint = torch.load(_ckpt('mae_pretrain_vit_huge.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)

    # MOCO
    elif embedding_name == 'moco_aug':
        model = moco_conv5(checkpoint_path=_ckpt('moco_aug.pth.tar'))
    elif embedding_name == 'moco_aug_habitat':
        model = moco_conv5(checkpoint_path=_ckpt('moco_aug_habitat_64.pth'))
    elif embedding_name == 'moco_aug_mujoco':
        model = moco_conv5(checkpoint_path=_ckpt('moco_aug_mujoco.pth'))
    elif embedding_name == 'moco_aug_uber':
        model = moco_conv5(checkpoint_path=_ckpt('moco_aug_uber.pth'))
    elif embedding_name == 'moco_aug_places':
        model = moco_conv5(checkpoint_path=_ckpt('moco_aug_places.pth.tar'))

    elif embedding_name == 'moco_aug_l4':
        model = moco_conv4_compressed(checkpoint_path=_ckpt('moco_aug_l4.pth'))
    elif embedding_name == 'moco_aug_places_l4':
        model = moco_conv4_compressed(checkpoint_path=_ckpt('moco_aug_places_l4.pth'))
    elif embedding_name == 'moco_aug_l3':
        model = moco_conv3_compressed(checkpoint_path=_ckpt('moco_aug_l3.pth'))
    elif embedding_name == 'moco_aug_places_l3':
        model = moco_conv3_compressed(checkpoint_path=_ckpt('moco_aug_places_l3.pth'))

    elif embedding_name == 'moco_croponly':
        model = moco_conv5(checkpoint_path=_ckpt('moco_croponly.pth'))
    elif embedding_name == 'moco_croponly_places':
        model = moco_conv5(checkpoint_path=_ckpt('moco_croponly_places.pth'))
    elif embedding_name == 'moco_croponly_habitat':
        model = moco_conv5(checkpoint_path=_ckpt('moco_croponly_habitat_64.pth'))
    elif embedding_name == 'moco_croponly_mujoco':
        model = moco_conv5(checkpoint_path=_ckpt('moco_croponly_mujoco.pth'))
    elif embedding_name == 'moco_croponly_uber':
        model = moco_conv5(checkpoint_path=_ckpt('moco_croponly_uber.pth'))

    elif embedding_name == 'moco_croponly_l4':
        model = moco_conv4_compressed(checkpoint_path=_ckpt('moco_croponly_l4.pth'))
    elif embedding_name == 'moco_croponly_l3':
        model = moco_conv3_compressed(checkpoint_path=_ckpt('moco_croponly_l3.pth'))
    elif embedding_name == 'moco_croponly_places_l4':
        model = moco_conv4_compressed(checkpoint_path=_ckpt('moco_croponly_places_l4.pth'))
    elif embedding_name == 'moco_croponly_places_l3':
        model = moco_conv3_compressed(checkpoint_path=_ckpt('moco_croponly_places_l3.pth'))

    elif embedding_name == 'moco_coloronly':
        model = moco_conv5(checkpoint_path=_ckpt('moco_coloronly.pth'))

    # MOCO UBER MODELS (AUG)
    elif embedding_name == 'moco_aug_places_uber_345':
        model = UberModel([
            _get_embedding('moco_aug_places_l3')[0],
            _get_embedding('moco_aug_places_l4')[0],
            _get_embedding('moco_aug_places')[0]
        ])
    elif embedding_name == 'moco_aug_uber_345':
        model = UberModel([
            _get_embedding('moco_aug_l3')[0],
            _get_embedding('moco_aug_l4')[0],
            _get_embedding('moco_aug')[0]
        ])
    elif embedding_name == 'moco_aug_places_uber_35':
        model = UberModel([
            _get_embedding('moco_aug_places_l3')[0],
            _get_embedding('moco_aug_places')[0]
        ])
    elif embedding_name == 'moco_aug_uber_35':
        model = UberModel([
            _get_embedding('moco_aug_l3')[0],
            _get_embedding('moco_aug')[0]
        ])
    elif embedding_name == 'moco_aug_places_uber_34':
        model = UberModel([
            _get_embedding('moco_aug_places_l3')[0],
            _get_embedding('moco_aug_places_l4')[0],
        ])
    elif embedding_name == 'moco_aug_uber_34':
        model = UberModel([
            _get_embedding('moco_aug_l3')[0],
            _get_embedding('moco_aug_l4')[0],
        ])
    elif embedding_name == 'moco_aug_places_uber_45':
        model = UberModel([
            _get_embedding('moco_aug_places_l4')[0],
            _get_embedding('moco_aug_places')[0]
        ])
    elif embedding_name == 'moco_aug_uber_45':
        model = UberModel([
            _get_embedding('moco_aug_l4')[0],
            _get_embedding('moco_aug')[0]
        ])

    # MOCO UBER MODELS (CROP)
    elif embedding_name == 'moco_croponly_places_uber_345':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3')[0],
            _get_embedding('moco_croponly_places_l4')[0],
            _get_embedding('moco_croponly_places')[0]
        ])
    elif embedding_name == 'moco_croponly_uber_345':
        model = UberModel([
            _get_embedding('moco_croponly_l3')[0],
            _get_embedding('moco_croponly_l4')[0],
            _get_embedding('moco_croponly')[0]
        ])
    elif embedding_name == 'moco_croponly_places_uber_35':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3')[0],
            _get_embedding('moco_croponly_places')[0]
        ])
    elif embedding_name == 'moco_croponly_uber_35':
        model = UberModel([
            _get_embedding('moco_croponly_l3')[0],
            _get_embedding('moco_croponly')[0]
        ])
    elif embedding_name == 'moco_croponly_places_uber_34':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3')[0],
            _get_embedding('moco_croponly_places_l4')[0],
        ])
    elif embedding_name == 'moco_croponly_uber_34':
        model = UberModel([
            _get_embedding('moco_croponly_l3')[0],
            _get_embedding('moco_croponly_l4')[0],
        ])
    elif embedding_name == 'moco_croponly_places_uber_45':
        model = UberModel([
            _get_embedding('moco_croponly_places_l4')[0],
            _get_embedding('moco_croponly_places')[0]
        ])
    elif embedding_name == 'moco_croponly_uber_45':
        model = UberModel([
            _get_embedding('moco_croponly_l4')[0],
            _get_embedding('moco_croponly')[0]
        ])

    # MASK
    elif embedding_name == 'maskrcnn_l3':
        if not _HAS_DETECTRON2:
            raise ImportError("maskrcnn_l3 requires detectron2. Install with: pip install git+https://github.com/facebookresearch/detectron2.git")
        # Input must be BGR and not normalized in [0, 1] (ie, keep them in [0, 255])
        class _rgb_to_bgr(nn.Module):
            def forward(self, x):
                x[:,:,[0,1,2]] = x[:,:,[2,1,0]]
                return x.float()
        transforms = nn.Sequential(
            _rgb_to_bgr(),
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize([103.530, 116.280, 123.675], [1.0, 1.0, 1.0]),
        )
        model = mask_rcnn_model(checkpoint_path=_ckpt('maskrcnn_l3.pth'))

    # CLIP
    elif 'clip' in embedding_name:
        # Custom transforms from
        # https://github.com/openai/CLIP/blob/573315e83f07b53a61ff5098757e8fc885f1703e/clip/clip.py#L76
        # My code avoids PIL and is faster, but works only with antialias=True
        # (see https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
        if embedding_name == 'clip_vit':
            model, _ = clip.load("ViT-B/32", device='cpu') # Will be moved to GPU later
        elif embedding_name == 'clip_rn50':
            model, _ = clip.load("RN50", device='cpu') # Will be moved to GPU later
        else:
            raise NotImplementedError("Requested model not available.")
        transforms = nn.Sequential(
            T.Resize(model.visual.input_resolution, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(model.visual.input_resolution),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        )

    # TRUE STATE (BASELINE)
    elif embedding_name == 'true_state':
        return nn.Sequential(Identity()), nn.Sequential(Identity())

    else:
        raise NotImplementedError("Requested model not available.")

    if train:
        model.train()
        for p in model.parameters():
            p.requires_grad = True
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model, transforms


# ==============================================================================
# EMBEDDING CLASS
# ==============================================================================

class EmbeddingNet(nn.Module):
    """
    Input shape must be (N, H, W, 3), where N is the number of frames.
    The class will then take care of transforming and normalizing frames.
    The output shape will be (N, O), where O is the embedding size.

    Augmentation (optional nn.Module) is applied to float [0, 1] tensors
    after spatial transforms (resize/crop/dtype conversion) but before
    mean/std normalisation — only when the model is in training mode.
    """
    def __init__(self, embedding_name, in_channels=3, pretrained=True, train=False,
                 disable_cuda=False, augmentation=None):
        super(EmbeddingNet, self).__init__()

        self.embedding_name = embedding_name

        if self.embedding_name == 'true_state':
            return

        self.in_channels = in_channels
        self.embedding, self.transforms = \
            _get_embedding(embedding_name, in_channels, pretrained, train)

        # Split transforms so augmentation runs on [0,1] floats, before normalisation.
        # All transform pipelines end with T.Normalize; split there.
        t_list = list(self.transforms.children())
        if isinstance(t_list[-1], T.Normalize):
            self._pre_aug = nn.Sequential(*t_list[:-1])
            self._normalize = t_list[-1]
        else:
            self._pre_aug = self.transforms
            self._normalize = nn.Identity()

        self.augmentation = augmentation

        dummy_in = torch.zeros(1, in_channels, 64, 64)
        dummy_in = self.transforms(dummy_in)
        self.in_shape = dummy_in.shape[1:]
        dummy_out = self._forward(dummy_in)
        self.out_size = np.prod(dummy_out.shape)

        if torch.cuda.is_available() and not disable_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.embedding = self.embedding.to(device=self.device)
        self.training = self.embedding.training

    def _forward(self, observation):
        if 'clip' in self.embedding_name:
            out = self.embedding.encode_image(observation)
        elif 'mae' in self.embedding_name:
            out, *_ = self.embedding.forward_encoder(observation, mask_ratio=0.0)
            out = out[:,0,:]
        else:
            out = self.embedding(observation)
            if self.embedding_name == 'maskrcnn_l3':
                out = out['res4']
        return out

    def forward(self, observation):
        if self.embedding_name == 'true_state':
            return observation.squeeze().cpu().numpy()

        # observation.shape -> (N, C, H, W)  [channels-first, as returned by gym_wrappers]
        observation = observation.to(device=self.device)
        if not observation.is_contiguous():
            observation = observation.contiguous()
        observation = self._pre_aug(observation)          # resize, crop, → float [0, 1]
        if self.training and self.augmentation is not None:
            observation = self.augmentation(observation)  # augment in [0, 1] space
        observation = self._normalize(observation)        # mean/std normalisation
        observation = observation.reshape(-1, *self.in_shape)

        if self.embedding.training:
            out = self._forward(observation)
            return out.reshape(-1, self.out_size).squeeze()
        else:
            with torch.no_grad():
                out = self._forward(observation)
                return out.view(-1, self.out_size).squeeze().cpu().numpy()


# ==============================================================================
# EMBEDDING WRAPPER
# ==============================================================================

class EmbeddingWrapper(gym.ObservationWrapper):
    def __init__(self, env, embedding):
        gym.ObservationWrapper.__init__(self, env)

        in_channels = env.observation_space.shape[2]
        assert in_channels % 3 == 0,  \
                """ Only RGB images are supported.
                    Be sure that observation shape is (H, W, n * 3),
                    where n is the number of frames per observation. """

        self.in_channels = 3
        self.n_frames = in_channels // 3

        self.embedding = embedding
        self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.embedding.out_size * self.n_frames,)
        )

    def observation(self, observation):
        # if self.n_frames > 1, each passes through the embedding separately
        observation = np.stack(np.split(observation, self.n_frames, axis=-1)) # (H, W, self.n_frames * 3) -> (self.n_frames, H, W, 3)
        return self.embedding(torch.from_numpy(observation)).flatten()
