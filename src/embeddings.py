import glob
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


def _resolve_model_dir_override(embedding_name, model_dir):
    """
    If model_dir is given, look there for a checkpoint file named
    <embedding_name>.<any extension> (e.g. resnet18.pth, moco_aug.pth.tar)
    instead of this embedding's normal default location/behavior. Crashes
    loudly if model_dir is given but no matching file is found -- an
    explicit model_dir means the caller wants exactly that file, not a
    silent fallback to the default. Returns None if model_dir is None,
    signalling "use the default behavior for this embedding_name".
    """
    if model_dir is None:
        return None
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_dir {model_dir!r} does not exist.")
    matches = glob.glob(os.path.join(model_dir, embedding_name + '.*'))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for {embedding_name!r} in {model_dir!r} "
            f"(looked for {embedding_name}.*)."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple checkpoints match {embedding_name!r} in {model_dir!r}: {matches}"
        )
    return matches[0]


def _load_custom_state_dict(model, path, embedding_name):
    """
    Loads a model_dir-provided checkpoint into an architecture built with
    weights=None (torchvision resnets, which otherwise have no local-file
    loading path at all -- they only ever auto-download). Unwraps the
    common 'state_dict'/'model' wrapper-dict conventions used elsewhere in
    this file (see the MAE branches below), then loads with strict=False
    since a custom checkpoint's exact key names aren't guaranteed to match
    torchvision's. Warns on partial mismatches; crashes if literally
    nothing matched, since that means the file is very likely the wrong
    architecture or format for embedding_name, not just a minor difference.
    """
    state_dict = torch.load(path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) == len(list(model.state_dict())):
        raise RuntimeError(
            f"Loaded {path!r} for {embedding_name!r} but zero parameters matched "
            f"the expected architecture -- this checkpoint is very likely the "
            f"wrong architecture or format for {embedding_name!r}."
        )
    if missing or unexpected:
        print(f"[embeddings] Warning: loading {path!r} for {embedding_name!r} -- "
              f"missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}, "
              f"unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

try:
    import clip
except:
    print('clip not found, launch pip install git+https://github.com/openai/CLIP.git')

try:
    import open_clip
    _HAS_OPENCLIP = True
except ImportError:
    _HAS_OPENCLIP = False

try:
    from r3m import load_r3m
    _HAS_R3M = True
except ImportError:
    _HAS_R3M = False

try:
    from vip import load_vip
    _HAS_VIP = True
except ImportError:
    _HAS_VIP = False

try:
    from liv import load_liv
    _HAS_LIV = True
except ImportError:
    _HAS_LIV = False

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
from src.vision_models.rl3d import rl3d_resnet18_backbone


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# ==============================================================================
# GET EMBEDDING
# ==============================================================================

# forward_fn(model, x) -> raw model output. _get_embedding() picks one of
# these per branch (explicitly, at the same place the model itself is
# built) instead of EmbeddingNet._forward() re-deriving "how do I call this
# model" from embedding_name substrings in a second, separate place that can
# drift out of sync with the loading logic.
def _forward_default(model, x):
    return model(x)


def _forward_clip(model, x):
    return model.encode_image(x)


def _forward_mae(model, x):
    out, *_ = model.forward_encoder(x, mask_ratio=0.0)
    return out[:, 0, :]


def _forward_maskrcnn(model, x):
    return model(x)['res4']


class _ScaleTo255(nn.Module):
    """R3M expects [0, 255]-range input with no external mean/std
    normalization -- that's baked into the model itself. This runs after
    T.ConvertImageDtype(torch.float) (which produces [0, 1]) to undo the
    usual /255 convention every other branch in this file relies on."""
    def forward(self, x):
        return x * 255.0


# One config-friendly name -> (open_clip arch, pretrained tag) per supported
# variant. embedding_name has to be a single self-contained string (it's
# what configs/embedding/*.yaml's `name:` field passes straight through to
# EmbeddingNet), so this stands in for open_clip's separate arch/pretrained
# arguments rather than trying to cram both into one free-form name.
OPENCLIP_CONFIGS = {
    'openclip_vit_b32': ('ViT-B-32', 'laion2b_s34b_b79k'),
    'openclip_vit_l14': ('ViT-L-14', 'laion2b_s32b_b82k'),
    'openclip_rn50': ('RN50', 'openai'),
}

# load_r3m() takes a bare torchvision resnet name ('resnet18'/'resnet34'/
# 'resnet50') and only ever loads R3M's own pretrained checkpoint for it --
# there's no random-init path, unlike the plain 'resnetNN' branches above.
R3M_ARCHS = {
    'r3m_resnet18': 'resnet18',
    'r3m_resnet34': 'resnet34',
    'r3m_resnet50': 'resnet50',
}

# embedding_name -> (this file's matching MAE-architecture builder, VC-1's
# own checkpoint filename). Same encoder classes as the mae_* branches --
# VC-1 is MAE pretraining on different data, not a different architecture.
VC1_ARCHS = {
    'vc1_vitb': (mae_vit_base_patch16, 'vc1_vitb.pth'),
    'vc1_vitl': (mae_vit_large_patch16, 'vc1_vitl.pth'),
}


def _extract_openclip_transform_params(preprocess_val):
    """
    open_clip.create_model_and_transforms() returns a torchvision Compose
    built for PIL inputs (Resize -> CenterCrop -> ToTensor -> Normalize).
    This file's pipeline instead runs on already-batched uint8 CHW tensors,
    so that Compose can't be reused directly -- pull just the resize size
    and normalize mean/std out of it and rebuild with this file's own
    transform idiom. Reading these two steps out of preprocess_val is more
    robust than reading a model attribute: the resolution attribute name on
    model.visual is inconsistent across architectures (image_size vs
    input_resolution), but every preprocess_val always contains a Resize
    and a Normalize step.
    """
    resize_size, mean, std = None, None, None
    for t in preprocess_val.transforms:
        if isinstance(t, T.Resize):
            size = t.size
            resize_size = size[0] if isinstance(size, (tuple, list)) else size
        elif isinstance(t, T.Normalize):
            mean, std = t.mean, t.std
    if resize_size is None or mean is None:
        raise RuntimeError(
            "Could not find a Resize/Normalize step in open_clip's "
            "preprocess_val pipeline -- open_clip's transform internals "
            "may have changed."
        )
    return resize_size, mean, std


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


def _get_embedding(embedding_name='random', in_channels=3, pretrained=True, train=False, model_dir=None):
    """
    See https://pytorch.org/vision/stable/models.html

    Args:
        embedding_name (str, 'random'): the name of the convolution model,
        in_channels (int, 3): number of channels of the input image,
        pretrained (bool, True): if True, the model's weights will be downloaded
            from torchvision (if possible),
        train (bool, False): if True the model will be trained during learning,
            if False its parameters will not change.
        model_dir (str, optional): if given, look here for a checkpoint file
            named <embedding_name>.<any extension> instead of this
            embedding's normal default location/behavior. Raises if no
            matching file is found. See _resolve_model_dir_override().

    """
    # Resolved lazily (only if a branch below actually needs a checkpoint
    # file) rather than once up front -- the MoCo "uber" ensemble branches
    # recursively call _get_embedding() for each sub-model instead of
    # loading a file directly themselves, so eagerly resolving a file for
    # e.g. 'moco_aug_uber_345' would crash looking for a file that was never
    # supposed to exist; each recursive call resolves its own sub-name instead.
    def _ckpt(filename):
        override = _resolve_model_dir_override(embedding_name, model_dir)
        return override if override is not None else os.path.join(MODELS_DIR, filename)

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
    forward_fn = _forward_default

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
    # These normally never touch a local file at all (torchvision
    # auto-downloads to its own cache) -- model_dir is the one way to make
    # them load a specific local checkpoint instead.
    elif embedding_name == 'resnet18':
        override = _resolve_model_dir_override(embedding_name, model_dir)
        if override is not None:
            model = models.resnet18(weights=None)
            _load_custom_state_dict(model, override, embedding_name)
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        model.fc = Identity()
    elif embedding_name == 'resnet34':
        override = _resolve_model_dir_override(embedding_name, model_dir)
        if override is not None:
            model = models.resnet34(weights=None)
            _load_custom_state_dict(model, override, embedding_name)
        else:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
        model.fc = Identity()
    elif embedding_name == 'resnet50':
        override = _resolve_model_dir_override(embedding_name, model_dir)
        if override is not None:
            model = models.resnet50(weights=None)
            _load_custom_state_dict(model, override, embedding_name)
        else:
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
        forward_fn = _forward_mae
    elif embedding_name == 'mae_large':
        model = mae_vit_large_patch16()
        checkpoint = torch.load(_ckpt('mae_pretrain_vit_large.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        forward_fn = _forward_mae
    elif embedding_name == 'mae_huge':
        model = mae_vit_huge_patch14()
        checkpoint = torch.load(_ckpt('mae_pretrain_vit_huge.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        forward_fn = _forward_mae

    # VC-1 (facebookresearch/eai-vc) -- MAE pretraining (egocentric video +
    # ImageNet + iNav) on the exact same ViT-B/16 / ViT-L/16 architecture as
    # the mae_* branches above, confirmed against their actual source
    # (vc_models/models/vit/vit.py's vit_base_patch16/vit_large_patch16 and
    # load_mae_encoder): same architecture class, same
    # torch.load(...)['model'] checkpoint format, same CLS-token forward.
    # Loaded directly here rather than via the vc_models package -- that
    # package pins timm==0.6.11, which conflicts with this repo's
    # timm==1.0.27; unnecessary anyway since the architecture is already
    # implemented in src/vision_models/mae.py. Weights: download
    # vc1_vitb.pth / vc1_vitl.pth from https://dl.fbaipublicfiles.com/eai-vc/
    # and place in MODELS_DIR (or point model_dir= at them) -- not
    # auto-downloaded, same as the mae_* checkpoints above.
    elif embedding_name in VC1_ARCHS:
        build_fn, ckpt_name = VC1_ARCHS[embedding_name]
        model = build_fn()
        checkpoint = torch.load(_ckpt(ckpt_name), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        # vc_models/transforms/__init__.py's vit_transforms: same ImageNet
        # mean/std as every other branch here, but BICUBIC resize (matching
        # this file's own 'mae' in embedding_name special case a few lines
        # up) -- built explicitly since 'vc1_vitb'/'vc1_vitl' don't contain
        # the substring that default block gates on.
        transforms = nn.Sequential(
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )
        forward_fn = _forward_mae

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
            _get_embedding('moco_aug_places_l3', model_dir=model_dir)[0],
            _get_embedding('moco_aug_places_l4', model_dir=model_dir)[0],
            _get_embedding('moco_aug_places', model_dir=model_dir)[0]
        ])
    elif embedding_name == 'moco_aug_uber_345':
        model = UberModel([
            _get_embedding('moco_aug_l3', model_dir=model_dir)[0],
            _get_embedding('moco_aug_l4', model_dir=model_dir)[0],
            _get_embedding('moco_aug', model_dir=model_dir)[0]
        ])
    elif embedding_name == 'moco_aug_places_uber_35':
        model = UberModel([
            _get_embedding('moco_aug_places_l3', model_dir=model_dir)[0],
            _get_embedding('moco_aug_places', model_dir=model_dir)[0]
        ])
    elif embedding_name == 'moco_aug_uber_35':
        model = UberModel([
            _get_embedding('moco_aug_l3', model_dir=model_dir)[0],
            _get_embedding('moco_aug', model_dir=model_dir)[0]
        ])
    elif embedding_name == 'moco_aug_places_uber_34':
        model = UberModel([
            _get_embedding('moco_aug_places_l3', model_dir=model_dir)[0],
            _get_embedding('moco_aug_places_l4', model_dir=model_dir)[0],
        ])
    elif embedding_name == 'moco_aug_uber_34':
        model = UberModel([
            _get_embedding('moco_aug_l3', model_dir=model_dir)[0],
            _get_embedding('moco_aug_l4', model_dir=model_dir)[0],
        ])
    elif embedding_name == 'moco_aug_places_uber_45':
        model = UberModel([
            _get_embedding('moco_aug_places_l4', model_dir=model_dir)[0],
            _get_embedding('moco_aug_places', model_dir=model_dir)[0]
        ])
    elif embedding_name == 'moco_aug_uber_45':
        model = UberModel([
            _get_embedding('moco_aug_l4', model_dir=model_dir)[0],
            _get_embedding('moco_aug', model_dir=model_dir)[0]
        ])

    # MOCO UBER MODELS (CROP)
    elif embedding_name == 'moco_croponly_places_uber_345':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3', model_dir=model_dir)[0],
            _get_embedding('moco_croponly_places_l4', model_dir=model_dir)[0],
            _get_embedding('moco_croponly_places', model_dir=model_dir)[0]
        ])
    elif embedding_name == 'moco_croponly_uber_345':
        model = UberModel([
            _get_embedding('moco_croponly_l3', model_dir=model_dir)[0],
            _get_embedding('moco_croponly_l4', model_dir=model_dir)[0],
            _get_embedding('moco_croponly', model_dir=model_dir)[0]
        ])
    elif embedding_name == 'moco_croponly_places_uber_35':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3', model_dir=model_dir)[0],
            _get_embedding('moco_croponly_places', model_dir=model_dir)[0]
        ])
    elif embedding_name == 'moco_croponly_uber_35':
        model = UberModel([
            _get_embedding('moco_croponly_l3', model_dir=model_dir)[0],
            _get_embedding('moco_croponly', model_dir=model_dir)[0]
        ])
    elif embedding_name == 'moco_croponly_places_uber_34':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3', model_dir=model_dir)[0],
            _get_embedding('moco_croponly_places_l4', model_dir=model_dir)[0],
        ])
    elif embedding_name == 'moco_croponly_uber_34':
        model = UberModel([
            _get_embedding('moco_croponly_l3', model_dir=model_dir)[0],
            _get_embedding('moco_croponly_l4', model_dir=model_dir)[0],
        ])
    elif embedding_name == 'moco_croponly_places_uber_45':
        model = UberModel([
            _get_embedding('moco_croponly_places_l4', model_dir=model_dir)[0],
            _get_embedding('moco_croponly_places', model_dir=model_dir)[0]
        ])
    elif embedding_name == 'moco_croponly_uber_45':
        model = UberModel([
            _get_embedding('moco_croponly_l4', model_dir=model_dir)[0],
            _get_embedding('moco_croponly', model_dir=model_dir)[0]
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
        forward_fn = _forward_maskrcnn

    # R3M
    elif embedding_name in R3M_ARCHS:
        if not _HAS_R3M:
            raise ImportError("r3m requires: pip install git+https://github.com/facebookresearch/r3m.git")
        if not pretrained:
            raise NotImplementedError("R3M has no random-init path -- load_r3m() always loads its pretrained checkpoint.")
        model = load_r3m(R3M_ARCHS[embedding_name])
        if isinstance(model, nn.DataParallel):
            # load_r3m() always wraps in DataParallel, even on a single GPU/CPU;
            # unwrap so this behaves like every other single-module branch in
            # this file (named_parameters(), .to(device), etc, without the
            # replicate-on-every-forward overhead DataParallel adds).
            model = model.module
        # load_r3m() also auto-moves the model onto cuda itself when a GPU
        # is present -- unlike every other branch here, which builds on CPU
        # and lets EmbeddingNet.__init__ move it to self.device uniformly,
        # after running a CPU dummy-forward sanity check. Force back to CPU
        # so that invariant holds (same reason clip.load(..., device='cpu')
        # is explicit below).
        model = model.cpu()
        transforms = nn.Sequential(
            T.Resize(256),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),  # -> [0, 1]
            _ScaleTo255(),                     # -> [0, 255]; R3M normalizes internally
        )
        # forward_fn stays _forward_default: R3M's forward() takes the [0, 255] tensor directly.

    # VIP
    # Same lab/codebase lineage as R3M (built directly on R3M's code) --
    # confirmed via their own example code (encoder_example.py): identical
    # [0, 255]-input, no-external-normalize convention. Only one released
    # variant (resnet50, 1024-dim), unlike R3M's resnet18/34/50 choice, so
    # load_vip() takes no size argument.
    elif embedding_name == 'vip_resnet50':
        if not _HAS_VIP:
            raise ImportError("vip requires: pip install git+https://github.com/facebookresearch/vip.git")
        if not pretrained:
            raise NotImplementedError("VIP has no random-init path -- load_vip() always loads its pretrained checkpoint.")
        model = load_vip()
        if isinstance(model, nn.DataParallel):
            # Same wrap-and-auto-cuda behavior as load_r3m() -- see R3M branch above.
            model = model.module
        model = model.cpu()
        transforms = nn.Sequential(
            T.Resize(256),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),  # -> [0, 1]
            _ScaleTo255(),                     # -> [0, 255]; VIP normalizes internally
        )
        # forward_fn stays _forward_default: VIP's forward() takes the [0, 255] tensor directly.

    # LIV (Language-Image Value learning) -- wraps CLIP RN50 internally
    # (their own bundled clip fork, loaded via clip.load("resnet50", ...)
    # inside load_liv()), not a plain torchvision resnet despite the name.
    # 'resnet50' is LIV's own (only) modelid, matching VIP's single-variant
    # convention -- named liv_resnet50 here to avoid confusion with a real
    # resnet50. Reaches into the underlying CLIP submodule (liv.model) and
    # reuses the existing CLIP dispatch (_forward_clip, same transform as
    # clip_rn50 below) rather than calling LIV's own forward(), which does
    # its own resize/crop/normalize AND auto-detects [0,1] vs [0,255] input
    # internally -- a different contract from every other branch in this
    # file, which all expect EmbeddingNet's external transform pipeline to
    # have already done that. Going through liv.model.encode_image()
    # directly sidesteps the mismatch entirely.
    elif embedding_name == 'liv_resnet50':
        if not _HAS_LIV:
            raise ImportError(
                "liv requires a two-step install (not a single pip line): "
                "pip install -e . && cd liv/models/clip && pip install -e . "
                "-- see https://github.com/penn-pal-lab/LIV"
            )
        if not pretrained:
            raise NotImplementedError("LIV has no random-init path -- load_liv() always loads its pretrained checkpoint.")
        liv = load_liv()
        if isinstance(liv, nn.DataParallel):
            # Same wrap-and-auto-cuda behavior as load_r3m()/load_vip() --
            # see R3M branch above.
            liv = liv.module
        model = liv.model  # the underlying CLIP RN50 submodule
        model = model.cpu()
        # Same transform as the clip_rn50 branch below: RN50's input
        # resolution is a fixed 224 across all official CLIP checkpoints
        # (including LIV's, which is trained from a CLIP RN50 init) --
        # hardcoded rather than read from model.visual.input_resolution,
        # since that attribute's presence on LIV's own bundled clip fork
        # isn't verified.
        transforms = nn.Sequential(
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        )
        forward_fn = _forward_clip

    # RL3D (YanjieZe/rl3d) -- CO3D-pretrained 2D backbone, vendored (see
    # src/vision_models/rl3d.py -- no pip package exists for this repo).
    # No auto-download: their own loader requires an already-existing
    # local checkpoint path and raises FileNotFoundError otherwise, same
    # as this repo's other manual-checkpoint embeddings (mae_*, moco_*).
    # Download videoae_co3d.tar from
    # https://github.com/YanjieZe/rl3d/tree/main/checkpoints, place at
    # MODELS_DIR/rl3d_resnet18.tar (or point model_dir= at a renamed copy).
    elif embedding_name == 'rl3d_resnet18':
        model = rl3d_resnet18_backbone()
        checkpoint = torch.load(_ckpt('rl3d_resnet18.tar'), map_location='cpu')
        # checkpoint['encoder_3d'] covers the FULL Encoder3D (this 2D
        # backbone under 'feature_extraction.*' keys, plus the 3D
        # ConvTranspose3d decoder head under 'conv3d_1.*'/'conv3d_2.*' --
        # dropped entirely here, see rl3d.py). 'module.' prefix: their own
        # loader does encoder_3d.load_state_dict(...) on a freshly built
        # nn.DataParallel(Encoder3D()) with strict=True (the default) --
        # DataParallel always prefixes its wrapped submodule's keys with
        # 'module.', so for that strict load to succeed at all, the
        # checkpoint's keys must already be 'module.'-prefixed. Strip both
        # so the remaining keys match this bare nn.Sequential's own naming.
        prefix = 'module.feature_extraction.'
        state_dict = {
            k[len(prefix):]: v for k, v in checkpoint['encoder_3d'].items()
            if k.startswith(prefix)
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) == len(list(model.state_dict())):
            raise RuntimeError(
                f"Loaded rl3d_resnet18.tar but zero parameters matched -- "
                "wrong checkpoint file, or the prefix-stripping above no "
                "longer matches this checkpoint's key structure."
            )
        if missing or unexpected:
            print(f"[embeddings] Warning: loading rl3d_resnet18.tar -- "
                  f"missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}, "
                  f"unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        # Verified against their src/augmentations.py and src/train.py:
        # both use 84x84 frames and plain /255 scaling, no ImageNet
        # mean/std normalize -- an RL-frame convention, not the 224px
        # internet-image convention every other PVR branch here follows.
        transforms = nn.Sequential(
            T.Resize((84, 84)),
            T.ConvertImageDtype(torch.float),
        )
        # forward_fn stays _forward_default: the vendored backbone is a
        # plain nn.Sequential, model(x) is already the right call.

    # OPENCLIP
    # Checked before the 'clip' in embedding_name branch below since
    # 'clip' is a substring of every 'openclip_*' name.
    elif embedding_name in OPENCLIP_CONFIGS:
        if not _HAS_OPENCLIP:
            raise ImportError("openclip requires: pip install open_clip_torch")
        arch, pretrained_tag = OPENCLIP_CONFIGS[embedding_name]
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained_tag if pretrained else None
        )
        resize_size, mean, std = _extract_openclip_transform_params(preprocess_val)
        transforms = nn.Sequential(
            T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(resize_size),
            T.ConvertImageDtype(torch.float),
            T.Normalize(list(mean), list(std)),
        )
        forward_fn = _forward_clip

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
        forward_fn = _forward_clip

    # TRUE STATE (BASELINE)
    # Unreachable via EmbeddingNet (it short-circuits before calling
    # _get_embedding for 'true_state') -- kept 3-tuple for consistency in
    # case anything ever calls this function directly with that name.
    elif embedding_name == 'true_state':
        return nn.Sequential(Identity()), nn.Sequential(Identity()), _forward_default

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

    return model, transforms, forward_fn


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
                 disable_cuda=False, augmentation=None, model_dir=None):
        super(EmbeddingNet, self).__init__()

        self.embedding_name = embedding_name

        if self.embedding_name == 'true_state':
            return

        self.in_channels = in_channels
        self.embedding, self.transforms, self._forward_fn = \
            _get_embedding(embedding_name, in_channels, pretrained, train, model_dir=model_dir)

        bad = [
            name for name, tensor in
            list(self.embedding.named_parameters()) + list(self.embedding.named_buffers())
            # OpenCLIP's text-tower causal attn_mask is a non-persistent,
            # runtime-constructed buffer deliberately filled with -inf
            # (never loaded from a checkpoint, and unused here since we
            # only call .encode_image()) -- not a sign of corruption.
            if not name.endswith('attn_mask') and not torch.isfinite(tensor).all()
        ]
        if bad:
            raise RuntimeError(
                f"Non-finite value(s) in {embedding_name!r} weights immediately "
                f"after loading, before any forward pass: {bad[:5]}"
                f"{'...' if len(bad) > 5 else ''}. The checkpoint/cache file is "
                "likely corrupted or truncated -- delete it and let it re-download."
            )

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
        return self._forward_fn(self.embedding, observation)

    def encode(self, observation, augment: bool = False):
        """
        Same computation as forward(), but returns a (N, out_size)
        torch.Tensor on this module's device -- batch dimension always
        intact (forward()'s squeeze() silently drops it when N==1) and no
        numpy conversion (forward() always detaches to numpy, even in
        training mode, which breaks callers that need a differentiable
        tensor -- e.g. a trainable head sitting on top of this embedding
        inside another nn.Module's forward pass, like an SB3 features
        extractor).

        augment: apply self.augmentation (if any) to this call only.
        Explicit per-call, not gated on self.training -- callers that keep
        the encoder permanently in eval() for BatchNorm-stats safety (see
        train_sac.py) would otherwise never be able to trigger it, and
        callers generally want augmentation on training-batch encodes only
        (never on rollout-time action selection or eval()).
        """
        # observation.shape -> (N, C, H, W)  [channels-first, as returned by gym_wrappers]
        observation = observation.to(device=self.device)
        if not observation.is_contiguous():
            observation = observation.contiguous()
        observation = self._pre_aug(observation)          # resize, crop, → float [0, 1]
        if augment and self.augmentation is not None:
            observation = self.augmentation(observation)  # augment in [0, 1] space
        observation = self._normalize(observation)        # mean/std normalisation
        observation = observation.reshape(-1, *self.in_shape)

        # Gradient tracking is entirely the caller's responsibility (ambient
        # torch.no_grad()/enable_grad() context + requires_grad on this
        # module's params) -- this used to branch on self.embedding.training
        # instead, which broke callers that keep the encoder in eval() (for
        # BatchNorm-stats safety) while still finetuning it via requires_grad
        # alone: an inner torch.no_grad() here always beat their outer
        # enable_grad(), silently cutting the graph. When the encoder really
        # is frozen (requires_grad=False on every param, non-grad-requiring
        # input), autograd already builds no backward graph on its own --
        # forcing no_grad() explicitly was redundant for that case anyway.
        out = self._forward(observation)
        return out.reshape(-1, self.out_size)

    def forward(self, observation):
        if self.embedding_name == 'true_state':
            return observation.squeeze().cpu().numpy()
        return self.encode(observation).squeeze().cpu().numpy()


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
