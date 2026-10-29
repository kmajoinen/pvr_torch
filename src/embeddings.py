import glob
import os
import numpy as np

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


def _read_state_dict(path):
    return _read_checkpoint(path)[0]


def _read_checkpoint(path):
    """torch.load a checkpoint and unwrap the usual wrapper dicts: 'encoder'
    (this repo's saved encoders), 'state_dict', 'model', and the official R3M
    code's 'r3m' snapshots (image encoder only). Returns
    (state_dict, build record): the record holds the embedding config the
    encoder was built with ('embedding' entry, or 'config'.embedding in a
    final_checkpoint.pt) and the frame size it encoded ('input_size'); both
    None for files that don't record them."""
    # weights_only=False: PyTorch 2.6+ defaults torch.load to a restricted
    # unpickler that rejects anything beyond plain tensors -- research
    # checkpoints routinely bundle non-tensor metadata (training args,
    # optimizer state) and fail to load at all otherwise. Fine here: paths
    # are either this repo's own MODELS_DIR or a user-provided file, not
    # arbitrary untrusted input.
    raw = torch.load(path, map_location='cpu', weights_only=False)
    record = {'embedding': None, 'input_size': None}
    if isinstance(raw, dict):
        # 'actor' marks train_sac.py's final_checkpoint.pt. A bare 'config'
        # entry doesn't: {"model": weights, "config": ...} is a common layout
        # for other checkpoints too, whose config isn't this repo's.
        is_sac_run = 'actor' in raw
        if is_sac_run and 'encoder' not in raw:
            raise ValueError(
                f"{path!r} is a training run's checkpoint without encoder weights: "
                "that run's encoder was frozen, so there was nothing to save. Use "
                "the encoder that run was configured with instead."
            )
        if isinstance(raw.get('embedding'), dict):
            record['embedding'] = raw['embedding']
        elif is_sac_run and isinstance(raw.get('config'), dict) \
                and isinstance(raw['config'].get('embedding'), dict):
            record['embedding'] = raw['config']['embedding']
        if isinstance(raw.get('input_size'), int):
            record['input_size'] = raw['input_size']
    if isinstance(raw, dict) and isinstance(raw.get('r3m'), dict):
        # The official R3M code's snapshots (its train_representation.py, and
        # the released ~/.r3m/r3m_*/model.pt): a DataParallel-wrapped model
        # whose image encoder is module.convnet.* -- a torchvision ResNet
        # with fc removed. Keep that; drop the language parts.
        prefix = 'module.convnet.'
        raw = {k[len(prefix):]: v for k, v in raw['r3m'].items() if k.startswith(prefix)}
    state_dict = raw
    for key in ('encoder', 'state_dict', 'model'):
        if isinstance(state_dict, dict) and isinstance(state_dict.get(key), dict):
            state_dict = state_dict[key]
            break
    if not isinstance(state_dict, dict):
        raise TypeError(f"{path!r} doesn't contain a state_dict (got {type(state_dict).__name__})")
    return state_dict, record


def _load_custom_state_dict(model, source, embedding_name, strict=False):
    """
    Loads a model_dir-provided checkpoint into an architecture built with
    weights=None (torchvision resnets, which otherwise have no local-file
    loading path at all -- they only ever auto-download). `source` is a
    path (unwrapped by _read_state_dict) or an already-loaded state_dict.
    Loads with strict=False since a custom checkpoint's exact key names
    aren't guaranteed to match torchvision's. Warns on partial mismatches;
    crashes if literally nothing matched, since that means the file is very
    likely the wrong architecture or format for embedding_name, not just a
    minor difference.

    strict=True (an explicit cfg.embedding.checkpoint) crashes on any
    missing or unexpected key instead of warning: that file is meant to be
    exactly the model being built, so a partial load is an error.
    """
    path = source if isinstance(source, str) else 'checkpoint'
    state_dict = _read_state_dict(source) if isinstance(source, str) else source
    # Compared before loading: BatchNorm fills in a missing
    # num_batches_tracked buffer by itself, so "every key missing" never
    # shows up in load_state_dict's report for a model with BatchNorm.
    if not set(state_dict) & set(model.state_dict()):
        raise RuntimeError(
            f"Loaded {path!r} for {embedding_name!r} but zero parameters matched "
            f"the expected architecture -- this checkpoint is very likely the "
            f"wrong architecture or format for {embedding_name!r}."
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.endswith('num_batches_tracked')]
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"{path!r} doesn't match the {embedding_name!r} model exactly -- "
            f"missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}, "
            f"unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
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

try:
    import mvp
    _HAS_MVP = True
except ImportError:
    _HAS_MVP = False

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
        vit_s16,
        vit_b16,
        vit_l16,
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
    # MAE's own forward_encoder(x, mask_ratio=0.0) still runs random_masking,
    # which keeps every patch but shuffles them with fresh noise on each call.
    # The CLS output doesn't care (attention is order-invariant), but a
    # truncated layer's tokens would arrive in a new random order every pass
    # (breaking a flatten/linear head) and every encode would consume RNG.
    # Same computation, patches kept in order:
    x = model.patch_embed(x) + model.pos_embed[:, 1:, :]
    cls = (model.cls_token + model.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    for blk in model.blocks:
        x = blk(x)
    return model.norm(x)[:, 0]


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

# torch.hub entrypoints of facebookresearch/dinov2 (ViT-S/B/L/g, 14px
# patches). forward() returns the final-layer CLS token: 384 / 768 / 1024 /
# 1536 dims.
DINOV2_ARCHS = ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')

# ViTs in MAE's layout (src/vision_models/mae.py's MAEEncoder; ViT-S/B/L
# with 16px patches as MVP defines them), random-init only: weights come from
# embedding.checkpoint -- a pretrain.py run (objective `mae`), or a released
# MAE-format file (MAE, MVP, VC-1 encoders).
VIT_ARCHS = {'vit_s16': vit_s16, 'vit_b16': vit_b16, 'vit_l16': vit_l16}

# Architectures cfg.embedding.checkpoint can load into: their branches build
# the model without touching any default weights when pretrained=False. The
# others (R3M, VIP, LIV, MVP, MAE, VC-1, MoCo, places, RL3D, demy, OpenAI
# CLIP) always load or download their own weights first.
CHECKPOINT_ARCHS = frozenset(
    ('random', 'resnet18', 'resnet34', 'resnet50')
    + DINOV2_ARCHS + tuple(OPENCLIP_CONFIGS) + tuple(VIT_ARCHS)
)

# embedding_name -> (this file's matching MAE-architecture builder, VC-1's
# own checkpoint filename). Same encoder classes as the mae_* branches --
# VC-1 is MAE pretraining on different data, not a different architecture.
VC1_ARCHS = {
    'vc1_vitb': (mae_vit_base_patch16, 'vc1_vitb.pth'),
    'vc1_vitl': (mae_vit_large_patch16, 'vc1_vitl.pth'),
}

# embedding_name -> mvp.load()'s own name string. Five released variants:
# hoi/in/egosoup = pretraining data (hand-object-interaction video /
# ImageNet / mixed egocentric video), mae/sup = pretraining objective
# (masked-autoencoding vs plain supervised classification -- vits-sup-in
# is NOT MAE-pretrained, unlike every other branch in this file). "256" in
# the name means mvp's own load() uses img_size=256 instead of 224 --
# preserved in the transform below via MVP_IMG_SIZE.
MVP_ARCHS = {
    'mvp_vits_mae_hoi': 'vits-mae-hoi',
    'mvp_vits_mae_in': 'vits-mae-in',
    'mvp_vits_sup_in': 'vits-sup-in',
    'mvp_vitb_mae_egosoup': 'vitb-mae-egosoup',
    'mvp_vitl256_mae_egosoup': 'vitl-256-mae-egosoup',
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
        # A ModuleList, not a plain list: otherwise the sub-models are
        # invisible to .parameters(), .modules(), .to() and state_dict(), so
        # nothing trains, BatchNorm can't be found and saved encoders are empty.
        self.models = nn.ModuleList(models)
        assert all(models[0].training == m.training for m in models)
        self.training = models[0].training

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

    # random is trained from scratch alongside the policy and has no
    # pretrained-weights first layer to be compatible with -- it's the
    # only branch that can accept a frame_stack-widened input (in_channels
    # a multiple of 3, not just 3). Every other branch below is a
    # pretrained backbone with a fixed 3-channel first layer.
    assert in_channels == 3 or embedding_name == 'random', \
        'Current models accept 3-channel inputs only.'

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
        # No ImageNet normalize here (unlike the shared default block
        # above): random has no pretrained-normalization convention to
        # match, and Normalize's 3-element mean/std would shape-mismatch
        # against a frame_stack-widened, non-3-channel input anyway.
        transforms = nn.Sequential(
            T.Resize(256),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
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
        checkpoint = torch.load(_ckpt('mae_pretrain_vit_base.pth'), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        forward_fn = _forward_mae
    elif embedding_name == 'mae_large':
        model = mae_vit_large_patch16()
        checkpoint = torch.load(_ckpt('mae_pretrain_vit_large.pth'), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        forward_fn = _forward_mae
    elif embedding_name == 'mae_huge':
        model = mae_vit_huge_patch14()
        checkpoint = torch.load(_ckpt('mae_pretrain_vit_huge.pth'), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        forward_fn = _forward_mae

    # ViT-S/B/L-16 in MAE's layout, from scratch (VIT_ARCHS). Features: the
    # CLS token after the final norm, as MAE/MVP/VC-1 use. Input pipeline:
    # MAE's evaluation transform (bicubic resize to 256, center crop 224,
    # ImageNet mean/std).
    elif embedding_name in VIT_ARCHS:
        if pretrained:
            raise ValueError(
                f"{embedding_name} has no default weights: set embedding.pretrained=false "
                "(random init, e.g. for pretrain.py) or embedding.checkpoint=<file>")
        model = VIT_ARCHS[embedding_name]()
        transforms = nn.Sequential(
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )

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
        checkpoint = torch.load(_ckpt(ckpt_name), map_location='cpu', weights_only=False)
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
            raise ImportError(
                "vip requires (without its dependencies -- its setup.py pins hydra-core 1.1.1, "
                "which breaks this repo): pip install gdown && "
                "pip install --no-deps git+https://github.com/facebookresearch/vip.git")
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
        checkpoint = torch.load(_ckpt('rl3d_resnet18.tar'), map_location='cpu', weights_only=False)
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

    # MVP (ir413/mvp) -- MAE-ViT lineage like VC-1, but genuinely
    # pip-installable (`pip install git+https://github.com/ir413/mvp`)
    # with no version-pin conflict (their setup.py requires bare `timm`,
    # unpinned) -- unlike VC-1, no reason to bypass their package here.
    # mvp.load() auto-downloads via its own cache_url() helper (Berkeley
    # Box-hosted), same auto-download category as R3M/VIP/LIV, not the
    # manual-checkpoint category MAE/VC-1/RL3D are in.
    elif embedding_name in MVP_ARCHS:
        if not _HAS_MVP:
            raise ImportError("mvp requires: pip install git+https://github.com/ir413/mvp")
        if not pretrained:
            raise NotImplementedError("MVP has no random-init path -- mvp.load() always loads its pretrained checkpoint.")
        model = mvp.load(MVP_ARCHS[embedding_name])
        # Verified against their own source (mvp/backbones/model_zoo.py,
        # vit.py): load() returns a bare model, no DataParallel wrapping or
        # .cuda() call anywhere in it -- unlike R3M/VIP/LIV, nothing to
        # unwrap or move back to CPU here.
        img_size = 256 if '256' in embedding_name else 224
        # Verified against mvp/bc/dataset.py: standard ImageNet mean/std,
        # /255 scaling -- same convention as every other MAE-family branch
        # in this file.
        transforms = nn.Sequential(
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )
        # forward_fn stays _forward_default: MVP's own forward() already
        # extracts and normalizes the CLS token internally (verified in
        # vit.py -- forward_norm(extract_feat(x))), so model(x) alone
        # returns the final embedding, no _forward_mae-style wrapper needed.

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

    # DINOv2 (Oquab et al., 2023) via torch.hub, which fetches the repo code
    # and weights on first use and caches both under $TORCH_HOME (default
    # ~/.cache/torch). On a cluster, point TORCH_HOME at scratch, and load
    # once from a login node if compute nodes have no internet. The repo is
    # pinned to ':main': without a branch, torch.hub asks GitHub for the
    # default branch on every load, with no timeout -- on a compute node
    # that silently drops outbound traffic, each build would hang.
    # Transforms follow DINOv2's own evaluation preprocessing (bicubic
    # resize to 256, center crop 224, ImageNet mean/std); input sizes must be
    # a multiple of the 14px patch, so a `preprocess` override has to keep that.
    elif embedding_name in DINOV2_ARCHS:
        override = _resolve_model_dir_override(embedding_name, model_dir)
        model = torch.hub.load('facebookresearch/dinov2:main', embedding_name,
                               pretrained=pretrained and override is None,
                               trust_repo=True)
        if override is not None:
            _load_custom_state_dict(model, override, embedding_name)
        transforms = nn.Sequential(
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )

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

# Steps allowed after the [0, 1] point: value rescaling only, nothing spatial.
_POST_AUG_STEPS = (T.Normalize, _ScaleTo255)


def _split_transforms(transforms: nn.Sequential):
    """Split an encoder's transform pipeline where the frame is a float in
    [0, 1] at the encoder's input resolution: after the last
    T.ConvertImageDtype. Returns (pre_aug, post_aug); augmentation runs in
    between (src/augmentations.py's contract).

    What follows the cut is value rescaling only: mean/std T.Normalize for
    most encoders, _ScaleTo255 for R3M/VIP, whose models expect [0, 255]
    input. Splitting at a trailing T.Normalize alone would leave R3M/VIP's
    preprocess() output in [0, 255], where colour transforms clamp the frame
    to [0, 1] without any error.

    A pipeline with no ConvertImageDtype (maskrcnn_l3, which works in BGR
    [0, 255] throughout) falls back to splitting before a trailing
    T.Normalize; its preprocess() output isn't in [0, 1], which
    augmentation's input check rejects.
    """
    steps = list(transforms.children())
    converts = [i for i, t in enumerate(steps) if isinstance(t, T.ConvertImageDtype)]
    if converts:
        cut = converts[-1] + 1
    elif steps and isinstance(steps[-1], T.Normalize):
        cut = len(steps) - 1
    else:
        cut = len(steps)
    post = steps[cut:]
    if not all(isinstance(t, _POST_AUG_STEPS) for t in post):
        raise RuntimeError(
            "transform steps after T.ConvertImageDtype must only rescale values "
            f"({[type(t).__name__ for t in _POST_AUG_STEPS]}), got "
            f"{[type(t).__name__ for t in post]} -- a spatial step there would "
            "run after augmentation."
        )
    return nn.Sequential(*steps[:cut]), (nn.Sequential(*post) if post else nn.Identity())


PREPROCESS_KEYS = ("resize", "crop", "interpolation", "mean", "std")


def _canonical_preprocess(spec):
    """A preprocess spec in one canonical form, so equivalent pipelines
    compare equal however they were written (missing vs null keys, ints vs
    floats, the default interpolation spelled out or not)."""
    if spec is None:
        return None
    spec = dict(spec)
    out = {k: spec.get(k) for k in PREPROCESS_KEYS}
    if out["resize"] is not None:
        out["resize"] = int(out["resize"])
        out["interpolation"] = str(out["interpolation"] or "bilinear").lower()
    if out["crop"] is not None:
        out["crop"] = int(out["crop"])
    for k in ("mean", "std"):
        if out[k] is not None:
            out[k] = [float(v) for v in out[k]]
    return out


def preprocess_transforms(spec) -> nn.Sequential:
    """Build an encoder input pipeline from a config spec, replacing the
    encoder's built-in one (cfg.embedding.preprocess):

        resize         resize the shorter side to this size (null: keep the
                       input size)
        crop           center crop to this square size (null: no crop)
        interpolation  for resize: bilinear (default) | bicubic | nearest
        mean, std      per-channel normalisation after scaling to [0, 1]
                       (both null: none)

    e.g. PIE-G feeds 84px frames as x/255 - 0.5:
        {resize: null, crop: null, mean: [0.5, 0.5, 0.5], std: [1, 1, 1]}

    Resizing happens on the uint8 frame, before conversion to float, like
    every built-in pipeline here. The result splits cleanly for augmentation
    (_split_transforms).
    """
    spec = dict(spec)
    unknown = sorted(set(spec) - set(PREPROCESS_KEYS))
    if unknown:
        raise ValueError(f"unknown preprocess keys {unknown}; expected a subset of {list(PREPROCESS_KEYS)}")
    steps = []
    if spec.get("resize") is not None:
        mode = str(spec.get("interpolation") or "bilinear").upper()
        if mode not in T.InterpolationMode.__members__:
            raise ValueError(f"preprocess.interpolation={spec['interpolation']!r}; "
                             "expected bilinear, bicubic or nearest")
        steps.append(T.Resize(int(spec["resize"]), interpolation=T.InterpolationMode[mode]))
    elif spec.get("interpolation") is not None:
        raise ValueError("preprocess.interpolation has no effect without preprocess.resize")
    if spec.get("crop") is not None:
        steps.append(T.CenterCrop(int(spec["crop"])))
    steps.append(T.ConvertImageDtype(torch.float))
    mean, std = spec.get("mean"), spec.get("std")
    if (mean is None) != (std is None):
        raise ValueError("preprocess needs both mean and std, or neither")
    if mean is not None:
        if len(mean) != len(std):
            raise ValueError(f"preprocess mean and std lengths differ: {list(mean)} vs {list(std)}")
        steps.append(T.Normalize([float(m) for m in mean], [float(s) for s in std]))
    return nn.Sequential(*steps)


class EmbeddingNet(nn.Module):
    """
    Input shape must be (N, C, H, W), where N is the number of frames.
    The class will then take care of transforming and normalizing frames.
    The output shape will be (N, O), where O is the embedding size.

    encode() = encode_preprocessed(preprocess(x)). The split exists so a
    caller can augment in between: preprocess() returns float [0, 1] frames
    at this encoder's input resolution (after resize/crop, before mean/std
    normalisation), which is the space src/augmentations.py operates in.

    Optional fields (cfg.embedding.arch / checkpoint / preprocess):
      arch        which architecture to build; defaults to embedding_name,
                  which then only labels the run (e.g. name: my_run,
                  arch: resnet18).
      checkpoint  load these weights instead of arch's default ones: an
                  encoder saved by this repo (best_encoder.pt, or
                  final_checkpoint.pt's 'encoder'; backbone + head), or a
                  bare backbone state_dict (e.g. a pretraining run). Only
                  for archs in CHECKPOINT_ARCHS. model_dir doesn't apply.
      preprocess  replace arch's built-in input pipeline; see
                  preprocess_transforms().

      layer       stop the backbone at this module; see src/truncation.py.
      head        turn its output into a vector: avgpool (default with
                  `layer`), cls, flatten, or linear / fc (trainable);
                  head_dim sizes those.

    input_size: side length of the frames that will be encoded. Only matters
    when the pipeline doesn't resize (a `preprocess` without resize/crop),
    where the output size can depend on the input size; it's measured once
    here with a dummy frame of this size (default 64).
    """
    def __init__(self, embedding_name, in_channels=3, pretrained=True, train=False,
                 disable_cuda=False, model_dir=None, arch=None, checkpoint=None,
                 preprocess=None, input_size=None, layer=None, head=None, head_dim=None,
                 frame_stack=1, frame_fusion=None):
        super(EmbeddingNet, self).__init__()

        self.embedding_name = embedding_name

        if self.embedding_name == 'true_state':
            return

        self.arch = arch or embedding_name
        self.checkpoint = checkpoint
        if checkpoint is not None and self.arch not in CHECKPOINT_ARCHS:
            raise ValueError(
                f"embedding.checkpoint can't be loaded into arch={self.arch!r}: its "
                "builder always loads or downloads its own default weights first. "
                f"Supported: {sorted(CHECKPOINT_ARCHS)}"
            )

        # Frame stacking: the input carries frame_stack frames of in_channels
        # each. 'channels' feeds the whole stack to the backbone as one wide
        # input (only `random` accepts that); 'concat'/'diff' encode each
        # frame separately and fuse the outputs before the head (see
        # src/truncation.py's fuse_frames).
        from src.truncation import FRAME_FUSIONS
        self.frame_stack = int(frame_stack or 1)
        if self.frame_stack > 1:
            frame_fusion = frame_fusion or ("channels" if self.arch == "random" else "concat")
            if frame_fusion not in FRAME_FUSIONS:
                raise ValueError(f"embedding.frame_fusion={frame_fusion!r}; expected one of {FRAME_FUSIONS}")
            if frame_fusion == "channels" and self.arch != "random":
                raise ValueError(
                    f"embedding.frame_fusion=channels feeds a {in_channels * self.frame_stack}-channel "
                    f"input to the backbone, which only `random` accepts (arch={self.arch!r} "
                    "has a fixed 3-channel first layer). Use concat or diff."
                )
        elif frame_fusion is not None:
            raise ValueError("embedding.frame_fusion needs embedding.frame_stack > 1")
        self.frame_fusion = frame_fusion if self.frame_stack > 1 else None
        self._per_frame = self.frame_fusion in ("concat", "diff")
        self._seq_first = None   # token layout of per-frame outputs, found below
        self.in_channels = in_channels * self.frame_stack
        backbone_channels = self.in_channels if self.frame_fusion == "channels" else in_channels

        # With a checkpoint: build without default weights (no download); the
        # file is loaded below, once the head exists.
        self.embedding, self.transforms, self._forward_fn = _get_embedding(
            self.arch, backbone_channels, pretrained and checkpoint is None, train,
            model_dir=None if checkpoint is not None else model_dir,
        )
        if preprocess is not None:
            self.transforms = preprocess_transforms(preprocess)
        self.layer = None
        if layer is not None:
            from src.truncation import truncate
            self._forward_fn, self.layer = truncate(self.embedding, self._forward_fn, layer)
            head = head or "avgpool"
        # How this encoder was built, beyond its weights; compared against a
        # loaded checkpoint's record of the same (_load_checkpoint).
        from src.truncation import DEFAULT_HEAD_DIM, TRAINABLE_HEADS
        self.build_spec = {
            "arch": self.arch, "layer": layer, "head": head,
            "head_dim": (head_dim or DEFAULT_HEAD_DIM) if head in TRAINABLE_HEADS else None,
            "preprocess": _canonical_preprocess(preprocess),
            "frame_stack": self.frame_stack,
            "frame_fusion": self.frame_fusion,
        }
        # Differences from a loaded checkpoint's build record; empty if none.
        # Kept so the training scripts can log them, not just print them.
        self.build_warnings = []

        self._pre_aug, self._normalize = _split_transforms(self.transforms)

        from src.truncation import make_head
        self.input_size = input_size or 64
        # A batch of 2, so make_head can tell a (T, N, D) token layout apart
        # from (N, T, D).
        dummy_in = torch.zeros(2, self.in_channels, self.input_size, self.input_size)
        with torch.no_grad():
            dummy_pre = self._preprocess_frames(dummy_in)
            self.in_shape = dummy_pre.shape[1:]
            dummy_raw = self._raw_output(dummy_pre)
            # Fused per-frame outputs are already batch-first.
            self.head = make_head(head, dummy_raw, batch=2, head_dim=head_dim,
                                  seq_first=False if self._per_frame else None)
            self.out_size = int(np.prod(self.head(dummy_raw).shape[1:]))

        # Whether the head's weights came from `checkpoint` (only a file saved
        # by this repo with head.* keys); a trainable head on a frozen encoder
        # needs them (encoder_config.check_head_is_trained).
        self.head_loaded = False
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

        bad = [
            name for name, tensor in
            list(self.named_parameters()) + list(self.named_buffers())
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

        if torch.cuda.is_available() and not disable_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.embedding = self.embedding.to(device=self.device)
        self.head = self.head.to(device=self.device)
        self.training = self.embedding.training

    def _forward(self, observation):
        return self._forward_fn(self.embedding, observation)

    def encode(self, observation):
        """
        Same computation as forward(), but returns a (N, out_size)
        torch.Tensor on this module's device -- batch dimension always
        intact (forward()'s squeeze() silently drops it when N==1) and no
        numpy conversion (forward() always detaches to numpy, even in
        training mode, which breaks callers that need a differentiable
        tensor -- e.g. a trainable head sitting on top of this embedding
        inside another nn.Module's forward pass, like an SB3 features
        extractor).
        """
        return self.encode_preprocessed(self.preprocess(observation))

    def normalize(self, x):
        """preprocess() output (maybe augmented) -> the backbone's input: the
        mean/std normalisation (or [0, 255] rescale) encode_preprocessed()
        applies itself. For objectives that run the backbone directly (MAE)."""
        return self._normalize(x)

    def preprocess(self, observation):
        """(N, C, H, W) frames [channels-first, as returned by gym_wrappers]
        -> float (N, C, h, w) in [0, 1] at this encoder's input resolution:
        resize/crop/dtype only, no mean/std normalisation."""
        observation = observation.to(device=self.device)
        if not observation.is_contiguous():
            observation = observation.contiguous()
        return self._preprocess_frames(observation)

    def _preprocess_frames(self, x):
        """_pre_aug, per frame when frames are encoded separately; the
        output keeps the stacked (N, k*C, h, w) layout, so an augmentation
        applied to it treats the whole stack alike (same shift, same
        overlay image)."""
        if not self._per_frame:
            return self._pre_aug(x)
        n, c = x.shape[:2]
        x = self._pre_aug(x.reshape(n * self.frame_stack, c // self.frame_stack, *x.shape[2:]))
        return x.reshape(n, c, *x.shape[2:])

    def _raw_output(self, x):
        """Preprocessed (and maybe augmented) frames -> the backbone's raw
        output, with stacked frames fused (batch-first)."""
        if not self._per_frame:
            x = self._normalize(x)                    # mean/std normalisation or [0, 255] rescale
            # Fold any extra leading dims into the batch. Taken from the
            # input, not in_shape, so frames at a size other than the one
            # measured in __init__ still pass when the pipeline doesn't resize.
            return self._forward(x.reshape(-1, *x.shape[-3:]))
        from src.truncation import fuse_frames, tokens_seq_first
        n, c = x.shape[:2]
        frames = x.reshape(n * self.frame_stack, c // self.frame_stack, *x.shape[2:])
        out = self._forward(self._normalize(frames))
        if out.ndim == 3:
            if self._seq_first is None:
                self._seq_first = tokens_seq_first(out, frames.shape[0])
            if self._seq_first:
                out = out.transpose(0, 1)
        return fuse_frames(out, n, self.frame_stack, self.frame_fusion)

    def encode_preprocessed(self, observation):
        """preprocess() output, optionally augmented -> (N, out_size)."""
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
        out = self.head(self._raw_output(observation))
        if out.shape[1] != self.out_size:
            # A plain reshape(-1, out_size) could silently fold features into
            # the batch dimension here instead of failing.
            raise RuntimeError(
                f"{self.embedding_name!r} produced {out.shape[1]} features per frame for "
                f"{tuple(observation.shape[-2:])} input, but {self.out_size} were measured "
                f"at input_size={self.input_size}. Without a resize in the pipeline the "
                "output size depends on the frame size: pass input_size=<frame side>."
            )
        return out

    def _load_checkpoint(self, path):
        """Two formats:
          - saved by this repo (EmbeddingNet.state_dict(), e.g. best_encoder.pt,
            or final_checkpoint.pt's 'encoder'): keys embedding.* and head.*.
            Loaded whole, so a trained head comes back too. A file without
            head weights loads the backbone and leaves the head fresh; a file
            whose head doesn't fit the configured one is an error.
          - a bare backbone state_dict (a pretraining run, torchvision): loaded
            into the backbone. Keys of modules this repo replaces with
            Identity (a torchvision ResNet's fc.*) are dropped first.

        Files saved by the training scripts also record the embedding config
        they were built with. A layer, head or preprocess that differs from
        the current one gives a different representation from the same
        weights, without any key mismatch (an avgpool head has no weights; a
        preprocess change never touches them), so that's warned about.
        """
        sd, record = _read_checkpoint(path)
        if record['embedding'] is not None:
            self._warn_on_build_differences(path, record)
        if any(k.startswith('embedding.') for k in sd):
            missing, unexpected = self.load_state_dict(sd, strict=False)
            missing = [k for k in missing if not k.endswith('num_batches_tracked')]
            file_has_head = any(k.startswith('head.') for k in sd)
            bad_missing = [k for k in missing if not k.startswith('head.') or file_has_head]
            if unexpected or bad_missing:
                raise RuntimeError(
                    f"{path!r} doesn't match this {self.embedding_name!r} encoder -- "
                    f"missing keys: {bad_missing[:5]}{'...' if len(bad_missing) > 5 else ''}, "
                    f"unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''} "
                    "(a head mismatch means embedding.head/head_dim differ from the saved run)"
                )
            self.head_loaded = file_has_head
            if missing:
                print(f"[embeddings] {path!r} has no head weights; the head starts fresh.")
            return
        dropped = [n for n, m in self.embedding.named_modules() if n and isinstance(m, nn.Identity)]
        sd = {k: v for k, v in sd.items() if not any(k.startswith(d + '.') for d in dropped)}
        if self.arch in VIT_ARCHS:
            # A whole MAE model (encoder + decoder, e.g. MAE's visualisation
            # checkpoints) or a timm ViT with its classifier: keep the encoder.
            sd = {k: v for k, v in sd.items()
                  if not k.startswith(('decoder_', 'mask_token', 'head.', 'fc_norm.'))}
        _load_custom_state_dict(self.embedding, sd, self.embedding_name, strict=True)

    def _warn_on_build_differences(self, path, record):
        from src.truncation import DEFAULT_HEAD_DIM, TRAINABLE_HEADS
        meta = record['embedding']
        layer = meta.get('layer')
        head = meta.get('head') or ('avgpool' if layer is not None else None)
        arch = meta.get('arch') or meta.get('name')
        frame_stack = int(meta.get('frame_stack') or 1)
        saved = {
            'arch': arch,
            'layer': layer,
            'head': head,
            'head_dim': (meta.get('head_dim') or DEFAULT_HEAD_DIM) if head in TRAINABLE_HEADS else None,
            'preprocess': _canonical_preprocess(meta.get('preprocess')),
            'frame_stack': frame_stack,
            'frame_fusion': (meta.get('frame_fusion') or ('channels' if arch == 'random' else 'concat'))
                            if frame_stack > 1 else None,
        }
        diffs = [f"{k}: saved {saved[k]!r}, now {self.build_spec[k]!r}"
                 for k in saved if saved[k] != self.build_spec[k]]
        # Without a resize, the frame size is part of what the encoder sees
        # (a linear/flatten head would fail on it anyway; avgpool and cls
        # would silently run at a different resolution).
        pre = self.build_spec['preprocess']
        if (pre is not None and pre['resize'] is None and record['input_size'] is not None
                and record['input_size'] != self.input_size):
            diffs.append(f"frame size (the pipeline doesn't resize): saved "
                         f"{record['input_size']}, now {self.input_size}")
        self.build_warnings = diffs
        if diffs:
            print(f"[embeddings] Warning: {path!r} was built differently from this "
                  f"encoder, so its weights give a different representation here -- "
                  + "; ".join(diffs))

    def _batchnorms(self):
        return [m for m in self.embedding.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def has_batchnorm(self) -> bool:
        return bool(self._batchnorms())

    def set_bn_train(self, train: bool) -> None:
        """BatchNorm layers only: train mode normalises with each batch's own
        statistics and updates the running ones (encoder.bn=adapt, as in
        PIE-G); eval mode uses the running statistics. Everything else
        (e.g. a ViT's dropout) stays in eval mode either way."""
        for m in self._batchnorms():
            m.train(train)

    def bn_stats(self):
        """BatchNorm running statistics (running_mean/var), as CPU copies."""
        return {n: b.detach().cpu().clone() for n, b in self.embedding.named_buffers()
                if n.endswith(("running_mean", "running_var"))}

    def bn_stat_drift(self, reference) -> float:
        """L2 distance of the running statistics from `reference`
        (a bn_stats() snapshot): how far encoder.bn=adapt has moved them."""
        current = self.bn_stats()
        return float(torch.sqrt(sum(((current[n] - reference[n]) ** 2).sum() for n in reference)
                                + torch.zeros(())))

    def set_trainable(self, mode: str) -> None:
        """Which parameters get requires_grad: 'none', 'all' (backbone and
        head), or 'head' (a linear head on a frozen backbone, as in PIE-G).
        Separate from train()/eval(): the training scripts keep the encoder
        in eval() so BatchNorm statistics stay frozen either way."""
        if mode not in ("none", "all", "head"):
            raise ValueError(f"trainable mode {mode!r}; expected none, all or head")
        for p in self.embedding.parameters():
            p.requires_grad_(mode == "all")
        for p in self.head.parameters():
            p.requires_grad_(mode != "none")

    def forward(self, observation):
        if self.embedding_name == 'true_state':
            return observation.squeeze().cpu().numpy()
        return self.encode(observation).squeeze().cpu().numpy()
