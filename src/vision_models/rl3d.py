"""
RL3D's 2D feature-extraction backbone (YanjieZe/rl3d, CO3D-pretrained
3D visual representation for RL). Vendored, not pip-installed: the repo
has no setup.py/pyproject.toml at all (conda-env-only, `load_3d.py` is a
bare script at the repo root) -- same situation as this repo's other
vendored third-party backbones (moco.py, mae.py, resnet.py), so the
relevant architecture code is adapted here instead.

Only the 2D encoder path is vendored (use_3d=False in their load_3d.py).
The full Encoder3D also has a ConvTranspose3d decoder head
(conv3d_1/conv3d_2) for the 3D path -- irrelevant for embedding
extraction, dropped entirely, not just left unused.

Verified against the actual source (github.com/YanjieZe/rl3d/blob/main/
load_3d.py) rather than assumed: get_resnet18() is NOT a plain resnet18
truncation -- it also overrides layer4's first block to stride=(1, 1),
removing the final spatial downsampling a standard resnet18 does there.
Reproduced exactly below; this can't be gotten via this repo's existing
resnet_conv5-style helpers, which don't have this stride override.
"""
import torch.nn as nn
import torchvision.models as models


def rl3d_resnet18_backbone():
    """
    weights=None here (not ImageNet-pretrained), unlike the original
    get_resnet18()'s torchvision.models.resnet18(pretrained=True) --
    RL3D's own checkpoint overwrites every one of this backbone's
    parameters via load_state_dict immediately after construction (see
    the rl3d_resnet18 branch in src/embeddings.py), so the ImageNet
    starting point in the original code becomes irrelevant once that
    completes. Also sidesteps torchvision's pretrained=True kwarg, which
    is removed entirely in newer torchvision releases (this repo pins
    torchvision==0.27.1) -- their original literal call would not run.
    """
    model = models.resnet18(weights=None)
    feature = nn.Sequential(*list(model.children())[:-2])
    feature[7][0].conv1.stride = (1, 1)
    feature[7][0].downsample[0].stride = (1, 1)
    return feature
