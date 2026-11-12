"""
Voltron's released reproductions (Karamcheti et al., "Language-Driven
Representation Learning for Robotics", RSS 2023; siddk/voltron-robotics, MIT
license), converted to this repo's architectures. Nothing from Voltron needs
to be installed.

Voltron saves (state_dict, optimizer_state); the state_dict uses its own
module names. Three released image encoders, all trained on
Something-Something v2:

  r-mvp       MVP's recipe (MAE): ViT-S/16, 6 heads, CLS token  -> vit_s16_h6
  r-r3m-vit   R3M's objectives:   ViT-S/16, 6 heads, no CLS token,
                                  features = mean of the normed tokens -> vit_s16_h6_gap
  r-r3m-rn50  R3M's objectives:   torchvision ResNet-50 (fc removed) -> resnet50

The encoders compute the same function as Voltron's code (tests/test_voltron.py
checks the converted weights against a reference port of its forward pass).
One exception, on Voltron's side: VMVP.encode() concatenates the CLS token
with torch.cat([cls, patches]) -- no dim=1 -- so as released it fails for any
batch; MVP's code, which it cites, uses dim=1, as MAEEncoder does.

Inputs (Voltron's materialize.py): r-mvp takes ImageNet-normalised images;
both R-R3M models take raw [0, 255] pixels with no normalisation (Voltron
trained them that way, unlike official R3M, which normalises internally).
The embedding configs (configs/embedding/voltron_*.yaml) set this.
"""

from typing import Dict

import torch

# Voltron block -> MAE/timm block (src/vision_models/mae.py's Block).
_BLOCK_RENAMES = (
    ("pre_norm_attn.", "norm1."),
    ("pre_norm_mlp.", "norm2."),
    ("mlp.0.0.", "mlp.fc1."),
    ("mlp.2.", "mlp.fc2."),
)
# Parts that aren't the image encoder: MAE decoder (R-MVP), language model and
# reward head (R-R3M).
_DROPPED = ("decoder_", "encoder2decoder.", "mask_token", "lm.", "language_reward.")


def is_voltron_checkpoint(raw) -> bool:
    """A torch.load()ed file in Voltron's format: (state_dict, optimizer_state)
    with Voltron's module names."""
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2 and isinstance(raw[0], dict)):
        return False
    keys = raw[0].keys()
    return any(k.startswith(("patch2embed.", "resnet.conv1.")) for k in keys)


def _rename_block(key: str) -> str:
    for old, new in _BLOCK_RENAMES:
        key = key.replace(old, new)
    return key


def convert_voltron(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Voltron state_dict -> the image encoder's state_dict in this repo's
    names (MAEEncoder for the ViTs, a torchvision ResNet-50 without fc)."""
    sd = {k: v for k, v in state_dict.items() if not k.startswith(_DROPPED)}
    if any(k.startswith("resnet.") for k in sd):
        return {k[len("resnet."):]: v for k, v in sd.items() if k.startswith("resnet.")}
    out = {}
    for k, v in sd.items():
        if k.startswith("patch2embed."):
            out["patch_embed." + k[len("patch2embed."):]] = v
        elif k in ("encoder_pe", "pe"):                       # R-MVP (with CLS) / R-R3M (without)
            out["pos_embed"] = v
        elif k == "cls_token":
            out["cls_token"] = v
        elif k.startswith(("encoder_blocks.", "blocks.")):
            rest = k.split(".", 1)[1]
            out["blocks." + _rename_block(rest)] = v
        elif k.startswith(("encoder_norm.", "norm.")):
            out["norm." + k.split(".", 1)[1]] = v
        else:
            raise KeyError(f"unexpected key in a Voltron checkpoint: {k!r}")
    return out
