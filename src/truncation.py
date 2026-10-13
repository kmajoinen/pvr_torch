"""
Layer truncation and output heads for EmbeddingNet (cfg.embedding.layer /
head / head_dim).

Truncation is generic: a forward hook on the chosen module captures its
output and stops the forward pass there, so later layers never run. The
encoder's own forward logic (R3M's input scaling, CLIP's encode_image, ...)
still runs unchanged up to that point, which is why this works across model
families without per-model code. If the hook never fires -- e.g. it landed
on CLIP's text tower while encode_image only runs the visual one -- that's
an error, not a silent fall-through to the full model.

`layer` names the module whose output becomes the representation:
  "layer1".."layer4"   ResNet stages (torchvision, MoCo, places, R3M/VIP,
                       CLIP/OpenCLIP/LIV RN50; RL3D's numbered Sequential
                       is mapped too)
  k (int >= 1)         the output of the first k transformer blocks (MAE,
                       MVP, VC-1, CLIP ViT, DINOv2)
  "a.b.c"              any exact module path, as a fallback

Truncated outputs are raw: no final norm layer (a ViT's last-block CLS token
therefore differs from the model's normal output, which applies the final
LayerNorm). The RL heads normalise their input anyway.

`head` turns that output into a vector:
  avgpool   ResNet maps: mean over H, W. Tokens: mean over patch tokens
            (token 0, the CLS token, excluded). No parameters. Default when
            `layer` is set.
  cls       tokens only: token 0. No parameters.
  flatten   the whole map or token sequence as one vector. No parameters, so
            the RL heads' own first layers do the projecting. Very wide:
            ResNet18 layer2 is 128 x 11 x 11 = 15.5k at 84px, 100k at 224px.
  linear    flatten, then a trainable Linear(-> head_dim) + LayerNorm: PIE-G's
            encoder projection (head_dim default 1024; PIE-G's actor and
            critic then each add their own Linear-LayerNorm-Tanh trunk, which
            this SAC's heads don't have). Needs the encoder to train;
            encoder.trainable=head trains it on a frozen backbone. Its size
            is (flattened width) x head_dim: ~16M parameters for ResNet18
            layer2 at 84px, ~100M at 224px.
  null      without `layer`: the model's own output, flattened (unchanged
            behaviour).
"""

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

HEADS = ("avgpool", "cls", "flatten", "linear")
DEFAULT_HEAD_DIM = 1024


class _Stop(Exception):
    def __init__(self, output):
        super().__init__()
        self.output = output


def _block_lists(model: nn.Module):
    # ModuleList in timm/DINOv2/MAE; nn.Sequential in OpenAI CLIP.
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.ModuleList, nn.Sequential))
            and n.split(".")[-1] in ("blocks", "resblocks")]


def resolve_layer(model: nn.Module, layer: Union[int, str]) -> Tuple[str, nn.Module]:
    """`layer` (see module docstring) -> (qualified name, module)."""
    modules = dict(model.named_modules())

    if isinstance(layer, bool) or not isinstance(layer, (int, str)):
        raise ValueError(f"embedding.layer must be a stage name, block count or module path, got {layer!r}")

    if isinstance(layer, int):
        lists = _block_lists(model)
        if len(lists) > 1:   # CLIP: text and visual towers both have resblocks
            visual = [(n, m) for n, m in lists if "visual" in n.split(".")]
            lists = visual or lists
        if len(lists) != 1:
            raise ValueError(
                f"embedding.layer={layer}: can't pick the transformer block list "
                f"(candidates: {[n for n, _ in lists] or 'none'}). Give a module path, "
                "e.g. embedding.layer=blocks.5"
            )
        name, blocks = lists[0]
        if not 1 <= layer <= len(blocks):
            raise ValueError(f"embedding.layer={layer}: {name} has {len(blocks)} blocks")
        return f"{name}.{layer - 1}", blocks[layer - 1]

    if layer in modules:
        return layer, modules[layer]
    matches = [n for n in modules if n.endswith("." + layer)]
    if len(matches) > 1:
        visual = [n for n in matches if "visual" in n.split(".")]
        matches = visual if len(visual) == 1 else matches
    if len(matches) == 1:
        return matches[0], modules[matches[0]]
    if not matches and layer.startswith("layer") and layer[5:].isdigit():
        # torchvision ResNet children flattened into a numbered Sequential
        # (RL3D): conv1, bn1, relu, maxpool, layer1..layer4 -> "4".."7".
        k = int(layer[5:])
        stage = modules.get(str(3 + k))
        if isinstance(model, nn.Sequential) and 1 <= k <= 4 and isinstance(stage, nn.Sequential):
            return str(3 + k), stage
    raise ValueError(
        f"embedding.layer={layer!r} doesn't match {'several modules: ' + str(matches) if matches else 'any module'}. "
        "Use a full module path."
    )


def truncate(model: nn.Module, forward_fn: Callable, layer: Union[int, str]) -> Tuple[Callable, str]:
    """Wrap forward_fn(model, x) so it returns `layer`'s output and stops
    there. Returns (new forward_fn, qualified layer name)."""
    name, module = resolve_layer(model, layer)

    def hook(_module, _inputs, output):
        raise _Stop(output[0] if isinstance(output, tuple) else output)

    def truncated_forward(m, x):
        handle = module.register_forward_hook(hook)
        try:
            forward_fn(m, x)
        except _Stop as stop:
            return stop.output
        finally:
            handle.remove()
        raise RuntimeError(
            f"embedding.layer resolved to {name!r}, but the encoder's forward pass never ran it"
        )

    return truncated_forward, name


# ------------------------------------------------------------------------------
# Heads
# ------------------------------------------------------------------------------

class _Flatten(nn.Module):
    def __init__(self, seq_first: bool = False):
        super().__init__()
        self.seq_first = seq_first

    def forward(self, x):
        if self.seq_first:
            x = x.transpose(0, 1)
        return x.reshape(x.shape[0], -1)


class _SpatialAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


class _TokenPool(nn.Module):
    def __init__(self, mode: str, seq_first: bool):
        super().__init__()
        self.mode, self.seq_first = mode, seq_first

    def forward(self, x):
        if self.seq_first:   # (T, N, D) -> (N, T, D), e.g. OpenAI CLIP
            x = x.transpose(0, 1)
        return x[:, 0] if self.mode == "cls" else x[:, 1:].mean(dim=1)


class _LinearHead(nn.Module):
    """PIE-G's projection: flatten -> Linear (orthogonal init, zero bias) ->
    LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, seq_first: bool):
        super().__init__()
        self.seq_first = seq_first
        self.fc = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        if self.seq_first:
            x = x.transpose(0, 1)
        return self.ln(self.fc(x.reshape(x.shape[0], -1)))


def make_head(head: Optional[str], sample: torch.Tensor, batch: int,
              head_dim: Optional[int] = None) -> nn.Module:
    """Head for an encoder whose raw output on a dummy batch of `batch`
    frames is `sample`."""
    if head is not None and head not in HEADS:
        raise ValueError(f"embedding.head={head!r}; expected one of {HEADS}")
    if head_dim is not None and head != "linear":
        raise ValueError(f"embedding.head_dim only applies to head=linear (got head={head})")

    if sample.ndim == 3:
        if sample.shape[0] == batch and sample.shape[1] != batch:
            seq_first = False
        elif sample.shape[1] == batch and sample.shape[0] != batch:
            seq_first = True
        else:
            raise RuntimeError(f"can't tell the token layout of a {tuple(sample.shape)} output")
        kind = "tokens"
    elif sample.ndim == 4:
        kind, seq_first = "map", False
    elif sample.ndim == 2:
        kind, seq_first = "vector", False
    else:
        raise RuntimeError(f"unsupported encoder output shape {tuple(sample.shape)}")

    if head == "linear":
        in_dim = sample[:, 0].numel() if seq_first else sample[0].numel()
        return _LinearHead(in_dim, head_dim or DEFAULT_HEAD_DIM, seq_first)
    if head == "flatten":
        if kind == "vector":
            raise ValueError("embedding.head=flatten: this output is already a vector")
        return _Flatten(seq_first)
    if head == "cls":
        if kind != "tokens":
            raise ValueError(f"embedding.head=cls needs transformer tokens; this layer outputs a {kind}")
        return _TokenPool("cls", seq_first)
    if head == "avgpool":
        if kind == "vector":
            raise ValueError("embedding.head=avgpool: this output is already a vector (nothing to pool)")
        return _SpatialAvgPool() if kind == "map" else _TokenPool("mean", seq_first)
    # head=None
    if kind == "tokens":
        raise ValueError("a transformer layer's tokens need embedding.head (avgpool, cls, flatten or linear)")
    return _Flatten()
