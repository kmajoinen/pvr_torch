"""
Encoder objectives, in one registry for both uses:
  - auxiliary losses that train the encoder alongside, or instead of, the
    RL/BC losses (train_sac.py, train_drqv2.py, train_bc.py: encoder.aux)
  - pretraining objectives (pretrain.py: objectives), e.g. R3M
Selected from config as a list:

    encoder:
      aux:
        - {name: l2sp, weight: 1.0}
        - {name: my_loss, weight: 0.1, scale: 2.0}   # extra keys -> __init__

Adding a loss:
    @register_aux_loss("my_loss")
    class MyLoss(AuxLoss):
        def __init__(self, encoder, scale: float = 1.0):
            super().__init__(encoder)
            self.scale = scale
            self.head = nn.Linear(encoder.out_size, 64)   # optional learnable head

        def forward(self, batch: AuxBatch) -> torch.Tensor:
            ...   # return an UNWEIGHTED scalar

Contract:
  - __init__(encoder, **params) runs once, after the encoder is built and
    before training. `encoder` is the EmbeddingNet. Snapshot from it if needed
    (l2sp does), but don't store it as an attribute -- nn.Module would
    register it as a submodule. Use batch.encoder in forward() instead.
  - forward(batch) returns an unweighted scalar tensor. The training script
    multiplies it by `weight`, sums the terms and backpropagates into the
    encoder. It must not step optimizers or touch the RL/BC networks.
  - Parameters the loss owns (decoders, predictor heads) are optimised by the
    encoder's optimizer, at encoder.lr.
  - Declare what the loss reads from AuxBatch with the class attributes
    below. The training loop fills only what some loss declared (each costs
    an encoder pass), and configs that can't supply it are rejected before
    anything expensive runs:
        needs_features  = True   # batch.obs / batch.feats      (clean view)
        needs_augmented = True   # batch.obs_aug / batch.feats_aug
        needs_clips     = True   # batch.clip_feats: several frames per video
        needs_text      = True   # batch.text: one description per video
    needs_augmented also requires an augmentation with stage aux or both
    (see src/views.py). Clips and text come from video datasets, so only
    pretrain.py supplies them; the RL/BC scripts reject such objectives. A
    clip objective also says which frames it wants: clip_sampler() returns a
    picklable FrameSampler (it runs in data-loader workers).
  - forward() may return (loss, metrics) instead of the loss alone: metrics
    is a dict of extra scalars to log (tensors or floats), logged as
    <name>/<key>.
  - Declare which of those views the loss backpropagates through with the
    grad_features / grad_augmented properties. They default to
    needs_features / needs_augmented. Override them when the loss reads a
    view only as a fixed target (aug_consistency does): that view is then
    encoded without gradient, which saves a full set of encoder activations,
    and it reaches the loss detached even when the task loss's encode of the
    same view carries gradient. The training scripts pass
    aux_grad_views(terms) to encode_views.
  - Param names are checked against __init__'s signature at startup, before
    the encoder loads. To check param values that early too, override the
    check_params classmethod (aug_consistency does, for `target`).
"""

import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AuxBatch:
    """What an aux loss can read. Fields a loss didn't declare a need for may
    be None."""
    encoder: nn.Module
    # Preprocessed observations: float (N, C, H, W) in [0, 1] at the encoder's
    # input resolution (EmbeddingNet.preprocess output).
    obs: Optional[torch.Tensor] = None
    obs_aug: Optional[torch.Tensor] = None
    # Encoder outputs for the above. A view carries gradient only if some loss
    # declared it (grad_features / grad_augmented); otherwise it is detached.
    feats: Optional[torch.Tensor] = None
    feats_aug: Optional[torch.Tensor] = None
    # Clip objectives (pretrain.py): (B, T, D) encoder outputs for the T
    # frames the objective's clip_sampler() picked from each of B videos, in
    # the order it returned them, with gradient; and each video's text.
    clip_feats: Optional[torch.Tensor] = None
    text: Optional[Sequence[str]] = None


class FrameSampler:
    """Which frames of a video a clip objective reads. Called in data-loader
    workers with the video's length (frames are numbered 1..length, as in
    src/data/clips.py) and a numpy Generator; returns frames_per_clip
    frame numbers. Videos shorter than min_length are skipped."""
    frames_per_clip: int = 1
    min_length: int = 1

    def __call__(self, length: int, rng) -> List[int]:
        raise NotImplementedError


class AuxLoss(nn.Module):
    needs_features: bool = False
    needs_augmented: bool = False
    needs_clips: bool = False
    needs_text: bool = False

    def __init__(self, encoder: nn.Module):
        super().__init__()

    @property
    def grad_features(self) -> bool:
        """Does this loss backpropagate through batch.feats?"""
        return self.needs_features

    @property
    def grad_augmented(self) -> bool:
        """Does this loss backpropagate through batch.feats_aug?"""
        return self.needs_augmented

    @classmethod
    def check_params(cls, **params) -> None:
        """Validate config param values without building the loss. Runs at
        startup, before the encoder loads. Raise ValueError on a bad value."""

    def clip_sampler(self, train: bool = True) -> FrameSampler:
        """Clip objectives: the frames to read per video. `train=False` is
        for evaluation batches (no randomness the objective wants only while
        training)."""
        raise NotImplementedError(f"{type(self).__name__} doesn't read clips")

    def forward(self, batch: AuxBatch):
        raise NotImplementedError


_REGISTRY: Dict[str, Type[AuxLoss]] = {}


def register_aux_loss(name: str):
    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, AuxLoss)):
            raise TypeError(f"{cls!r} must subclass AuxLoss")
        if name in _REGISTRY:
            raise ValueError(f"aux loss name {name!r} is already taken")
        _REGISTRY[name] = cls
        return cls
    return decorator


def available_aux_losses():
    return sorted(_REGISTRY)


# ------------------------------------------------------------------------------
# Built-in losses
# ------------------------------------------------------------------------------

def _backbone(encoder: nn.Module) -> nn.Module:
    """EmbeddingNet's pretrained backbone (.embedding), without a trainable
    head; any other module as-is."""
    return getattr(encoder, "embedding", encoder)


@register_aux_loss("l2sp")
class L2SP(AuxLoss):
    """||theta - theta_0||^2 over the backbone's parameters (Xuhong et al.,
    2018): adapt the encoder while penalising drift from its pretrained
    weights. A trainable head (embedding.head=linear) is excluded: it starts
    from a random init, which L2-SP has no reason to anchor to."""

    def __init__(self, encoder: nn.Module):
        super().__init__(encoder)
        self._anchor = {n: p.detach().clone() for n, p in _backbone(encoder).named_parameters()}

    def forward(self, batch: AuxBatch) -> torch.Tensor:
        return sum(
            ((p - self._anchor[n]) ** 2).sum()
            for n, p in _backbone(batch.encoder).named_parameters()
        )


@register_aux_loss("aug_consistency")
class AugConsistency(AuxLoss):
    """MSE between the embeddings of the clean and augmented views of the
    same observations, on the raw encoder output.

    target: which branch is the fixed target (detached); the other one
    receives the gradient. 'clean' (default) pulls augmented embeddings
    toward clean ones -- the usual invariance direction. The target branch
    is declared gradient-free (grad_features / grad_augmented), so it is
    encoded without building a graph.

    On its own (encoder.grads=none, no other aux loss) this objective is
    degenerate: mapping every input to the same embedding gives zero loss, so
    that's the optimum, and nothing in it preserves information. It's
    SimSiam without the predictor SimSiam showed is necessary. Pair it with
    l2sp or with RL gradients (grads != none), and watch
    train/embedding_std_ratio, which the training scripts warn on.
    """
    needs_features = True
    needs_augmented = True

    @classmethod
    def check_params(cls, target: str = "clean") -> None:
        if target not in ("clean", "augmented"):
            raise ValueError(f"aug_consistency target must be 'clean' or 'augmented', got {target!r}")

    def __init__(self, encoder: nn.Module, target: str = "clean"):
        super().__init__(encoder)
        self.check_params(target=target)
        self.target = target

    @property
    def grad_features(self) -> bool:
        return self.target == "augmented"

    @property
    def grad_augmented(self) -> bool:
        return self.target == "clean"

    def forward(self, batch: AuxBatch) -> torch.Tensor:
        clean, aug = batch.feats, batch.feats_aug
        # encode_views already hands the target branch over detached; detach
        # here too so direct callers get the same semantics.
        if self.target == "clean":
            clean = clean.detach()
        else:
            aug = aug.detach()
        return F.mse_loss(aug, clean)


# ------------------------------------------------------------------------------
# Pretraining objectives (pretrain.py)
# ------------------------------------------------------------------------------

class R3MFrameSampler(FrameSampler):
    """R3M's five frames per video (R3MBuffer._sample in the official
    r3m/utils/data_loaders.py), in this order:
      start  within the first `alpha` of the video
      goal   within the last `alpha`
      s0 < s1 <= s2, anywhere
    Same index formulas as the official loader, on frames numbered
    1..length. Evaluation batches use alpha=0 (first and second-to-last
    frame), as the official code does."""
    frames_per_clip = 5

    def __init__(self, alpha: float):
        self.alpha = alpha
        # s1 needs a frame strictly between 1 and length, and the goal
        # window's lower bound int((1 - alpha) * length) - 1 must not reach
        # frame 0 (it can for 3-frame videos once alpha > 1/3).
        self.min_length = 3
        while int((1 - alpha) * self.min_length) < 2:
            self.min_length += 1

    def __call__(self, length: int, rng) -> List[int]:
        a = self.alpha
        start = rng.integers(1, 2 + int(a * length))
        goal = rng.integers(int((1 - a) * length) - 1, length)
        s1 = rng.integers(2, length)
        s0 = rng.integers(1, s1)
        s2 = rng.integers(s1, length + 1)
        return [int(i) for i in (start, goal, s0, s1, s2)]


class _SentenceEncoder:
    """R3M's frozen DistilBERT sentence embedding (LangEncoder in the
    official r3m/models/models_language.py): tokens padded to the longest
    sentence in the batch, last hidden state averaged over ALL positions,
    padding included -- so a sentence's embedding depends on the batch it's
    in, and caching per sentence would change it. Never trained (no_grad),
    but in the official code it's a submodule of the model, so model.train()
    turns its dropout on. train() here does the same (R3M.train passes its
    mode on). Loaded on first use, and a plain object rather than a
    submodule: its weights are neither trained nor saved with the
    objective."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = self.model = None
        self.training = False

    def train(self, mode: bool = True):
        self.training = mode
        if self.model is not None:
            self.model.train(mode)

    def tokenize(self, sentences, device):
        """Token ids and attention mask on `device` (deterministic, so a step
        can tokenize once and embed() several times)."""
        if self.model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as e:
                raise ImportError("R3M's language loss needs transformers: pip install transformers "
                                  "(or set lang_weight: 0)") from e
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).train(self.training)
        self.model.to(device)
        enc = self.tokenizer(list(sentences), return_tensors="pt", padding=True)
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def embed(self, tokens) -> torch.Tensor:
        input_ids, attention_mask = tokens
        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        return out.mean(1)

    def __call__(self, sentences, device) -> torch.Tensor:
        return self.embed(self.tokenize(sentences, device))


class _LanguageReward(nn.Module):
    """G(e_0, e_t, l): R3M's progress score (LanguageReward in the official
    code; same layer layout, so its state_dict keys match)."""

    def __init__(self, im_dim: int, hidden_dim: int, lang_dim: int):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Linear(im_dim * 2 + lang_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, e0, et, le):
        return self.pred(torch.cat([e0, et, le], -1)).squeeze(-1)


@register_aux_loss("r3m")
class R3M(AuxLoss):
    """R3M (Nair et al., 2022): a port of the loss in the official
    r3m/trainer.py, term for term, with the same random draws in the same
    order (negatives via torch.randperm).

      tcn   time-contrastive: within a video, s1 is closer to s2 than s0 is,
            and s0 closer to s1 than to s2; num_negatives frames from other
            videos in the batch are negatives
      lang  video-language alignment: G(start, later frame, text) beats
            G(start, earlier frame, text) and other videos' frame pairs
            (InfoNCE); videos with empty text are masked out
      l1/l2 mean L1 and L2 norms of all embeddings (sparsity)

    text_dropout (default, as the official code): while training, DistilBERT
    runs in train mode (dropout 0.1) and the text is re-encoded at every
    reward call -- 15 encodings per step at 3 negatives, each with its own
    dropout mask. false: encoded once per step without dropout (cheaper,
    deterministic). Evaluation batches always encode once without dropout.

    Weights are R3M's own; the registry `weight` scales the whole objective.
    Defaults are the released models' (README: langweight 1, l1weight 1e-5;
    config: l2weight 1e-5, tcnweight 1). Metrics use the official names
    (tcnloss, rewloss, rewacc1-3, aligned, l0/l1/l2loss).
    """
    needs_clips = True
    needs_text = True

    @classmethod
    def check_params(cls, tcn_weight=1.0, lang_weight=1.0, l1_weight=1e-5, l2_weight=1e-5,
                     num_negatives=3, hidden_dim=1024, l2dist=True, alpha=0.2,
                     text_encoder="distilbert-base-uncased", text_dropout=True) -> None:
        for name, w in (("tcn_weight", tcn_weight), ("lang_weight", lang_weight),
                        ("l1_weight", l1_weight), ("l2_weight", l2_weight)):
            if w < 0:
                raise ValueError(f"r3m {name} must be >= 0, got {w}")
        if int(num_negatives) != num_negatives or num_negatives < 0:
            raise ValueError(f"r3m num_negatives must be a whole number >= 0, got {num_negatives}")
        if not 0.0 <= alpha < 0.5:
            raise ValueError(f"r3m alpha must be in [0, 0.5), got {alpha}")

    def __init__(self, encoder: nn.Module, tcn_weight: float = 1.0, lang_weight: float = 1.0,
                 l1_weight: float = 1e-5, l2_weight: float = 1e-5, num_negatives: int = 3,
                 hidden_dim: int = 1024, l2dist: bool = True, alpha: float = 0.2,
                 text_encoder: str = "distilbert-base-uncased", text_dropout: bool = True):
        super().__init__(encoder)
        self.check_params(tcn_weight, lang_weight, l1_weight, l2_weight, num_negatives,
                          hidden_dim, l2dist, alpha, text_encoder, text_dropout)
        self.text_dropout = text_dropout
        self.tcn_weight, self.lang_weight = tcn_weight, lang_weight
        self.l1_weight, self.l2_weight = l1_weight, l2_weight
        self.num_negatives = int(num_negatives)
        self.l2dist = l2dist
        self.alpha = alpha
        self.encode_text = _SentenceEncoder(text_encoder)
        if lang_weight > 0:
            self.lang_rew = _LanguageReward(encoder.out_size, hidden_dim, lang_dim=768)
        # nn.Module starts in training mode without calling train(): sync the
        # text encoder's mode now, not only at the first train()/eval().
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encode_text.train(mode and self.text_dropout)
        return self

    def clip_sampler(self, train: bool = True) -> FrameSampler:
        return R3MFrameSampler(self.alpha if train else 0.0)

    def sim(self, a, b):
        return -torch.linalg.norm(a - b, dim=-1) if self.l2dist else F.cosine_similarity(a, b, dim=-1)

    def forward(self, batch: AuxBatch):
        eps = 1e-8
        alle = batch.clip_feats                                   # (B, 5, D)
        bs = alle.shape[0]
        alles = alle.reshape(bs * 5, -1)
        e0, eg, es0, es1, es2 = alle.unbind(1)
        metrics = {}

        l2loss = torch.linalg.norm(alles, ord=2, dim=-1).mean()
        l1loss = torch.linalg.norm(alles, ord=1, dim=-1).mean()
        metrics["l2loss"] = l2loss
        metrics["l1loss"] = l1loss
        metrics["l0loss"] = torch.linalg.norm(alles.detach(), ord=0, dim=-1).mean()
        full_loss = self.l2_weight * l2loss + self.l1_weight * l1loss

        if self.lang_weight > 0:
            if self.training and self.text_dropout:
                # Official: get_reward re-encodes the text on every call (each
                # with its own dropout mask); tokenizing is deterministic, so
                # it's done once.
                tokens = self.encode_text.tokenize(batch.text, alles.device)
                reward = lambda a, b: self.lang_rew(
                    a, b, self.encode_text.embed(tokens).to(alles.dtype))
            else:
                le = self.encode_text(batch.text, alles.device).to(alles.dtype)
                reward = lambda a, b: self.lang_rew(a, b, le)
            sim_pos = [reward(e0, eg), reward(e0, es1), reward(e0, es2)]
            negs = [[reward(e0, e0)], [reward(e0, es0)], [reward(e0, es1)]]
            for _ in range(self.num_negatives):
                for neg, later in zip(negs, (eg, es1, es2)):
                    perm = torch.randperm(bs)
                    neg.append(reward(e0[perm], later[perm]))
            rewlosses = []
            for i, (pos, neg) in enumerate(zip(sim_pos, negs), 1):
                neg = torch.stack(neg, -1)
                rewlosses.append(-torch.log(eps + torch.exp(pos) / (eps + torch.exp(pos) + torch.exp(neg).sum(-1))))
                metrics[f"rewacc{i}"] = (1.0 * (neg.max(-1)[0] < pos)).mean().detach()
            rewloss = (rewlosses[0] + rewlosses[1] + rewlosses[2]) / 3
            mask = torch.tensor([1.0 * (t != "") for t in batch.text], device=rewloss.device)
            rewloss = (rewloss * mask).mean()
            metrics["rewloss"] = rewloss
            full_loss = full_loss + self.lang_weight * rewloss

        if self.tcn_weight > 0:
            sim_0_2 = self.sim(es2, es0)
            sim_1_2 = self.sim(es2, es1)
            sim_0_1 = self.sim(es1, es0)
            neg0, neg2 = [], []
            for _ in range(self.num_negatives):
                es0_shuf = es0[torch.randperm(bs)]
                es2_shuf = es2[torch.randperm(bs)]
                neg0.append(self.sim(es0, es0_shuf))
                neg2.append(self.sim(es2, es2_shuf))
            exp_neg0 = torch.exp(torch.stack(neg0, -1)).sum(-1) if neg0 else 0.0
            exp_neg2 = torch.exp(torch.stack(neg2, -1)).sum(-1) if neg2 else 0.0
            smooth1 = -torch.log(eps + torch.exp(sim_1_2) / (
                eps + torch.exp(sim_0_2) + torch.exp(sim_1_2) + exp_neg2))
            smooth2 = -torch.log(eps + torch.exp(sim_0_1) / (
                eps + torch.exp(sim_0_1) + torch.exp(sim_0_2) + exp_neg0))
            tcnloss = ((smooth1 + smooth2) / 2.0).mean()
            metrics["tcnloss"] = tcnloss
            metrics["aligned"] = ((1.0 * (sim_0_2 < sim_1_2)) * (1.0 * (sim_0_1 > sim_0_2))).mean().detach()
            full_loss = full_loss + self.tcn_weight * tcnloss

        return full_loss, {k: v.detach() for k, v in metrics.items()}


# ------------------------------------------------------------------------------
# Config -> losses
# ------------------------------------------------------------------------------

@dataclass
class AuxTerm:
    name: str
    weight: float
    loss: AuxLoss


def _split_spec(spec) -> Tuple[str, float, dict]:
    if not isinstance(spec, dict):
        raise ValueError(f"aux entries look like {{name: l2sp, weight: 1.0}}, got {spec!r}")
    spec = dict(spec)
    name = spec.pop("name", None)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown aux loss {name!r}. Available: {available_aux_losses()} "
            "(add your own in src/losses.py)"
        )
    weight = spec.pop("weight", None)
    if not isinstance(weight, (int, float)) or weight < 0:
        raise ValueError(f"aux loss {name!r} needs a non-negative numeric `weight`, got {weight!r}")
    return name, float(weight), spec


def aux_loss_classes(specs) -> List[Type[AuxLoss]]:
    """Validate a config list of aux specs without building anything (names,
    weights, duplicates, param names against __init__, param values via
    check_params) and return the loss classes, for checking their declared
    needs. Runs at startup, before the encoder loads."""
    split = [_split_spec(s) for s in specs]
    names = [name for name, _, _ in split]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"aux loss listed more than once: {dupes}")
    classes = []
    for name, _, params in split:
        cls = _REGISTRY[name]
        try:
            inspect.signature(cls.__init__).bind(None, None, **params)  # self, encoder
        except TypeError as e:
            raise TypeError(f"bad params for aux loss {name!r}: {e}") from None
        cls.check_params(**params)
        classes.append(cls)
    return classes


def build_aux_losses(specs, encoder: nn.Module) -> List[AuxTerm]:
    aux_loss_classes(specs)
    terms = []
    for spec in specs:
        name, weight, params = _split_spec(spec)
        try:
            loss = _REGISTRY[name](encoder, **params)
        except TypeError as e:
            raise TypeError(f"bad params for aux loss {name!r}: {e}") from e
        # grad_* depend on params, so this is the earliest they can be checked.
        # encode_views would otherwise raise only at the first training update.
        if (loss.grad_features and not loss.needs_features) or \
                (loss.grad_augmented and not loss.needs_augmented):
            raise ValueError(
                f"aux loss {name!r} declares gradient through a view it doesn't "
                f"read: grad_features={loss.grad_features} with needs_features="
                f"{loss.needs_features}, grad_augmented={loss.grad_augmented} "
                f"with needs_augmented={loss.needs_augmented}"
            )
        terms.append(AuxTerm(name, weight, loss))
    return terms


def aux_grad_views(terms: List[AuxTerm]) -> Tuple[bool, bool]:
    """(clean, augmented): does any built loss backpropagate through that
    view? Pass to encode_views as aux_grad_clean / aux_grad_aug."""
    return (any(t.loss.grad_features for t in terms),
            any(t.loss.grad_augmented for t in terms))


def objective_terms(terms: List[AuxTerm], batch: AuxBatch):
    """Weighted sum of all terms, plus each term's unweighted value and any
    metrics it returned ('<name>/<key>'), as detached tensors or floats --
    convert them only when logging (each .item() is a GPU sync)."""
    total, values = 0.0, {}
    for t in terms:
        out = t.loss(batch)
        value, metrics = out if isinstance(out, tuple) else (out, {})
        values[t.name] = value.detach()
        for k, v in metrics.items():
            values[f"{t.name}/{k}"] = v.detach() if torch.is_tensor(v) else v
        total = total + t.weight * value
    return total, values


def total_aux_loss(terms: List[AuxTerm], batch: AuxBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted sum of all terms, plus each term's unweighted value (and
    metrics) as floats for logging."""
    total, values = objective_terms(terms, batch)
    return total, {k: float(v) for k, v in values.items()}
