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
        needs_views     = 2      # batch.views: this many augmented views per image
        needs_labels    = True   # batch.labels: one class per image
    needs_augmented also requires an augmentation with stage aux or both
    (see src/views.py). Clips and text come from video datasets, so only
    pretrain.py supplies them; the RL/BC scripts reject such objectives. A
    clip objective also says which frames it wants: clip_sampler() returns a
    picklable FrameSampler (it runs in data-loader workers). Image
    objectives (needs_views) get the views as preprocessed, augmented
    images and run batch.encoder on them themselves, so they control how
    (MoCo: in chunks, for its BatchNorm shuffling).
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
    # Image objectives (pretrain.py): needs_views views of each of B images,
    # each float (B, 3, H, W) at the encoder's input size, independently
    # cropped and augmented, before mean/std normalisation (encode them
    # with batch.encoder.encode_preprocessed); and class labels (B,).
    views: Optional[List[torch.Tensor]] = None
    labels: Optional[torch.Tensor] = None


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
    needs_views: int = 0
    needs_labels: bool = False

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


class VIPFrameSampler(FrameSampler):
    """VIP's four frames per video (VIPBuffer._sample in the official
    vip/utils/data_loaders.py), in this order:
      start  anywhere but the last two frames
      goal   after start
      s0     in [start, goal)
      s1     s0 + 1
    -- a random sub-trajectory start..goal and one transition inside it. The
    official loader numbers frames 0..length-1; these are its draws, shifted
    to frames 1..length. Evaluation batches sample the same way (the
    official validation loader differs only in having no augmentation)."""
    frames_per_clip = 4
    min_length = 3

    def __call__(self, length: int, rng) -> List[int]:
        start = rng.integers(0, length - 2)
        goal = rng.integers(start + 1, length)
        s0 = rng.integers(start, goal)
        s1 = min(s0 + 1, goal)
        return [int(i) + 1 for i in (start, goal, s0, s1)]


@register_aux_loss("vip")
class VIP(AuxLoss):
    """VIP (Ma et al., 2023): a port of the loss in the official
    vip/trainer.py, term for term, with the same random draws in the same
    order (negatives via torch.randperm).

    The negative embedding distance V(o; g) = -||phi(o) - phi(g)|| is a
    goal-conditioned value function, trained with VIP's dual offline-RL
    objective under a reward of -1 per step:
      vip    (1 - gamma) * -V(start; goal), which pulls a sub-trajectory's
             start towards its goal, plus log-mean-exp of
             -(r + gamma V(s1; goal) - V(s0; goal)) over the batch: a soft
             temporal-difference term on the transition (s0, s1). With
             num_negatives > 0 the same term again for transitions taken
             from other videos in the batch, against this video's goal.
      l1/l2  mean L1 and L2 norms of all embeddings
    No text: VIP learns from frames alone, so it reads any clip dataset.

    Defaults are the official VIP model class's: gamma 0.98, l1 and l2
    weights 1e-3 (its released config), no negatives. The released model
    trained with num_negatives: 3 (the paper, Appendix D.1, and
    config_vip_ego4d.yaml, which the VIP README names as the released
    model's config); the vip preset sets that. VIP's representation is a
    ResNet-50 with its fc resized to 1024, i.e. embedding.head=fc (the
    preset sets it too). The official loader's reward tensor is float64,
    which promotes its TD term to double precision; here it stays in the
    embeddings' dtype. With float32 rewards the official trainer gives
    exactly this loss and update; with float64 ones the losses agree to
    ~1e-7, but Adam's first step (about lr * sign(grad)) can flip on
    near-zero gradients. Metrics use the official names (vip_loss, l1loss,
    l2loss).
    """
    needs_clips = True

    @classmethod
    def check_params(cls, gamma=0.98, l1_weight=1e-3, l2_weight=1e-3, num_negatives=0) -> None:
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"vip gamma must be in (0, 1), got {gamma}")
        for name, w in (("l1_weight", l1_weight), ("l2_weight", l2_weight)):
            if w < 0:
                raise ValueError(f"vip {name} must be >= 0, got {w}")
        if int(num_negatives) != num_negatives or num_negatives < 0:
            raise ValueError(f"vip num_negatives must be a whole number >= 0, got {num_negatives}")

    def __init__(self, encoder: nn.Module, gamma: float = 0.98, l1_weight: float = 1e-3,
                 l2_weight: float = 1e-3, num_negatives: int = 0):
        super().__init__(encoder)
        self.check_params(gamma, l1_weight, l2_weight, num_negatives)
        self.gamma, self.l1_weight, self.l2_weight = gamma, l1_weight, l2_weight
        self.num_negatives = int(num_negatives)

    def clip_sampler(self, train: bool = True) -> FrameSampler:
        return VIPFrameSampler()

    @staticmethod
    def sim(a, b):
        return -torch.linalg.norm(a - b, dim=-1)

    def forward(self, batch: AuxBatch):
        eps = 1e-8
        alle = batch.clip_feats                                   # (B, 4, D)
        bs = alle.shape[0]
        alles = alle.reshape(bs * 4, -1)
        e0, eg, es0, es1 = alle.unbind(1)
        metrics = {}

        l2loss = torch.linalg.norm(alles, ord=2, dim=-1).mean()
        l1loss = torch.linalg.norm(alles, ord=1, dim=-1).mean()
        metrics["l2loss"] = l2loss
        metrics["l1loss"] = l1loss
        full_loss = self.l2_weight * l2loss + self.l1_weight * l1loss

        g = self.gamma
        r = -1.0                    # official: float(s0 == goal) - 1, and s0 < goal always
        V_0 = self.sim(e0, eg)
        V_s = self.sim(es0, eg)
        V_s_next = self.sim(es1, eg)
        V_loss = (1 - g) * -V_0.mean() + torch.log(eps + torch.mean(torch.exp(-(r + g * V_s_next - V_s))))
        if self.num_negatives > 0:
            V_s_neg, V_s_next_neg = [], []
            for _ in range(self.num_negatives):
                perm = torch.randperm(bs)
                V_s_neg.append(self.sim(es0[perm], eg))
                V_s_next_neg.append(self.sim(es1[perm], eg))
            V_s_neg, V_s_next_neg = torch.cat(V_s_neg), torch.cat(V_s_next_neg)
            V_loss = V_loss + torch.log(eps + torch.mean(torch.exp(-(-1.0 + g * V_s_next_neg - V_s_neg))))
        metrics["vip_loss"] = V_loss
        full_loss = full_loss + V_loss
        return full_loss, {k: v.detach() for k, v in metrics.items()}


def _topk_accuracy(logits, labels, ks=(1, 5)):
    ks = [k for k in ks if k <= logits.shape[1]]
    hit = logits.topk(max(ks), dim=1).indices == labels[:, None]
    return {f"top{k}": hit[:, :k].any(1).float().mean() for k in ks}


@register_aux_loss("supervised")
class Supervised(AuxLoss):
    """Image classification: cross-entropy of a linear classifier on the
    encoder's output (torchvision's ImageNet ResNets: fc on the pooled
    features). The classifier belongs to the objective, so the saved encoder
    stays a feature extractor."""
    needs_views = 1
    needs_labels = True

    @classmethod
    def check_params(cls, num_classes=1000, label_smoothing=0.0) -> None:
        if int(num_classes) != num_classes or num_classes < 2:
            raise ValueError(f"supervised num_classes must be a whole number >= 2, got {num_classes}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"supervised label_smoothing must be in [0, 1), got {label_smoothing}")

    def __init__(self, encoder: nn.Module, num_classes: int = 1000, label_smoothing: float = 0.0):
        super().__init__(encoder)
        self.check_params(num_classes, label_smoothing)
        self.fc = nn.Linear(encoder.out_size, int(num_classes))
        self.label_smoothing = label_smoothing

    def forward(self, batch: AuxBatch):
        logits = self.fc(batch.encoder.encode_preprocessed(batch.views[0])).float()
        loss = F.cross_entropy(logits, batch.labels, label_smoothing=self.label_smoothing)
        return loss, {k: v.detach() for k, v in _topk_accuracy(logits, batch.labels).items()}


@register_aux_loss("moco")
class MoCo(AuxLoss):
    """MoCo v2 (He et al., 2020; Chen et al., 2020): contrastive learning
    between two augmented views of each image, against a queue of keys from
    past batches. The query view goes through the encoder and a projection
    head; the key view through momentum (EMA) copies of both, updated at the
    start of every training forward pass, as in the paper's code. Loss:
    InfoNCE over (positive key, queue) at `temperature`; keys are enqueued
    after each training batch.

    bn_splits: the paper trains on 8 GPUs and shuffles the key batch across
    them, so a sample's query and key BatchNorm statistics come from
    different images (otherwise BatchNorm leaks which samples belong
    together). Here both views are encoded in bn_splits chunks, the keys
    after a random permutation -- the same thing on one GPU (batch 256 / 8 =
    32 images per BatchNorm, as in the paper).

    Defaults are MoCo v2's: 128-dim MLP head, queue 65536, momentum 0.999,
    temperature 0.2. Metric: acc1, how often the positive key scores
    highest (the official code's training accuracy)."""
    needs_views = 2

    @classmethod
    def check_params(cls, dim=128, queue_size=65536, momentum=0.999, temperature=0.2, mlp=True,
                     bn_splits=8) -> None:
        for name, v in (("dim", dim), ("queue_size", queue_size), ("bn_splits", bn_splits)):
            if int(v) != v or v < 1:
                raise ValueError(f"moco {name} must be a whole number >= 1, got {v}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"moco momentum must be in [0, 1), got {momentum}")
        if temperature <= 0:
            raise ValueError(f"moco temperature must be > 0, got {temperature}")

    def __init__(self, encoder: nn.Module, dim: int = 128, queue_size: int = 65536,
                 momentum: float = 0.999, temperature: float = 0.2, mlp: bool = True,
                 bn_splits: int = 8):
        super().__init__(encoder)
        self.check_params(dim, queue_size, momentum, temperature, mlp, bn_splits)
        import copy
        d = encoder.out_size
        self.head = (nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, dim))
                     if mlp else nn.Linear(d, dim))
        # Momentum copies: part of the objective's state (saved, resumed),
        # never optimised.
        self.key_encoder = copy.deepcopy(encoder).requires_grad_(False)
        self.key_head = copy.deepcopy(self.head).requires_grad_(False)
        self.register_buffer("queue", F.normalize(torch.randn(int(dim), int(queue_size)), dim=0))
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))
        self.momentum, self.temperature, self.bn_splits = momentum, temperature, int(bn_splits)

    @torch.no_grad()
    def _momentum_update(self, encoder):
        m = self.momentum
        for q, k in ((encoder, self.key_encoder), (self.head, self.key_head)):
            for pq, pk in zip(q.parameters(), k.parameters()):
                pk.mul_(m).add_(pq.detach(), alpha=1 - m)

    def _project(self, encoder, head, x, perm=None):
        if perm is not None:
            x = x[perm]
        out = torch.cat([head(encoder.encode_preprocessed(c)) for c in x.chunk(self.bn_splits)])
        if perm is not None:
            out = out[torch.argsort(perm)]
        return F.normalize(out.float(), dim=1)

    @torch.no_grad()
    def _enqueue(self, k):
        size = self.queue.shape[1]
        idx = (self.queue_ptr + torch.arange(len(k), device=k.device)) % size
        self.queue[:, idx] = k.T
        self.queue_ptr.copy_((self.queue_ptr + len(k)) % size)

    def forward(self, batch: AuxBatch):
        x_q, x_k = batch.views[0], batch.views[1]
        q = self._project(batch.encoder, self.head, x_q)
        with torch.no_grad():
            if self.training:
                self._momentum_update(batch.encoder)
            k = self._project(self.key_encoder, self.key_head, x_k,
                              perm=torch.randperm(len(x_k), device=x_k.device))
        logits = torch.cat([(q * k).sum(1, keepdim=True), q @ self.queue.clone()], dim=1) / self.temperature
        target = torch.zeros(len(q), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, target)
        if self.training:
            self._enqueue(k)
        return loss, {"acc1": (logits.argmax(1) == 0).float().mean().detach()}


@register_aux_loss("mae")
class MAE(AuxLoss):
    """MAE (He et al., 2022): mask most of an image's patches, encode the
    visible ones, and reconstruct the masked ones' pixels with a light
    decoder; the loss is the mean squared error on the masked patches. The
    MVP and VC-1 representations are this objective on (egocentric) video
    frames.

    The encoder is the run's ViT in MAE's layout (embedding=vit_s16, vit_b16
    or vit_l16), whole: no embedding.layer or head. The decoder belongs to
    the objective, so the saved encoder is the feature extractor (its CLS
    token after the final norm). Encoder, decoder and loss run the official
    models_mae.py's own code (src/vision_models/mae.py), so with the same
    weights and masking noise the loss is the official model's. The input
    is the view after the encoder's mean/std normalisation, which is also
    the reconstruction target, as in MAE's training code.

    Defaults are MAE's: mask ratio 0.75, per-patch normalised targets
    (norm_pix_loss, as its released models), decoder 512 wide, 8 blocks of
    16 heads.
    """
    needs_views = 1

    @classmethod
    def check_params(cls, mask_ratio=0.75, norm_pix_loss=True, decoder_dim=512, decoder_depth=8,
                     decoder_heads=16) -> None:
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mae mask_ratio must be in [0, 1), got {mask_ratio}")
        for name, v in (("decoder_dim", decoder_dim), ("decoder_depth", decoder_depth),
                        ("decoder_heads", decoder_heads)):
            if int(v) != v or v < 1:
                raise ValueError(f"mae {name} must be a whole number >= 1, got {v}")
        if decoder_dim % decoder_heads:
            raise ValueError(f"mae decoder_dim ({decoder_dim}) must be a multiple of decoder_heads ({decoder_heads})")

    def __init__(self, encoder: nn.Module, mask_ratio: float = 0.75, norm_pix_loss: bool = True,
                 decoder_dim: int = 512, decoder_depth: int = 8, decoder_heads: int = 16):
        super().__init__(encoder)
        self.check_params(mask_ratio, norm_pix_loss, decoder_dim, decoder_depth, decoder_heads)
        from functools import partial
        from src.vision_models.mae import MAEDecoder
        vit = getattr(encoder, "embedding", None)
        spec = getattr(encoder, "build_spec", {})
        if not hasattr(vit, "forward_encoder") or spec.get("layer") or spec.get("head") \
                or spec.get("frame_stack", 1) > 1:
            raise ValueError("mae trains a whole ViT in MAE's layout: embedding=vit_s16, vit_b16 or "
                             "vit_l16, without embedding.layer, head or frame_stack")
        self.mask_ratio = float(mask_ratio)
        self.decoder = MAEDecoder(
            vit.patch_embed.num_patches, vit.patch_embed.patch_size[0], embed_dim=vit.embed_dim,
            decoder_embed_dim=int(decoder_dim), decoder_depth=int(decoder_depth),
            decoder_num_heads=int(decoder_heads), norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_pix_loss=bool(norm_pix_loss))

    def forward(self, batch: AuxBatch):
        imgs = batch.encoder.normalize(batch.views[0])
        latent, mask, ids_restore = batch.encoder.embedding.forward_encoder(imgs, self.mask_ratio)
        pred = self.decoder.forward_decoder(latent, ids_restore)
        return self.decoder.forward_loss(imgs, pred, mask)


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
