"""
Multi-GPU plumbing for pretrain.py (DDP on one node), started as one process
per GPU:

    python -m torch.distributed.run --standalone --nproc_per_node=4 pretrain.py ...

Without that launcher (no WORLD_SIZE in the environment) nothing here does
anything, so a single-GPU run is the plain training loop.

Each process holds a copy of the encoder and objectives and trains on its own
share of every batch; gradients are averaged in backward(), so all copies stay
the same. Rank 0 does what only one process should: printing, wandb,
validation and checkpoints.

DDP syncs gradients only for what runs through its own forward(), and the
objectives call the encoder's methods directly, so the whole forward pass goes
through one module (ObjectivesForward).

Which objectives can run this way is declared by AuxLoss.multi_gpu
(src/losses.py).
"""

import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass(frozen=True)
class Processes:
    rank: int = 0
    world: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def active(self) -> bool:
        return self.world > 1


def setup(device_name: str) -> Processes:
    """Join the process group if started by the launcher (WORLD_SIZE set), and
    claim this process's GPU: "cuda" (which EmbeddingNet uses) then means it.
    Only rank 0 prints; errors still reach stderr."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world <= 1:
        return Processes()
    rank, local_rank = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", 0))
    cuda = device_name != "cpu" and torch.cuda.is_available()
    if cuda:
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"{world} processes, but this node shows only {torch.cuda.device_count()} "
                               "GPUs: --nproc_per_node must match the job's --gpus")
        torch.cuda.set_device(local_rank)
    # The timeout covers every collective, gradient all-reduces included, so a
    # hung rank holds the GPUs this long before the job fails (a crashed one
    # stops the rest at once). 30 minutes leaves room for the others' wait in
    # barrier() while rank 0 validates and saves (minutes).
    dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=world, timeout=timedelta(minutes=30))
    if rank != 0:
        sys.stdout = open(os.devnull, "w")
    return Processes(rank, world, local_rank)


def device_of(procs: Processes, device_name: str) -> torch.device:
    if device_name == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", procs.local_rank) if procs.active else torch.device(device_name)


class ObjectivesForward(nn.Module):
    """The encoder and objectives as one module, with `forward_fn` (the
    training forward pass) as its forward, for DDP to wrap."""

    def __init__(self, encoder: nn.Module, losses, forward_fn):
        super().__init__()
        self.encoder = encoder
        self.losses = nn.ModuleList(losses)
        self.forward_fn = forward_fn

    def forward(self, *args, **kwargs):
        return self.forward_fn(*args, **kwargs)


def wrap(procs: Processes, device: torch.device, encoder: nn.Module, losses, forward_fn,
         find_unused_parameters: bool = False):
    """A callable for the training forward pass: forward_fn itself in a
    single-process run, otherwise it through DDP. Wrapping copies rank 0's
    weights to every process. BatchNorm is per GPU, with rank 0's running
    statistics copied to the others at each forward (DDP's default)."""
    if not procs.active:
        return forward_fn
    return nn.parallel.DistributedDataParallel(
        ObjectivesForward(encoder, losses, forward_fn),
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
    )


def mean_over_ranks(procs: Processes, values: Dict[str, float], device: torch.device) -> Dict[str, float]:
    """Average logged values across processes (every rank must call this, with
    the same keys)."""
    if not procs.active:
        return values
    keys = sorted(values)
    t = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return {k: float(v) / procs.world for k, v in zip(keys, t)}


def from_main(procs: Processes, flag: bool, device: torch.device) -> bool:
    """Rank 0's decision, on every rank: for what depends on its clock."""
    if not procs.active:
        return flag
    t = torch.tensor([int(flag)], device=device)
    dist.broadcast(t, src=0)
    return bool(t.item())


def barrier(procs: Processes) -> None:
    if procs.active:
        dist.barrier()


def check_same(procs: Processes, name: str, value: str) -> None:
    """Raise if the processes disagree on a string (e.g. the run directory)."""
    if not procs.active:
        return
    values = [None] * procs.world
    dist.all_gather_object(values, value)
    if len(set(values)) > 1:
        raise ValueError(f"the {procs.world} processes got different values for {name}: {sorted(set(values))}")


def close(procs: Processes) -> None:
    if procs.active:
        dist.destroy_process_group()
