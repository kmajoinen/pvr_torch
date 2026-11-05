"""
How a pretraining data stream picks its items (pretrain.py's
schedule.sampling), for both image and clip datasets:

  sweep   epoch by epoch, each item once per epoch in a fresh random order
          (a permutation seeded from (seed, epoch))
  random  uniform with replacement; batch b's items come from a generator
          seeded from (seed, b)

Both are split across data-loader workers so that a resumed run continues at
the exact item (same batch_size and num_workers): workers hand out whole
batches in turn, starting with worker 0, so a fresh loader started after
start_batch batches gets batch start_batch + k from worker k % workers. This
covers which items come next, not what's drawn from each (a clip's frames,
crops): those use the workers' RNGs, reseeded on resume so they don't repeat.
"""

import math

import numpy as np


def epoch_stream(n: int, seed: int, wid: int, workers: int, start_batch: int, batch_size: int):
    """Worker wid's share of sweep sampling over items 0..n-1, as (epoch,
    item). Each worker takes every workers-th item of an epoch's
    permutation; the batch after start_batch belongs to role
    start_batch % workers, so a fresh loader's worker i takes role
    (i + start_batch) % workers, and role r had produced
    ceil((start_batch - r) / workers) batches."""
    wid = (wid + start_batch) % workers
    skip = math.ceil(max(start_batch - wid, 0) / workers) * batch_size
    epoch = 0
    while True:
        share = len(range(wid, n, workers))
        if skip >= share:
            skip -= share
            epoch += 1
            continue
        for i in np.random.default_rng([seed, epoch]).permutation(n)[wid::workers][skip:]:
            yield epoch, int(i)
        skip = 0
        epoch += 1


def random_stream(n: int, seed: int, wid: int, workers: int, start_batch: int, batch_size: int):
    """Worker wid's share of random sampling over items 0..n-1, as (batch
    number, item): batches start_batch + wid, + workers, ..., each drawn
    with replacement from its own (seed, batch number) generator -- so a
    batch's items don't depend on the worker count."""
    b = start_batch + wid
    while True:
        for i in np.random.default_rng([seed, b]).integers(0, n, batch_size):
            yield b, int(i)
        b += workers
