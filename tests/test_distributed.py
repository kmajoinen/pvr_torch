"""Multi-GPU pretraining (src/distributed.py, pretrain.py), on CPU: DDP runs the
same way over processes with the gloo backend. Checks the data split across
processes, that DDP's gradient equals the single-process one, and real runs
through the launcher. What only GPUs show (nccl, speed) needs debug_ddp.sh."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision.io import write_jpeg

from src import distributed
from src.data.images import ImageDataset
from src.data.order import epoch_stream, random_stream, split_across_processes
from src.losses import AuxBatch, build_aux_losses, objective_terms

ROOT = Path(__file__).resolve().parents[1]


# ---- the data split ----------------------------------------------------------

class _Stream(IterableDataset):
    """The order helpers through a real DataLoader, for one process's share."""

    def __init__(self, fn, n, rank, world, batch_size=2, start_batch=0, seed=3):
        self.fn, self.n, self.rank, self.world = fn, n, rank, world
        self.batch_size, self.start_batch, self.seed = batch_size, start_batch, seed

    def __iter__(self):
        info = get_worker_info()
        wid, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        wid, workers, start = split_across_processes(wid, workers, self.start_batch, self.rank, self.world)
        for _, i in self.fn(self.n, self.seed, wid, workers, start, self.batch_size):
            yield i


def _steps(fn, n, world, workers, steps, batch_size=2, start_batch=0):
    """For each step, each process's batch: [step][rank] -> items."""
    its = [iter(DataLoader(_Stream(fn, n, r, world, batch_size, start_batch), batch_size=batch_size,
                           num_workers=workers)) for r in range(world)]
    return [[next(it).tolist() for it in its] for _ in range(steps)]


def test_one_process_is_the_plain_split():
    assert split_across_processes(1, 4, 7, 0, 1) == (1, 4, 7)


@pytest.mark.parametrize("world, workers", [(2, 0), (2, 2), (3, 0), (3, 2), (4, 3)])
def test_sweep_gives_every_item_once_per_epoch_across_processes(world, workers):
    # Every worker's share of an epoch a whole number of batches (as in
    # test_schedule.py), so epochs end on a step.
    batch_size = 2
    roles = world * max(workers, 1)
    n = roles * batch_size * 3
    steps = _steps(epoch_stream, n, world, workers, 2 * n // (world * batch_size), batch_size)   # 2 epochs
    flat = [i for step in steps for batch in step for i in batch]
    assert sorted(flat[:n]) == list(range(n)) and sorted(flat[n:]) == list(range(n))


@pytest.mark.parametrize("fn", [epoch_stream, random_stream])
@pytest.mark.parametrize("world, workers, start", [(2, 0, 5), (2, 2, 5), (3, 2, 7), (4, 3, 4)])
def test_resume_continues_each_process_exactly(fn, world, workers, start):
    fresh = _steps(fn, 24, world, workers, start + 6)
    assert _steps(fn, 24, world, workers, 6, start_batch=start) == fresh[start:]


def test_processes_draw_different_random_batches():
    (a, b), = _steps(random_stream, 1000, 2, 0, 1, batch_size=8)
    assert a != b


class _ImageIds(ImageDataset):
    def _item(self, i):
        return i


def _image_folder(root, n=24, side=32):
    torch.manual_seed(0)
    for split, count in (("train", n), ("val", 8)):
        for i in range(count):
            d = root / split / f"c{i % 2}"
            d.mkdir(parents=True, exist_ok=True)
            write_jpeg(torch.randint(0, 255, (3, side, side), dtype=torch.uint8), str(d / f"{i}.jpg"))
    return root


def test_image_dataset_splits_across_processes(tmp_path):
    root = _image_folder(tmp_path, n=24, side=8)
    got = []
    for rank in range(2):
        ds = _ImageIds(str(root), "train", train=True, seed=1, batch_size=3, rank=rank, world=2)
        it = iter(DataLoader(ds, batch_size=3, num_workers=2))
        got += [i for _ in range(4) for i in next(it).tolist()]
    assert sorted(got) == list(range(24))


# ---- gradients ---------------------------------------------------------------

class _Enc(nn.Module):
    """No BatchNorm, so the batch's split can't change the result."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.out_size = 8
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, stride=2), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
                                 nn.Flatten(), nn.Linear(16, 8))

    def encode_preprocessed(self, x):
        return self.net(x)


def _model():
    enc = _Enc()
    terms = build_aux_losses([{"name": "supervised", "weight": 1.0, "num_classes": 4}], enc)
    return enc, terms


def _full_batch():
    g = torch.Generator().manual_seed(5)
    return torch.rand(8, 3, 16, 16, generator=g), torch.randint(0, 4, (8,), generator=g)


def _ddp_rank(rank, world, port, out):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    procs = distributed.setup("cpu")
    device = distributed.device_of(procs, "cpu")
    enc, terms = _model()
    forward = distributed.wrap(procs, device, enc, [t.loss for t in terms],
                               lambda x, y: objective_terms(terms, AuxBatch(encoder=enc, views=[x], labels=y)))
    x, y = _full_batch()
    x, y = x.chunk(world)[rank], y.chunk(world)[rank]
    params = list(enc.parameters()) + [p for t in terms for p in t.loss.parameters()]
    opt = torch.optim.SGD(params, lr=0.1)
    total, _ = forward(x, y)
    opt.zero_grad()
    total.backward()
    grads = [p.grad.clone() for p in params]
    opt.step()
    means = distributed.mean_over_ranks(procs, {"rank": float(rank)}, device)
    torch.save({"grads": grads, "params": [p.detach().clone() for p in params], "mean_rank": means["rank"]},
               f"{out}.{rank}")
    distributed.close(procs)


@pytest.mark.parametrize("world", [2, 4])
def test_ddp_gradient_equals_the_full_batch_gradient(tmp_path, world):
    mp.spawn(_ddp_rank, args=(world, 29600 + world, str(tmp_path / "r")), nprocs=world, join=True)
    enc, terms = _model()
    x, y = _full_batch()
    params = list(enc.parameters()) + [p for t in terms for p in t.loss.parameters()]
    objective_terms(terms, AuxBatch(encoder=enc, views=[x], labels=y))[0].backward()
    results = [torch.load(tmp_path / f"r.{r}") for r in range(world)]
    for res in results:
        assert res["mean_rank"] == pytest.approx((world - 1) / 2)
        for got, p in zip(res["grads"], params):
            assert torch.allclose(got, p.grad, atol=1e-6)
    for res in results[1:]:                                       # the copies stayed the same
        for a, b in zip(res["params"], results[0]["params"]):
            assert torch.equal(a, b)


def test_single_process_helpers_do_nothing():
    procs = distributed.Processes()
    assert not procs.active and procs.is_main
    fn = lambda: None    # noqa: E731
    assert distributed.wrap(procs, torch.device("cpu"), nn.Identity(), [], fn) is fn
    assert distributed.mean_over_ranks(procs, {"a": 1.0}, torch.device("cpu")) == {"a": 1.0}
    assert distributed.from_main(procs, True, torch.device("cpu")) is True


# ---- real runs through the launcher ------------------------------------------

def _launch(run_dir, *overrides, world=2):
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    return subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}", "pretrain.py",
         "device=cpu", "num_workers=0", f"hydra.run.dir={run_dir}", *overrides],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=600)


SUPERVISED = ["+experiment=imagenet_supervised", "embedding=resnet18", "objectives.0.num_classes=2", "epochs=null",
              "lr_schedule.name=constant", "eval_every_epochs=null", "save.keep_every_epochs=null",
              "batch_size=4", "log_frequency=2", "eval_frequency=2"]


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    return _image_folder(tmp_path_factory.mktemp("ddp_images"))


def test_supervised_runs_saves_once_and_resumes(images, tmp_path):
    run = tmp_path / "run"
    out = _launch(run, *SUPERVISED, f"pretrain_data.root={images}", "train_steps=4")
    assert out.returncode == 0, out.stderr[-3000:]
    assert "Multi-GPU: 2 processes, batch_size 4 = 2 per GPU" in out.stdout
    assert out.stdout.count("EVAL loss=") == 2                    # rank 0 only: steps 2 and 4
    assert out.stdout.count("Device: cpu (one of 2 processes)") == 1          # only rank 0 prints
    ckpt = torch.load(run / "checkpoint.pt", weights_only=False)
    assert ckpt["step"] == 4 and ckpt["world_size"] == 2
    assert (run / "encoder_step0000004.pt").is_file()
    assert json.loads((run / "timing.json").read_text())["info"]["gpus"] == 2

    out = _launch(run, *SUPERVISED, f"pretrain_data.root={images}", "train_steps=6")
    assert out.returncode == 0, out.stderr[-3000:]
    assert "at step 4" in out.stdout and "'GPUs'" not in out.stdout
    assert torch.load(run / "checkpoint.pt", weights_only=False)["step"] == 6


def test_resuming_on_another_gpu_count_warns(images, tmp_path):
    run = tmp_path / "run"
    out = _launch(run, *SUPERVISED, f"pretrain_data.root={images}", "train_steps=2")
    assert out.returncode == 0, out.stderr[-3000:]
    out = subprocess.run(
        [sys.executable, "pretrain.py", "device=cpu", "num_workers=0", f"hydra.run.dir={run}", *SUPERVISED,
         f"pretrain_data.root={images}", "train_steps=4"],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "'GPUs': (2, 1)" in out.stdout


def test_mae_runs_on_a_whole_vit(images, tmp_path):
    # Also shows every ViT parameter gets a gradient: DDP stops on the second
    # step otherwise.
    out = _launch(tmp_path / "run", "+experiment=mae", "embedding=vit_s16", "epochs=null", "lr_schedule.name=constant",
                  "eval_every_epochs=null", "save.keep_every_epochs=null", "batch_size=4", "log_frequency=2",
                  "eval_frequency=3", "train_steps=3", "objectives.0.decoder_depth=1", "objectives.0.decoder_dim=64",
                  "objectives.0.decoder_heads=4", f"pretrain_data.root={images}", "pretrain_data.eval_batches=1",
                  "perf.amp_bf16=false")
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.count("EVAL loss=") == 1


def test_benchmark_stops_every_process_and_writes_once(images, tmp_path):
    run = tmp_path / "run"
    out = _launch(run, *SUPERVISED, f"pretrain_data.root={images}", "train_steps=100", "timing.benchmark_steps=4",
                  "eval_frequency=50")
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.count("Predicted time for this job") == 1
    assert (run / "timing_benchmark.json").is_file() and not (run / "checkpoint.pt").exists()


@pytest.mark.parametrize("name, extra", [
    ("moco", ["+experiment=moco_v2", "embedding=resnet18"]),
    ("r3m", ["+experiment=r3m", "embedding=resnet18"]),
    ("vip", ["+experiment=vip", "embedding=resnet18"]),
])
def test_objectives_that_compare_samples_are_refused(images, tmp_path, name, extra):
    out = _launch(tmp_path / "run", *extra, f"pretrain_data.root={images}")
    assert out.returncode != 0
    assert f"objectives ['{name}'] run on one GPU" in out.stderr + out.stdout


def test_batch_must_divide_over_gpus(images, tmp_path):
    out = _launch(tmp_path / "run", *SUPERVISED, "batch_size=5", f"pretrain_data.root={images}")
    assert out.returncode != 0 and "doesn't divide" in out.stderr + out.stdout


def test_processes_must_share_a_run_directory(images, tmp_path):
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    # A folder per process is what a default, timestamped hydra.run.dir can become.
    script = tmp_path / "launch.py"
    script.write_text("import os, runpy, sys\n"
                      f"sys.argv = ['pretrain.py', 'hydra.run.dir={tmp_path}/r' + os.environ['RANK']] + sys.argv[1:]\n"
                      f"sys.path.insert(0, {str(ROOT)!r})\n"
                      "runpy.run_path('pretrain.py', run_name='__main__')\n")
    out = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(script),
         "device=cpu", "num_workers=0", *SUPERVISED, f"pretrain_data.root={images}"],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=300)
    assert out.returncode != 0 and "set hydra.run.dir to a fixed folder" in out.stderr + out.stdout
