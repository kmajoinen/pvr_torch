"""
Run-time estimation for the training scripts (train_sac.py, train_drqv2.py,
pretrain.py): how long a run will take, while it runs and before committing
to it.

RunTimer only reads the clock when it's called -- at log points, and around
evaluations and checkpoint saves -- so it costs nothing per step and never
synchronises the GPU. It keeps:
  - the training rate: steps per second of wall time spent training, i.e.
    excluding evaluations and saves (timed separately, by `excluded()`),
    averaged over the last few log intervals so it follows the current
    speed rather than the run's start;
  - a warm-up phase (SAC/DrQ-v2's learning_starts: random actions, no
    updates), whose rate is kept separate -- it's much faster and would
    otherwise skew the estimate;
  - the mean cost of each kind of excluded work (eval, save).
The estimate is
    remaining steps / training rate + remaining evaluations x mean eval time
and is unknown (None) until the training phase has a measured rate.

Each log also rewrites <run dir>/timing.json: the measurements, the
estimate, the GPU, and the settings that drive the cost (`info`), so later
runs can be predicted from it (python -m src.timing predict). It's written
once more at the very end, so a finished run's startup_seconds +
elapsed_seconds is the job's whole time. A benchmark writes
timing_benchmark.json instead, so it never counts as a resumed run's earlier
time; benchmarks also run with wandb off.

Benchmark mode (timing.benchmark_steps=N in the configs): train N steps
past the warm-up, run one evaluation, print the predicted total for the
configured run and stop -- a few minutes on the real node instead of a guess.

Command line:
    python -m src.timing summary <dir> [<dir> ...]      # measured runs under these folders
    python -m src.timing predict <run dir or timing.json> [--steps N] [--eval-every M]
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import deque
from contextlib import contextmanager
from typing import Optional


def _hms(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    seconds = int(round(seconds))
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    h, rem = divmod(seconds, 3600)
    return f"{h}h{rem // 60:02d}m"


def gpu_name() -> Optional[str]:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return None


def slurm_time(seconds: float) -> str:
    """sbatch's --time format: HH:MM:SS, or D-HH:MM:SS from a day on."""
    seconds = int(math.ceil(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    return (f"{d}-" if d else "") + f"{h:02d}:{rem // 60:02d}:{rem % 60:02d}"


def count_evals(a: int, b: int, total: int, eval_every, warmup: int = 0, final_eval: bool = False) -> int:
    """Evaluations a script runs at steps in (a, b]: every multiple of
    eval_every past the warm-up (SAC/DrQ-v2 skip them during it), plus a
    final one at `total` when the script runs that too (pretrain.py) and it
    isn't already a multiple."""
    if not eval_every:
        return 0
    lo = max(a, warmup)
    n = max(0, b // eval_every - lo // eval_every)
    if final_eval and a < total <= b and total % eval_every:
        n += 1
    return n


class RunTimer:
    """See the module docstring. final_eval: the script also evaluates at the
    last step (pretrain.py). save_per_eval: it saves a checkpoint with every
    evaluation (pretrain.py); otherwise one final save is counted (SAC,
    DrQ-v2). prior_seconds: time spent by earlier jobs of a resumed run."""

    def __init__(self, total_steps: int, eval_every: Optional[int] = None, warmup_steps: int = 0,
                 start_step: int = 0, window: int = 10, path: Optional[str] = None,
                 info: Optional[dict] = None, enabled: bool = True, final_eval: bool = False,
                 save_per_eval: bool = False, prior_seconds: float = 0.0, clock=time.time):
        self.total_steps, self.eval_every = int(total_steps), eval_every
        self.warmup_steps, self.enabled = int(warmup_steps), enabled
        self.final_eval, self.save_per_eval = final_eval, save_per_eval
        self.path, self.info, self.clock = path, dict(info or {}), clock
        self.prior_seconds = float(prior_seconds)
        self.t_start = clock()
        self.excluded_total = 0.0                    # seconds of eval/save so far
        self.excluded = {}                           # kind -> [count, seconds]
        self.timing_only_seconds = 0.0               # evals/saves run only to time them
        # (step, training clock) at log points, per phase: a rate is measured
        # between log points of the same phase only.
        self._samples = {"warmup": deque(maxlen=window), "train": deque(maxlen=window)}
        self._first_train = None                     # (step, training clock) at the first training log
        self._one_time = None                        # fixed once the first training rate exists
        self.start_step = int(start_step)
        self.last_step = int(start_step)

    # ---- measurement -----------------------------------------------------------
    def _train_clock(self) -> float:
        return self.clock() - self.t_start - self.excluded_total

    @contextmanager
    def exclude(self, kind: str, timing_only: bool = False):
        """Time a block (an evaluation, a save) apart from training.
        timing_only: the real run wouldn't do it here (benchmark mode)."""
        t = self.clock()
        try:
            yield
        finally:
            dt = self.clock() - t
            self.excluded_total += dt
            n, s = self.excluded.get(kind, [0, 0.0])
            self.excluded[kind] = [n + 1, s + dt]
            if timing_only:
                self.timing_only_seconds += dt

    def phase(self, step: int) -> str:
        return "warmup" if step <= self.warmup_steps else "train"

    def mark(self, step: int):
        """Record (step, training clock); call at log points (and at the
        warm-up's last step)."""
        self.last_step = int(step)
        if not self.enabled:
            return
        ph = self.phase(step)
        if ph == "warmup" and not self._samples["warmup"]:
            self._samples["warmup"].append((self.start_step, 0.0))   # the warm-up starts with the run
        if self._samples[ph] and self._samples[ph][-1][0] == step:
            return
        sample = (int(step), self._train_clock())
        if ph == "train" and self._first_train is None:
            self._first_train = sample
        self._samples[ph].append(sample)
        if ph == "train" and self._one_time is None and self.rate("train") is not None:
            self._one_time = self._unexplained_before_first_train_log()

    def rate(self, phase: str = "train") -> Optional[float]:
        """Steps per second of training time over the recent window."""
        s = self._samples[phase]
        if len(s) < 2 or s[-1][1] <= s[0][1]:
            return None
        return (s[-1][0] - s[0][0]) / (s[-1][1] - s[0][1])

    def mean_excluded(self, kind: str) -> Optional[float]:
        n, s = self.excluded.get(kind, [0, 0.0])
        return s / n if n else None

    # ---- estimates --------------------------------------------------------------
    def _evals_between(self, a: int, b: int) -> int:
        return count_evals(a, b, self.total_steps, self.eval_every, self.warmup_steps, self.final_eval)

    def _eval_cost(self) -> float:
        cost = self.mean_excluded("eval") or 0.0
        if self.save_per_eval:
            cost += self.mean_excluded("save") or 0.0
        return cost

    def remaining_seconds(self, step: int) -> Optional[float]:
        tr = self.rate("train")
        if tr is None:
            return None
        rest = 0.0
        if step < self.warmup_steps:
            wr = self.rate("warmup")
            if wr is None:
                return None
            rest += (self.warmup_steps - step) / wr
        rest += (self.total_steps - max(step, self.warmup_steps)) / tr
        rest += self._evals_between(step, self.total_steps) * self._eval_cost()
        if not self.save_per_eval and step < self.total_steps:
            rest += self.mean_excluded("save") or 0.0        # the final checkpoint
        return rest

    def elapsed(self) -> float:
        return self.clock() - self.t_start

    def one_time_seconds(self) -> float:
        """Work done once -- data-loader start-up and prefetching, the first
        CUDA/cuDNN steps: the training-clock time up to the first training
        log that the rates don't explain. Fixed when the first training rate
        is measured (from that early rate), so a rate that drifts later in
        the run isn't mistaken for it."""
        return self._one_time or 0.0

    def _unexplained_before_first_train_log(self) -> float:
        tr, wr = self.rate("train"), self.rate("warmup")
        step, clock = self._first_train
        done_warm = max(0, min(step, self.warmup_steps) - self.start_step)
        done_train = max(0, step - max(self.warmup_steps, self.start_step))
        explained = (done_warm / wr if (done_warm and wr) else 0.0) + (done_train / tr if tr else 0.0)
        return max(0.0, clock - explained)

    def predicted_job_total(self, step: int) -> Optional[float]:
        """This job, start to finish, measured so far plus estimated rest:
        startup + elapsed (every one-time cost so far included) + the rest,
        minus evaluations/saves the real run wouldn't have done."""
        rem = self.remaining_seconds(step)
        if rem is None:
            return None
        return self.info.get("startup_seconds", 0.0) + self.elapsed() + rem - self.timing_only_seconds

    # ---- reporting ------------------------------------------------------------------
    def status(self, step: int):
        """(stdout fragment, wandb metrics) at a log point (after mark())."""
        if not self.enabled:
            return "", {}
        rem = self.remaining_seconds(step)
        total_elapsed = self.prior_seconds + self.elapsed()
        metrics = {"charts/elapsed_hours": total_elapsed / 3600}
        if self.rate("train") is not None:
            metrics["charts/steps_per_sec_recent"] = self.rate("train")
        if rem is not None:
            metrics["charts/eta_hours"] = rem / 3600
        return f"  elapsed={_hms(total_elapsed)}  eta={_hms(rem)}", metrics

    def record(self, step: int) -> dict:
        return {
            "step": int(step), "start_step": self.start_step, "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps, "eval_every": self.eval_every,
            "final_eval": self.final_eval, "save_per_eval": self.save_per_eval,
            "elapsed_seconds": self.elapsed(), "prior_seconds": self.prior_seconds,
            "elapsed_total_seconds": self.prior_seconds + self.elapsed(),
            "train_steps_per_sec": self.rate("train"), "warmup_steps_per_sec": self.rate("warmup"),
            "excluded": {k: {"count": n, "mean_seconds": s / n} for k, (n, s) in self.excluded.items() if n},
            "one_time_seconds": self.one_time_seconds(),
            "remaining_seconds": self.remaining_seconds(step),
            "predicted_job_seconds": self.predicted_job_total(step),
            "gpu": gpu_name(), "info": self.info, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def write(self, step: int):
        if not (self.enabled and self.path):
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.record(step), f, indent=1)
        os.replace(tmp, self.path)

    def benchmark_report(self, step: int) -> str:
        total = self.predicted_job_total(step)
        rec = self.record(step)
        fmt = lambda x: "not measured" if x is None else (f"{x:.1f}s" if x < 60 else _hms(x))
        ev, sv = self.mean_excluded("eval"), self.mean_excluded("save")
        lines = [
            f"Benchmark ({rec['gpu']}): measured {step - max(self.warmup_steps, self.start_step):,} "
            f"training steps" + (" after the warm-up" if self.warmup_steps else ""),
            f"  training      {self.rate('train') or float('nan'):.2f} steps/s",
        ]
        if self.warmup_steps > self.start_step:
            lines.append(f"  warm-up       {self.rate('warmup') or float('nan'):.2f} steps/s "
                         f"({self.warmup_steps:,} steps)")
        lines.append(f"  evaluation    {fmt(ev)} each, {self._evals_between(step, self.total_steps)} still to come")
        lines.append(f"  checkpoint    {fmt(sv)} " + ("with every evaluation" if self.save_per_eval else "(final)"))
        lines.append(f"  one-time      {fmt(self.one_time_seconds())} (start-up, first steps; already elapsed)")
        span = f"from step {self.start_step:,} " if self.start_step else ""
        lines.append(f"Predicted time for this job {span}to step {self.total_steps:,}: {_hms(total)}"
                     + (f"  -> e.g. --time={slurm_time(total * 1.2 + 600)} (20% + 10 min margin)" if total else ""))
        return "\n".join(lines)


def benchmark_stop_step(bench: int, base: int, log_frequency: int) -> int:
    """The step a benchmark actually stops at: the first log step at least
    `bench` steps past `base` (the end of the warm-up, or a resumed run's
    start) that also has two training logs behind it (a measured rate)."""
    reach = -(-(base + bench) // log_frequency) * log_frequency      # first log step >= base + bench
    second_log = (base // log_frequency + 2) * log_frequency         # second training log after base
    return max(reach, second_log)


def timer_from_config(cfg, total_steps: int, eval_every, warmup_steps: int, run_dir: str, info: dict,
                      t_script: float, start_step: int = 0, final_eval: bool = False,
                      save_per_eval: bool = False, log_frequency: int = 1, main_process: bool = True):
    """(RunTimer, benchmark_steps) from cfg.timing ({enabled, benchmark_steps});
    benchmark mode turns timing on regardless. A resumed run (start_step > 0)
    carries over the time its earlier jobs recorded in timing.json. With several
    processes only the main one times anything or writes files."""
    tcfg = cfg.get("timing") or {}
    bench = tcfg.get("benchmark_steps")
    if bench:
        base = max(warmup_steps, start_step)
        stop = benchmark_stop_step(int(bench), base, int(log_frequency))
        if stop >= total_steps:
            raise ValueError(
                f"timing.benchmark_steps={bench} would stop at step {stop:,} (the first log step "
                f"{bench} past step {base:,}, with two training logs for a rate; logs every "
                f"{log_frequency:,}), but the run ends at {total_steps:,}: it would train the whole run "
                "and save nothing. Use fewer benchmark steps, or run normally (eta= is in the logs)")
    info = {**info, "startup_seconds": round(time.time() - t_script, 1)}
    path = os.path.join(run_dir, "timing.json")
    prior = 0.0
    if start_step > 0 and os.path.isfile(path):
        try:
            with open(path) as f:
                old = json.load(f)
            prior = old.get("elapsed_total_seconds", old.get("elapsed_seconds", 0.0)) \
                + old.get("info", {}).get("startup_seconds", 0.0)
        except (OSError, ValueError):
            pass
    # A benchmark writes its own file: timing.json is what the next job of a
    # resumed run reads as its earlier jobs' time.
    timer = RunTimer(total_steps, eval_every=eval_every, warmup_steps=warmup_steps, start_step=start_step,
                     path=os.path.join(run_dir, "timing_benchmark.json") if bench else path, info=info,
                     enabled=(bool(tcfg.get("enabled", True)) or bool(bench)) and main_process,
                     final_eval=final_eval, save_per_eval=save_per_eval, prior_seconds=prior)
    if bench:
        print(f"Benchmark mode: {int(bench):,} training steps after the warm-up, one evaluation, "
              "then a prediction for the configured run (nothing is saved)")
    return timer, (int(bench) if bench else None)


# ---- command line ------------------------------------------------------------------

def _load(path):
    if os.path.isdir(path):
        run_file = os.path.join(path, "timing.json")
        path = run_file if os.path.isfile(run_file) else os.path.join(path, "timing_benchmark.json")
    with open(path) as f:
        return json.load(f)


def predict(rec: dict, steps: Optional[int] = None, eval_every: Optional[int] = None) -> Optional[float]:
    """Total time for a run like `rec`'s but `steps` long (and/or evaluating
    every `eval_every` steps), from its measured rates, evaluation/save
    costs and one-time costs."""
    steps = int(steps or rec["total_steps"])
    eval_every = eval_every if eval_every is not None else rec.get("eval_every")
    tr = rec.get("train_steps_per_sec")
    if not tr:
        return None
    warm = min(rec.get("warmup_steps") or 0, steps)
    wr = rec.get("warmup_steps_per_sec")
    exc = rec.get("excluded", {})
    ev = exc.get("eval", {}).get("mean_seconds", 0.0)
    sv = exc.get("save", {}).get("mean_seconds", 0.0)
    total = rec.get("info", {}).get("startup_seconds", 0.0) + rec.get("one_time_seconds", 0.0)
    total += warm / wr if (warm and wr) else 0.0
    total += (steps - warm) / tr
    n = count_evals(0, steps, steps, eval_every, warm, rec.get("final_eval", False))
    total += n * (ev + (sv if rec.get("save_per_eval") else 0.0))
    if not rec.get("save_per_eval"):
        total += sv                                          # the final checkpoint
    return total


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="list measured runs under folders")
    s.add_argument("dirs", nargs="+")
    q = sub.add_parser("predict", help="predict a run's total time from a measured one")
    q.add_argument("run", help="a run directory or its timing.json")
    q.add_argument("--steps", type=int, help="length of the run to predict (default: the measured run's)")
    q.add_argument("--eval-every", type=int, help="evaluation interval of the run to predict")
    a = p.parse_args(argv)

    if a.cmd == "predict":
        rec = _load(a.run)
        total = predict(rec, a.steps, a.eval_every)
        info = ", ".join(f"{k}={v}" for k, v in rec.get("info", {}).items() if k != "startup_seconds")
        print(f"measured on {rec.get('gpu')}: {info}")
        print(f"  training {rec.get('train_steps_per_sec') or float('nan'):.2f} steps/s"
              + (f", evaluation {_hms(rec['excluded']['eval']['mean_seconds'])} each"
                 if rec.get("excluded", {}).get("eval") else ""))
        print(f"predicted total for {a.steps or rec['total_steps']:,} steps: {_hms(total)}"
              + ("" if rec.get("excluded", {}).get("eval") else "  (evaluations not measured yet: not included)"))
        return
    rows = []
    for d in a.dirs:
        for path in sorted(glob.glob(os.path.join(d, "**", "timing*.json"), recursive=True)):
            r = _load(path)
            rows.append((path, r))
    if not rows:
        sys.exit("no timing.json found")
    for path, r in rows:
        info = r.get("info", {})
        print(f"{path}\n  {r.get('gpu')}  {info.get('script', '?')}  "
              + "  ".join(f"{k}={v}" for k, v in info.items() if k not in ("script", "startup_seconds"))
              + f"\n  step {r['step']:,}/{r['total_steps']:,}  "
              f"elapsed {_hms(r.get('elapsed_total_seconds', r['elapsed_seconds']) + info.get('startup_seconds', 0.0))}  "
              f"train {r.get('train_steps_per_sec') or float('nan'):.2f} steps/s  "
              f"remaining {_hms(r.get('remaining_seconds'))}  predicted total {_hms(predict(r))}")


if __name__ == "__main__":
    main()
