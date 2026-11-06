"""RunTimer (src/timing.py) on a fake clock: rates, phases, evaluation cost,
estimates and predictions."""

import json

import pytest

from src.timing import RunTimer, count_evals, predict, slurm_time


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _run(timer, clock, steps, per_step, log_every=100, eval_every=None, eval_seconds=0.0, until=None):
    """Advance step by step with constant per-step time (a dict per phase)."""
    for step in range(timer.last_step + 1, (until or steps) + 1):
        clock.t += per_step[timer.phase(step)]
        if step % log_every == 0:
            timer.mark(step)
        if eval_every and step % eval_every == 0:
            with timer.exclude("eval"):
                clock.t += eval_seconds


def test_rate_excludes_evaluations_and_estimates_the_rest():
    clock = Clock()
    t = RunTimer(10_000, eval_every=1000, clock=clock)
    _run(t, clock, 10_000, {"warmup": 0, "train": 0.5}, eval_every=1000, eval_seconds=60, until=3000)
    assert t.rate() == pytest.approx(2.0)                 # 0.5 s/step, evals not counted
    assert t.mean_excluded("eval") == pytest.approx(60)
    # 7000 steps x 0.5 s + 7 evaluations x 60 s
    assert t.remaining_seconds(3000) == pytest.approx(7000 * 0.5 + 7 * 60)


def test_warmup_is_a_separate_phase():
    clock = Clock()
    t = RunTimer(5000, warmup_steps=1000, clock=clock)
    _run(t, clock, 5000, {"warmup": 0.1, "train": 1.0}, until=500)
    assert t.remaining_seconds(500) is None                # training rate not measured yet
    _run(t, clock, 5000, {"warmup": 0.1, "train": 1.0}, until=1500)
    assert t.rate("warmup") == pytest.approx(10.0) and t.rate("train") == pytest.approx(1.0)
    assert t.remaining_seconds(1500) == pytest.approx(3500)
    assert t.predicted_job_total(1500) == pytest.approx(1000 * 0.1 + 4000 * 1.0)


def test_rate_follows_a_slowdown():
    clock = Clock()
    t = RunTimer(100_000, window=5, clock=clock)
    _run(t, clock, 100_000, {"warmup": 0, "train": 1.0}, until=2000)
    _run(t, clock, 100_000, {"warmup": 0, "train": 2.0}, until=3000)   # twice as slow now
    assert t.rate() == pytest.approx(0.5)                 # the recent window, not the average


def test_timing_json_and_prediction_for_another_length(tmp_path):
    clock = Clock()
    path = str(tmp_path / "timing.json")
    t = RunTimer(10_000, eval_every=2000, warmup_steps=1000, path=path,
                 info={"script": "train_sac", "startup_seconds": 30.0}, clock=clock)
    _run(t, clock, 10_000, {"warmup": 0.1, "train": 0.5}, eval_every=2000, eval_seconds=20, until=4000)
    t.write(4000)
    rec = json.load(open(path))
    assert rec["train_steps_per_sec"] == pytest.approx(2.0) and rec["info"]["script"] == "train_sac"
    full = 30 + 1000 * 0.1 + 9000 * 0.5 + 5 * 20
    assert predict(rec) == pytest.approx(full)
    assert rec["predicted_job_seconds"] == pytest.approx(full)
    # The same setup, 4x as long, evaluated half as often.
    assert predict(rec, steps=40_000, eval_every=4000) == pytest.approx(30 + 100 + 39_000 * 0.5 + 10 * 20)


def test_disabled_timer_reports_nothing(tmp_path):
    t = RunTimer(100, enabled=False, path=str(tmp_path / "timing.json"))
    t.mark(50)
    assert t.status(50) == ("", {})
    t.write(50)
    assert not (tmp_path / "timing.json").exists()


def test_cli_summary_and_predict(tmp_path, capsys):
    from src.timing import main
    clock = Clock()
    run = tmp_path / "runs" / "a"
    run.mkdir(parents=True)
    t = RunTimer(1000, eval_every=500, path=str(run / "timing.json"), info={"script": "pretrain"}, clock=clock)
    _run(t, clock, 1000, {"warmup": 0, "train": 2.0}, eval_every=500, eval_seconds=10, until=400)
    t.write(400)
    main(["summary", str(tmp_path / "runs")])
    assert "step 400/1,000" in capsys.readouterr().out
    main(["predict", str(run), "--steps", "2000"])
    out = capsys.readouterr().out                   # no evaluation yet at step 400: 2000 x 2 s only
    assert "predicted total for 2,000 steps: 1h06m" in out and "not measured yet" in out
    _run(t, clock, 1000, {"warmup": 0, "train": 2.0}, eval_every=500, eval_seconds=10, until=600)
    t.write(600)
    main(["predict", str(run), "--steps", "2000"])
    assert "predicted total for 2,000 steps: 1h07m" in capsys.readouterr().out   # 4000 s + 4 x 10 s = 1h07m20s


def test_eval_counting_follows_each_script():
    # SAC/DrQ-v2: none during the warm-up. pretrain.py: also at the last step.
    assert count_evals(0, 10_000, 10_000, 1000, warmup=2500) == 8          # 3000, ..., 10000
    assert count_evals(0, 10_500, 10_500, 1000, final_eval=True) == 11     # + the final one
    assert count_evals(0, 10_000, 10_000, 1000, final_eval=True) == 10     # already a multiple


def test_one_time_costs_and_timing_only_work():
    clock = Clock()
    t = RunTimer(2000, eval_every=1000, save_per_eval=True, final_eval=True, clock=clock)
    clock.t += 120                                     # start-up: loader workers, first CUDA steps
    _run(t, clock, 2000, {"warmup": 0, "train": 1.0}, until=400)
    assert t.rate() == pytest.approx(1.0) and t.one_time_seconds() == pytest.approx(120)
    with t.exclude("eval", timing_only=True):          # benchmark: off-schedule evaluation...
        clock.t += 30
    with t.exclude("save", timing_only=True):          # ... and a test save
        clock.t += 10
    # elapsed 120 + 400 + 40, minus the 40 s timing-only; then 1600 steps + 2 x (eval + save)
    assert t.predicted_job_total(400) == pytest.approx(120 + 400 + 1600 + 2 * 40)
    rec = t.record(400)
    assert predict(rec) == pytest.approx(120 + 2000 + 2 * 40)


def test_final_save_counted_once_for_rl_scripts():
    clock = Clock()
    t = RunTimer(1000, eval_every=500, clock=clock)
    _run(t, clock, 1000, {"warmup": 0, "train": 1.0}, eval_every=500, eval_seconds=5, until=600)
    with t.exclude("save", timing_only=True):
        clock.t += 20
    assert t.remaining_seconds(600) == pytest.approx(400 + 1 * 5 + 20)


def test_resumed_run_carries_earlier_jobs():
    clock = Clock()
    t = RunTimer(1000, start_step=400, prior_seconds=3600, clock=clock)
    _run(t, clock, 1000, {"warmup": 0, "train": 1.0}, until=600)
    text, metrics = t.status(600)
    assert metrics["charts/elapsed_hours"] == pytest.approx((3600 + 200) / 3600)
    assert t.predicted_job_total(600) == pytest.approx(600)           # this job: 400 -> 1000


def test_slurm_time_format():
    assert slurm_time(59) == "00:00:59" and slurm_time(4000) == "01:06:40"
    assert slurm_time(30 * 3600 + 125) == "1-06:02:05"


def test_one_time_cost_ignores_later_rate_drift():
    clock = Clock()
    t = RunTimer(10_000, window=3, clock=clock)
    clock.t += 50                                       # one-time start-up
    _run(t, clock, 10_000, {"warmup": 0, "train": 1.0}, until=2000)
    assert t.one_time_seconds() == pytest.approx(50)
    _run(t, clock, 10_000, {"warmup": 0, "train": 2.0}, until=5000)   # slows down later
    assert t.one_time_seconds() == pytest.approx(50, abs=1)          # not inflated by the drift


def test_benchmark_config_checks_and_files(tmp_path):
    from omegaconf import OmegaConf
    from src.timing import timer_from_config
    cfg = OmegaConf.create({"timing": {"enabled": True, "benchmark_steps": 5000}})
    with pytest.raises(ValueError, match="run ends at 6,000"):
        timer_from_config(cfg, 6000, 1000, 1000, str(tmp_path), {}, 0.0, log_frequency=250)   # 5000 left
    # Passes a naive "steps left" check (4800 > 4500), but the benchmark could
    # only stop at the log step 10,000 -- past the end.
    cfg.timing.benchmark_steps = 4500
    with pytest.raises(ValueError, match="would stop at step 10,000"):
        timer_from_config(cfg, 9800, 1000, 5000, str(tmp_path), {}, 0.0, log_frequency=1000)
    cfg.timing.benchmark_steps = 1000
    timer, bench = timer_from_config(cfg, 6000, 1000, 1000, str(tmp_path), {}, 0.0, log_frequency=250)
    assert bench == 1000 and timer.path.endswith("timing_benchmark.json")    # never timing.json


def test_resume_reads_only_the_real_runs_record(tmp_path):
    from omegaconf import OmegaConf
    from src.timing import timer_from_config
    (tmp_path / "timing.json").write_text(json.dumps({"elapsed_total_seconds": 3000.0,
                                                      "info": {"startup_seconds": 20.0}}))
    (tmp_path / "timing_benchmark.json").write_text(json.dumps({"elapsed_seconds": 999.0}))
    cfg = OmegaConf.create({"timing": {"enabled": True, "benchmark_steps": None}})
    timer, _ = timer_from_config(cfg, 10_000, 1000, 0, str(tmp_path), {}, 0.0, start_step=4000)
    assert timer.prior_seconds == pytest.approx(3020.0)


def test_benchmark_stop_step():
    from src.timing import benchmark_stop_step
    assert benchmark_stop_step(1000, 1000, 250) == 2000      # 1000 past the warm-up, on a log step
    assert benchmark_stop_step(100, 1000, 250) == 1500       # needs two training logs: 1250, 1500
    assert benchmark_stop_step(4500, 5000, 1000) == 10_000   # the first log step at/after 9500
