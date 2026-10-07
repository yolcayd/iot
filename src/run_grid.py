"""Runs the experiment grid in configs/grid.yaml (P3: baseline and ae_feasibility stages; P5/F4: full stage).

Every run: train -> calibrate the decision threshold on validation normals (inside ``src.train``, K-035) ->
evaluate the stage's ``evaluate_splits``. Runs of models listed in the stage's ``quantize`` additionally get
Q0/Q1/Q2 and their K-036 evaluation (``src.quantize``). Only the run ids that the CURRENT configuration plans are
used (older runs under results/raw are ignored).

Full stage (P5, K-053): one job per training run id; the steps of a job run in order and every step is skipped
when its output exists (resumable: DONE, metrics_<split>.csv, quant/quant_meta.json, ``evaluated_utc`` in
quant_meta.json; an interrupted deep training continues from its epoch checkpoint). Seed-0 runs of P3/P4 have
the same run ids and are therefore reused, not retrained. Every job runs in its own subprocess (``--job``), at
most ``--workers`` at a time; a failing job is logged and the grid moves on; failures are listed at the end and
in results/tables/grid_failures.csv. All output goes to logs/grid_<date>.log. No latency is measured per seed
(K-052).

Usage:
    python -m src.run_grid --stage full --dry-run [--workers N]     # job list and estimated duration
    python -m src.run_grid --stage full [--workers N] [--threads T] [--only RUN_ID ...]
    python -m src.run_grid --stage full --status
    python -m src.run_grid --stage full --table      # grid_quant.csv, grid_classical.csv, grid_k041_units.csv
    python -m src.run_grid --stage baseline ae_feasibility [--threads N]     # P3 stages (in-process)
    python -m src.run_grid --ae-check     # autoencoder protocol on validation ROC-AUC (K-030, K-033)
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from src import evaluate as E
from src import train as T
from src.data import sources as S

GRID = S.PROJECT_ROOT / "configs" / "grid.yaml"
TABLES = S.PROJECT_ROOT / "results" / "tables"
AE_TABLE = TABLES / "ae_feasibility.csv"
FAILURES = TABLES / "grid_failures.csv"
LOG_DIR = S.PROJECT_ROOT / "logs"
LOCK = LOG_DIR / "grid.lock"
JOB_FILE = "grid_job.json"
BENCH_DIR = S.PROJECT_ROOT / "data" / "processed" / "bench"
# K-033 candidate order: (name, model, overrides reference)
AE_CANDIDATES = [("baseline", "lstm_ae", None), ("retry", "lstm_ae", "ae_feasibility.retry"), ("conv_ae", "conv_ae", None)]
DATASETS = ("car_hacking", "road")
STEPS = ("train", "evaluate", "quantize", "quant_eval")
TAIL_LINES = 40          # lines of a failed job's output repeated in the failure summary
K041_FLOOR = 0.10        # K-041: a unit leaves the primary AS-1 test if the mean Q0 recall of either side is below


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_grid() -> dict:
    return yaml.safe_load(GRID.read_text(encoding="utf-8"))


def stage_runs(stage: str) -> list[dict]:
    g = load_grid()[stage]
    mcfg = T.load_models_config()
    entries = g.get("runs") or [{"model": m} for m in g["models"]]
    quant = set(g.get("quantize", []))
    out = []
    for d in g["datasets"]:
        for e in entries:
            for s in g["seeds"]:
                ov = T.resolve_overrides(e.get("overrides"), mcfg)
                out.append({"stage": stage, "model": e["model"], "dataset": d, "seed": s, "overrides": ov,
                            "overrides_ref": e.get("overrides") or "",
                            "run_id": T.planned_run_id(e["model"], d, s, ov),
                            "evaluate_splits": g.get("evaluate_splits", ["val", "test"]),
                            "quantize": e["model"] in quant})
    return out


def planned_dirs(stage: str) -> list[Path]:
    return [T.RAW_DIR / r["run_id"] for r in stage_runs(stage)]


# --------------------------------------------------------------------------------------------
# Job state (resumability)
# --------------------------------------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def pending_steps(r: dict, raw: Path = T.RAW_DIR) -> list[str]:
    """Steps of a job whose output is missing, in execution order."""
    d = raw / r["run_id"]
    out = []
    if not (d / "DONE").exists():
        out.append("train")
    if any(not (d / f"metrics_{s}.csv").exists() for s in r["evaluate_splits"]):
        out.append("evaluate")
    if r.get("quantize"):
        meta = _read_json(d / "quant" / "quant_meta.json")
        if meta is None:
            out.append("quantize")
        # evaluated_utc is written after quant/metrics.csv, so a half-finished evaluation is redone
        if meta is None or "evaluated_utc" not in meta or not (d / "quant" / "metrics.csv").exists():
            out.append("quant_eval")
    return out


def job_state(r: dict, raw: Path = T.RAW_DIR) -> str:
    d = raw / r["run_id"]
    steps = pending_steps(r, raw)
    if not steps:
        return "done"
    if not d.exists():
        return "todo"
    if "train" in steps and (d / "checkpoint.pt").exists():
        return "interrupted"
    return "partial"


def status(stages: list[str]) -> pd.DataFrame:
    rows = []
    for st in stages:
        for r in stage_runs(st):
            d = T.RAW_DIR / r["run_id"]
            job = _read_json(d / JOB_FILE) or {}
            rows.append({"stage": st, "model": r["model"], "overrides": r["overrides_ref"], "dataset": r["dataset"],
                         "seed": r["seed"], "state": job_state(r), "pending": ",".join(pending_steps(r)),
                         "run": r["run_id"],
                         "evaluated": ",".join(s for s in r["evaluate_splits"] if (d / f"metrics_{s}.csv").exists()),
                         "job_wall_s": job.get("wall_s")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# One job (runs inside its own subprocess in the full stage)
# --------------------------------------------------------------------------------------------


def run_job(r: dict, threads: int, allow_dirty: bool = False, log=log) -> Path:
    """train -> evaluate -> (quantize -> K-036 evaluation); every step skipped when its output exists."""
    T.set_threads(threads)
    started, t0 = datetime.now(timezone.utc), time.perf_counter()
    todo = pending_steps(r)
    log(f"job {r['run_id']}: pending steps {todo or 'none'}")
    if not todo:
        return T.RAW_DIR / r["run_id"]
    git = T.git_info()
    if git["dirty"] and not allow_dirty:
        raise RuntimeError("working tree has uncommitted changes (CLAUDE.md rule 7); commit or pass --allow-dirty")
    steps = {}

    def step(name, fn):
        s = time.perf_counter()
        fn()
        steps[name] = round(time.perf_counter() - s, 1)
        log(f"job {r['run_id']}: step {name} done in {steps[name]} s")

    run_dir = T.RAW_DIR / r["run_id"]

    def train():
        d = T.train(r["model"], r["dataset"], r["seed"], threads=threads, overrides=r["overrides"],
                    allow_dirty=allow_dirty, log=log)
        if d.name != r["run_id"]:
            raise RuntimeError(f"run id mismatch: planned {r['run_id']}, trained {d.name}")

    if "train" in todo:
        step("train", train)
    if "evaluate" in todo:
        step("evaluate", lambda: [E.evaluate_run(run_dir, s, log=log) for s in r["evaluate_splits"]
                                  if not (run_dir / f"metrics_{s}.csv").exists()])
    if "quantize" in todo or "quant_eval" in todo:
        from src import quantize as Q

        qcfg = Q.load_qconfig()
        if "quantize" in todo:
            step("quantize", lambda: Q.quantize_run(run_dir, qcfg, threads, log=log))
        step("quant_eval", lambda: Q.evaluate_run(run_dir, qcfg, threads, log=log))
    left = pending_steps(r)
    if left:
        raise RuntimeError(f"job {r['run_id']} finished with missing outputs: {left}")
    finished = datetime.now(timezone.utc)
    job = {"run_id": r["run_id"], "stage": r["stage"], "model": r["model"], "dataset": r["dataset"],
           "seed": r["seed"], "steps_run": list(steps), "step_s": steps,
           "wall_s": round(time.perf_counter() - t0, 1), "started_utc": started.isoformat(timespec="seconds"),
           "finished_utc": finished.isoformat(timespec="seconds"), "threads": threads, "git": git,
           "pid": os.getpid()}
    (run_dir / JOB_FILE).write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    log(f"job {r['run_id']}: finished in {job['wall_s']} s")
    return run_dir


# --------------------------------------------------------------------------------------------
# Duration estimates (no invented constants: measured grid jobs, else the recorded seed-0 durations)
# --------------------------------------------------------------------------------------------


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def reference_durations(runs: list[dict], raw: Path = T.RAW_DIR) -> dict[tuple[str, str], tuple[float, str]]:
    """(model, dataset) -> (seconds, source). Source ``measured``: mean wall time of finished grid jobs of that
    pair that really trained (grid_job.json). Else ``seed0_record``: training ``duration_s`` of the done run +
    the quant evaluation time (evaluated_utc - created_utc of quant_meta.json); it excludes interpreter start,
    data loading, the post-training threshold and evaluation and the quantization step, which are not recorded
    for those runs, so it is a lower bound."""
    out = {}
    for key in {(r["model"], r["dataset"]) for r in runs}:
        same = [r for r in runs if (r["model"], r["dataset"]) == key]
        jobs = [j for j in (_read_json(raw / r["run_id"] / JOB_FILE) for r in same) if j and "train" in j["steps_run"]]
        if jobs:
            out[key] = (sum(j["wall_s"] for j in jobs) / len(jobs), f"measured (n={len(jobs)})")
            continue
        for r in sorted(same, key=lambda r: r["seed"]):
            rc = _read_json(raw / r["run_id"] / "run_config.json")
            if not (raw / r["run_id"] / "DONE").exists() or not rc or "duration_s" not in rc:
                continue
            sec = float(rc["duration_s"])
            qm = _read_json(raw / r["run_id"] / "quant" / "quant_meta.json")
            if r.get("quantize") and qm and "evaluated_utc" in qm:
                sec += (_utc(qm["evaluated_utc"]) - _utc(qm["created_utc"])).total_seconds()
            out[key] = (sec, f"seed{r['seed']}_record (lower bound)")
            break
    return out


def makespan(durations: list[float], workers: int) -> float:
    """Longest-processing-time-first schedule length on ``workers`` identical workers."""
    loads = [0.0] * max(1, workers)
    for d in sorted(durations, reverse=True):
        heapq.heapreplace(loads, loads[0] + d)
    return max(loads)


def _fmt(sec: float | None) -> str:
    if sec is None:
        return "?"
    sec = int(round(sec))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def plan_table(runs: list[dict], workers: int) -> tuple[pd.DataFrame, float | None, list[str]]:
    ref = reference_durations(runs)
    rows, todo_est, unknown = [], [], []
    for r in runs:
        steps = pending_steps(r)
        est, src = ref.get((r["model"], r["dataset"]), (None, "none"))
        if steps:
            if est is None:
                unknown.append(r["run_id"])
            else:
                todo_est.append(est)
        rows.append({"model": r["model"], "dataset": r["dataset"], "seed": r["seed"], "run_id": r["run_id"],
                     "state": job_state(r), "steps": ",".join(steps) or "-",
                     "est": _fmt(est) if steps else "-", "est_source": src if steps else "-"})
    total = makespan(todo_est, workers) if todo_est else 0.0
    return pd.DataFrame(rows), (None if unknown else total), unknown


# --------------------------------------------------------------------------------------------
# Grid log, lock, scheduler
# --------------------------------------------------------------------------------------------


class GridLog:
    """Thread-safe log: logs/grid_<date>.log (UTF-8, append) and the console."""

    def __init__(self, path: Path | None, echo: bool = True):
        self.path, self.echo, self.lock = path, echo, threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str, raw: bool = False) -> None:
        line = msg if raw else f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        with self.lock:
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            if self.echo:
                print(line, flush=True)


def _lock_holder_alive(pid: int) -> bool:
    import psutil

    try:
        p = psutil.Process(pid)
        return p.is_running() and "src.run_grid" in " ".join(p.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def acquire_lock(path: Path = LOCK, alive=_lock_holder_alive) -> None:
    """Only one grid process at a time; a lock of a process that no longer runs is replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    old = _read_json(path)
    if old and alive(int(old["pid"])):
        raise SystemExit(f"another grid is running (pid {old['pid']}, started {old['started']}); see {path}")
    path.write_text(json.dumps({"pid": os.getpid(), "started": datetime.now().isoformat(timespec="seconds"),
                                "argv": sys.argv}) + "\n", encoding="utf-8")


def release_lock(path: Path = LOCK) -> None:
    old = _read_json(path)
    if old and int(old["pid"]) == os.getpid():
        path.unlink(missing_ok=True)


def job_command(r: dict, threads: int, allow_dirty: bool) -> list[str]:
    cmd = [sys.executable, "-m", "src.run_grid", "--stage", r["stage"], "--job", r["run_id"], "--threads", str(threads)]
    return cmd + (["--allow-dirty"] if allow_dirty else [])


def run_jobs(jobs: list[dict], workers: int, make_cmd, glog, estimates: dict[str, float] | None = None,
             poll: float = 1.0) -> list[dict]:
    """Runs every job in its own subprocess, at most ``workers`` at a time, longest estimate first. A failing
    job does not stop the others. Returns the failures (run_id, exit code, output tail)."""
    est = dict(estimates or {})
    measured: dict[tuple, list[float]] = {}

    def guess(r):
        m = measured.get((r["model"], r["dataset"]))
        return sum(m) / len(m) if m else est.get(r["run_id"])

    queue = sorted(jobs, key=lambda r: -(est.get(r["run_id"]) or 0.0))
    running, failures, n_done, t_start = {}, [], 0, time.perf_counter()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}

    def reader(rid, proc, tail):
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            tail.append(line)
            glog(f"[{rid}] {line}", raw=True)

    try:
        while queue or running:
            while queue and len(running) < workers:
                r = queue.pop(0)
                proc = subprocess.Popen(make_cmd(r), cwd=S.PROJECT_ROOT, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                        env=env)
                tail = deque(maxlen=TAIL_LINES)
                th = threading.Thread(target=reader, args=(r["run_id"], proc, tail), daemon=True)
                th.start()
                running[r["run_id"]] = (r, proc, th, tail, time.perf_counter())
                glog(f"START {r['run_id']} (pid {proc.pid}; running {len(running)}/{workers}, queued {len(queue)})")
            time.sleep(poll)
            for rid in [k for k, v in running.items() if v[1].poll() is not None]:
                r, proc, th, tail, t0 = running.pop(rid)
                th.join(timeout=10)
                wall = time.perf_counter() - t0
                n_done += 1
                if proc.returncode == 0:
                    measured.setdefault((r["model"], r["dataset"]), []).append(wall)
                    glog(f"OK {rid} in {_fmt(wall)}")
                else:
                    failures.append({"run_id": rid, "model": r["model"], "dataset": r["dataset"], "seed": r["seed"],
                                     "exit_code": proc.returncode, "wall_s": round(wall, 1),
                                     "finished": datetime.now().isoformat(timespec="seconds"), "tail": "\n".join(tail)})
                    glog(f"FAILED {rid} (exit code {proc.returncode}) after {_fmt(wall)}; last lines:\n"
                         + "\n".join(f"    {x}" for x in tail))
                left = [guess(q) for q in queue]
                now = time.perf_counter()
                left += [max(0.0, (guess(v[0]) or 0.0) - (now - v[4])) for v in running.values()]
                eta = None if any(x is None for x in left) else makespan(left, workers)
                fin = "?" if eta is None else f"{datetime.now() + timedelta(seconds=eta):%H:%M}"
                glog(f"PROGRESS {n_done}/{len(jobs)} finished ({len(failures)} failed), running {len(running)}, "
                     f"queued {len(queue)}; elapsed {_fmt(now - t_start)}, est. remaining {_fmt(eta)}, "
                     f"est. finish {fin}")
    except KeyboardInterrupt:
        glog("interrupted: terminating running jobs (rerun the same command to resume)")
        for _, proc, *_ in running.values():
            proc.terminate()
        raise
    return failures


def write_failures(failures: list[dict], path: Path = FAILURES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["run_id", "model", "dataset", "seed", "exit_code", "wall_s", "finished", "tail"]
    pd.DataFrame(failures, columns=cols).to_csv(path, index=False)


def preflight(runs: list[dict]) -> None:
    """Cheap checks before any job starts."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available (deep models, XGBoost and the Q1 fp16 simulation use the GPU, K-031/K-043)")
    ae = [r for r in runs if r["model"] in T.AUTOENCODERS and r["seed"] == 0]
    if ae:
        sel = E.selected_autoencoders()
        for r in ae:
            if sel.get(r["dataset"]) != r["run_id"]:
                raise SystemExit(f"{r['dataset']}: the K-033 protocol selected {sel.get(r['dataset'])}, "
                                 f"the grid plans {r['run_id']} (K-039)")
    for d in {r["dataset"] for r in runs if r.get("quantize")}:
        if not (BENCH_DIR / f"{d}.npy").exists():   # written once in P4; parallel jobs must not race on it
            raise SystemExit(f"missing benchmark inputs {BENCH_DIR / f'{d}.npy'} (python -m src.quantize --pilot)")


def run_full(stage: str, workers: int, threads: int, only: list[str] | None, allow_dirty: bool) -> int:
    runs = stage_runs(stage)
    if only:
        unknown = set(only) - {r["run_id"] for r in runs}
        if unknown:
            raise SystemExit(f"not in stage {stage}: {sorted(unknown)}")
        runs = [r for r in runs if r["run_id"] in only]
    glog = GridLog(LOG_DIR / f"grid_{datetime.now():%Y-%m-%d}.log")
    acquire_lock()
    try:
        glog(f"=== grid stage {stage}: {len(runs)} runs; workers {workers} x threads {threads} = "
             f"{workers * threads} CPU threads; commit {T.git_info()['commit'][:7]}; pid {os.getpid()}")
        if T.git_info()["dirty"] and not allow_dirty:
            raise SystemExit("working tree has uncommitted changes (CLAUDE.md rule 7); commit or pass --allow-dirty")
        preflight(runs)
        todo = []
        for r in runs:
            steps = pending_steps(r)
            if steps:
                todo.append(r)
                glog(f"queued {r['run_id']} ({job_state(r)}; steps {','.join(steps)})")
            else:
                glog(f"skip {r['run_id']} (done)")
        ref = reference_durations(stage_runs(stage))
        est = {r["run_id"]: ref.get((r["model"], r["dataset"]), (None, ""))[0] for r in todo}
        known = [v for v in est.values() if v is not None]
        glog(f"{len(todo)} jobs to run; estimated {_fmt(makespan(known, workers) if known else None)}"
             + ("" if len(known) == len(todo) else " (some jobs without estimate)"))
        failures = run_jobs(todo, workers, lambda r: job_command(r, threads, allow_dirty), glog, est)
        write_failures(failures)
        if failures:
            glog(f"=== finished; {len(failures)} FAILED: {[f['run_id'] for f in failures]} "
                 f"(details {FAILURES.relative_to(S.PROJECT_ROOT)}); rerun the same command to retry them")
        else:
            glog(f"=== finished; all {len(todo)} jobs succeeded")
        return 1 if failures else 0
    finally:
        release_lock()


# --------------------------------------------------------------------------------------------
# Grid tables (after the grid; F5 aggregates further)
# --------------------------------------------------------------------------------------------


def grid_tables(stage: str = "full") -> dict[str, pd.DataFrame]:
    from src import quantize as Q
    from src.data import dataset as D

    runs = stage_runs(stage)
    missing = [r["run_id"] for r in runs if pending_steps(r)]
    if missing:
        raise SystemExit(f"grid not complete, missing: {missing}")
    keep = ["attack_type", "attack_free", "pooled"]
    deep, classical, pairs = [], [], []
    win_cache = {}
    for r in runs:
        d = T.RAW_DIR / r["run_id"]
        if r["quantize"]:
            m = Q.quant_rows(d)
            m["latency"] = "not per seed: seed-0 host measurements (K-052; quant_pilot.csv, benchmark_ratios_host.csv)"
            deep.append(m)
            full = pd.read_csv(d / "quant" / "metrics.csv")
            q0 = full[(full.quant == "q0") & (full.threshold_mode == "recalibrated") & (full.split == "test")]
            if r["dataset"] not in win_cache:
                win_cache[r["dataset"]] = D.eval_windows(D.load_window_index(), r["dataset"], "test")
            p = E.pairs_table(q0, win_cache[r["dataset"]])
            p["q0_source"] = "q0_onnx_cpu (K-045)"
        else:
            for s in r["evaluate_splits"]:
                m = pd.read_csv(d / f"metrics_{s}.csv")
                classical.append(m[m["level"].isin(keep)])
            p = pd.read_csv(d / "pairs_test.csv")
            p["q0_source"] = "native model"
        if len(p):
            p.insert(0, "seed", r["seed"])
            p.insert(0, "model", r["model"])
            p.insert(0, "dataset", r["dataset"])
            pairs.append(p)
    drop = ["tp", "fp", "fn", "tn", "captures", "attack_types"]
    out = {"grid_quant.csv": pd.concat(deep, ignore_index=True).drop(columns=drop, errors="ignore").sort_values(
               ["dataset", "model", "threshold_mode", "split", "level", "name", "quant", "seed"]),
           "grid_classical.csv": pd.concat(classical, ignore_index=True).drop(columns=drop, errors="ignore").sort_values(
               ["dataset", "model", "split", "level", "name", "seed"])}
    if pairs:
        p = pd.concat(pairs, ignore_index=True)
        u = p.groupby(["dataset", "model", "source_capture", "fabrication", "masquerade", "role", "q0_source"],
                      as_index=False).agg(n_seeds=("seed", "nunique"),
                                          q0_recall_fabrication_mean=("recall_fabrication", "mean"),
                                          q0_recall_masquerade_mean=("recall_masquerade", "mean"))
        floor = (u.q0_recall_fabrication_mean < K041_FLOOR) | (u.q0_recall_masquerade_mean < K041_FLOOR)
        u["in_primary_test"] = (u.role == "primary") & ~floor & (u.n_seeds == 3)
        u["excluded_reason"] = ""
        u.loc[u.role != "primary", "excluded_reason"] = "not in primary pool (K-019)"
        u.loc[(u.role == "primary") & floor, "excluded_reason"] = f"floor: mean Q0 recall < {K041_FLOOR} (K-041)"
        out["grid_k041_units.csv"] = u
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, t in out.items():
        t.to_csv(TABLES / name, index=False, float_format="%.6g")
    return out


# --------------------------------------------------------------------------------------------
# Autoencoder feasibility protocol (K-033)
# --------------------------------------------------------------------------------------------


def val_auc(run_dir: Path) -> float | None:
    f = run_dir / "metrics_val.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f)
    return float(m.loc[m["level"] == "pooled", "roc_auc"].iloc[0])


def ae_decision(auc: dict[tuple[str, str], float | None], min_auc: float) -> dict:
    """Inspection order and selection of K-033 from validation ROC-AUC values (None = not available yet).

    candidate 2 is inspected only if candidate 1 fails on a dataset, candidate 3 only if candidate 2 fails on a
    dataset. Selected = the first inspected candidate that passes on BOTH datasets; if there is none, per dataset
    the first inspected candidate that passes on that dataset; a dataset with no passing candidate is excluded
    (exit rule: the autoencoder leaves the ROAD analyses, AS-3 on Car-Hacking only).
    """
    names = [c[0] for c in AE_CANDIDATES]

    def passes(c, d):
        return auc.get((c, d)) is not None and auc[(c, d)] >= min_auc

    inspected = []
    for c in names:
        inspected.append(c)
        if any(auc.get((c, d)) is None for d in DATASETS):
            return {"inspected": inspected, "pending": [c], "selected": None}
        if all(passes(c, d) for d in DATASETS):
            break                                  # later candidates are not inspected
    both = [c for c in inspected if all(passes(c, d) for d in DATASETS)]
    if both:
        sel = {d: both[0] for d in DATASETS}
    else:                                          # all three inspected
        sel = {d: next((c for c in inspected if passes(c, d)), None) for d in DATASETS}
    return {"inspected": inspected, "pending": [], "selected": sel}


def ae_check(min_auc: float) -> tuple[pd.DataFrame, dict]:
    mcfg = T.load_models_config()
    auc, rows = {}, {}
    for name, model, ref in AE_CANDIDATES:
        for d in DATASETS:
            rid = T.planned_run_id(model, d, 0, T.resolve_overrides(ref, mcfg))
            auc[(name, d)] = val_auc(T.RAW_DIR / rid)
            rows[(name, d)] = rid
    dec = ae_decision(auc, min_auc)
    shown = [{"candidate": c, "dataset": d, "run": rows[(c, d)], "val_roc_auc": auc[(c, d)],
              "passes": None if auc[(c, d)] is None else auc[(c, d)] >= min_auc}
             for c in dec["inspected"] for d in DATASETS]       # uninspected candidates are not shown (K-033)
    return pd.DataFrame(shown), dec


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def run_stages_inprocess(stages: list[str], threads: int | None) -> int:
    """P3 stages (baseline, ae_feasibility): sequential, in this process."""
    failed = []
    for st in stages:
        for r in stage_runs(st):
            log(f"=== [{st}] {r['model']} {r['overrides_ref']} / {r['dataset']} / seed {r['seed']} ({r['run_id']})")
            try:
                run_dir = T.train(r["model"], r["dataset"], r["seed"], threads=threads, overrides=r["overrides"], log=log)
                if run_dir.name != r["run_id"]:
                    raise RuntimeError(f"run id mismatch: planned {r['run_id']}, trained {run_dir.name}")
                for split in r["evaluate_splits"]:
                    if not (run_dir / f"metrics_{split}.csv").exists():
                        E.evaluate_run(run_dir, split, log=log)
            except Exception:
                log(f"FAILED {r['model']} / {r['dataset']} / seed {r['seed']}\n{traceback.format_exc()}")
                failed.append((st, r["model"], r["dataset"], r["seed"]))
        if st == "baseline":
            try:
                E.baseline_table()
            except RuntimeError as e:
                log(str(e))
    log(f"finished; {len(failed)} failed: {failed}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", nargs="+", default=["baseline"])
    ap.add_argument("--threads", type=int, help="CPU threads PER JOB (default: configs/models.yaml runtime.threads)")
    ap.add_argument("--workers", type=int, default=1, help="parallel jobs (full stage); CPU use = workers x threads")
    ap.add_argument("--only", nargs="+", help="run only these run ids of the stage (e.g. to retry failures)")
    ap.add_argument("--dry-run", action="store_true", help="show the jobs and the estimated duration; run nothing")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--table", action="store_true", help="grid tables after the grid has finished")
    ap.add_argument("--job", help=argparse.SUPPRESS)          # internal: one job in this process
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--ae-check", action="store_true")
    a = ap.parse_args(argv)
    threads = a.threads or T.load_models_config()["runtime"]["threads"]
    if a.status:
        print(status(a.stage).to_string(index=False))
        return 0
    if a.ae_check:
        t, dec = ae_check(T.load_models_config()["ae_feasibility"]["min_val_roc_auc"])
        print(t.to_string(index=False))
        print(json.dumps(dec, indent=2))
        AE_TABLE.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(AE_TABLE, index=False, float_format="%.4f")
        return 0
    if a.job:
        (stage,) = a.stage
        r = next((r for r in stage_runs(stage) if r["run_id"] == a.job), None)
        if r is None:
            raise SystemExit(f"{a.job} is not a run of stage {stage}")
        try:
            run_job(r, threads, a.allow_dirty)
        except Exception:
            log(f"job {a.job} FAILED\n{traceback.format_exc()}")
            return 1
        return 0
    if a.table:
        for name, t in grid_tables(a.stage[0]).items():
            print(f"{TABLES / name}: {len(t)} rows")
        return 0
    if a.dry_run:
        (stage,) = a.stage
        runs = stage_runs(stage)
        if a.only:
            runs = [r for r in runs if r["run_id"] in a.only]
        t, total, unknown = plan_table(runs, a.workers)
        print(t.to_string(index=False))
        n = int((t.steps != "-").sum())
        print(f"\n{len(t)} runs, {len(t) - n} done (skipped), {n} to run; workers {a.workers} x threads {threads} "
              f"= {a.workers * threads} CPU threads")
        if total is None:
            print(f"estimate unavailable for {unknown}")
        else:
            print(f"estimated duration {_fmt(total)} -> finish ~{datetime.now() + timedelta(seconds=total):%H:%M} "
                  f"if started now (est_source: see table)")
        print(f"log: {LOG_DIR / f'grid_{datetime.now():%Y-%m-%d}.log'}")
        return 0
    if a.stage == ["full"] or load_grid()[a.stage[0]].get("quantize"):
        (stage,) = a.stage
        return run_full(stage, a.workers, threads, a.only, a.allow_dirty)
    return run_stages_inprocess(a.stage, a.threads)


if __name__ == "__main__":
    sys.exit(main())
