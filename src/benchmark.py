"""Inference latency, model size and peak memory measurement (P4, plan F3; docs/DECISIONS.md K-046, K-047).

Targets come from results/tables/benchmark_manifest.json (written by ``python -m src.quantize --pilot``): the
Q0/Q1/Q2 ONNX files of the deep models and the native RF (joblib) / XGBoost (ubj) files. Q1 targets marked
``cpu_unsupported`` (no float16 kernels on the CPU execution provider, K-043) are not timed. This module needs
neither PyTorch nor the data pipeline, so it also runs in the constrained Docker container (docker/).

Protocol (configs/quantization.yaml ``benchmark``):
* batch = 1, one window per inference; ``warmup`` (100) inferences excluded, then ``repeats`` (>= 1000) timed
  with ``time.perf_counter_ns``; inputs are a fixed sample of validation windows (data/processed/bench/).
* the RF/XGBoost time includes the window summary features (the deep models take the raw window).
* every target runs in its own subprocess: peak memory = peak resident set of that process (Windows peak working
  set, Linux ru_maxrss), reported absolute and above the level measured right before the model is loaded.
* threads fixed: ONNX Runtime intra-op = T, inter-op = 1, sequential; OMP/MKL/OpenBLAS = T; XGBoost and RF use T.
* the measurement environment (CPU, cores, affinity, memory, cgroup limits, ONNX Runtime provider, threads, power
  state, git commit, library versions) is stored with every result.
* ``--compare``: two sessions of a profile are consistent if every target's median and p95 latency differ by at
  most the pre-registered relative tolerances.

Usage:
    python -m src.benchmark --profile host --session 1          # resumable: finished targets are skipped
    python -m src.benchmark --profile host --session 2
    python -m src.benchmark --profile host --status
    python -m src.benchmark --profile host --compare            # results/tables/benchmark_repeatability_host.csv
    python -m src.benchmark --profile host --table              # results/tables/benchmark_host.csv
    python -m src.benchmark --smoke --out <dir outside results/>    # 20 timed runs, not a measurement
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QCONFIG = PROJECT_ROOT / "configs" / "quantization.yaml"
MANIFEST = PROJECT_ROOT / "results" / "tables" / "benchmark_manifest.json"
TABLES = PROJECT_ROOT / "results" / "tables"
INPUT = "x"
SMOKE = {"warmup": 5, "repeats": 20}
VERSIONED = ["numpy", "onnxruntime", "scikit-learn", "xgboost", "psutil", "joblib"]
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
# Files the quantization/benchmark pipeline WRITES (excluded from the dirty check here and in src.train)
GENERATED_OUTPUTS = ("results/raw", "results/tables/quant_*.csv", "results/tables/benchmark_*.csv",
                     "results/tables/benchmark_manifest.json")
SUMMARY_COLUMNS = ["latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_p99_ms", "latency_std_ms",
                   "peak_rss_mb", "peak_rss_delta_mb", "bench_threads", "bench_session"]


def load_qconfig(path: Path = QCONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found; run `python -m src.quantize --pilot` first")
    return json.loads(path.read_text(encoding="utf-8"))["targets"]


def result_path(out_dir: Path, profile: str, session: int, quant: str) -> Path:
    return Path(out_dir) / f"benchmark_{profile}_s{session}_{quant}.json"


def load_result(out_dir: Path, profile: str, session: int, quant: str) -> dict | None:
    p = result_path(out_dir, profile, session, quant)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------


def latency_stats(samples_ns, n_warmup: int, min_measured: int) -> dict:
    """Statistics of the timed inferences after dropping the first ``n_warmup`` (CLAUDE.md rule 8)."""
    s = np.asarray(samples_ns, dtype=np.float64)
    if n_warmup < 0 or n_warmup >= len(s):
        raise ValueError("warm-up must be shorter than the sample")
    ms = s[n_warmup:] / 1e6
    if len(ms) < min_measured:
        raise ValueError(f"{len(ms)} timed inferences after warm-up; at least {min_measured} required")
    return {"n_warmup": int(n_warmup), "n_measured": int(len(ms)), "mean_ms": float(ms.mean()),
            "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99)), "std_ms": float(ms.std(ddof=1)), "min_ms": float(ms.min()),
            "max_ms": float(ms.max())}


def summary_columns(res: dict) -> dict:
    lat, mem = res["latency"], res["memory"]
    return {"latency_mean_ms": lat["mean_ms"], "latency_median_ms": lat["median_ms"], "latency_p95_ms": lat["p95_ms"],
            "latency_p99_ms": lat["p99_ms"], "latency_std_ms": lat["std_ms"], "peak_rss_mb": mem["peak_rss_mb"],
            "peak_rss_delta_mb": mem["peak_rss_delta_mb"], "bench_threads": res["threads"],
            "bench_session": res["session"]}


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / ((a + b) / 2) if a + b else 0.0


def repeatability(r1: dict, r2: dict, tol: dict) -> dict:
    """Session-to-session agreement of one target (pre-registered tolerances, K-046)."""
    m = rel_diff(r1["latency"]["median_ms"], r2["latency"]["median_ms"])
    p = rel_diff(r1["latency"]["p95_ms"], r2["latency"]["p95_ms"])
    return {"median_s1_ms": r1["latency"]["median_ms"], "median_s2_ms": r2["latency"]["median_ms"],
            "median_rel_diff": m, "p95_s1_ms": r1["latency"]["p95_ms"], "p95_s2_ms": r2["latency"]["p95_ms"],
            "p95_rel_diff": p, "consistent": bool(m <= tol["median_rel_tol"] and p <= tol["p95_rel_tol"])}


# --------------------------------------------------------------------------------------------
# Measurement environment
# --------------------------------------------------------------------------------------------


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def cpu_features() -> list[str]:
    try:
        from numpy._core._multiarray_umath import __cpu_features__
    except ImportError:
        return []
    return sorted(k for k, v in __cpu_features__.items() if v)


def cgroup_limits() -> dict:
    out = {}
    for name, f in (("cpu.max", "/sys/fs/cgroup/cpu.max"), ("memory.max", "/sys/fs/cgroup/memory.max"),
                    ("cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
                    ("memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        p = Path(f)
        if p.exists():
            out[name] = p.read_text().strip()
    return out


def power_state() -> dict:
    import psutil

    out = {}
    try:
        b = psutil.sensors_battery()
        out["battery"] = None if b is None else {"percent": b.percent, "power_plugged": b.power_plugged}
    except (AttributeError, NotImplementedError):
        out["battery"] = None
    if sys.platform == "win32":
        try:
            out["power_scheme"] = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True,
                                                 timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out["power_scheme"] = None
    return out


def git_state() -> dict:
    """Commit and dirty flag; inside the container they come from GIT_COMMIT / GIT_DIRTY (no git there)."""
    if os.environ.get("GIT_COMMIT"):
        return {"commit": os.environ["GIT_COMMIT"], "dirty": os.environ.get("GIT_DIRTY", "1") != "0",
                "source": "environment"}
    excludes = [f":(exclude,glob){p}" if "*" in p else f":(exclude){p}" for p in GENERATED_OUTPUTS]
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True,  # noqa: E731
                                        check=True).stdout
        return {"commit": run("rev-parse", "HEAD").strip(),
                "dirty": bool(run("status", "--porcelain", "--", ".", *excludes).strip()), "source": "git",
                "excluded_from_dirty_check": list(GENERATED_OUTPUTS)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": True, "source": "unavailable"}


def environment_record(profile: str, threads: int, qcfg: dict) -> dict:
    import psutil

    proc = psutil.Process()
    versions = {}
    for p in VERSIONED:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    freq = psutil.cpu_freq()
    return {"profile": profile, "profile_note": qcfg["benchmark"]["profiles"][profile]["note"],
            "threads": threads, "thread_env": {v: os.environ.get(v) for v in THREAD_VARS},
            "platform": platform.platform(), "python": sys.version.split()[0], "cpu": cpu_name(),
            "cpu_features": cpu_features(), "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_affinity": proc.cpu_affinity() if hasattr(proc, "cpu_affinity") else None,
            "cpu_freq_mhz": None if freq is None else {"current": freq.current, "max": freq.max},
            "memory_total_mb": psutil.virtual_memory().total / 2**20, "cgroup": cgroup_limits(),
            "in_container": Path("/.dockerenv").exists(), "power": power_state(), "git": git_state(),
            "versions": versions}


# --------------------------------------------------------------------------------------------
# One target (runs in its own subprocess)
# --------------------------------------------------------------------------------------------


def peak_rss_bytes(proc) -> tuple[int, str]:
    if sys.platform == "win32":
        return int(proc.memory_info().peak_wset), "windows peak working set"
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (int(r) if sys.platform == "darwin" else int(r) * 1024), "getrusage ru_maxrss"


def _ort_session(path: Path, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def measure(target: dict, threads: int, warmup: int, repeats: int, min_repeats: int) -> dict:
    import psutil

    if target["kind"] == "onnx":
        import onnxruntime as ort
    else:
        from src.models.classical import summary_features
        if target["kind"] == "rf":
            import joblib
        else:
            from xgboost import XGBClassifier
    proc = psutil.Process()
    x = np.load(PROJECT_ROOT / target["input_path"])
    model_path = PROJECT_ROOT / target["model_path"]
    rss0 = proc.memory_info().rss                       # libraries imported, inputs loaded, model not yet
    extra = {}
    if target["kind"] == "onnx":
        sess = _ort_session(model_path, threads)

        def fn(xi):
            return sess.run(None, {INPUT: xi})[0]
        extra["onnxruntime"] = {"version": ort.__version__, "providers_used": sess.get_providers(),
                                "intra_op_threads": threads, "inter_op_threads": 1, "execution_mode": "sequential",
                                "graph_optimization": "ORT_ENABLE_ALL"}
    elif target["kind"] == "rf":
        clf = joblib.load(model_path)
        clf.set_params(n_jobs=threads)

        def fn(xi):
            return clf.predict_proba(summary_features(xi))[:, 1]
        extra["runtime"] = {"library": "scikit-learn", "n_jobs": threads}
    else:
        clf = XGBClassifier()
        clf.load_model(model_path)
        clf.set_params(device="cpu", n_jobs=threads)
        rng = (0, int(target["best_iteration"]) + 1)

        def fn(xi):
            return clf.predict(summary_features(xi), output_margin=True, iteration_range=rng)
        extra["runtime"] = {"library": "xgboost", "device": "cpu", "n_jobs": threads, "iteration_range": list(rng)}
    rss1 = proc.memory_info().rss
    n = len(x)
    lat = []
    for i in range(warmup + repeats):
        xi = x[i % n: i % n + 1]
        t0 = time.perf_counter_ns()
        fn(xi)
        lat.append(time.perf_counter_ns() - t0)
    peak, method = peak_rss_bytes(proc)
    return {"latency": latency_stats(lat, warmup, min_repeats), "latencies_us": [round(v / 1e3, 1) for v in lat],
            "memory": {"rss_before_load_mb": rss0 / 2**20, "rss_after_load_mb": rss1 / 2**20,
                       "peak_rss_mb": peak / 2**20, "peak_rss_delta_mb": (peak - rss0) / 2**20, "peak_method": method},
            "model_size_bytes": model_path.stat().st_size, "model_size_mb": model_path.stat().st_size / 2**20,
            "batch_size": 1, "n_inputs": int(n), **extra}


def worker(target_id: str, profile: str, session: int, threads: int, warmup: int, repeats: int, min_repeats: int,
           out: Path, manifest: Path) -> None:
    target = next(t for t in load_manifest(manifest) if t["id"] == target_id)
    started = _now()
    res = measure(target, threads, warmup, repeats, min_repeats)
    res = {"target": target, "profile": profile, "session": session, "threads": threads, "warmup": warmup,
           "repeats": repeats, "started_utc": started, "finished_utc": _now(),
           "smoke": min_repeats < load_qconfig()["benchmark"]["min_repeats"],
           "environment": environment_record(profile, threads, load_qconfig()), **res}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, out)


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


def thread_env(threads: int) -> dict:
    env = dict(os.environ)
    for v in THREAD_VARS:
        env[v] = str(threads)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_all(profile: str, session: int, threads: int, qcfg: dict, force: bool = False, smoke_out: Path | None = None,
            only: str | None = None, manifest: Path = MANIFEST, timeout: int = 3600) -> list[tuple[str, str]]:
    b = qcfg["benchmark"]
    warmup, repeats, min_rep = b["warmup"], b["repeats"], b["min_repeats"]
    if smoke_out is not None:
        warmup, repeats, min_rep = SMOKE["warmup"], SMOKE["repeats"], SMOKE["repeats"]
    if repeats < min_rep:
        raise ValueError(f"repeats {repeats} < {min_rep} (CLAUDE.md rule 8)")
    done = []
    for t in load_manifest(manifest):
        if only and only not in t["id"]:
            continue
        if t["status"] != "measure":
            log(f"skip {t['id']}: {t['status']}")
            done.append((t["id"], t["status"]))
            continue
        out_dir = smoke_out if smoke_out is not None else PROJECT_ROOT / t["out_dir"]
        out = result_path(out_dir, profile, session, t["quant"])
        if smoke_out is not None:
            out = out.with_name(f"{t['run_id']}_{out.name}")
        if out.exists() and not force:
            log(f"skip {t['id']}: already measured ({out.name})")
            done.append((t["id"], "exists"))
            continue
        log(f"measure {t['id']} ({profile}, session {session}, {threads} threads, {warmup}+{repeats})")
        cmd = [sys.executable, "-m", "src.benchmark", "--worker", t["id"], "--profile", profile, "--session",
               str(session), "--threads", str(threads), "--warmup", str(warmup), "--repeats", str(repeats),
               "--min-repeats", str(min_rep), "--out", str(out), "--manifest", str(manifest)]
        try:
            subprocess.run(cmd, cwd=PROJECT_ROOT, env=thread_env(threads), check=True, timeout=timeout)
            r = json.loads(out.read_text(encoding="utf-8"))
            log(f"  median {r['latency']['median_ms']:.4f} ms, p95 {r['latency']['p95_ms']:.4f} ms, "
                f"peak +{r['memory']['peak_rss_delta_mb']:.1f} MB")
            done.append((t["id"], "measured"))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log(f"FAILED {t['id']}: {e}")
            done.append((t["id"], "failed"))
    return done


def status(profile: str, manifest: Path = MANIFEST) -> list[dict]:
    rows = []
    for t in load_manifest(manifest):
        row = {"target": t["id"], "status": t["status"]}
        for s in (1, 2):
            row[f"session{s}"] = result_path(PROJECT_ROOT / t["out_dir"], profile, s, t["quant"]).exists()
        rows.append(row)
    return rows


def compare(profile: str, qcfg: dict, manifest: Path = MANIFEST):
    import pandas as pd

    tol = qcfg["benchmark"]["repeatability"]
    rows = []
    for t in load_manifest(manifest):
        if t["status"] != "measure":
            continue
        r1, r2 = (load_result(PROJECT_ROOT / t["out_dir"], profile, s, t["quant"]) for s in (1, 2))
        if r1 and r2:
            rows.append({"target": t["id"], "dataset": t["dataset"], "model": t["model"], "quant": t["quant"],
                         **repeatability(r1, r2, tol)})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(TABLES / f"benchmark_repeatability_{profile}.csv", index=False, float_format="%.6g")
    return df, bool(len(df)) and bool(df["consistent"].all())


def table(profile: str, manifest: Path = MANIFEST):
    import pandas as pd

    targets = load_manifest(manifest)
    q0_size = {t["run_id"]: (PROJECT_ROOT / t["model_path"]).stat().st_size for t in targets if t["quant"] == "q0"}
    rows = []
    for t in targets:
        size = (PROJECT_ROOT / t["model_path"]).stat().st_size
        base = {"dataset": t["dataset"], "model": t["model"], "quant": t["quant"], "run_id": t["run_id"],
                "status": t["status"], "model_size_mb": size / 2**20,
                "size_ratio_vs_q0": size / q0_size[t["run_id"]] if t["run_id"] in q0_size else np.nan}
        found = False
        for s in (1, 2):
            r = load_result(PROJECT_ROOT / t["out_dir"], profile, s, t["quant"])
            if r:
                found = True
                env = r["environment"]
                rows.append({**base, "profile": profile, **summary_columns(r), "cpu": env["cpu"],
                             "in_container": env["in_container"], "git_commit": env["git"]["commit"],
                             "finished_utc": r["finished_utc"]})
        if not found:
            rows.append({**base, "profile": profile})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"benchmark_{profile}.csv", index=False, float_format="%.6g")
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default="host", help="host | container (configs/quantization.yaml)")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--threads", type=int, help="default: the profile's thread count")
    ap.add_argument("--only", help="substring filter on target ids")
    ap.add_argument("--force", action="store_true", help="re-measure targets that already have a result")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="20 timed runs into --out (outside results/); not a measurement")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--warmup", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--repeats", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--min-repeats", type=int, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    qcfg = load_qconfig()
    if a.profile not in qcfg["benchmark"]["profiles"]:
        raise SystemExit(f"unknown profile {a.profile!r}")
    threads = a.threads or qcfg["benchmark"]["profiles"][a.profile]["threads"]
    if a.worker:
        worker(a.worker, a.profile, a.session, threads, a.warmup, a.repeats, a.min_repeats, a.out, a.manifest)
        return 0
    if a.status:
        for r in status(a.profile, a.manifest):
            print(r)
        return 0
    if a.compare:
        df, ok = compare(a.profile, qcfg, a.manifest)
        print(df.to_string(index=False) if len(df) else "no target measured in both sessions")
        print("CONSISTENT" if ok else "NOT CONSISTENT (or incomplete)")
        return 0 if ok else 1
    if a.table:
        print(table(a.profile, a.manifest).to_string(index=False))
        return 0
    smoke_out = None
    if a.smoke:
        if a.out is None:
            raise SystemExit("--smoke needs --out")
        smoke_out = a.out.resolve()
        if smoke_out.is_relative_to((PROJECT_ROOT / "results").resolve()):
            raise SystemExit("smoke output must not go under results/ (it is not a measurement)")
    else:
        g = git_state()
        if g["dirty"] and not a.allow_dirty:
            raise SystemExit("working tree has uncommitted changes (CLAUDE.md rule 7); commit or pass --allow-dirty")
    done = run_all(a.profile, a.session, threads, qcfg, a.force, smoke_out, a.only, a.manifest)
    failed = [d for d in done if d[1] == "failed"]
    log(f"finished: {len(done)} targets, {len(failed)} failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
