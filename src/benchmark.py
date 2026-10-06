"""Inference latency, model size and peak memory measurement (P4, plan F3; docs/DECISIONS.md K-046, K-047).

Targets come from results/tables/benchmark_manifest.json (written by ``python -m src.quantize --pilot``): the
Q0/Q1/Q2 ONNX files of the deep models and the native RF (joblib) / XGBoost (ubj) files. Q1 targets marked
``cpu_unsupported`` (no float16 kernels on the CPU execution provider, K-043) are not timed. This module needs
neither PyTorch nor the data pipeline, so it also runs in the constrained Docker container (docker/).

Protocol K-049 (configs/quantization.yaml ``benchmark``; supersedes the run length and warm-up of K-046):
* batch = 1, one window per inference; warm-up of at least ``warmup`` (100) inferences AND at least
  ``warmup_min_seconds`` (1 s), excluded; then ``repeats`` (10,000) timed with ``time.perf_counter_ns``. Inputs are
  a fixed sample of validation windows (data/processed/bench/), cycled.
* the RF/XGBoost time includes the window summary features (the deep models take the raw window).
* every target runs in its own subprocess: peak memory = peak resident set of that process (Windows peak working
  set, Linux ru_maxrss), reported absolute and above the level measured right before the model is loaded.
* threads fixed: ONNX Runtime intra-op = T, inter-op = 1, sequential; OMP/MKL/OpenBLAS = T; XGBoost uses T; the
  random forest predicts single-threaded (n_jobs = 1, K-048: sklearn's threaded prediction races on the
  process-wide warning filters and then prints a warning per tree inside the timed loop).
* host profile: the process is pinned to one logical CPU of each of 4 performance cores (core 0 skipped) and the
  run is refused unless the Windows power mode is "Best performance" (K-049).
* a measurement is rejected if any warning is shown during the timed inferences or the warning filters change;
  results marked invalid (``--invalidate``) or measured under another protocol are kept, ignored by every reader and
  archived before re-measuring.
* the measurement environment (CPU, cores, affinity, memory, cgroup limits, ONNX Runtime provider, threads, power
  plan and mode, git commit, library versions) is stored with every result.
* ``--compare``: two sessions of a profile are consistent if every target's median and p95 latency differ by at
  most the pre-registered relative tolerances. Pre-registered: no re-measurement after the two K-049 sessions;
  inconsistent targets are reported with both values and flagged, comparisons rest on within-session ratios.

Usage:
    python -m src.benchmark --profile host --session 1          # resumable: finished targets are skipped
    python -m src.benchmark --profile host --session 2
    python -m src.benchmark --profile host --status
    python -m src.benchmark --profile host --compare            # results/tables/benchmark_repeatability_host.csv
    python -m src.benchmark --profile host --table              # results/tables/benchmark_host.csv
    python -m src.benchmark --profile host --session 1 --only rf_ --invalidate "<reason>" --decision K-048
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
import warnings
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
# K-048: scikit-learn's threaded prediction (n_jobs > 1, threading backend) enters warnings.catch_warnings() in several
# worker threads at once. catch_warnings is not thread-safe, so the process-wide filter list can be swapped for another
# list (observed: emptied); from then on sklearn prints "`sklearn.utils.parallel.delayed` should be used with ..." for
# every tree task, inside the timed loop (host session 1, RF on ROAD: median 23 ms -> 578 ms). The random forest
# therefore predicts with n_jobs=1 (no worker threads, no race), this one message is ignored, and a measurement is
# rejected if any warning is shown during the timed inferences or the warning filters change.
RF_N_JOBS = 1
SKLEARN_DELAYED_WARNING = r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`"


class MeasurementInvalid(RuntimeError):
    """Something other than the inference ran inside the timed loop (warning output, changed warning filters)."""


# Windows 11 power mode overlays (PowerGetEffectiveOverlayScheme)
POWER_MODES = {"00000000-0000-0000-0000-000000000000": "balanced",
               "ded574b5-45a0-4f42-8737-46345c09c238": "best_performance",
               "961cc777-2547-4f9d-8174-7d86181b8a7a": "best_power_efficiency",
               "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "better_battery"}


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


def current_protocol() -> str:
    return load_qconfig()["benchmark"]["protocol"]


def load_result(out_dir: Path, profile: str, session: int, quant: str, protocol: str | None = None) -> dict | None:
    """The result of a target, or None if it is missing, marked invalid or measured under another protocol."""
    p = result_path(out_dir, profile, session, quant)
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    if r.get("invalid") or r.get("protocol") != (protocol or current_protocol()):
        return None
    return r


def invalidate(path: Path, reason: str, decision: str) -> None:
    """Mark a result invalid in place (kept, never deleted; ignored by compare, table and the pilot table)."""
    r = json.loads(path.read_text(encoding="utf-8"))
    r["invalid"] = {"reason": reason, "decision": decision, "marked_utc": _now()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def archive_invalid(path: Path, protocol: str | None = None) -> Path | None:
    """Before re-measuring: keep an invalid result, or one of another protocol when ``protocol`` is given, under a
    new name instead of overwriting it."""
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    if r.get("invalid") or (protocol is not None and r.get("protocol") != protocol):
        dst = path.with_name(f"{path.stem}.invalid-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
        os.replace(path, dst)
        return dst
    return None


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
    out["power_mode"] = power_mode()
    return out


def power_mode() -> str | None:
    """Windows power mode overlay (Settings > Power > Power mode): balanced, best_performance, ...; None elsewhere."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        import uuid

        g = ctypes.create_string_buffer(16)
        if ctypes.WinDLL("powrprof").PowerGetEffectiveOverlayScheme(g) != 0:
            return None
        s = str(uuid.UUID(bytes_le=g.raw))
        return POWER_MODES.get(s, s)
    except (OSError, AttributeError):
        return None


def cpu_sets() -> list[dict]:
    """Logical processors with their physical core and efficiency class (Windows GetSystemCpuSetInformation;
    a higher class is a faster core). Empty where unavailable."""
    if sys.platform != "win32":
        return []
    import ctypes
    import struct
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    n = wintypes.ULONG(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
    buf = ctypes.create_string_buffer(n.value)
    if not k32.GetSystemCpuSetInformation(buf, n, ctypes.byref(n), None, 0):
        return []
    raw, off, out = buf.raw, 0, []
    while off < n.value:                       # SYSTEM_CPU_SET_INFORMATION records
        size, _ = struct.unpack_from("<II", raw, off)
        _, group, logical, core, _, _, eff = struct.unpack_from("<IHBBBBB", raw, off + 8)
        out.append({"logical": logical, "core": core, "efficiency_class": eff, "group": group})
        off += size
    return out


def performance_cpus(sets: list[dict], n: int, skip_core0: bool) -> list[int]:
    """One logical CPU (the lowest index) of each of the first ``n`` performance cores (highest efficiency class,
    processor group 0); core 0 is skipped if requested and enough performance cores remain (K-049)."""
    sets = [s for s in sets if s["group"] == 0]
    if not sets:
        raise RuntimeError("CPU set information unavailable: cannot pin to performance cores")
    top = max(s["efficiency_class"] for s in sets)
    cores = sorted({s["core"] for s in sets if s["efficiency_class"] == top})
    if skip_core0 and 0 in cores and len(cores) - 1 >= n:
        cores.remove(0)
    if len(cores) < n:
        raise RuntimeError(f"only {len(cores)} performance cores available, {n} requested")
    return [min(s["logical"] for s in sets if s["core"] == c) for c in cores[:n]]


def pin_to_performance_cores(n: int, skip_core0: bool) -> dict:
    import psutil

    sets = cpu_sets()
    cpus = performance_cpus(sets, n, skip_core0)
    proc = psutil.Process()
    proc.cpu_affinity(cpus)
    return {"requested_cores": n, "skip_core0": skip_core0, "cpus": cpus, "applied": proc.cpu_affinity() == cpus,
            "cpu_sets": sets}


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


def measure(target: dict, threads: int, warmup: int, repeats: int, min_repeats: int,
            warmup_min_s: float = 0.0) -> dict:
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
        # only this message, and only for the random forest (K-048); n_jobs=1 removes its cause
        warnings.filterwarnings("ignore", message=SKLEARN_DELAYED_WARNING, category=UserWarning)
        clf = joblib.load(model_path)
        clf.set_params(n_jobs=RF_N_JOBS)

        def fn(xi):
            return clf.predict_proba(summary_features(xi))[:, 1]
        extra["runtime"] = {"library": "scikit-learn", "n_jobs": RF_N_JOBS,
                            "note": "single-threaded prediction: threaded prediction races on warning filters (K-048)"}
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
    warm, lat, shown, at = [], [], [], [-1]          # at[0] = -1 during warm-up, else the timed inference index
    show_orig, filters0 = warnings.showwarning, list(warnings.filters)

    def record(message, category, filename, lineno, file=None, line=None):    # still printed, but counted
        shown.append({"inference": at[0], "category": category.__name__, "message": str(message)[:200]})
        show_orig(message, category, filename, lineno, file, line)

    def timed(i):
        xi = x[i % n: i % n + 1]                     # inputs are cycled
        t0 = time.perf_counter_ns()
        fn(xi)
        return time.perf_counter_ns() - t0

    warnings.showwarning = record
    try:
        # warm-up: at least `warmup` inferences AND at least `warmup_min_s` seconds (K-049); excluded (rule 8)
        t_start = time.perf_counter()
        while len(warm) < warmup or time.perf_counter() - t_start < warmup_min_s:
            warm.append(timed(len(warm)))
        k0 = len(warm)
        for j in range(repeats):
            at[0] = j
            lat.append(timed(k0 + j))
    finally:
        warnings.showwarning = show_orig
    guard = {"warning_filters_intact": list(warnings.filters) == filters0, "n_warnings_shown": len(shown),
             "n_warnings_shown_timed": sum(w["inference"] >= 0 for w in shown), "warnings": shown[:20]}
    if not guard["warning_filters_intact"] or guard["n_warnings_shown_timed"]:
        raise MeasurementInvalid(f"{target['id']}: output or warning-filter change inside the timed loop: {guard}")
    peak, method = peak_rss_bytes(proc)
    w = np.asarray(warm, dtype=np.float64)
    return {"latency": latency_stats(warm + lat, len(warm), min_repeats),
            "warmup_run": {"n": int(len(w)), "seconds": float(w.sum() / 1e9), "min_inferences": warmup,
                           "min_seconds": warmup_min_s, "median_ms": float(np.median(w) / 1e6)},
            "latencies_us": [round(v / 1e3, 1) for v in lat],          # timed inferences only (warm-up excluded)
            "warnings_guard": guard,
            "memory": {"rss_before_load_mb": rss0 / 2**20, "rss_after_load_mb": rss1 / 2**20,
                       "peak_rss_mb": peak / 2**20, "peak_rss_delta_mb": (peak - rss0) / 2**20, "peak_method": method},
            "model_size_bytes": model_path.stat().st_size, "model_size_mb": model_path.stat().st_size / 2**20,
            "batch_size": 1, "n_inputs": int(n), **extra}


def worker(target_id: str, profile: str, session: int, threads: int, warmup: int, repeats: int, min_repeats: int,
           out: Path, manifest: Path, warmup_min_s: float = 0.0, smoke: bool = False) -> None:
    qcfg = load_qconfig()
    prof = qcfg["benchmark"]["profiles"][profile]
    target = next(t for t in load_manifest(manifest) if t["id"] == target_id)
    pinning = None
    if prof.get("pin_performance_cores"):
        # before the model is loaded, so every runtime thread inherits the affinity (K-049)
        try:
            pinning = pin_to_performance_cores(prof["pin_performance_cores"], prof.get("skip_core0", False))
        except (RuntimeError, OSError) as e:
            if not smoke:
                raise
            pinning = {"requested_cores": prof["pin_performance_cores"], "applied": False, "reason": str(e)}
        if not pinning["applied"] and not smoke:
            raise RuntimeError(f"could not pin the process: {pinning}")
    started = _now()
    res = measure(target, threads, warmup, repeats, min_repeats, warmup_min_s)
    res = {"target": target, "protocol": qcfg["benchmark"]["protocol"], "profile": profile, "session": session,
           "threads": threads, "warmup": warmup, "warmup_min_seconds": warmup_min_s, "repeats": repeats,
           "pinning": pinning, "started_utc": started, "finished_utc": _now(), "smoke": smoke,
           "environment": environment_record(profile, threads, qcfg), **res}
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
    warmup, repeats, min_rep, warm_s = b["warmup"], b["repeats"], b["min_repeats"], b["warmup_min_seconds"]
    if smoke_out is not None:
        warmup, repeats, min_rep, warm_s = SMOKE["warmup"], SMOKE["repeats"], SMOKE["repeats"], 0.0
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
        kept = archive_invalid(out, b["protocol"])
        if kept:
            log(f"{t['id']}: previous result is invalid or of another protocol, kept as {kept.name}")
        if out.exists() and not force:
            log(f"skip {t['id']}: already measured ({out.name})")
            done.append((t["id"], "exists"))
            continue
        log(f"measure {t['id']} ({profile}, session {session}, {threads} threads, "
            f"warm-up >= {warmup} and >= {warm_s} s, {repeats} timed)")
        cmd = [sys.executable, "-m", "src.benchmark", "--worker", t["id"], "--profile", profile, "--session",
               str(session), "--threads", str(threads), "--warmup", str(warmup), "--repeats", str(repeats),
               "--min-repeats", str(min_rep), "--warmup-seconds", str(warm_s), "--out", str(out),
               "--manifest", str(manifest)] + (["--smoke"] if smoke_out is not None else [])
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
            p = result_path(PROJECT_ROOT / t["out_dir"], profile, s, t["quant"])
            if not p.exists():
                row[f"session{s}"] = "missing"
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            row[f"session{s}"] = "invalid" if r.get("invalid") else \
                "done" if r.get("protocol") == current_protocol() else f"other protocol ({r.get('protocol')})"
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
                         "protocol": r1["protocol"], **repeatability(r1, r2, tol)})
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
                rows.append({**base, "profile": profile, "protocol": r["protocol"], **summary_columns(r),
                             "pinned_cpus": (r.get("pinning") or {}).get("cpus"),
                             "power_mode": env["power"].get("power_mode"), "cpu": env["cpu"],
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
    ap.add_argument("--invalidate", metavar="REASON",
                    help="mark the results of --profile/--session (filtered by --only) invalid; they are kept")
    ap.add_argument("--decision", default="", help="decision id recorded with --invalidate (e.g. K-048)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--warmup", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--repeats", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--min-repeats", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--warmup-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    qcfg = load_qconfig()
    if a.profile not in qcfg["benchmark"]["profiles"]:
        raise SystemExit(f"unknown profile {a.profile!r}")
    threads = a.threads or qcfg["benchmark"]["profiles"][a.profile]["threads"]
    if a.worker:
        worker(a.worker, a.profile, a.session, threads, a.warmup, a.repeats, a.min_repeats, a.out, a.manifest,
               a.warmup_seconds, a.smoke)
        return 0
    if a.invalidate:
        for t in load_manifest(a.manifest):
            p = result_path(PROJECT_ROOT / t["out_dir"], a.profile, a.session, t["quant"])
            if (not a.only or a.only in t["id"]) and p.exists():
                invalidate(p, a.invalidate, a.decision)
                print(f"marked invalid: {p.relative_to(PROJECT_ROOT)}")
        return 0
    if a.status:
        for r in status(a.profile, a.manifest):
            print(r)
        return 0
    if a.compare:
        df, ok = compare(a.profile, qcfg, a.manifest)
        print(df.to_string(index=False) if len(df) else "no target measured in both sessions")
        print("CONSISTENT" if ok else "NOT CONSISTENT (or incomplete)")
        if len(df) and not ok:
            print("Pre-registered (K-049): no re-measurement. Report both sessions, flag the inconsistent targets and "
                  "base latency comparisons on within-session ratios (Qk/Q0).")
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
        need = qcfg["benchmark"]["profiles"][a.profile].get("required_power_mode")
        if need and power_mode() != need:
            raise SystemExit(f"power mode is {power_mode()!r}, the {a.profile} profile requires {need!r} (K-049): "
                             "Settings > System > Power & battery > Power mode")
    done = run_all(a.profile, a.session, threads, qcfg, a.force, smoke_out, a.only, a.manifest)
    failed = [d for d in done if d[1] == "failed"]
    log(f"finished: {len(done)} targets, {len(failed)} failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
