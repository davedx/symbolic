"""Memory safeguards for training/eval processes (the Mac Studio has 128 GB, shared CPU/GPU).

Why: MLX keeps freed GPU buffers in a cache for reuse. By default the cache limit is the
memory limit, which defaults to 1.5x the GPU's recommended working set: ~173 GB on the
M5 Max, more than the machine has. Evaluation uses many batch shapes, so a single S run's
cache grew to 51 GB in 6M tokens while its live memory never exceeded 1.5 GB. Four such
processes crashed the machine (~200 GB). Measurements: README, "Memory safeguards".

What this module does (all on by default, tunable through environment variables):

1. Per-process MLX caps  SYMBOLIC_CACHE_LIMIT_GB (default 6) and
                         SYMBOLIC_MEM_LIMIT_GB (default 16).
2. Job slots             at most SYMBOLIC_MAX_JOBS (default 2) jobs at once, enforced by
                         OS file locks in ~/.cache/symbolic/slots, shared by every
                         checkout and entry point (released automatically when a
                         process exits, even if it crashes). A job waits for a free slot.
3. Admission check       a job only starts once the system has at least
                         SYMBOLIC_MEM_LIMIT_GB + SYMBOLIC_RESERVE_GB (default 16) free.
4. Watchdog              ``check()`` is called from the training loop. It raises
                         MemoryGuardError (after which the trainer saves a checkpoint and
                         exits) if this process's live MLX memory exceeds its limit, or if
                         the system's free memory falls below SYMBOLIC_RESERVE_GB.

SYMBOLIC_MEMGUARD=0 disables slots/admission (caps and watchdog stay on).
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

GB = 1024**3


class MemoryGuardError(RuntimeError):
    pass


def _env_gb(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def limits() -> dict:
    return {
        "cache_gb": _env_gb("SYMBOLIC_CACHE_LIMIT_GB", 6),
        "mem_gb": _env_gb("SYMBOLIC_MEM_LIMIT_GB", 16),
        "reserve_gb": _env_gb("SYMBOLIC_RESERVE_GB", 16),
        "max_jobs": int(os.environ.get("SYMBOLIC_MAX_JOBS", 2)),
    }


_configured = False


def configure() -> dict:
    """Cap this process's MLX cache and memory. Idempotent."""
    global _configured
    lim = limits()
    if not _configured:
        mx.set_cache_limit(int(lim["cache_gb"] * GB))
        mx.set_memory_limit(int(lim["mem_gb"] * GB))
        _configured = True
    return lim


def total_gb() -> float:
    return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / GB


def system_free_gb() -> float:
    """Free memory as macOS reports it (kern.memorystatus_level = % available)."""
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True).stdout
    return float(out.strip()) / 100 * total_gb()


def process_gb() -> float:
    return (mx.get_active_memory() + mx.get_cache_memory()) / GB


class Slot:
    """Holds one of SYMBOLIC_MAX_JOBS job slots for the life of the process."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.environ.get("SYMBOLIC_SLOTS_DIR", Path.home() / ".cache/symbolic/slots"))
        self.fh = None

    def acquire(self, label: str = "", poll: float = 15.0, timeout: float | None = None) -> None:
        if os.environ.get("SYMBOLIC_MEMGUARD", "1") == "0":
            return
        t0 = time.monotonic()
        lim = limits()
        self.root.mkdir(parents=True, exist_ok=True)
        need = lim["mem_gb"] + lim["reserve_gb"]
        waited = False
        while True:
            for i in range(lim["max_jobs"]):
                fh = open(self.root / f"slot{i}.lock", "a+")
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    fh.close()
                    continue
                free = system_free_gb()
                if free < need:
                    fcntl.flock(fh, fcntl.LOCK_UN)
                    fh.close()
                    if not waited:
                        print(f"[memguard] waiting: {free:.0f} GB free, need {need:.0f} GB", file=sys.stderr, flush=True)
                        waited = True
                    break
                fh.seek(0)
                fh.truncate()
                fh.write(f"{os.getpid()} {label}\n")
                fh.flush()
                self.fh = fh
                return
            else:
                if not waited:
                    print(f"[memguard] waiting for one of {lim['max_jobs']} job slots ({label})", file=sys.stderr, flush=True)
                    waited = True
            if timeout is not None and time.monotonic() - t0 > timeout:
                raise TimeoutError(f"no job slot / memory for {label} after {timeout:.0f}s")
            time.sleep(poll)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

    def release(self) -> None:
        if self.fh is not None:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None


class Watchdog:
    """Cheap per-step check; the system query runs at most every ``every`` seconds."""

    def __init__(self, every: float = 10.0):
        self.every = every
        self.last = 0.0

    def check(self) -> None:
        lim = limits()
        live = mx.get_active_memory() / GB
        if live > lim["mem_gb"]:
            raise MemoryGuardError(f"live MLX memory {live:.1f} GB exceeds SYMBOLIC_MEM_LIMIT_GB={lim['mem_gb']:.0f}")
        if mx.get_cache_memory() / GB > lim["cache_gb"]:
            mx.clear_cache()
        now = time.monotonic()
        if now - self.last >= self.every:
            self.last = now
            free = system_free_gb()
            if free < lim["reserve_gb"]:
                raise MemoryGuardError(
                    f"system free memory {free:.1f} GB is below SYMBOLIC_RESERVE_GB={lim['reserve_gb']:.0f}"
                )
