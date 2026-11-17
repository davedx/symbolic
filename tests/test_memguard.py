"""Memory safeguards (train/memguard.py)."""

import subprocess
import sys
import time

import mlx.core as mx
import pytest

from train import memguard
from train.memguard import GB, MemoryGuardError, Slot, Watchdog


def test_cache_is_capped_across_many_shapes(monkeypatch):
    monkeypatch.setenv("SYMBOLIC_CACHE_LIMIT_GB", "1")
    monkeypatch.setattr(memguard, "_configured", False)
    lim = memguard.configure()
    # many distinct ~64-256 MB buffers: without a cap, freed buffers pile up in the cache
    for i in range(40):
        a = mx.ones((1024 + 37 * i, 16384))
        mx.eval(a)
        del a
    # reclaimed on the next allocation, so it may overshoot by up to one buffer (<= 0.3 GB here);
    # uncapped, this loop leaves ~6 GB cached
    assert mx.get_cache_memory() <= lim["cache_gb"] * GB + 0.3 * GB
    assert mx.get_memory_limit() <= lim["mem_gb"] * GB * 1.01
    monkeypatch.delenv("SYMBOLIC_CACHE_LIMIT_GB")
    memguard._configured = False
    memguard.configure()  # restore the defaults for the rest of the session


def test_watchdog_stops_on_live_memory(monkeypatch):
    monkeypatch.setenv("SYMBOLIC_MEM_LIMIT_GB", "0.01")
    a = mx.ones((4096, 4096))  # 64 MB live
    mx.eval(a)
    with pytest.raises(MemoryGuardError):
        Watchdog().check()


def test_watchdog_stops_on_low_system_memory(monkeypatch):
    monkeypatch.setattr(memguard, "system_free_gb", lambda: 1.0)
    with pytest.raises(MemoryGuardError):
        Watchdog().check()


def test_slots_limit_concurrent_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMBOLIC_MEMGUARD", "1")
    monkeypatch.setenv("SYMBOLIC_MAX_JOBS", "1")
    monkeypatch.setenv("SYMBOLIC_SLOTS_DIR", str(tmp_path))
    monkeypatch.setattr(memguard, "system_free_gb", lambda: 1e6)
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import os,sys,time; os.environ['SYMBOLIC_MEMGUARD']='1';"
         "from train.memguard import Slot; s=Slot(sys.argv[1]); s.acquire('holder'); print('held', flush=True); time.sleep(30)",
         str(tmp_path)],
        stdout=subprocess.PIPE, text=True, env={**__import__("os").environ, "SYMBOLIC_MAX_JOBS": "1"},
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(TimeoutError):
            Slot(tmp_path).acquire("second", poll=0.2, timeout=1.0)
    finally:
        holder.kill()
        holder.wait()
    # the lock is released when the holder dies, even without cleanup
    s = Slot(tmp_path)
    s.acquire("third", poll=0.2, timeout=5.0)
    s.release()


def test_slot_waits_for_free_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMBOLIC_MEMGUARD", "1")
    monkeypatch.setenv("SYMBOLIC_SLOTS_DIR", str(tmp_path))
    monkeypatch.setattr(memguard, "system_free_gb", lambda: 5.0)
    with pytest.raises(TimeoutError):
        Slot().acquire("needs memory", poll=0.2, timeout=0.5)


def test_trainer_checkpoints_and_stops_on_memory_guard(tmp_path, monkeypatch):
    from tests.test_train import smoke_cfg
    from train.trainer import Trainer

    t = Trainer(smoke_cfg(), runs_dir=tmp_path)
    monkeypatch.setattr(memguard, "system_free_gb", lambda: 1.0)
    with pytest.raises(MemoryGuardError):
        t.train(verbose=False)
    assert (t.run_dir / "ckpt").exists()
    assert not (t.run_dir / "results.json").exists()
