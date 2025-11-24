from __future__ import annotations

"""
Feature collection for PhenoVector.

This module defines a ``BehaviourFeatures`` dataclass capturing a
variety of runtime metrics for a given process, along with helper
functions to safely collect these metrics using ``psutil``. It also
provides a lightweight entropy approximation for executables. These
features form the raw input for gene scoring.
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class BehaviourFeatures:
    """A collection of behavioural characteristics for a single process."""

    pid: int
    name: str
    exe: Optional[str]
    create_time: float
    cpu_percent: float
    memory_rss_mb: float
    num_threads: int
    num_open_files: int
    num_connections: int
    binary_entropy: Optional[float]
    is_system_process: bool
    is_user_temp_exe: bool
    cpu_time_total: float
    lifetime_seconds: float


def _safe_entropy(path: str, max_bytes: int = 200_000) -> Optional[float]:
    """Approximate Shannon entropy of the first ``max_bytes`` of a file.

    Returns a value between 0.0 and 1.0, or ``None`` if the file cannot
    be read. Entropy is normalised by dividing by the maximum possible
    entropy for a byte stream (8 bits).
    """
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
    except OSError:
        return None

    if not data:
        return None

    counts = [0] * 256
    for b in data:
        counts[b] += 1

    total = len(data)
    probs = [c / total for c in counts if c]
    # Shannon entropy in bits
    entropy_bits = -sum(p * math.log2(p) for p in probs)
    # normalised to [0,1]
    return entropy_bits / 8.0


def collect_behaviour_features(proc: psutil.Process) -> BehaviourFeatures:
    """Safely gather behavioural metrics for a given process.

    ``psutil`` calls can raise ``NoSuchProcess`` or ``AccessDenied``. We
    catch and default values to zero where appropriate. CPU percent is
    sampled over a short 50 ms interval to capture transient usage.
    """
    # basic identity
    try:
        name = proc.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        name = f"pid_{proc.pid}"

    try:
        exe = proc.exe()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        exe = None

    try:
        create_time = proc.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        create_time = 0.0

    # resource usage
    try:
        cpu_percent = proc.cpu_percent(interval=0.05)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        cpu_percent = 0.0

    try:
        memory_rss_mb = proc.memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        memory_rss_mb = 0.0

    try:
        num_threads = proc.num_threads()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        num_threads = 0

    try:
        num_open_files = len(proc.open_files())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        num_open_files = 0

    try:
        num_connections = len(proc.connections(kind="inet"))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        num_connections = 0

    # CPU time since process start
    try:
        times = proc.cpu_times()
        cpu_time_total = float(times.user + times.system)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        cpu_time_total = 0.0

    now = time.time()
    lifetime_seconds = max(now - create_time, 1.0)

    # classify
    exe_lower = (exe or "").lower()
    if os.name == "nt":
        win_dir = os.path.expandvars(r"%windir%").lower()
        is_system = exe_lower.startswith(win_dir)
        is_temp = ("temp" in exe_lower) or ("appdata" in exe_lower)
    else:
        # treat common system dirs on Unix as system processes
        is_system = exe_lower.startswith( ("/usr", "/bin", "/sbin", "/lib") )
        is_temp = any(tok in exe_lower for tok in ("/tmp", "/var/tmp", "/run"))

    entropy = _safe_entropy(exe) if exe else None

    return BehaviourFeatures(
        pid=proc.pid,
        name=name,
        exe=exe,
        create_time=create_time,
        cpu_percent=cpu_percent,
        memory_rss_mb=memory_rss_mb,
        num_threads=num_threads,
        num_open_files=num_open_files,
        num_connections=num_connections,
        binary_entropy=entropy,
        is_system_process=is_system,
        is_user_temp_exe=is_temp,
        cpu_time_total=cpu_time_total,
        lifetime_seconds=lifetime_seconds,
    )