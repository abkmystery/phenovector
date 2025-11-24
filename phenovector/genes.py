"""
phenovector.genes
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable

from features import BehaviourFeatures


@dataclass
class PopulationStats:

    max_cpu: float
    max_rss_mb: float
    max_open_files: int
    max_connections: int
    max_threads: int


def compute_population_stats(features: Iterable[BehaviourFeatures]) -> PopulationStats:
 
    return PopulationStats(
        max_cpu=max((f.cpu_percent for f in features), default=0.0) or 1.0,
        max_rss_mb=max((f.memory_rss_mb for f in features), default=0.0) or 1.0,
        max_open_files=max((f.num_open_files for f in features), default=0) or 1,
        max_connections=max((f.num_connections for f in features), default=0) or 1,
        max_threads=max((f.num_threads for f in features), default=0) or 1,
    )


def _norm(value: float, maximum: float) -> float:
   
    if maximum <= 0:
        return 0.0
    v = value / maximum
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return float(v)


# -----------------------------------------------------------------------------
# Gene implementations
# -----------------------------------------------------------------------------

def gene_resource_abuse(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Composite of CPU and memory utilisation.

    Both CPU and resident memory are scaled to [0, 1] based on the
    maxima seen in the current scan, then combined with a weighted
    average (60% CPU, 40% memory).
    """
    cpu_norm = _norm(f.cpu_percent, stats.max_cpu)
    mem_norm = _norm(f.memory_rss_mb, stats.max_rss_mb)
    score = 0.6 * cpu_norm + 0.4 * mem_norm
    return float(min(score, 1.0))


def gene_entropy(f: BehaviourFeatures, _stats: PopulationStats) -> float:
    """Return precomputed executable entropy.

    ``BehaviourFeatures.binary_entropy`` is computed by the collector and
    normalised to [0, 1].  If no entropy is available, this gene
    returns 0.0.
    """
    return float(f.binary_entropy or 0.0)


def gene_impersonation(f: BehaviourFeatures, _stats: PopulationStats) -> float:
    """Heuristic for mismatched process name and executable path.

    Many malware samples rename themselves to look like core system
    components (e.g. ``svchost.exe``) but reside outside of the
    expected directories.  This gene returns a score from 0.0 to 1.0
    based on two rules:

      * If the basename of the executable does not match the process
        name, add 0.4.
      * If the name is in a set of suspicious names and the executable
        does not reside in ``system32`` on Windows (or ``/bin``,
        ``/usr`` on POSIX), add 0.6.
    """
    score = 0.0
    name = (f.name or "").lower()
    exe = (f.exe or "").lower()
    exe_base = os.path.basename(exe) if exe else ""

    if exe_base and exe_base != name:
        score += 0.4

    suspicious = {"svchost.exe", "lsass.exe", "services.exe", "explorer.exe"}
    if name in suspicious:
        if os.name == "nt":
            # Windows: expect suspicious names only in system32
            if "system32" not in exe:
                score += 0.6
        else:
            # POSIX: no exact equivalent but treat it suspiciously if we hit one
            score += 0.6

    return float(min(score, 1.0))


def gene_exfiltration(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Normalised count of outbound network connections."""
    return _norm(f.num_connections, stats.max_connections)


def gene_tracking(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Normalised number of open file handles."""
    return _norm(f.num_open_files, stats.max_open_files)


def gene_persistence(f: BehaviourFeatures, _stats: PopulationStats) -> float:
    """Boot time proximity heuristic.

    Processes that start very close to boot time (within a ten minute
    window) are considered highly persistent.  The score decays
    linearly from 1.0 at boot to 0.0 at 600 seconds.
    """
    import psutil

    try:
        boot_time = psutil.boot_time()
    except Exception:
        return 0.0

    age = max(f.create_time - boot_time, 0.0)
    window = 600.0  # 10 minutes
    if age >= window:
        return 0.0
    return float(max(0.0, 1.0 - (age / window)))


def gene_mutation(f: BehaviourFeatures, _stats: PopulationStats) -> float:
    """Composite of temp directory execution and high entropy.

    Executables running from temp directories or AppData on Windows are
    often transient or malicious.  Very high entropy also suggests a
    packed or encrypted binary.  This gene adds 0.5 for each of these
    conditions, capping at 1.0.
    """
    score = 0.0
    if f.is_user_temp_exe:
        score += 0.5
    if f.binary_entropy and f.binary_entropy > 0.85:
        score += 0.5
    return float(min(score, 1.0))


def gene_stealth(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """IO/network activity with low CPU usage.

    Processes that perform noticeable amounts of IO or network activity
    but consume very little CPU may be "quiet" but doing work behind
    the scenes.  This gene is proportional to IO/network activity and
    inversely proportional to CPU utilisation.  Very low IO (<0.2) is
    considered non-stealthy.
    """
    io_norm = max(
        _norm(f.num_open_files, stats.max_open_files),
        _norm(f.num_connections, stats.max_connections),
    )
    cpu_norm = _norm(f.cpu_percent, stats.max_cpu)
    if io_norm < 0.2:
        return 0.0
    score = io_norm * (1.0 - cpu_norm)
    # clamp to [0, 1]
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)


def gene_latency(f: BehaviourFeatures, _stats: PopulationStats) -> float:
    """Inverse CPU utilisation over lifetime.

    High latency indicates that the process has been alive for a long
    time but consumed relatively little CPU (ratio of total CPU time to
    wall clock lifetime).  Sleepy or dormant processes will have
    latency scores near 1.0.
    """
    ratio = f.cpu_time_total / max(f.lifetime_seconds, 1.0)
    if ratio < 0.0:
        ratio = 0.0
    elif ratio > 1.0:
        ratio = 1.0
    return float(1.0 - ratio)


# Register all available genes here.  The keys of this mapping become
# column names (prefixed with ``gene_``) and radar axis labels.

# ---------------------------------------------------------------------------
# Advanced behaviour genes (research-inspired)
# ---------------------------------------------------------------------------

def syscall_diversity_gene(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Approximate syscall diversity via activity-type diversity.

    We do not have raw syscalls, so we proxy using the presence of
    different activity classes: CPU use, file IO, networking and
    multithreading. The more classes a process actively uses, the
    higher its diversity score.
    """
    try:
        cpu_active = float(getattr(f, "cpu_percent", 0.0)) > 0.1
        has_files = int(getattr(f, "num_open_files", 0) or 0) > 0
        has_net = int(getattr(f, "num_connections", 0) or 0) > 0
        has_threads = int(getattr(f, "num_threads", 0) or 0) > 1
        active_types = sum([cpu_active, has_files, has_net, has_threads])
        return min(1.0, max(0.0, active_types / 4.0))
    except Exception:
        return 0.0


def burst_density_gene(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Heuristic burstiness based on instantaneous vs. lifetime CPU usage.

    With only snapshot metrics, we approximate bursts as the mismatch
    between current CPU utilisation and long‑term average
    (cpu_time_total / lifetime_seconds).
    """
    try:
        cpu_pct = float(getattr(f, "cpu_percent", 0.0))
        lifetime = float(getattr(f, "lifetime_seconds", 0.0))
        cpu_time = float(getattr(f, "cpu_time_total", 0.0))
        if lifetime <= 0.0 or cpu_time <= 0.0:
            return 0.0

        inst_norm = cpu_pct / max(1.0, float(getattr(stats, "max_cpu_percent", 100.0)))
        avg_cpu = cpu_time / lifetime
        # both in [0, 1] ish; burstiness is divergence between them
        score = abs(inst_norm - avg_cpu)
        return float(min(1.0, max(0.0, score)))
    except Exception:
        return 0.0


def thread_intensity_gene(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Ratio of active threads relative to population maximum."""
    try:
        threads = float(getattr(f, "num_threads", 0.0) or 0.0)
        max_threads = float(getattr(stats, "max_num_threads", threads) or threads or 1.0)
        return float(min(1.0, max(0.0, threads / max_threads)))
    except Exception:
        return 0.0


def registry_touch_gene(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Placeholder for future registry behaviour.

    Registry access is not currently instrumented, so this gene returns
    0.0 (no suspicious registry activity observed).
    """
    return 0.0


def io_intensity_gene(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Disk IO intensity proxied by number of open files."""
    try:
        open_files = float(getattr(f, "num_open_files", 0) or 0)
        max_open = float(getattr(stats, "max_num_open_files", open_files) or open_files or 1.0)
        return float(min(1.0, max(0.0, open_files / max_open)))
    except Exception:
        return 0.0


def network_activity_gene(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Network activity based on number of connections."""
    try:
        conns = float(getattr(f, "num_connections", 0) or 0)
        max_conns = float(getattr(stats, "max_num_connections", conns) or conns or 1.0)
        return float(min(1.0, max(0.0, conns / max_conns)))
    except Exception:
        return 0.0


def file_entropy_gene(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Shannon entropy of the executable image, scaled to [0, 1].

    Typical byte entropy lies in [0, 8] bits. We simply normalise by 8.
    """
    try:
        ent = float(getattr(f, "binary_entropy", 0.0) or 0.0)
        return float(min(1.0, max(0.0, ent / 8.0)))
    except Exception:
        return 0.0


def handle_abuse_gene(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Placeholder for handle abuse (LSASS / WINLOGON, etc.).

    Not currently instrumented; returns 0.0.
    """
    return 0.0


def injection_sus_gene(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Placeholder for injection techniques (WriteProcessMemory, etc.).

    Not currently instrumented; returns 0.0.
    """
    return 0.0


def dll_sideload_gene(f: BehaviourFeatures, stats: PopulationStats) -> float:
    """Placeholder for DLL sideloading behaviour.

    Not currently instrumented; returns 0.0.
    """
    return 0.0


# Mapping of gene names to their implementations. The key name becomes the
# column suffix (stored as ``gene_<name>``) and is also used on radar plots.
GENE_FUNCTIONS: Dict[str, callable] = {
    # Existing lightweight genes
    "resource_abuse": gene_resource_abuse,
    "entropy": gene_entropy,
    "impersonation": gene_impersonation,
    "exfiltration": gene_exfiltration,
    "tracking": gene_tracking,
    "persistence": gene_persistence,
    "mutation": gene_mutation,
    "stealth": gene_stealth,
    "latency": gene_latency,
    # Advanced / EDR‑style genes
    "syscall_diversity": syscall_diversity_gene,
    "burst_density": burst_density_gene,
    "thread_intensity": thread_intensity_gene,
    "registry_touch": registry_touch_gene,
    "io_intensity": io_intensity_gene,
    "network_activity": network_activity_gene,
    "file_entropy": file_entropy_gene,
    "handle_abuse": handle_abuse_gene,
    "injection_sus": injection_sus_gene,
    "dll_sideload": dll_sideload_gene,
}


def compute_genes(f: BehaviourFeatures, stats: PopulationStats) -> Dict[str, float]:
    """Return a mapping of gene name to score for a given process.

    Genes are computed using their registered functions and passed
    normalisation statistics to scale inputs to [0, 1]. All genes must
    be total: they should handle missing values and never raise.
    """
    return {name: func(f, stats) for name, func in GENE_FUNCTIONS.items()}
