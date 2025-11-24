"""
PhenoVector genome module.

This module defines the core logic for analyzing running processes and producing
per‑process genome vectors along with a simple risk score and risk level.

Functions:
    analyze_system(limit: int = 200) -> list[ProcessGenome]
        Inspect up to ``limit`` processes on the host and return their genome
        representations.

Classes:
    ProcessGenome: Holds all derived information for a single process.

The implementation here intentionally keeps the model simple. Risk levels are
determined by a linear combination of gene scores. If you wish to use more
advanced anomaly detection or supervised classification, build on top of the
returned genome vectors in your own code or via the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil

from features import BehaviourFeatures, collect_behaviour_features
from genes import compute_genes, compute_population_stats, PopulationStats


@dataclass
class ProcessGenome:
    """Container for the derived behaviour genome of a process."""

    pid: int
    name: str
    exe: Optional[str]
    features: BehaviourFeatures
    genes: Dict[str, float]
    risk_score: float
    risk_level: str  # one of: 'benign', 'suspicious', 'high'


def _compute_risk(genes: Dict[str, float]) -> float:
    """Combine gene scores into a single risk score.

    The weights here were chosen heuristically. Feel free to tune them for your
    environment or replace this with a learned model. The score is clamped to
    the range [0, 1].
    """
    weights = {
        "resource_abuse": 0.20,
        "exfiltration": 0.15,
        "tracking": 0.10,
        "mutation": 0.10,
        "impersonation": 0.10,
        "stealth": 0.10,
        "latency": 0.05,

        "syscall_diversity": 0.05,
        "burst_density": 0.05,
        "thread_intensity": 0.03,
        "io_intensity": 0.03,
        "network_activity": 0.03,
        "file_entropy": 0.03,
    }

    score = 0.0
    for key, weight in weights.items():
        score += genes.get(key, 0.0) * weight
    # clamp
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return float(score)


def _risk_label(score: float) -> str:
    """Map a risk score into a discrete risk level."""
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "suspicious"
    return "benign"


def analyze_system(limit: int = 200) -> List[ProcessGenome]:
    """Gather genome data for up to ``limit`` running processes.

    This walks the list of visible processes and collects features for each one,
    handling common exceptions such as access denial. The behaviour features
    collected here feed into gene computation and risk scoring.

    Parameters
    ----------
    limit : int, optional
        Maximum number of processes to inspect. Processes beyond this limit
        are silently ignored. Defaults to 200.

    Returns
    -------
    List[ProcessGenome]
        One genome object per visible process (up to the limit).
    """
    processes: List[psutil.Process] = []
    for proc in psutil.process_iter():
        try:
            processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if len(processes) >= limit:
            break

    # Collect behaviour snapshots for each process
    features: List[BehaviourFeatures] = []
    for p in processes:
        try:
            features.append(collect_behaviour_features(p))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Compute population statistics used by the gene functions
    stats: PopulationStats = compute_population_stats(features)

    genomes: List[ProcessGenome] = []
    for f in features:
        genes = compute_genes(f, stats)
        risk = _compute_risk(genes)
        label = _risk_label(risk)
        genomes.append(
            ProcessGenome(
                pid=f.pid,
                name=f.name,
                exe=f.exe,
                features=f,
                genes=genes,
                risk_score=risk,
                risk_level=label,
            )
        )
    return genomes