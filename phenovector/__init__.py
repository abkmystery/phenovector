"""
PhenoVector: Behavioural genome library for running processes.

This package provides functions to collect behavioural features from
running processes and compute a set of heuristic "gene" scores that
capture resource abuse, entropy, impersonation, exfiltration, tracking,
persistence, mutation, stealth and latency characteristics. These gene
vectors can be used to cluster processes, perform anomaly detection
and generate system behaviour visualisations.

Top‑level usage:

    from phenovector.genome import analyze_system

    genomes = analyze_system(limit=100)
    for g in genomes:
        print(g.pid, g.name, g.risk_score, g.genes)

See the Streamlit UI in ``app.py`` for an interactive console that
scans the local system, visualises the results and supports PID
whitelisting.
"""

from .genome import analyze_system, ProcessGenome  # noqa: F401

__all__ = ["analyze_system", "ProcessGenome"]