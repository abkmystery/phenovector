"""
PhenoVector Streamlit UI
========================

Streamlit-based dashboard for inspecting the behavioural genomes of
running processes on the host machine.

All analysis happens locally – process data is never sent off-box.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from genome import analyze_system, ProcessGenome

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stable ordering for radar genes so polygon layout is consistent
GENE_ORDER = [
    "resource_abuse",
    "entropy",
    "impersonation",
    "exfiltration",
    "tracking",
    "persistence",
    "mutation",
    "stealth",
    "latency",
]


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PhenoVector – System Behaviour Genome",
    layout="wide",
    initial_sidebar_state="expanded",
)


def genomes_to_dataframe(genomes: List[ProcessGenome]) -> pd.DataFrame:
    """Convert a list of ProcessGenome objects into a pandas DataFrame."""
    rows: List[Dict[str, float | int | str]] = []
    for g in genomes:
        row: Dict[str, float | int | str] = {
            "pid": int(g.pid),
            "name": g.name,
            "exe": getattr(g, "exe", "") or "",
            "risk_score": float(getattr(g, "risk_score", 0.0)),
            "risk_level": getattr(g, "risk_level", "benign"),
            "is_system_process": bool(
                getattr(getattr(g, "features", None), "is_system_process", False)
            ),
        }
        # flatten gene dictionary with a prefix
        for gene_name, gene_val in getattr(g, "genes", {}).items():
            row[f"gene_{gene_name}"] = float(gene_val)
        rows.append(row)
    return pd.DataFrame(rows)


def build_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 2-D projection and anomaly scores for the DataFrame.

    Steps:
    * Min-max normalise all gene columns across the current population.
    * Use PCA for the main 2-D projection.
    * If PCA collapses (second component ~0 variance), fall back to t-SNE.
    * Score anomalies with IsolationForest on the normalised gene vectors.
    """
    gene_cols = [c for c in df.columns if c.startswith("gene_")]
    if not gene_cols or df.empty:
        df["pc1"] = 0.0
        df["pc2"] = 0.0
        df["anomaly_score"] = 0.0
        df["is_anomaly"] = False
        return df

    X = df[gene_cols].values.astype(float)

    # handle degenerate case: very small population
    if X.shape[0] < 3:
        df["pc1"] = 0.0
        df["pc2"] = 0.0
        df["anomaly_score"] = 0.0
        df["is_anomaly"] = False
        return df

    # 1) Normalise all gene columns across this scan
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    df[gene_cols] = X_scaled

    # 2) PCA for 2-D scatter
    use_tsne = False
    pcs = None
    try:
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(X_scaled)
        if getattr(pca, "explained_variance_ratio_", None) is not None:
            # if second component carries almost no variance, use t-SNE
            if pca.explained_variance_ratio_[1] < 0.02:
                use_tsne = True
    except Exception:
        use_tsne = True

    if use_tsne:
        try:
            perplexity = max(2, min(30, X_scaled.shape[0] - 1))
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                init="random",
                learning_rate="auto",
            )
            pcs = tsne.fit_transform(X_scaled)
        except Exception:
            pcs = np.zeros((X_scaled.shape[0], 2), dtype=float)

    if pcs is None:
        pcs = np.zeros((X_scaled.shape[0], 2), dtype=float)

    df["pc1"] = pcs[:, 0]
    df["pc2"] = pcs[:, 1]

    # 3) Isolation Forest for anomaly scoring (on normalised genes)
    iso = IsolationForest(contamination=0.15, random_state=42)
    iso.fit(X_scaled)
    scores = -iso.score_samples(X_scaled)
    df["anomaly_score"] = scores
    threshold = float(np.percentile(scores, 85))
    df["is_anomaly"] = df["anomaly_score"] >= threshold
    return df


def radar_chart(row: pd.Series) -> go.Figure:
    """Create a radar chart for a given row of gene values."""
    # Build ordered list of gene columns using the stable GENE_ORDER
    available_genes = [c.replace("gene_", "") for c in row.index if c.startswith("gene_")]
    ordered_genes = [g for g in GENE_ORDER if g in available_genes]
    # Include any additional genes at the end, sorted for stability
    extra_genes = sorted(set(available_genes) - set(ordered_genes))
    ordered_genes.extend(extra_genes)

    labels = ordered_genes
    values = [float(row[f"gene_{g}"]) for g in ordered_genes]

    # close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    # Outer glow
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            line=dict(color="rgba(0, 230, 118, 0.6)", width=6),
            opacity=0.35,
            hoverinfo="skip",
        )
    )

    # Main filled polygon
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(0, 230, 118, 0.25)",
            line=dict(color="rgba(0, 230, 118, 1.0)", width=2),
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def main() -> None:
    """Entry point for the Streamlit app."""
    # Session state for caching scans and whitelists
    if "scan_counter" not in st.session_state:
        st.session_state["scan_counter"] = 0
    if "whitelist_pids" not in st.session_state:
        st.session_state["whitelist_pids"] = []

    # Sidebar controls
    with st.sidebar:
        st.markdown("## 🧬 PhenoVector")
        if st.button("🔬 Scan system processes"):
            st.session_state["scan_counter"] += 1
        max_processes = st.slider("Max processes to scan", 10, 1000, 100, step=50)
        st.info(
            "Runs locally using psutil & scikit-learn. No process data ever leaves this machine.",
            icon="ℹ️",
        )

    # Perform scan (cached)
    @st.cache_data(show_spinner=True)
    def _scan(counter: int, limit: int) -> List[ProcessGenome]:
        return analyze_system(limit=limit)

    genomes = _scan(st.session_state["scan_counter"], max_processes)

    if not genomes:
        st.warning("No processes found to analyse.")
        return

    # Convert to DataFrame and cluster
    df = genomes_to_dataframe(genomes)
    df = build_cluster(df)

    # Whitelist selection: after scanning, we list options from df
    with st.sidebar:
        st.markdown("### PID whitelist")
        options = []
        label_to_pid: Dict[str, int] = {}
        for _, row in df.iterrows():
            label = f"{int(row['pid'])} – {row['name']}"
            options.append(label)
            label_to_pid[label] = int(row["pid"])
        # Preselect previously stored values
        default_labels = [
            label
            for label, pid in label_to_pid.items()
            if pid in st.session_state["whitelist_pids"]
        ]
        selected_labels = st.multiselect(
            "Never mark these PIDs as anomalies",
            options=options,
            default=default_labels,
        )
        # st.session_state["whitelist_pids"] = [label_to_pid[l] for l in selected_labels]
        st.session_state["whitelist_pids"] = list({label_to_pid[l] for l in selected_labels})

    # Apply whitelist at the very end of the pipeline
    df["is_whitelisted"] = df["pid"].astype(int).isin(st.session_state["whitelist_pids"])
    whitelist_mask = df["is_whitelisted"]

    # Enforce whitelist: never flag as anomaly, force benign label and zero risk
    df.loc[whitelist_mask, "is_anomaly"] = False
    df.loc[whitelist_mask, "anomaly_score"] = 0.0
    df.loc[whitelist_mask, "risk_level"] = "benign"
    df.loc[whitelist_mask, "risk_score"] = 0.0

    # Ensure a clean, contiguous index for all downstream UI interactions
    df = df.reset_index(drop=True)

    # Page header and metrics
    st.title("System Behaviour Genome")
    st.caption("Runtime behaviour vectors for process anomaly detection.")

    st.markdown("### Analysis status")
    st.success("Analysis complete")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total scanned", len(df))
    c2.metric("Anomalies", int(df["is_anomaly"].sum()))
    avg_imp = float(df.get("gene_impersonation", pd.Series([0.0])).mean())
    c3.metric("Avg impersonation", f"{avg_imp:.2f}")

    # Scatter and radar charts
    st.markdown("---")
    st.subheader("Genome cluster map (PCA / t-SNE)")
    scatter_col, radar_col = st.columns([2, 1])

    with scatter_col:
        fig_scatter = px.scatter(
            df,
            x="pc1",
            y="pc2",
            color="risk_level",
            size="risk_score",
            hover_data=["pid", "name", "exe", "risk_score", "anomaly_score"],
            color_discrete_map={
                "benign": "#00c896",
                "suspicious": "#ffb020",
                "high": "#ff4d4f",
            },
            title="Behavioural clustering",
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_scatter, theme="streamlit")

    with radar_col:
        st.markdown("### Genome radar")
        # Create selection list of pid–name
        pid_options = df["pid"].astype(int).tolist()

        def format_pid(pid_: int) -> str:
            name = df.loc[df["pid"] == pid_, "name"].iloc[0]
            return f"{pid_} – {name}"

        selected_pid = st.selectbox(
            "Select process",
            options=pid_options,
            format_func=format_pid,
        )
        active_row = df[df["pid"] == selected_pid].iloc[0]
        st.markdown(f"**PID {int(active_row['pid'])} – {active_row['name']}**")
        st.plotly_chart(radar_chart(active_row), theme="streamlit")

    # Raw behaviour table
    st.markdown("---")
    st.subheader("Raw behaviour data")
    if bool(df.loc[df["pid"] == selected_pid, "is_anomaly"].iloc[0]):
        st.error(f"⚠ Anomaly detected (PID: {int(selected_pid)})")
    else:
        st.success(f"✓ No anomaly detected (PID: {int(selected_pid)})")

    st.dataframe(df, height=400)

    # Export data
    st.markdown("---")
    with st.expander("Export genome data"):
        data_json = json.dumps(df.to_dict(orient="records"), indent=2)
        st.download_button(
            "Download JSON",
            data=data_json,
            file_name="phenovector_genomes.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
