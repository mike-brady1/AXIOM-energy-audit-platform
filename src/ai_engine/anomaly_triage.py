"""
AXIOM — K-Means Anomaly Triage
================================
Clusters buildings by residual (actual − predicted EUI) into:
  Normal / Elevated / Critical
Fed by residuals from baseline_model.py LinearRegression.

Usage:
    from src.ai_engine.anomaly_triage import train_kmeans, triage_building
    triage = train_kmeans(residuals)
    label  = triage_building(residual_value, triage)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CLUSTER_LABELS = ["Normal", "Elevated", "Critical"]
N_CLUSTERS = 3


@dataclass
class TriageResult:
    model:          KMeans
    scaler:         StandardScaler
    cluster_stats:  pd.DataFrame   # mean/std residual per cluster
    label_map:      dict           # cluster_id → label
    thresholds:     dict           # Normal/Elevated/Critical boundaries


def train_kmeans(
    residuals: np.ndarray,
    *,
    n_clusters: int = N_CLUSTERS,
    random_state: int = 42,
) -> TriageResult:
    """
    Fit K-Means on 1-D residuals (actual − predicted EUI).
    Clusters are auto-labelled by ascending mean residual:
      lowest mean  → Normal
      middle mean  → Elevated
      highest mean → Critical
    """
    X = residuals.reshape(-1, 1)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(X_s)
    labels = km.labels_

    # Build stats per cluster — sort by mean residual ascending
    stats = []
    for c in range(n_clusters):
        mask = labels == c
        stats.append({
            "cluster_id":   c,
            "count":        int(mask.sum()),
            "mean_residual": float(residuals[mask].mean()),
            "std_residual":  float(residuals[mask].std()),
            "min_residual":  float(residuals[mask].min()),
            "max_residual":  float(residuals[mask].max()),
        })
    stats_df = pd.DataFrame(stats).sort_values("mean_residual").reset_index(drop=True)

    # Map cluster_id → triage label
    label_map = {}
    for i, row in stats_df.iterrows():
        label_map[int(row["cluster_id"])] = CLUSTER_LABELS[i]
    stats_df["triage_label"] = stats_df["cluster_id"].map(label_map)

    # Simple numeric thresholds for single-building inference
    normal_max   = stats_df[stats_df["triage_label"] == "Normal"]["max_residual"].values[0]
    elevated_max = stats_df[stats_df["triage_label"] == "Elevated"]["max_residual"].values[0]
    thresholds = {
        "Normal":   (-np.inf, normal_max),
        "Elevated": (normal_max, elevated_max),
        "Critical": (elevated_max, np.inf),
    }

    logger.info("K-Means triage fitted. Cluster distribution:")
    for _, row in stats_df.iterrows():
        logger.info(
            "  %-8s  n=%d  mean_res=%.1f  range=[%.1f, %.1f]",
            row["triage_label"], row["count"],
            row["mean_residual"], row["min_residual"], row["max_residual"]
        )

    return TriageResult(
        model=km,
        scaler=scaler,
        cluster_stats=stats_df,
        label_map=label_map,
        thresholds=thresholds,
    )


def triage_building(
    actual_eui: float,
    predicted_eui: float,
    triage: TriageResult,
) -> dict:
    """
    Classify a single building's anomaly level.
    Returns label (Normal/Elevated/Critical) + residual + recommended action.
    """
    residual = actual_eui - predicted_eui

    label = "Critical"  # default
    for lbl, (lo, hi) in triage.thresholds.items():
        if lo <= residual <= hi:
            label = lbl
            break

    actions = {
        "Normal":   "No immediate action required. Schedule next audit per ISO 50002 cycle.",
        "Elevated": "Investigate HVAC scheduling and BMS setpoints. ASHRAE Level II audit recommended.",
        "Critical": "Urgent: anomalous consumption detected. Immediate ASHRAE Level III audit + sub-metering required.",
    }

    return {
        "actual_eui":    round(actual_eui, 1),
        "predicted_eui": round(predicted_eui, 1),
        "residual":      round(residual, 1),
        "triage_label":  label,
        "action":        actions[label],
        "severity_pct":  round(residual / max(predicted_eui, 1) * 100, 1),
    }


def print_summary(triage: TriageResult) -> None:
    print("\n" + "="*55)
    print("AXIOM K-MEANS ANOMALY TRIAGE")
    print("="*55)
    print(triage.cluster_stats[["triage_label","count","mean_residual","std_residual"]].to_string(index=False))
    print("\nThresholds:")
    for lbl, (lo, hi) in triage.thresholds.items():
        lo_s = f"{lo:.1f}" if lo != -np.inf else "-∞"
        hi_s = f"{hi:.1f}" if hi != np.inf else "+∞"
        print(f"  {lbl:10s}: residual in [{lo_s}, {hi_s}]")
