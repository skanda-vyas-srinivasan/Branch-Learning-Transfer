#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from _common import ROOT, ensure_results_dir, load_config


def safe_corr(x, y, method: str):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return float("nan"), float("nan")
    if method == "pearson":
        return pearsonr(x[mask], y[mask])
    return spearmanr(x[mask], y[mask])


def binary_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    total = 0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        total += len(neg)
    return float(wins / total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setcover_final_100.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    results_dir = ensure_results_dir(config)
    raw_csv = results_dir / "raw_results.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"missing {raw_csv}; run scripts/run_experiment.py first")

    df = pd.read_csv(raw_csv)
    if "relative_node_count" not in df.columns and "degradation" in df.columns:
        df["relative_node_count"] = df["degradation"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["distance", "relative_node_count"])
    df["log10_rnc"] = np.log10(df["relative_node_count"].clip(lower=1e-12))
    df["ml_worse"] = df["relative_node_count"] > 1.0

    summary = (
        df.groupby("class")
        .agg(
            n=("relative_node_count", "size"),
            mean_distance=("distance", "mean"),
            median_distance=("distance", "median"),
            median_scip_nodes=("scip_nodes", "median"),
            median_ml_nodes=("ml_nodes", "median"),
            mean_rnc=("relative_node_count", "mean"),
            q1_rnc=("relative_node_count", lambda s: s.quantile(0.25)),
            median_rnc=("relative_node_count", "median"),
            q3_rnc=("relative_node_count", lambda s: s.quantile(0.75)),
            pct_ml_worse=("ml_worse", "mean"),
        )
        .reset_index()
        .sort_values("median_distance")
    )
    summary.to_csv(results_dir / "summary_by_class.csv", index=False)

    r_raw, p_raw = safe_corr(df["distance"].to_numpy(float), df["relative_node_count"].to_numpy(float), "pearson")
    r_log, p_log = safe_corr(df["distance"].to_numpy(float), df["log10_rnc"].to_numpy(float), "pearson")
    r_spear, p_spear = safe_corr(df["distance"].to_numpy(float), df["relative_node_count"].to_numpy(float), "spearman")
    auc = binary_auc(df["distance"].to_numpy(float), df["ml_worse"].to_numpy(bool))
    pd.DataFrame(
        [
            {"metric": "pearson_distance_rnc", "r": r_raw, "p_value": p_raw},
            {"metric": "pearson_distance_log10_rnc", "r": r_log, "p_value": p_log},
            {"metric": "spearman_distance_rnc", "r": r_spear, "p_value": p_spear},
            {"metric": "auroc_predict_ml_worse", "r": auc, "p_value": np.nan},
        ]
    ).to_csv(results_dir / "correlation_summary.csv", index=False)

    order = summary["class"].tolist()

    plt.figure(figsize=(10, 7))
    for class_name, subset in df.groupby("class"):
        plt.scatter(subset["distance"], subset["relative_node_count"], label=class_name, alpha=0.75)
    if df["distance"].nunique() > 1:
        x = df["distance"].to_numpy(float)
        y = df["log10_rnc"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        plt.plot(xs, 10 ** (slope * xs + intercept), color="black", linewidth=2, label="linear fit on log10 ratio")
    plt.axhline(1, color="red", linestyle="--", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Mean Maudet distance to set-cover reference")
    plt.ylabel("Relative node count: ML nodes / vanilla SCIP nodes")
    plt.title("MILP Distance vs ML Branching Transfer")
    plt.text(0.02, 0.98, f"Pearson log r={r_log:.3f}\nSpearman rho={r_spear:.3f}\nAUROC={auc:.3f}", transform=plt.gca().transAxes, va="top")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "scatter_distance_rnc_log.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.boxplot([df.loc[df["class"] == cls, "relative_node_count"] for cls in order], labels=order, showfliers=True)
    plt.axhline(1, color="red", linestyle="--", linewidth=1)
    plt.yscale("log")
    plt.ylabel("Relative node count")
    plt.title("Relative Node Count by Class")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "box_rnc_by_class.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(summary["median_distance"], summary["median_rnc"], s=90)
    for _, row in summary.iterrows():
        plt.annotate(row["class"], (row["median_distance"], row["median_rnc"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    plt.axhline(1, color="red", linestyle="--", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Class median Maudet distance")
    plt.ylabel("Class median relative node count")
    plt.title("Class-Level Transfer Summary")
    plt.tight_layout()
    plt.savefig(results_dir / "class_median_distance_vs_rnc.png", dpi=200)
    plt.close()

    print(summary.to_string(index=False), flush=True)
    print(f"saved plots and summaries to {results_dir}", flush=True)


if __name__ == "__main__":
    main()
