#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from _common import ROOT, ensure_results_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setcover_final_100.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    results_dir = ensure_results_dir(config)
    summary_csv = results_dir / "summary_by_class.csv"
    corr_csv = results_dir / "correlation_summary.csv"
    if not summary_csv.exists() or not corr_csv.exists():
        raise FileNotFoundError("Run scripts/make_plots.py first.")

    summary = pd.read_csv(summary_csv)
    correlations = pd.read_csv(corr_csv)
    print("Class summary")
    print(summary.to_string(index=False))
    print("\nCorrelations")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
