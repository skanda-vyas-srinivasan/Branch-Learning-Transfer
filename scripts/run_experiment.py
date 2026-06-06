#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, ensure_results_dir, instance_files, instances_root, load_config


def load_reference_reps(reference_paths: list[Path], cache_path: Path):
    from milp_distance.distance import extract_normalized_representation

    expected = [str(path.resolve()) for path in reference_paths]
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("paths") == expected:
            print(f"loaded cached reference reps from {cache_path}", flush=True)
            return payload["reps"]

    reps = []
    for idx, path in enumerate(reference_paths, start=1):
        print(f"reference {idx}/{len(reference_paths)}: {path.name}", flush=True)
        reps.append(extract_normalized_representation(path))
    with cache_path.open("wb") as handle:
        pickle.dump({"paths": expected, "reps": reps}, handle)
    return reps


def load_distance_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["path"], df["distance"]))


def save_distance_cache(path: Path, cache: dict[str, float]) -> None:
    rows = [{"path": key, "distance": value} for key, value in sorted(cache.items())]
    pd.DataFrame(rows).to_csv(path, index=False)


def distribution_distance(instance_path: Path, reference_reps, distance_cache: dict[str, float], cache_path: Path) -> float:
    from milp_distance.distance import extract_normalized_representation, greedy_instance_distance

    resolved = str(instance_path.resolve())
    if resolved in distance_cache:
        return float(distance_cache[resolved])
    rep = extract_normalized_representation(instance_path)
    distances = [greedy_instance_distance(rep, ref) for ref in reference_reps if ref is not None]
    finite = [distance for distance in distances if np.isfinite(distance)]
    distance = float(np.mean(finite)) if finite else float("inf")
    distance_cache[resolved] = distance
    if len(distance_cache) % 10 == 0:
        save_distance_cache(cache_path, distance_cache)
    return distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setcover_final_100.yaml")
    args = parser.parse_args()

    from branching_eval.evaluate import evaluate_instance
    from branching_eval.model_loader import load_policy

    config = load_config(ROOT / args.config)
    results_dir = ensure_results_dir(config)
    data_root = instances_root(config)
    n = int(config["n_per_class"])
    reference_size = int(config["reference_size"])
    time_limit = float(config["time_limit"])
    node_limit = int(config["node_limit"])
    seed = int(config.get("seed", 0))

    policy, device = load_policy(ROOT / config["model_path"])
    print(f"loaded policy on {device}", flush=True)

    folders = config["instances"]
    reference_paths = instance_files(data_root / folders["reference_setcover"])[:reference_size]
    if len(reference_paths) < reference_size:
        raise RuntimeError(f"expected {reference_size} reference instances; found {len(reference_paths)}")
    reference_reps = load_reference_reps(reference_paths, results_dir / "reference_reps.pkl")

    test_sets = {
        class_name: instance_files(data_root / folder)[:n]
        for class_name, folder in folders["eval_classes"].items()
    }
    for class_name, paths in test_sets.items():
        if len(paths) < n:
            raise RuntimeError(f"expected {n} instances for {class_name}; found {len(paths)}")
        print(f"{class_name}: {len(paths)} instances", flush=True)

    raw_csv = results_dir / "raw_results.csv"
    failures_csv = results_dir / "failures.csv"
    distance_cache_csv = results_dir / "distance_cache.csv"
    distance_cache = load_distance_cache(distance_cache_csv)

    if raw_csv.exists():
        results_df = pd.read_csv(raw_csv)
        rows = results_df.to_dict("records")
        completed = set(results_df["path"].astype(str))
        print(f"loaded {len(rows)} existing results", flush=True)
    else:
        rows = []
        completed = set()

    if failures_csv.exists():
        failures_df = pd.read_csv(failures_csv)
        failure_rows = failures_df.to_dict("records")
        failed = set(failures_df["path"].astype(str))
    else:
        failure_rows = []
        failed = set()

    for class_name, paths in test_sets.items():
        for idx, path in enumerate(paths, start=1):
            resolved = str(path.resolve())
            if resolved in completed or resolved in failed:
                continue
            try:
                distance = distribution_distance(path, reference_reps, distance_cache, distance_cache_csv)
                metrics = evaluate_instance(path, policy, device, time_limit=time_limit, node_limit=node_limit, seed=seed)
                if metrics is None:
                    failure_rows.append({"class": class_name, "path": resolved, "reason": "evaluate_instance returned None"})
                    pd.DataFrame(failure_rows).to_csv(failures_csv, index=False)
                    failed.add(resolved)
                    continue
                row = {
                    "class": class_name,
                    "path": resolved,
                    "distance": distance,
                    **metrics,
                }
                rows.append(row)
                completed.add(resolved)
                pd.DataFrame(rows).to_csv(raw_csv, index=False)
                save_distance_cache(distance_cache_csv, distance_cache)
                print(
                    f"{class_name} {idx}/{len(paths)} distance={distance:.4f} rnc={metrics['relative_node_count']:.3f} "
                    f"scip={metrics['scip_nodes']} ml={metrics['ml_nodes']}",
                    flush=True,
                )
            except Exception as exc:
                print(f"FAILED {class_name} {path.name}: {type(exc).__name__}: {exc}", flush=True)
                failure_rows.append({"class": class_name, "path": resolved, "reason": f"{type(exc).__name__}: {exc}"})
                pd.DataFrame(failure_rows).to_csv(failures_csv, index=False)
                failed.add(resolved)

    pd.DataFrame(rows).to_csv(raw_csv, index=False)
    save_distance_cache(distance_cache_csv, distance_cache)
    print(f"saved {len(rows)} rows to {raw_csv}", flush=True)


if __name__ == "__main__":
    main()
