#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from _common import ROOT, instance_files, instances_root, load_config


def load_l2b_generator(learn2branch_dir: Path):
    generator_path = learn2branch_dir / "01_generate_instances.py"
    if not generator_path.exists():
        raise FileNotFoundError(
            f"Missing {generator_path}. Clone Learn2Branch first: "
            "git clone https://github.com/ds4dm/learn2branch.git external/learn2branch"
        )
    sys.path.insert(0, str(generator_path.parent))
    spec = importlib.util.spec_from_file_location("learn2branch_instance_generator", generator_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def generate_setcover(module, out_dir: Path, count: int, seed: int, params: dict, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    for idx in range(count):
        path = out_dir / f"{prefix}_{idx:03d}.lp"
        if path.exists():
            continue
        module.generate_setcover(
            nrows=int(params["nrows"]),
            ncols=int(params["ncols"]),
            density=float(params["density"]),
            filename=str(path),
            rng=rng,
            max_coef=int(params.get("max_coef", 100)),
        )
        if (idx + 1) % 10 == 0 or idx + 1 == count:
            print(f"{out_dir.name}: generated {idx + 1}/{count}", flush=True)


def generate_cauctions(module, out_dir: Path, count: int, seed: int, params: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    for idx in range(count):
        path = out_dir / f"cauctions_{idx:03d}.lp"
        if path.exists():
            continue
        module.generate_cauctions(
            rng,
            str(path),
            n_items=int(params["n_items"]),
            n_bids=int(params["n_bids"]),
            add_item_prob=float(params.get("add_item_prob", 0.7)),
        )
        if (idx + 1) % 10 == 0 or idx + 1 == count:
            print(f"{out_dir.name}: generated {idx + 1}/{count}", flush=True)


def generate_facilities(module, out_dir: Path, count: int, seed: int, params: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    # The upstream function uses a global variable named rng internally.
    module.rng = rng
    for idx in range(count):
        path = out_dir / f"facilities_{idx:03d}.lp"
        if path.exists():
            continue
        module.generate_capacited_facility_location(
            rng,
            str(path),
            n_customers=int(params["n_customers"]),
            n_facilities=int(params["n_facilities"]),
            ratio=float(params["ratio"]),
        )
        if (idx + 1) % 10 == 0 or idx + 1 == count:
            print(f"{out_dir.name}: generated {idx + 1}/{count}", flush=True)


def generate_indset(module, out_dir: Path, count: int, seed: int, params: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    for idx in range(count):
        path = out_dir / f"indset_{idx:03d}.lp"
        if path.exists():
            continue
        graph = module.Graph.barabasi_albert(
            int(params["n_nodes"]),
            int(params["affinity"]),
            rng,
        )
        module.generate_indset(graph, str(path))
        if (idx + 1) % 10 == 0 or idx + 1 == count:
            print(f"{out_dir.name}: generated {idx + 1}/{count}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setcover_final_100.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    module = load_l2b_generator(ROOT / config["learn2branch_dir"])
    data_root = instances_root(config)
    data_root.mkdir(parents=True, exist_ok=True)
    n = int(config["n_per_class"])
    reference_size = int(config["reference_size"])
    seed = int(config.get("seed", 0))
    generation = config["generation"]
    folders = config["instances"]

    generate_setcover(module, data_root / folders["reference_setcover"], reference_size, seed, generation["setcover"], "setcover_reference")
    generate_setcover(module, data_root / folders["eval_classes"]["setcover_valid"], n, seed + 1, generation["setcover"], "setcover")
    generate_cauctions(module, data_root / folders["eval_classes"]["cauctions"], n, seed + 2, generation["cauctions"])
    generate_facilities(module, data_root / folders["eval_classes"]["facilities"], n, seed + 3, generation["facilities"])
    generate_indset(module, data_root / folders["eval_classes"]["indset"], n, seed + 4, generation["indset"])

    for name, folder in folders["eval_classes"].items():
        print(f"{name}: {len(instance_files(data_root / folder))} instances", flush=True)
    print(f"reference_setcover: {len(instance_files(data_root / folders['reference_setcover']))} instances", flush=True)


if __name__ == "__main__":
    main()
