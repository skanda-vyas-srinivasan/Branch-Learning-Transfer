from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def default_scip_params(time_limit: float, node_limit: int) -> dict[str, Any]:
    return {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": float(time_limit),
        "limits/nodes": int(node_limit),
        "timing/clocktype": 1,
    }


def tensorize_node_observation(obs, device: str):
    return (
        torch.from_numpy(np.nan_to_num(obs.row_features).astype(np.float32)).to(device),
        torch.from_numpy(obs.edge_features.indices.astype(np.int64)).to(device),
        torch.from_numpy(np.nan_to_num(obs.edge_features.values).astype(np.float32)).view(-1, 1).to(device),
        torch.from_numpy(np.nan_to_num(obs.variable_features).astype(np.float32)).to(device),
    )


def solve_vanilla_scip(instance_path: str | Path, time_limit: float, node_limit: int, seed: int = 0) -> dict[str, Any]:
    import ecole

    env = ecole.environment.Configuring(scip_params=default_scip_params(time_limit, node_limit))
    env.seed(seed)
    wall_start = time.perf_counter()
    env.reset(str(instance_path))
    env.step({})
    walltime = time.perf_counter() - wall_start
    model = env.model.as_pyscipopt()
    return {
        "scip_nodes": int(model.getNNodes()),
        "scip_lps": int(model.getNLPs()),
        "scip_time": float(model.getSolvingTime()),
        "scip_walltime": float(walltime),
        "scip_status": str(model.getStatus()),
        "scip_gap": float(model.getGap()),
    }


def solve_ml_branching(instance_path: str | Path, policy, device: str, time_limit: float, node_limit: int, seed: int = 0) -> dict[str, Any]:
    import ecole

    policy.eval()
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params=default_scip_params(time_limit, node_limit),
        pseudo_candidates=False,
    )
    env.seed(seed)
    torch.manual_seed(seed)

    wall_start = time.perf_counter()
    obs, action_set, _, done, _ = env.reset(str(instance_path))
    while not done:
        if action_set is None or len(action_set) == 0:
            break
        with torch.no_grad():
            tensors = tensorize_node_observation(obs, device)
            logits = policy(*tensors)
            action_indices = torch.as_tensor(action_set.astype(np.int64), device=device)
            selected = int(torch.argmax(logits[action_indices]).item())
            action = int(action_set[selected])
        obs, action_set, _, done, _ = env.step(action)

    if not done:
        raise RuntimeError("ML branching exited before solve completion")

    walltime = time.perf_counter() - wall_start
    model = env.model.as_pyscipopt()
    return {
        "ml_nodes": int(model.getNNodes()),
        "ml_lps": int(model.getNLPs()),
        "ml_time": float(model.getSolvingTime()),
        "ml_walltime": float(walltime),
        "ml_status": str(model.getStatus()),
        "ml_gap": float(model.getGap()),
    }


def evaluate_instance(instance_path: str | Path, policy, device: str, time_limit: float, node_limit: int, seed: int = 0) -> dict[str, Any] | None:
    try:
        scip = solve_vanilla_scip(instance_path, time_limit=time_limit, node_limit=node_limit, seed=seed)
        ml = solve_ml_branching(instance_path, policy, device=device, time_limit=time_limit, node_limit=node_limit, seed=seed)
        scip_nodes = max(1, int(scip["scip_nodes"]))
        ml_nodes = int(ml["ml_nodes"])
        return {
            **scip,
            **ml,
            "relative_node_count": float(ml_nodes / scip_nodes),
        }
    except Exception as exc:
        print(f"FAILED {instance_path}: {type(exc).__name__}: {exc}", flush=True)
        return None
