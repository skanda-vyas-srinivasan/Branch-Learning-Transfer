from __future__ import annotations

from pathlib import Path

import torch

from .learn2branch_gnn import GNNPolicy


def load_policy(weights_path: str | Path, device: str | None = None) -> tuple[GNNPolicy, str]:
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = GNNPolicy().to(resolved_device)
    state = torch.load(Path(weights_path), map_location=resolved_device)
    policy.load_state_dict(state)
    policy.eval()
    return policy, resolved_device
