from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def instance_files(path: Path) -> list[Path]:
    files = list(path.glob("*.lp")) + list(path.glob("*.mps"))
    return sorted((p for p in files if " " not in p.stem), key=natural_key)


def ensure_results_dir(config: dict) -> Path:
    results_dir = ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def instances_root(config: dict) -> Path:
    return ROOT / config["instances"]["root"]
