from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict) -> dict:
    """Later dicts override earlier ones."""
    result = {}
    for cfg in configs:
        _deep_update(result, cfg)
    return result


def _deep_update(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
