from __future__ import annotations
import os
import yaml
from copy import deepcopy


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dict(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_config(cfg_path: str) -> dict:
    cfg = load_yaml(cfg_path)
    base_path = cfg.get("base", None)
    if base_path:
        base = load_yaml(base_path)
        cfg = merge_dict(base, cfg)
    return cfg


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
