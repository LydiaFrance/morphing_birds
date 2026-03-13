"""Config loading utilities for builtin skeleton definitions."""

import os
import yaml


_CONFIG_DIR = os.path.dirname(__file__)

_BUILTIN_CONFIGS = {
    "hawk": "hawk.yaml",
    "pigeon": "pigeon.yaml",
    "kestrel": "kestrel.yaml",
    "spider": "spider.yaml",
}


def load_config(name: str) -> dict:
    """Load a builtin config by name (e.g. 'hawk', 'pigeon').

    Parameters
    ----------
    name : str
        One of the builtin config names.

    Returns
    -------
    dict
        Parsed YAML config dictionary.

    Raises
    ------
    ValueError
        If the name is not a recognised builtin config.
    """
    if name not in _BUILTIN_CONFIGS:
        raise ValueError(
            f"Unknown builtin config '{name}'. "
            f"Available: {list(_BUILTIN_CONFIGS.keys())}"
        )
    path = os.path.join(_CONFIG_DIR, _BUILTIN_CONFIGS[name])
    with open(path, "r") as f:
        return yaml.safe_load(f)


def list_configs() -> list[str]:
    """Return a list of available builtin config names."""
    return list(_BUILTIN_CONFIGS.keys())
