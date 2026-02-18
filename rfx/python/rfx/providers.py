"""
Provider package loader for the Python SDK.

Allows users to stay on a single import surface (`import rfx`) while
optionally enabling extension packages.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Dict, Iterable

_PROVIDERS = {
    "sim": "rfx_sim",
    "go2": "rfx_go2",
    "lerobot": "rfx_lerobot",
}


def available_providers() -> Dict[str, bool]:
    return {
        name: importlib.util.find_spec(module) is not None for name, module in _PROVIDERS.items()
    }


def use(*providers: str, strict: bool = False) -> Dict[str, bool]:
    """
    Load extension provider packages from the main `rfx` namespace.

    Example:
        >>> import rfx
        >>> rfx.use("sim", "go2")
    """
    if not providers:
        providers = tuple(_PROVIDERS.keys())

    loaded: Dict[str, bool] = {}
    for name in providers:
        if name not in _PROVIDERS:
            raise ValueError(f"Unknown provider '{name}'. Valid providers: {sorted(_PROVIDERS)}")

        module = _PROVIDERS[name]
        try:
            importlib.import_module(module)
            loaded[name] = True
        except Exception:
            loaded[name] = False
            if strict:
                raise
    return loaded


def require(provider: str) -> None:
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Valid providers: {sorted(_PROVIDERS)}")

    module = _PROVIDERS[provider]
    if importlib.util.find_spec(module) is not None:
        return

    raise ImportError(
        f"Provider '{provider}' is not installed.\n"
        f"Install with: uv pip install {module.replace('_', '-')}"
    )


def enabled(provider: str) -> bool:
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Valid providers: {sorted(_PROVIDERS)}")
    return importlib.util.find_spec(_PROVIDERS[provider]) is not None
