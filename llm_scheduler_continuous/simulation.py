"""Simulation helpers for the LLM scheduler continuous-time experiments."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

try:
    from llm_scheduler_timeseries import simulation as _timeseries_sim
except ModuleNotFoundError:  # pragma: no cover - runtime safeguard
    import sys

    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from llm_scheduler_timeseries import simulation as _timeseries_sim

try:
    from .config import DEFAULT_CONFIG
except ImportError:  # pragma: no cover - runtime safeguard
    from llm_scheduler_continuous.config import DEFAULT_CONFIG


def _build_experiment_override(
    config_override: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Merge the package defaults with any user supplied overrides."""

    merged = deepcopy(DEFAULT_CONFIG)
    if config_override:
        merged.update(config_override)
    return merged


def run_simulation(*args, **kwargs):
    """Delegate to the timeseries simulation runner for compatibility."""

    return _timeseries_sim.run_simulation(*args, **kwargs)


def run_experiment(
    config_override: Optional[Dict[str, object]] = None,
    epochs: int = 1,
    save_results: bool = True,
):
    """Execute the continuous-time experiment with the requested overrides."""

    override = _build_experiment_override(config_override)
    return _timeseries_sim.run_experiment(
        experiment_config_override=override,
        epochs=epochs,
        save_results=save_results,
    )


__all__ = ["run_simulation", "run_experiment"]



if __name__ == "__main__":
    run_experiment(epochs=10, save_results=True)