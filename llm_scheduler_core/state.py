"""Simulation state tracking for the LLM scheduler core experiment suite."""
from __future__ import annotations

import heapq
import sys
from typing import Any, List, Tuple

# --- Simulation Event Queue and Time (managed by the main simulation script) ---
event_queue: List[Tuple[float, str, Tuple[Any, ...]]] = []
current_time: float = 0.0
requests_generated: int = 0
max_requests_to_generate: int = 0
request_lookup: dict[int, Any] = {}


def add_event(time: float, event_type: str, *details: Any) -> None:
    """Adds an event to the global event queue."""
    heapq.heappush(event_queue, (time, event_type, details))


def reset_state() -> None:
    """Reset all simulation state variables.

    The llm scheduler core package re-exports several of these globals for
    convenience (e.g. ``llm_scheduler_core.current_time``).  Those re-exported
    attributes are plain values, so reassignments within this module do not
    automatically propagate back to the package namespace.  When the state is
    reset we therefore need to explicitly mirror the fresh values on the
    package module to keep subsequent consumers in sync.
    """
    global current_time, requests_generated, max_requests_to_generate
    event_queue.clear()
    request_lookup.clear()
    current_time = 0.0
    requests_generated = 0
    max_requests_to_generate = 0

    for module_name in ("llm_scheduler_core", "llm_scheduler_timeseries"):
        module = sys.modules.get(module_name)
        if module is None:
            continue

        module.event_queue = event_queue
        module.current_time = current_time
        module.requests_generated = requests_generated
        module.max_requests_to_generate = max_requests_to_generate
        module.request_lookup = request_lookup


__all__ = [
    "event_queue",
    "current_time",
    "requests_generated",
    "max_requests_to_generate",
    "request_lookup",
    "add_event",
    "reset_state",
]
