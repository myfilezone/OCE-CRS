"""Processing node implementations for the LLM scheduler core simulations."""
from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional

from .config import CONFIG
from .state import request_lookup
from .request_model import Request


class Node:
    """Represents a processing node for INDIVIDUAL requests (e.g., Cloud Node)."""

    def __init__(self, name: str, concurrency: int):
        self.name = name
        self.concurrency = concurrency
        self.active_requests_info: Dict[int, float] = {}
        self.historical_processing_times: deque[float] = deque(maxlen=100)
        self.avg_processing_time: float = 0.001
        logging.info("Node '%s' initialized with concurrency %s.", name, concurrency)

    @property
    def available_slots(self) -> int:
        return self.concurrency - len(self.active_requests_info)

    def is_available(self) -> bool:
        return self.available_slots > 0

    def assign_request(self, request: Request, current_time: float, processing_time_on_node: float):
        """Assigns an individual request to this node."""
        if not self.is_available():
            raise RuntimeError(f"Node {self.name} unavailable for new request {request.id}.")

        completion_time = current_time + processing_time_on_node

        self.active_requests_info[request.id] = completion_time
        request.assigned_node = self.name
        request.start_time = current_time
        request.simulated_processing_time = processing_time_on_node
        request.completion_time = completion_time

        logging.log(
            logging.getLevelName(CONFIG.get("log_level", "INFO")),
            "T=%0.4f: Req %s assigned to %s, ProcT=%0.4f, Finish at %0.4f",
            current_time,
            request.id,
            self.name,
            processing_time_on_node,
            completion_time,
        )
        return request.id, completion_time

    def release_slot(self, request_id: int, current_time_for_check: float) -> bool:
        """Releases a slot if the request is completed by ``current_time_for_check``."""
        if request_id in self.active_requests_info:
            if self.active_requests_info[request_id] <= current_time_for_check + 1e-9:
                request = request_lookup.get(request_id)
                if request and request.start_time != -1:
                    actual_proc_time = current_time_for_check - request.start_time
                    if self.name == "cloud":
                        actual_proc_time -= CONFIG.get("network_rtt", 0.05)
                    self.add_historical_processing_time(max(0.001, actual_proc_time))

                del self.active_requests_info[request_id]
                logging.log(
                    logging.getLevelName(CONFIG.get("log_level", "INFO")),
                    "T=%0.4f: Slot released on %s by req %s",
                    current_time_for_check,
                    self.name,
                    request_id,
                )
                return True
        return False

    def add_historical_processing_time(self, processing_time: float) -> None:
        self.historical_processing_times.append(processing_time)
        if self.historical_processing_times:
            self.avg_processing_time = sum(self.historical_processing_times) / len(self.historical_processing_times)
        else:
            self.avg_processing_time = 0.001
        logging.debug("Node %s: Updated avg_processing_time=%0.4f", self.name, self.avg_processing_time)

    def estimate_queueing_delay(self, current_time_for_estimate: float) -> float:
        if self.available_slots == self.concurrency:
            return 0.0
        if self.active_requests_info:
            max_busy_time = max(self.active_requests_info.values())
            return max(0.0, max_busy_time - current_time_for_estimate)
        return 0.0


class EdgeNodeBatching:
    """Represents an Edge Node that processes requests in batches."""

    def __init__(self, name: str, concurrency: int, batch_capacity_k: int):
        self.name = name
        self.concurrency = concurrency
        self.batch_capacity_k = batch_capacity_k
        self.active_batches: List[Dict[str, object]] = []
        self._next_batch_internal_id_counter = 0
        self.historical_processing_times: deque[float] = deque(maxlen=100)
        self.avg_processing_time: float = 0.001
        logging.info(
            "EdgeNodeBatching '%s' initialized: Concurrency=%s batches, Capacity/batch=%s reqs.",
            name,
            concurrency,
            batch_capacity_k,
        )

    def _get_next_batch_internal_id(self) -> str:
        batch_id = f"{self.name}_b{self._next_batch_internal_id_counter}"
        self._next_batch_internal_id_counter += 1
        return batch_id

    def _calculate_actual_batch_processing_time(self, batch_requests: List[Request]) -> float:
        if not batch_requests:
            return 0.0
        return max(req.base_edge_inference_time for req in batch_requests)

    @property
    def available_slots(self) -> int:
        return self.concurrency - len(self.active_batches)

    def is_available(self) -> bool:
        return self.available_slots > 0

    def assign_batch(self, batch_requests: List[Request], current_time: float):
        if not self.is_available():
            logging.warning("T=%0.4f: Attempted to assign batch to busy %s.", current_time, self.name)
            return None
        if not batch_requests:
            logging.warning("T=%0.4f: assign_batch called with empty batch_requests on %s.", current_time, self.name)
            return None
        if len(batch_requests) > self.batch_capacity_k:
            raise ValueError(f"Batch size {len(batch_requests)} for {self.name} exceeds capacity {self.batch_capacity_k}.")

        actual_batch_processing_time = self._calculate_actual_batch_processing_time(batch_requests)
        batch_completion_time = current_time + actual_batch_processing_time
        batch_internal_id = self._get_next_batch_internal_id()

        batch_job_info = {
            "batch_internal_id": batch_internal_id,
            "completion_time": batch_completion_time,
            "requests": batch_requests,
        }
        self.active_batches.append(batch_job_info)

        for req in batch_requests:
            req.assigned_node = self.name
            req.start_time = current_time
            req.simulated_processing_time = req.base_edge_inference_time
            req.completion_time = batch_completion_time

        logging.log(
            logging.getLevelName(CONFIG.get("log_level", "INFO")),
            "T=%0.4f: Batch %s (size %s) assigned to %s. BatchProcT=%0.4f, Finish at %0.4f",
            current_time,
            batch_internal_id,
            len(batch_requests),
            self.name,
            actual_batch_processing_time,
            batch_completion_time,
        )

        return batch_internal_id, batch_completion_time

    def release_slot(self, batch_internal_id_to_release: str, current_time_for_check: float):
        released_batch_job_info: Optional[Dict[str, object]] = None
        remaining_active_batches: List[Dict[str, object]] = []
        found_batch = False

        for batch_job in self.active_batches:
            if batch_job["batch_internal_id"] == batch_internal_id_to_release:
                found_batch = True
                if batch_job["completion_time"] <= current_time_for_check + 1e-9:
                    released_batch_job_info = batch_job
                    batch_start_time = released_batch_job_info["requests"][0].start_time
                    if batch_start_time != -1:
                        actual_batch_proc_time = current_time_for_check - batch_start_time
                        self.add_historical_processing_time(max(0.001, actual_batch_proc_time))

                    logging.log(
                        logging.getLevelName(CONFIG.get("log_level", "INFO")),
                        "T=%0.4f: Batch %s COMPLETED on %s. Releasing slot.",
                        current_time_for_check,
                        batch_job["batch_internal_id"],
                        self.name,
                    )
                else:
                    remaining_active_batches.append(batch_job)
                    logging.warning(
                        "T=%0.4f: Batch %s on %s release called at %0.4f but completion is at %0.4f.",
                        current_time_for_check,
                        batch_job["batch_internal_id"],
                        self.name,
                        current_time_for_check,
                        batch_job["completion_time"],
                    )
            else:
                remaining_active_batches.append(batch_job)

        if not found_batch:
            logging.warning(
                "T=%0.4f: Attempted to release non-existent or already released batch ID %s on %s.",
                current_time_for_check,
                batch_internal_id_to_release,
                self.name,
            )

        self.active_batches = remaining_active_batches
        return released_batch_job_info

    def add_historical_processing_time(self, processing_time: float) -> None:
        self.historical_processing_times.append(processing_time)
        if self.historical_processing_times:
            self.avg_processing_time = sum(self.historical_processing_times) / len(self.historical_processing_times)
        else:
            self.avg_processing_time = 0.001
        logging.debug(
            "EdgeNodeBatching %s: Updated avg_processing_time=%0.4f",
            self.name,
            self.avg_processing_time,
        )

    def estimate_queueing_delay(self, current_time_for_estimate: float) -> float:
        if self.available_slots == self.concurrency:
            return 0.0
        if self.active_batches:
            max_batch_completion_time = max(batch_job["completion_time"] for batch_job in self.active_batches)
            return max(0.0, max_batch_completion_time - current_time_for_estimate)
        return 0.0


__all__ = ["Node", "EdgeNodeBatching"]
