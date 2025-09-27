"""Request model used throughout the LLM scheduler core simulations."""
from __future__ import annotations


class Request:
    """Represents a single LLM request in the simulation."""

    _id_counter = 0

    def __init__(self, arrival_time, prompt, reference_output, target_model, embedding, base_edge_inference_time):
        self.id = Request._id_counter
        Request._id_counter += 1
        self.arrival_time = arrival_time
        self.prompt = prompt
        self.reference_output = reference_output
        self.target_model = target_model
        self.embedding = embedding
        self.base_edge_inference_time = float(base_edge_inference_time)

        self.start_time = -1.0
        self.completion_time = -1.0
        self.assigned_node = None
        self.simulated_processing_time = -1.0

        self.predicted_processing_time = None
        self.base_predicted_processing_time = None

    def __lt__(self, other):
        """Comparison for priority queue (heapq)."""
        if self.predicted_processing_time is not None and other.predicted_processing_time is not None:
            return self.predicted_processing_time < other.predicted_processing_time
        return self.arrival_time < other.arrival_time


__all__ = ["Request"]
