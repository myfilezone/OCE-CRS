"""Metric collection utilities for the LLM scheduler core simulations."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG
from .nodes import EdgeNodeBatching, Node
from .request_model import Request


class MetricsCollector:
    """Collects and calculates simulation metrics."""

    def __init__(self):
        self.completed_request_data: List[Dict[str, object]] = []
        # Timeseries tracking is disabled for the core simulations to avoid the
        # additional bookkeeping overhead observed in the addbaseline runner.
        # Keep an empty container to preserve the public attributes that the
        # schedulers expect, but do not populate it.
        self.timeseries_log: List[Dict[str, object]] = []
        self.simulation_start_time = 0.0
        self.simulation_end_time = 0.0

    def set_simulation_times(self, start_time: float, end_time_config: float) -> None:
        self.simulation_start_time = start_time
        self.simulation_end_time = end_time_config

    def record_completion(self, completed_request: Request, current_time: float) -> None:
        """Records the completion of an individual request and calculates related metrics."""
        completion_time = current_time
        queueing_delay = (
            max(0.0, completed_request.start_time - completed_request.arrival_time)
            if completed_request.start_time != -1.0
            else 0.0
        )
        prediction_error = None
        is_edge_processed = (
            completed_request.assigned_node
            and completed_request.assigned_node.lower().startswith(
                CONFIG.get("edge_node_name", "edge").lower()
            )
        )
        is_predicted = completed_request.predicted_processing_time is not None

        if is_edge_processed and is_predicted:
            actual_proc_time = completed_request.simulated_processing_time
            predicted_proc_time = completed_request.predicted_processing_time
            prediction_error = actual_proc_time - predicted_proc_time

        self.completed_request_data.append(
            {
                "id": completed_request.id,
                "arrival_time": completed_request.arrival_time,
                "start_time": completed_request.start_time,
                "completion_time": completion_time,
                "assigned_node": completed_request.assigned_node,
                "simulated_processing_time_on_node": completed_request.simulated_processing_time,
                "base_edge_inference_time": completed_request.base_edge_inference_time,
                "predicted_processing_time": completed_request.predicted_processing_time,
                "queueing_delay": queueing_delay,
                "prediction_error": prediction_error,
            }
        )
        logging.log(
            logging.getLevelName(CONFIG.get("log_level", "INFO")),
            "T=%0.4f: Req %s COMPLETED on %s. Arr:%0.4f, Start:%0.4f, Comp:%0.4f, NodeProcT:%0.4f, WaitT:%0.4f",
            current_time,
            completed_request.id,
            completed_request.assigned_node,
            completed_request.arrival_time,
            completed_request.start_time,
            completion_time,
            completed_request.simulated_processing_time,
            queueing_delay,
        )
        if prediction_error is not None:
            logging.log(
                logging.getLevelName(CONFIG.get("log_level", "INFO")),
                "  Req %s PredErr: %0.4f (Actual:%0.4f, Predicted:%0.4f)",
                completed_request.id,
                prediction_error,
                actual_proc_time,
                predicted_proc_time,
            )

    def record_decision_point(
        self,
        timestamp: float,
        scheduler_name: str,
        waiting_queue_len_after_scheduling: int,
        edge_node: EdgeNodeBatching,
        cloud_node: Node,
        scheduled_edge_batch_requests: List[Request],
        scheduled_cloud_requests: List[Request],
    ) -> None:
        """Records system state at a scheduling decision point."""
        # The core simulations no longer persist per-decision timeseries data to
        # avoid the extra computation overhead.  We keep a light-weight debug
        # message so that developers can still trace high-level queue length
        # behaviour when the log level is verbose, but skip all further
        # bookkeeping.
        logging.debug(
            "T=%0.4f: Scheduler '%s' decision point. Queue (after): %s",
            timestamp,
            scheduler_name,
            waiting_queue_len_after_scheduling,
        )
        return

    def get_final_metrics_and_timeseries(
        self, actual_simulation_end_time: float
    ) -> Tuple[Dict[str, object], List[Dict[str, object]], pd.DataFrame]:
        """Calculates summary metrics and returns collected logs."""
        self.simulation_end_time = actual_simulation_end_time

        summary_metrics: Dict[str, object]
        completed_requests_df = pd.DataFrame(self.completed_request_data)

        if completed_requests_df.empty:
            summary_metrics = {"error": "No requests completed"}
        else:
            total_completed = len(completed_requests_df)
            edge_node_name_substr = CONFIG.get("edge_node_name", "edge").lower()

            completed_requests_df["is_edge"] = (
                completed_requests_df["assigned_node"].astype(str).str.lower().str.contains(edge_node_name_substr, na=False)
            )
            edge_completed = int(completed_requests_df["is_edge"].sum())
            cloud_completed = total_completed - edge_completed
            completed_requests_df["jct"] = (
                completed_requests_df["completion_time"] - completed_requests_df["arrival_time"]
            )
            avg_queueing_delay_s = (
                completed_requests_df["queueing_delay"].mean() if not completed_requests_df.empty else 0
            )
            df_edge_predicted = completed_requests_df[
                (completed_requests_df["is_edge"] == True)  # noqa: E712
                & (completed_requests_df["prediction_error"].notna())
            ].copy()

            avg_prediction_error_s = (
                df_edge_predicted["prediction_error"].mean() if not df_edge_predicted.empty else float("nan")
            )
            p95_prediction_error_s = (
                df_edge_predicted["prediction_error"].quantile(0.95) if not df_edge_predicted.empty else float("nan")
            )
            p99_prediction_error_s = (
                df_edge_predicted["prediction_error"].quantile(0.99) if not df_edge_predicted.empty else float("nan")
            )

            summary_metrics = {
                "total_completed_requests": total_completed,
                "edge_completed_requests": edge_completed,
                "cloud_completed_requests": cloud_completed,
                "simulation_duration_actual": self.simulation_end_time,
                "total_throughput_rps": total_completed / self.simulation_end_time if self.simulation_end_time > 0 else 0,
                "edge_throughput_rps": edge_completed / self.simulation_end_time if self.simulation_end_time > 0 else 0,
                "cloud_offload_rate": cloud_completed / total_completed if total_completed > 0 else 0,
                "avg_jct_s": completed_requests_df["jct"].mean() if not completed_requests_df.empty else 0,
                "p95_jct_s": completed_requests_df["jct"].quantile(0.95) if not completed_requests_df.empty else 0,
                "p99_jct_s": completed_requests_df["jct"].quantile(0.99) if not completed_requests_df.empty else 0,
                "avg_queueing_delay_s": avg_queueing_delay_s,
                "avg_prediction_error_s_edge_nn": avg_prediction_error_s,
                "p95_prediction_error_s_edge_nn": p95_prediction_error_s,
                "p99_prediction_error_s_edge_nn": p99_prediction_error_s,
            }

        # Timeseries data collection is disabled, so always return an empty list
        # for backward compatibility with callers that unpack three values.
        return summary_metrics, [], completed_requests_df


__all__ = ["MetricsCollector"]
