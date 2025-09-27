"""Configuration and runtime state for the LLM scheduler core experiment suite."""
from __future__ import annotations

import logging
import sys
from copy import deepcopy
from typing import Dict, Optional

import torch

LOGGER = logging.getLogger(__name__)

# --- Default Configuration ---
DEFAULT_CONFIG: Dict[str, object] = {
    # Simulation Parameters
    "simulation_duration": 7200,
    "random_seed": 42,
    "llm_name": "Qwen", 

    # Node Capacities
    "edge_concurrency": 1,  # Number of parallel BATCHES edge can process
    "edge_batch_capacity_k": 4,  # Max number of requests PER BATCH on edge
    "cloud_concurrency": 8,  # Number of parallel individual requests cloud can process
    "network_rtt": 0.05,

    # Real Data Processing Time Config
    "cloud_time_multiplier": 0.7,
    "inference_time_scale_factor": 10.0,  # Scaling for TRAINING ONLY
    "inference_time_base_offset": 0.1,   # Scaling for TRAINING ONLY

    # Workload Generation
    "request_rate_lambda": 5.0,
    "dataset_path": "<DATASET_PATH_PLACEHOLDER>",
    # Example alternative dataset paths:
    # "dataset_path": "<DATASET_PATH_PLACEHOLDER>/inference_results/inference_data_Qwen2_5-3B.jsonl",
    "bert_model_name": "<BERT_MODEL_PATH_PLACEHOLDER>",
    "max_dataset_samples": 50000,

    # Embedding Model Configuration
    "embedding_model_type": "bert",
    "bert_embedding_dim": 768,

    # Neural Network Estimator
    "nn_hidden_layers": [256, 256],
    "nn_learning_rate": 0.001,
    "nn_batch_size": 32,
    "nn_train_interval": 5,
    "nn_training_buffer_maxlen": 10000,  # Max length for the AsyncTrainer's training buffer

    # Bandit Parameters
    "clinucb_alpha": 0.1,
    "clinucb_lambda_reg": 1,
    "cngreedy_epsilon": 0.1,
    "cnducb_nu": 0.001,
    "linucb_feedback_buffer_max": 5000,  # Max length for LinUCB feedback buffer

    # Scheduler Parameters
    "candidate_pool_size_m_factor": 8,  # m = candidate_pool_size_m_factor * edge_batch_capacity_k
    "edge_node_name": "edge0",  # Name used to identify edge completions in metrics

    # Asynchronous Training
    "max_training_workers": 1,

    # Logging Level Control
    "log_level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR
    "device": "auto",  # Options: "auto", "cpu", "cuda:0", ...
}

CONFIG: Dict[str, object] = {}
CURRENT_EMBEDDING_DIM: Optional[int] = None
DEVICE: Optional[torch.device] = None


def _resolve_device(device_request: object) -> torch.device:
    """Translate configuration input into a concrete :class:`torch.device`."""

    if isinstance(device_request, torch.device):
        if device_request.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning(
                "CUDA requested via torch.device(%s) but no GPU is available; falling back to CPU.",
                device_request,
            )
            return torch.device("cpu")
        return device_request

    if device_request is None:
        device_request = "auto"

    request_str = str(device_request).lower()
    if request_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        resolved = torch.device(str(device_request))
        if resolved.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning(
                "CUDA device '%s' requested but torch.cuda.is_unavailable(); using CPU instead.",
                device_request,
            )
            return torch.device("cpu")
        return resolved
    except (TypeError, ValueError):
        LOGGER.warning("Invalid device specification '%s'; defaulting to CPU.", device_request)
        return torch.device("cpu")


def _propagate_runtime_state() -> None:
    """Synchronise exported module-level symbols after config mutations."""

    for module_name in ("llm_scheduler_core", "llm_scheduler_timeseries"):
        module = sys.modules.get(module_name)
        if module is None:
            continue

        module.CONFIG = CONFIG
        module.CURRENT_EMBEDDING_DIM = CURRENT_EMBEDDING_DIM
        module.DEVICE = DEVICE


def apply_config(config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Apply a configuration and initialise derived globals.

    The function mirrors the behaviour of the original module-level globals by
    mutating ``CONFIG`` in-place while also computing ``DEVICE`` and
    ``CURRENT_EMBEDDING_DIM`` so that the rest of the simulation code can rely
    on them.
    """
    global CONFIG, CURRENT_EMBEDDING_DIM, DEVICE

    CONFIG = deepcopy(config) if config is not None else deepcopy(DEFAULT_CONFIG)

    DEVICE = _resolve_device(CONFIG.get("device", "auto"))

    CURRENT_EMBEDDING_DIM = CONFIG.get("bert_embedding_dim")

    _propagate_runtime_state()
    return CONFIG


def reset_runtime_state() -> None:
    """Reset derived runtime globals to their default uninitialised values."""
    global CONFIG, CURRENT_EMBEDDING_DIM, DEVICE
    CONFIG = {}
    CURRENT_EMBEDDING_DIM = None
    DEVICE = None

    _propagate_runtime_state()


__all__ = [
    "DEFAULT_CONFIG",
    "CONFIG",
    "CURRENT_EMBEDDING_DIM",
    "DEVICE",
    "apply_config",
    "reset_runtime_state",
]
