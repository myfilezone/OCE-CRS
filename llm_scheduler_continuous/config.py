"""Configuration helpers for the LLM scheduler continuous-time experiments."""
from __future__ import annotations

from copy import deepcopy
from typing import Dict


DEFAULT_CONFIG: Dict[str, object] = {
    # Simulation Parameters
    "simulation_duration": 7200,  # Default duration
    "random_seed": 42,
    "llm_name": "Qwen", 

    # Node Capacities
    "edge_concurrency": 4,
    "cloud_concurrency": 8,
    "network_rtt": 0.05,

    # Real Data Processing Time Config
    "cloud_time_multiplier": 0.7,
    # Scaling for TRAINING ONLY
    "inference_time_scale_factor": 10.0,
    "inference_time_base_offset": 0.1,

    # Workload Generation
    "request_rate_lambda": 5.0,
    # --- PATHS ARE NOW EXPECTED HERE ---
    "dataset_path": "<DATASET_PATH_PLACEHOLDER>",
    # Example alternative dataset paths:
    # "dataset_path": "<DATASET_PATH_PLACEHOLDER>/inference_results/inference_data_Qwen2_5-3B.jsonl",
    "bert_model_name": "<BERT_MODEL_PATH_PLACEHOLDER>",  # Default path <- UPDATE THIS
    # --- END PATHS ---
    "max_dataset_samples": 50000,  # Limit samples loaded

    # Embedding Model Configuration
    "embedding_model_type": "bert", 
    # BERT Specific (bert_model_name is now the path)
    "bert_embedding_dim": 768,

    # Neural Network Estimator (Predicts scaled/offset time)
    "nn_hidden_layers": [256, 256],  # Excluded from hyperparameter search by user request
    "nn_learning_rate": 0.001,
    "nn_batch_size": 32,
    "nn_train_interval": 5,

    # Bandit Parameters
    "clinucb_alpha": 0.1,
    "clinucb_lambda_reg": 1,
    "cngreedy_epsilon": 0.1,
    "cnducb_nu": 0.001,

    # Scheduler Parameters
    "candidate_pool_size_m_factor": 2,

    # Asynchronous Training
    "max_training_workers": 1,

    # Logging Level Control
    "log_level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR

    # --- NEW: Dynamics Logging ---
    "dynamics_log_interval": 1.0,  # Log system state every 1.0 simulation seconds
    "enable_dynamics_logging": True,  # Flag to enable/disable dynamics logging

    # Device Control
    "device": "auto",  # Options: "auto", "cpu", "cuda:0", "cuda:1", ...
}


def apply_default_override() -> Dict[str, object]:
    """Return a deep copy of the default configuration for safe mutation."""

    return deepcopy(DEFAULT_CONFIG)


__all__ = ["DEFAULT_CONFIG", "apply_default_override"]
