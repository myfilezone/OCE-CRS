"""Utility helpers for the LLM scheduler core experiment suite."""
from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import torch
from transformers import BertTokenizer

from .config import CONFIG


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info("Random seed set to %s", seed)


def _scale_time_for_training(original_time: float) -> float:
    """Applies scaling and offset for training targets. For NN training data."""
    scale_factor = CONFIG.get("inference_time_scale_factor", 1.0)
    base_offset = CONFIG.get("inference_time_base_offset", 0.0)
    return max(0.001, (original_time * scale_factor) + base_offset)


def _unscale_time_from_prediction(scaled_prediction: float) -> float:
    """Removes scaling and offset from NN predictions. For simulation use."""
    scale_factor = CONFIG.get("inference_time_scale_factor", 1.0)
    base_offset = CONFIG.get("inference_time_base_offset", 0.0)
    if scale_factor == 0:
        return max(0.001, scaled_prediction - base_offset)
    return max(0.001, (scaled_prediction - base_offset) / scale_factor)


def calculate_individual_processing_time(base_edge_time: float, assigned_node_type: str) -> float:
    """Calculates processing time for an INDIVIDUAL request on a given node type."""
    cloud_multiplier = CONFIG.get("cloud_time_multiplier", 1.0)
    if assigned_node_type == "edge":
        proc_time = base_edge_time
    elif assigned_node_type == "cloud":
        proc_time = base_edge_time * cloud_multiplier
    else:
        proc_time = base_edge_time
    return max(0.001, proc_time)


def get_bert_tokenizer_for_filter() -> Optional[BertTokenizer]:
    """Loads BERT tokenizer based on global config, used for dataset filtering."""
    bert_model_path = CONFIG.get("bert_model_name")
    if not bert_model_path:
        return None
    try:
        return BertTokenizer.from_pretrained(bert_model_path)
    except Exception:  # pylint: disable=broad-except
        return None


scale_time_for_training = _scale_time_for_training
unscale_time_from_prediction = _unscale_time_from_prediction

__all__ = [
    "set_seed",
    "_scale_time_for_training",
    "_unscale_time_from_prediction",
    "scale_time_for_training",
    "unscale_time_from_prediction",
    "calculate_individual_processing_time",
    "get_bert_tokenizer_for_filter",
]
