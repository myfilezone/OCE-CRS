"""Dataset loading helpers for the LLM scheduler core simulations."""
from __future__ import annotations

import json
import os

import pandas as pd

from .config import CONFIG
from .utils import scale_time_for_training


def load_and_process_jsonl_dataset(jsonl_path, max_samples_to_load, bert_tokenizer_for_filtering=None):
    """Loads data from a JSONL file, processes it, and optionally filters by token length."""
    if not os.path.exists(jsonl_path):
        return None
    processed_data_list = []
    lines_processed, lines_filtered_by_length, lines_with_errors = 0, 0, 0
    bert_filter_max_length = 512

    try:
        with open(jsonl_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                lines_processed += 1
                if max_samples_to_load > 0 and len(processed_data_list) >= max_samples_to_load:
                    break
                try:
                    data_point = json.loads(line.strip())
                    request_text = data_point.get("request_text")
                    inference_time = data_point.get("inference_time")
                    if not isinstance(request_text, str) or not request_text.strip():
                        lines_with_errors += 1
                        continue
                    if not isinstance(inference_time, (int, float)) or inference_time < 0:
                        lines_with_errors += 1
                        continue
                    if bert_tokenizer_for_filtering:
                        try:
                            tokens = bert_tokenizer_for_filtering.encode(
                                request_text,
                                add_special_tokens=True,
                                truncation=False,
                            )
                            if len(tokens) >= bert_filter_max_length:
                                lines_filtered_by_length += 1
                                continue
                        except Exception:  # pylint: disable=broad-except
                            pass

                    processed_data_list.append(
                        {
                            "prompt": request_text,
                            "reference_output": data_point.get("response_text", ""),
                            "target_model": data_point.get("model_name", "default_model"),
                            "base_edge_inference_time": float(inference_time),
                        }
                    )
                except json.JSONDecodeError:
                    lines_with_errors += 1
                except Exception:  # pylint: disable=broad-except
                    lines_with_errors += 1
    except Exception:  # pylint: disable=broad-except
        return None
    if not processed_data_list:
        return None
    try:
        dataset_df = pd.DataFrame(processed_data_list)
        if "base_edge_inference_time" in dataset_df.columns and CONFIG:
            _ = [scale_time_for_training(t) for t in dataset_df["base_edge_inference_time"]]
        return dataset_df
    except Exception:  # pylint: disable=broad-except
        return None


__all__ = ["load_and_process_jsonl_dataset"]
