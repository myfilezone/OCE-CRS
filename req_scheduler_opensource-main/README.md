# Request Scheduler Experiments

This repository provides the data preparation pipeline and three complementary simulation suites for evaluating large language model (LLM) request schedulers under realistic workloads, as described in the paper "Cloud-Edge System for Scheduling Unpredictable LLM Requests with Combinatorial Bandit". The workflow is designed to:

1. Collect ground-truth inference latency measurements for a target LLM on your hardware.
2. Feed the empirical latency distribution into the **static-batching** and **continuous-batching** simulators to benchmark alternative scheduling strategies.

---

## Installation

The project relies on the Python packages listed in `requirements.txt`. We recommend creating a virtual environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** GPU-backed runs require a CUDA-enabled PyTorch build. Adjust `requirements.txt` or reinstall `torch`/`torchvision` as needed for your environment.

---

## Data Preparation with `build_inference_dataset.py`

The simulator expects a JSONL dataset that records, for each prompt, the inference latency observed on the deployment hardware. Use `build_inference_dataset.py` to generate this dataset from the [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) prompts.

1. **Point the script at the Alpaca dataset.** Replace the placeholder path in `LOCAL_DATASET_PATH` with the absolute or repository-relative directory that holds the downloaded dataset. In this repository checkout, the dataset lives under:
   ```python
   LOCAL_DATASET_PATH = "dataset--tatsu-lab--alpaca"
   ```

2. **Declare the models to benchmark.** Update `MODELS_TO_PROCESS` to map a human-readable model name to the local directory that contains the weights. For example:
   ```python
   MODELS_TO_PROCESS = {
       "Qwen2.5-3B": "/models/Qwen2.5-3B"
   }
   ```
   Multiple entries are supported; each model produces a dedicated JSONL output file.

3. **Run the collector.** Execute the script after ensuring the selected model(s) and tokenizer(s) are available locally:
   ```bash
   python build_inference_dataset.py
   ```

4. **Inspect the outputs.** Results are written to `inference_results/` as files named `inference_data_<MODEL_NAME>.jsonl`, where `<MODEL_NAME>` matches the sanitized key in `MODELS_TO_PROCESS` (for instance, `inference_data_Qwen2_5-3B.jsonl`). Each line records the prompt, response, device metadata, and the measured latency in seconds.

These JSONL files become the input datasets for all downstream schedulers.

---

## Quick Start: Running the Simulators

All three simulators accept configuration overrides via Python dictionaries. The essential fields to customise before running are:

- `dataset_path`: Absolute path to one of the JSONL files produced by `build_inference_dataset.py`.
- `bert_model_name`: Hugging Face identifier or local directory for a BERT model (e.g., `"bert-base-uncased"`). The embedder downloads weights automatically if they are not cached locally.
- Optional overrides for simulation duration, request arrival rate, logging verbosity, etc.

Below are minimal examples that assume you generated `inference_results/inference_data_Qwen2_5-3B.jsonl` and want to run three epochs per simulator.

### 1. Static-Batching (Core) Scheduler Benchmark

```bash
python - <<'PY'
from pathlib import Path
from llm_scheduler_core import simulation

dataset = Path("inference_results/inference_data_Qwen2_5-3B.jsonl").resolve()
override = {
    "dataset_path": str(dataset),
    "bert_model_name": "bert-base-uncased",
    "simulation_duration": 7200,
    "log_level": "INFO",
}

simulation.run_experiment(experiment_config_override=override, epochs=10, save_results=True)
PY
```

### 2. Static-Batching (Time-Series) Scheduler Benchmark

```bash
python - <<'PY'
from pathlib import Path
from llm_scheduler_timeseries import simulation

dataset = Path("inference_results/inference_data_Qwen2_5-3B.jsonl").resolve()
override = {
    "dataset_path": str(dataset),
    "bert_model_name": "bert-base-uncased",
    "simulation_duration": 7200,  # seconds
    "log_level": "INFO",
}

simulation.run_experiment(experiment_config_override=override, epochs=10, save_results=True)
PY
```

### 3. Continuous-Time Scheduler Benchmark

```bash
python - <<'PY'
from pathlib import Path
from llm_scheduler_continuous import simulation

dataset = Path("inference_results/inference_data_Qwen2_5-3B.jsonl").resolve()
override = {
    "dataset_path": str(dataset),
    "bert_model_name": "bert-base-uncased",
    "simulation_duration": 7200,
    "log_level": "INFO",
}

simulation.run_experiment(config_override=override, epochs=10, save_results=True)
PY
```

Each run prints per-epoch summaries to stdout and persists CSV artifacts (summary metrics, time-series traces, and completed-request logs) in the working directory. Modify the overrides to explore alternative schedulers, change arrival rates, or point to different latency datasets.

---

## File Structure

```
├── README.md                       # Project overview and usage guide
├── requirements.txt                # Python dependency list
├── build_inference_dataset.py      # Generates empirical latency datasets from Alpaca prompts
├── dataset--tatsu-lab--alpaca/     # Local snapshot of the tatsu-lab/alpaca dataset
│   ├── README.md
│   └── data/                       # Parquet shard(s) consumed by the collector script
├── llm_scheduler_core/             # Core simulator with batching-aware schedulers
│   ├── __init__.py                 # Public API exports
│   ├── config.py                   # Global configuration and device helpers
│   ├── data.py                     # Dataset loading and preprocessing utilities
│   ├── embedding.py                # BERT-based prompt embedding
│   ├── metrics.py                  # Metric aggregation and logging
│   ├── nodes.py                    # Edge/cloud node abstractions
│   ├── request_model.py            # Request data structures
│   ├── schedulers/                 # Scheduling strategy implementations
│   │   └── strategies.py
│   ├── simulation.py               # Experiment orchestration loop
│   ├── state.py                    # Global simulation state management
│   ├── training.py                 # Neural estimator training helpers
│   └── utils.py                    # Shared utility functions
├── llm_scheduler_timeseries/       # Time-series focused variant of the simulator
│   ├── __init__.py
│   ├── metrics.py                  # Extended metrics tailored for temporal analysis
│   └── simulation.py               # Wrapper around the core simulator with TS logging
└── llm_scheduler_continuous/       # Continuous-time abstraction built atop the time-series engine
    ├── __init__.py
    ├── config.py                   # Defaults specific to the continuous experiments
    └── simulation.py               # Adaptor into the shared time-series runner
```

---

## Troubleshooting & Tips

- **CUDA memory pressure:** The data collection script generates responses without sampling to record precise latency. Reduce `MAX_NEW_TOKENS` or the dataset slice (`DATASET_SPLIT`) if you encounter out-of-memory errors.
- **BERT downloads:** When specifying a model ID (e.g., `"bert-base-uncased"`), the first run downloads weights to the Hugging Face cache. Set `HF_HOME` if you need a custom cache directory.
- **Reproducibility:** Seeds and simulation durations are governed by the config dictionaries. Override `"random_seed"`, `"simulation_duration"`, and `"request_rate_lambda"` to control workload variability.

With the dataset in place and the configuration paths updated, you can iterate rapidly on scheduler designs and compare their effectiveness under shared, reproducible workloads.
