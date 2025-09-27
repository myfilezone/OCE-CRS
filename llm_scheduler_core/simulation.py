"""Simulation orchestration for LLM scheduler core experiments."""

from __future__ import annotations

import copy
import gc
import heapq
import json
import logging
import os
import pathlib
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llm_scheduler_core as cc
from llm_scheduler_core import (
    BertEmbedder,
    MetricsCollector,
    load_and_process_jsonl_dataset,
    set_seed,
)
from llm_scheduler_core.request_model import Request
from llm_scheduler_core.schedulers.strategies import (
    CNDUCBScheduler,
    CNGreedyScheduler,
    CombinatorialLinUCBScheduler,
    EdgeCloudFCFSScheduler,
    HistoricalLRTScheduler,
    LeastConnectionsScheduler,
    QueueLengthBasedScheduler,
    RandomOffloadScheduler,
    RoundRobinOffloadScheduler,
)

CONFIG = cc.CONFIG
CURRENT_EMBEDDING_DIM = cc.CURRENT_EMBEDDING_DIM
DEVICE = cc.DEVICE


def run_simulation(scheduler_instance, embedder_instance, processed_dataset_df):
    """Runs a single simulation with the given scheduler and data."""
    # Reset global simulation state for a clean run
    cc.reset_state()
    Request._id_counter = 0 # Reset request ID counter
    gc.collect() # Clean up memory from previous runs

    metrics_coll = scheduler_instance.metrics_collector
    metrics_coll.set_simulation_times(0.0, cc.CONFIG["simulation_duration"])

    if processed_dataset_df is None or processed_dataset_df.empty:
        logging.error("Simulation failed: Invalid or empty dataset provided.")
        # Now returns 3 values: summary_metrics, timeseries_log, completed_requests_df
        return {"error": "Invalid dataset"}, [], pd.DataFrame()
    if embedder_instance is None:
        logging.error("Simulation failed: Invalid embedder provided.")
        # Now returns 3 values
        return {"error": "Invalid embedder"}, [], pd.DataFrame()

    # Determine the total number of requests to generate
    sim_duration = cc.CONFIG["simulation_duration"]
    request_rate_lambda = cc.CONFIG["request_rate_lambda"]
    # Generate enough requests to cover the simulation duration plus a buffer
    cc.max_requests_to_generate = int(sim_duration * request_rate_lambda * 1.5) + 100

    # Schedule the first arrival event if request rate is positive
    if request_rate_lambda > 0:
        first_arrival_time = cc.current_time + random.expovariate(request_rate_lambda)
        cc.add_event(first_arrival_time, "arrival")
        logging.info(f"Scheduled first arrival event at T={first_arrival_time:.4f}")
    else:
        logging.warning("Request rate lambda is 0. No requests will be generated.")


    logging.info(f"Starting simulation for '{scheduler_instance.scheduler_name}' for {sim_duration} seconds.")
    logging.info(f"Max requests to generate: {cc.max_requests_to_generate}")


    # Main simulation loop
    while cc.event_queue:
        # Get the next event from the priority queue
        try:
            event_time, event_type, details = heapq.heappop(cc.event_queue)
        except IndexError:
            # Event queue is empty, end simulation
            logging.info("Event queue is empty. Ending simulation loop.")
            break

        # If the event time is beyond the simulation duration, stop
        if event_time > sim_duration:
            cc.current_time = sim_duration # Advance time to the simulation end
            logging.info(f"Simulation duration ({sim_duration}s) reached at T={cc.current_time:.4f}. Ending simulation loop.")
            break # Exit the loop

        # Advance the simulation time to the event time
        cc.current_time = max(cc.current_time, event_time) # Ensure time doesn't go backward


        # Process the event
        try:
            if event_type == "arrival":
                logging.debug(f"T={cc.current_time:.4f}: Processing arrival event.")
                # Check if we should generate more requests
                if cc.requests_generated < cc.max_requests_to_generate:
                    # Get data for the new request from the dataset (wrap around if needed)
                    # Ensure the dataset_df is not empty before accessing iloc
                    if not processed_dataset_df.empty:
                        row_idx = cc.requests_generated % len(processed_dataset_df)
                        data_row = processed_dataset_df.iloc[row_idx]

                        # Get embedding for the request
                        req_emb = embedder_instance.get_embedding(data_row['prompt'])

                        # Ensure embedding is valid and has the correct dimension
                        # Use cc.CURRENT_EMBEDDING_DIM which is set by run_experiment after embedder init
                        if req_emb is None or cc.CURRENT_EMBEDDING_DIM is None or req_emb.shape[0] != cc.CURRENT_EMBEDDING_DIM:
                             logging.error(f"T={cc.current_time:.4f}: Failed to get valid embedding for request {cc.requests_generated}. Skipping request generation.")
                             # Still schedule the next arrival event to maintain rate, but skip this request.
                             if request_rate_lambda > 0:
                                cc.add_event(cc.current_time + random.expovariate(request_rate_lambda), "arrival")
                             continue # Skip to the next event


                        # Create a new Request object
                        new_req = Request(
                            arrival_time=cc.current_time, # Use current time as arrival time
                            prompt=data_row['prompt'],
                            reference_output=data_row['reference_output'],
                            target_model=data_row['target_model'],
                            embedding=req_emb, # Use the generated embedding
                            base_edge_inference_time=float(data_row['base_edge_inference_time']) # Use base edge time from dataset
                        )

                        cc.requests_generated += 1
                        cc.request_lookup[new_req.id] = new_req # Add to lookup dictionary

                        # Add the new request to the scheduler's waiting queue
                        scheduler_instance.add_request(new_req)

                        # Schedule the next arrival event
                        if request_rate_lambda > 0:
                           cc.add_event(cc.current_time + random.expovariate(request_rate_lambda), "arrival")
                           # logging.debug(f"T={cc.current_time:.4f}: Scheduled next arrival at {cc.current_time + random.expovariate(request_rate_lambda):.4f}")

                    else:
                        # Dataset is empty, cannot generate requests. Stop arrival events.
                        logging.warning(f"T={cc.current_time:.4f}: Dataset is empty. Cannot generate more requests. Stopping arrival events.")
                        # Remove any pending arrival events
                        cc.event_queue[:] = [e for e in cc.event_queue if e[1] != "arrival"]
                        heapq.heapify(cc.event_queue)


                # Trigger the scheduler to make a decision based on the updated queue
                # The scheduler's schedule method will now call record_decision_point internally
                scheduled_any = scheduler_instance.schedule(cc.current_time)
                # The return value (scheduled_any) is not strictly needed here anymore
                # as logging happens within schedule().


            elif event_type == "edge_batch_completion":
                # logging.debug(f"T={cc.current_time:.4f}: Processing edge batch completion event for batch ID: {details[0]}")
                batch_internal_id, scheduler_name_completed = details # Get batch ID and scheduler name from event details

                # Release the slot on the edge node and get info about the completed batch
                # We need the scheduler instance to access the edge node
                completed_batch_info = scheduler_instance.edge_node.release_slot(batch_internal_id, cc.current_time)

                if completed_batch_info:
                    # Record completion for each request in the batch
                    for completed_req_in_batch in completed_batch_info['requests']:
                        # Ensure the request is in the lookup before recording completion (should be)
                        if completed_req_in_batch.id in cc.request_lookup:
                            metrics_coll.record_completion(completed_req_in_batch, cc.current_time)
                        else:
                            logging.warning(f"T={cc.current_time:.4f}: Completed edge batch contains request ID {completed_req_in_batch.id} not found in lookup.")

                    # Provide feedback to the scheduler (if it's a learning scheduler)
                    scheduler_instance.get_feedback(completed_batch_info, is_batch_feedback=True)
                else:
                    logging.warning(f"T={cc.current_time:.4f}: Received completion event for unknown or already released edge batch ID: {batch_internal_id}. Ignoring.")

                # Trigger the scheduler after a slot is freed on the edge node
                # The scheduler's schedule method will now call record_decision_point internally
                scheduled_any = scheduler_instance.schedule(cc.current_time)


            elif event_type == "cloud_individual_completion":
                # logging.debug(f"T={cc.current_time:.4f}: Processing cloud individual completion event for request ID: {details[0]}")
                request_id_completed, scheduler_name_completed = details # Get request ID and scheduler name

                # Get the request object from the lookup
                completed_req_obj = cc.request_lookup.get(request_id_completed)

                if completed_req_obj:
                    # Release the slot on the cloud node
                     # We need the scheduler instance to access the cloud node
                    if scheduler_instance.cloud_node.release_slot(request_id_completed, cc.current_time):
                        # Record completion for the individual request
                        metrics_coll.record_completion(completed_req_obj, cc.current_time)
                        # Provide feedback to the scheduler (if it's a learning scheduler)
                        scheduler_instance.get_feedback(completed_req_obj, is_batch_feedback=False)
                    else:
                        # Should not happen if the event timing is correct, but handle defensively
                        logging.warning(f"T={cc.current_time:.4f}: Cloud node slot release failed for Req {request_id_completed}. Slot might be busy or already released.")
                        # Still record completion and provide feedback even if slot release check failed
                        metrics_coll.record_completion(completed_req_obj, cc.current_time)
                        scheduler_instance.get_feedback(completed_req_obj, is_batch_feedback=False)

                    # Request object can be potentially removed from lookup after processing
                    # del cc.request_lookup[request_id_completed] # Consider if removing is safe/necessary

                else:
                    logging.warning(f"T={cc.current_time:.4f}: Received completion event for unknown request ID: {request_id_completed}. Ignoring.")


                # Trigger the scheduler after a slot is freed on the cloud node
                # The scheduler's schedule method will now call record_decision_point internally
                scheduled_any = scheduler_instance.schedule(cc.current_time)

            else:
                logging.error(f"T={cc.current_time:.4f}: Unknown event type: {event_type}")

        except Exception as e:
            logging.error(f"T={cc.current_time:.4f}: Error processing event {event_type} for details {details}: {e}", exc_info=True)
            # Continue simulation loop even on event processing error


    # Simulation loop finished. Finalize metrics.
    logging.info(f"Simulation ended at T={cc.current_time:.4f}. Collecting final metrics.")

    # Shutdown the scheduler to allow any background processes (like NN training) to finish
    try:
        scheduler_instance.shutdown()
    except Exception as e:
        logging.error(f"Error during scheduler shutdown: {e}", exc_info=True)

    # Get final metrics, including optional timeseries and completed request data.
    summary_metrics, timeseries_log, completed_requests_df = metrics_coll.get_final_metrics_and_timeseries(
        cc.current_time
    )

    # Clear request lookup and run garbage collection
    cc.request_lookup.clear()
    gc.collect()

    logging.info("Simulation run completed.")

    return summary_metrics, timeseries_log, completed_requests_df


# Add the epochs parameter to the run_experiment function signature
def run_experiment(experiment_config_override={}, epochs=1, save_results=True):
    """
    Sets up and runs simulations for multiple schedulers based on the configuration.
    Collects results and saves them to CSV files.

    Args:
        experiment_config_override (dict): Dictionary to override default configuration.
        epochs (int): Number of times to run the entire set of simulations.
        save_results (bool): Whether to persist aggregated results to CSV files.
    """
    global CONFIG, CURRENT_EMBEDDING_DIM, DEVICE # Allow modification of global config and device
    pid = os.getpid() # Get process ID for logging

    # --- Setup Logging (moved here, once per experiment run) ---
    log_level_str = experiment_config_override.get("log_level", cc.DEFAULT_CONFIG.get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = f'%(asctime)s - %(levelname)s - PID:{pid} - %(filename)s:%(lineno)d - %(message)s'
    # Remove existing handlers to avoid duplication if called multiple times
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=log_level, format=log_format)
    # --- End Logging Setup ---

    logging.info(f"--- Starting Experiment Run (Epochs: {epochs}) --- PID: {pid}")
    logging.info(f"Override Config: {json.dumps(experiment_config_override, indent=2)}")

    # --- Initialize Aggregation Lists (before epoch loop) ---
    all_epochs_sum_list = []
    all_epochs_timeseries_records_list = []
    all_epochs_completed_requests_dfs_list = []

    exp_start_t = time.time()

    # --- Epoch Loop ---
    for epoch in range(epochs):
        logging.info(f"\n====== Starting Epoch {epoch + 1}/{epochs} ======")

        # --- Update Global Config and Set Seed for this Epoch ---
        epoch_config = copy.deepcopy(cc.DEFAULT_CONFIG)
        epoch_config.update(experiment_config_override)
        CONFIG = cc.apply_config(epoch_config)
        CURRENT_EMBEDDING_DIM = cc.CURRENT_EMBEDDING_DIM
        DEVICE = cc.DEVICE

        # Set seed for this epoch, adding epoch number and pid for variation and reproducibility
        cc.set_seed(cc.CONFIG["random_seed"] + epoch + pid)

        logging.info(f"Epoch {epoch+1}: Using device: {cc.DEVICE}")

        # --- Initialize Embedder for this Epoch ---
        embedder_type_config = cc.CONFIG.get("embedding_model_type", "bert").lower()
        embedder_instance = None
        bert_tokenizer_for_filter = None
        try:
            if embedder_type_config != "bert":
                raise ValueError(
                    f"Unsupported embedding type '{embedder_type_config}'. Only 'bert' is available."
                )

            embedding_dim = cc.CURRENT_EMBEDDING_DIM
            if embedding_dim is None:
                raise ValueError("CURRENT_EMBEDDING_DIM is not set. Did apply_config run?")

            bert_model_path = cc.CONFIG.get("bert_model_name")
            if not bert_model_path or not os.path.exists(bert_model_path):
                raise FileNotFoundError(f"BERT path not found: {bert_model_path}")

            embedder_instance = BertEmbedder(
                model_path=bert_model_path,
                embedding_dim=embedding_dim,
            )
            bert_tokenizer_for_filter = embedder_instance.tokenizer

            logging.info(
                "Epoch %s: Embedder: %s, Dim: %s",
                epoch + 1,
                embedder_type_config,
                embedding_dim,
            )

        except Exception as e:
            logging.error(f"Epoch {epoch+1}: Embedder init fail: {e}", exc_info=True)
            return {"error": f"Epoch {epoch+1} Embedder Init Fail: {e}"}, [], []  # Return empty lists on error


        # --- Load Dataset for this Epoch ---
        logging.info(f"Epoch {epoch+1}: Loading dataset...")
        dataset_path = cc.CONFIG.get("dataset_path")
        max_dataset_samples = cc.CONFIG.get("max_dataset_samples", -1)
        if not dataset_path:
             logging.error(f"Epoch {epoch+1}: Dataset path is not configured.")
             if embedder_instance and hasattr(embedder_instance, 'shutdown'):
                  try: embedder_instance.shutdown()
                  except Exception: pass
             return {"error": f"Epoch {epoch+1} Dataset path missing."}, [], []

        processed_dataset = cc.load_and_process_jsonl_dataset(
            dataset_path, max_dataset_samples, bert_tokenizer_for_filter
        )
        if processed_dataset is None or processed_dataset.empty:
             logging.error(f"Epoch {epoch+1}: Dataset loading or processing failed.")
             if embedder_instance and hasattr(embedder_instance, 'shutdown'):
                  try: embedder_instance.shutdown()
                  except Exception: pass
             return {"error": f"Epoch {epoch+1} Dataset prep failed."}, [], []

        # Dispose of tokenizer if loaded separately (no-op for BERT)

        # --- Define NN Components Creation (uses current global CONFIG/DEVICE) ---
        def create_nn_components():
            model = cc.TimeEstimatorNN(cc.CURRENT_EMBEDDING_DIM, cc.CONFIG["nn_hidden_layers"]).to(cc.DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=cc.CONFIG["nn_learning_rate"])
            criterion = torch.nn.MSELoss()
            return model, optimizer, criterion
        # --- End Define NN Components ---


        # --- Define Schedulers (uses current global CONFIG/DEVICE/CURRENT_EMBEDDING_DIM) ---
        schedulers_to_run = {
            "B1_Edge_Cloud_FCFS": lambda: EdgeCloudFCFSScheduler(
                cc.EdgeNodeBatching(
                    cc.CONFIG["edge_node_name"],
                    cc.CONFIG["edge_concurrency"],
                    cc.CONFIG["edge_batch_capacity_k"],
                ),
                cc.Node("cloud0", cc.CONFIG["cloud_concurrency"]),
                MetricsCollector(),
            ),
            "B2_Random_Offload": lambda: RandomOffloadScheduler(
                cc.EdgeNodeBatching(
                    cc.CONFIG["edge_node_name"],
                    cc.CONFIG["edge_concurrency"],
                    cc.CONFIG["edge_batch_capacity_k"],
                ),
                cc.Node("cloud0", cc.CONFIG["cloud_concurrency"]),
                MetricsCollector(),
            ),
            "B3_Round_Robin_Offload": lambda: RoundRobinOffloadScheduler(
                cc.EdgeNodeBatching(
                    cc.CONFIG["edge_node_name"],
                    cc.CONFIG["edge_concurrency"],
                    cc.CONFIG["edge_batch_capacity_k"],
                ),
                cc.Node("cloud0", cc.CONFIG["cloud_concurrency"]),
                MetricsCollector(),
            ),
            "B4_CLinUCB_Diag": lambda: CombinatorialLinUCBScheduler(
                cc.EdgeNodeBatching(
                    cc.CONFIG["edge_node_name"],
                    cc.CONFIG["edge_concurrency"],
                    cc.CONFIG["edge_batch_capacity_k"],
                ),
                cc.Node("cloud0", cc.CONFIG["cloud_concurrency"]),
                MetricsCollector(),
                cc.CURRENT_EMBEDDING_DIM,
            ),
            "B5_CNGreedy": lambda: CNGreedyScheduler(
                cc.EdgeNodeBatching(
                    cc.CONFIG["edge_node_name"],
                    cc.CONFIG["edge_concurrency"],
                    cc.CONFIG["edge_batch_capacity_k"],
                ),
                cc.Node("cloud0", cc.CONFIG["cloud_concurrency"]),
                MetricsCollector(),
                cc.CURRENT_EMBEDDING_DIM,
            ),
            "Proposed_CN_DUCB": lambda: CNDUCBScheduler(
                cc.EdgeNodeBatching(
                    cc.CONFIG["edge_node_name"],
                    cc.CONFIG["edge_concurrency"],
                    cc.CONFIG["edge_batch_capacity_k"],
                ),
                cc.Node("cloud0", cc.CONFIG["cloud_concurrency"]),
                MetricsCollector(),
                cc.CURRENT_EMBEDDING_DIM,
            ),
            # --- NEW BASELINE SCHEDULERS ---
            "B6_Least_Connections": lambda: LeastConnectionsScheduler(
                cc.EdgeNodeBatching(
                    cc.CONFIG["edge_node_name"],
                    cc.CONFIG["edge_concurrency"],
                    cc.CONFIG["edge_batch_capacity_k"],
                ),
                cc.Node("cloud0", cc.CONFIG["cloud_concurrency"]),
                MetricsCollector(),
            ),
            "B7_Historical_Least_Response_Time": lambda: HistoricalLRTScheduler(
                cc.EdgeNodeBatching(
                    cc.CONFIG["edge_node_name"],
                    cc.CONFIG["edge_concurrency"],
                    cc.CONFIG["edge_batch_capacity_k"],
                ),
                cc.Node("cloud0", cc.CONFIG["cloud_concurrency"]),
                MetricsCollector(),
            ),
            "B8_Queue_Length_Based": lambda: QueueLengthBasedScheduler(
                cc.EdgeNodeBatching(
                    cc.CONFIG["edge_node_name"],
                    cc.CONFIG["edge_concurrency"],
                    cc.CONFIG["edge_batch_capacity_k"],
                ),
                cc.Node("cloud0", cc.CONFIG["cloud_concurrency"]),
                MetricsCollector(),
            ),
        }
        # --- End Define Schedulers ---


        # --- Run Simulation for each scheduler within the current epoch ---
        for sch_name, sch_fact_fn in schedulers_to_run.items():
            logging.info(f"\n--- Starting Epoch {epoch + 1}, Scheduler: {sch_name} ---")
            sch_inst = None
            sim_run_start_t = time.time()
            try:
                # Create scheduler instance (which creates its own MetricsCollector)
                sch_inst = sch_fact_fn()
                if embedder_instance is None: raise ValueError("Embedder instance is None")

                summary_data_this_run, timeseries_log_this_run, completed_req_df_this_run = run_simulation(
                    sch_inst, embedder_instance, processed_dataset
                )

                # --- Aggregate results for the current epoch and add epoch/scheduler info ---

                # Summary data (add epoch and scheduler name)
                if summary_data_this_run and "error" not in summary_data_this_run:
                    summary_data_this_run['epoch'] = epoch # Add epoch number
                    summary_data_this_run['scheduler_name'] = sch_name # Add scheduler name
                    all_epochs_sum_list.append(summary_data_this_run) # Append the modified summary dict
                    logging.info(f"Summary for Epoch {epoch + 1}, {sch_name}:")
                    # Log summary data for this scheduler in this epoch
                    for k, v in summary_data_this_run.items():
                         if isinstance(v, (int, float, np.number)) and not isinstance(v, (list, dict, pd.Series, pd.DataFrame)):
                              logging.info(f"  {k}: {v:.4f}")
                         elif isinstance(v, (list, dict)):
                              log_level_current = logging.getLogger().getEffectiveLevel()
                              if log_level_current <= logging.DEBUG:
                                   logging.debug(f"  {k}: {v}")
                              else:
                                   logging.info(f"  {k}: (list/dict, len={len(v) if hasattr(v, '__len__') else '?'})")
                         else: logging.info(f"  {k}: {v}")

                elif summary_data_this_run and "error" in summary_data_this_run:
                    logging.error(f"Error for Epoch {epoch + 1}, {sch_name}: {summary_data_this_run['error']}")
                    error_summary = {'epoch': epoch, 'scheduler_name': sch_name, 'error': summary_data_this_run['error']}
                    all_epochs_sum_list.append(error_summary)
                else:
                    logging.warning(f"No summary data returned for Epoch {epoch + 1}, {sch_name}.")
                    no_data_summary = {'epoch': epoch, 'scheduler_name': sch_name, 'status': 'No data returned'}
                    all_epochs_sum_list.append(no_data_summary)

                # Timeseries data (add epoch and scheduler name to each record)
                if timeseries_log_this_run:
                    for rec in timeseries_log_this_run:
                        rec_c = rec.copy()
                        rec_c["epoch"] = epoch
                        rec_c["scheduler_name"] = sch_name
                        all_epochs_timeseries_records_list.append(rec_c)
                    logging.info(
                        "Collected %s timeseries data points for Epoch %s, %s",
                        len(timeseries_log_this_run),
                        epoch + 1,
                        sch_name,
                    )

                # Completed requests data (add epoch and scheduler name columns)
                if isinstance(completed_req_df_this_run, pd.DataFrame) and not completed_req_df_this_run.empty:
                    completed_df_copy = completed_req_df_this_run.copy()
                    completed_df_copy["epoch"] = epoch
                    completed_df_copy["scheduler_name"] = sch_name
                    all_epochs_completed_requests_dfs_list.append(completed_df_copy)


                sim_run_end_t = time.time()
                logging.info(f"--- Finished Epoch {epoch + 1}, Scheduler {sch_name} (Took {sim_run_end_t - sim_run_start_t:.2f}s) ---")


            except Exception as e:
                sim_run_end_t = time.time()
                logging.error(f"Exception during simulation for Epoch {epoch + 1}, {sch_name} (Took {sim_run_end_t - sim_run_start_t:.2f}s): {e}", exc_info=True)
                error_summary = {'epoch': epoch, 'scheduler_name': sch_name, 'error': f"Runtime error: {str(e)}"}
                all_epochs_sum_list.append(error_summary)
                # No timeseries or completed requests data to add in case of simulation crash

            finally:
                # Ensure the scheduler instance is shut down properly
                if sch_inst:
                    try:
                        sch_inst.shutdown()
                    except Exception as e:
                        logging.error(f"Error during shutdown for scheduler {sch_name} in Epoch {epoch + 1}: {e}", exc_info=True)
                    # Explicitly delete the scheduler instance to help with memory management
                    del sch_inst
                    sch_inst = None

                # Clear CUDA cache and run garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Clean up dataset and embedder after each epoch run
        if 'processed_dataset' in locals() and processed_dataset is not None:
            del processed_dataset
            processed_dataset = None # Explicitly set to None
        if 'embedder_instance' in locals() and embedder_instance is not None:
            # Ensure embedder is shut down if it has a shutdown method
            if hasattr(embedder_instance, 'shutdown') and callable(embedder_instance.shutdown):
                try: embedder_instance.shutdown()
                except Exception as e: logging.error(f"Error during embedder shutdown in Epoch {epoch + 1}: {e}", exc_info=True)
            del embedder_instance
            embedder_instance = None # Explicitly set to None
        gc.collect() # Final garbage collection for the epoch

        logging.info(f"\n====== Finished Epoch {epoch + 1}/{epochs} ======")
        # Optional: Add a small delay between epochs if needed for system stability
        # time.sleep(1) # Uncomment and adjust if needed


    # --- End Epoch Loop ---

    exp_end_t = time.time()
    logging.info(f"--- Experiment Run Finished (Total time: {exp_end_t - exp_start_t:.2f}s across {epochs} epochs) ---")

    # --- Process and Save Aggregated Results ---
    if save_results:
        try:
            # Save Summary Metrics
            if all_epochs_sum_list:
                sum_df = pd.DataFrame(all_epochs_sum_list)
                cols = sum_df.columns.tolist()
                if "epoch" in cols:
                    cols.remove("epoch")
                if "scheduler_name" in cols:
                    cols.remove("scheduler_name")
                cols = ["epoch", "scheduler_name"] + cols
                sum_df = sum_df[cols]

                numeric_cols_sum = sum_df.select_dtypes(include=np.number).columns
                float_cols_to_round = numeric_cols_sum.drop(["epoch"], errors="ignore")
                if not float_cols_to_round.empty:
                    sum_df[float_cols_to_round] = sum_df[float_cols_to_round].round(4)

                summary_filename = (
                    f"exp_summary_batch_addbaseline_{cc.CONFIG.get('llm_name', 'unknown_llm')}"
                    f"_epoch_{cc.CONFIG.get('cnducb_nu', '')}nu_{cc.CONFIG.get('clinucb_lambda_reg', '')}lam.csv"
                )
                sum_df.to_csv(summary_filename, index=False)
                logging.info("Summary metrics saved to %s", summary_filename)
                print(f"\nSummary results saved: {summary_filename}")
                print("\nSummary Results Table (first few rows):")
                print(sum_df.head().to_string())
                if len(sum_df) > 5:
                    print("...")
            else:
                logging.warning("No summary results to save.")
                print("\nNo summary results to save.")

            # Save Timeseries Log
            if all_epochs_timeseries_records_list:
                ts_df = pd.DataFrame(all_epochs_timeseries_records_list)
                cols = ts_df.columns.tolist()
                key_cols = ["epoch", "scheduler_name", "timestamp"]
                for col in key_cols:
                    if col in cols:
                        cols.remove(col)
                cols = key_cols + cols
                ts_df = ts_df[cols]

                numeric_cols_ts = ts_df.select_dtypes(include=np.number).columns
                float_cols_ts_to_round = numeric_cols_ts.drop(["epoch", "timestamp"], errors="ignore")
                if not float_cols_ts_to_round.empty:
                    ts_df[float_cols_ts_to_round] = ts_df[float_cols_ts_to_round].round(4)

                timeseries_filename = (
                    f"exp_timeseries_batch_addbaseline_{cc.CONFIG.get('llm_name', 'unknown_llm')}"
                    f"_epoch_{cc.CONFIG.get('cnducb_nu', '')}nu_{cc.CONFIG.get('clinucb_lambda_reg', '')}lam.csv"
                )
                ts_df.to_csv(timeseries_filename, index=False)
                logging.info("Timeseries data saved to %s", timeseries_filename)
                print(f"Time-series results saved: {timeseries_filename}")
            else:
                logging.info("Timeseries logging is disabled for llm_scheduler_core runs.")
                print("\nTimeseries logging is disabled for the core simulation run.")

            # Save Completed Requests Data
            if all_epochs_completed_requests_dfs_list:
                all_completed_df = pd.concat(all_epochs_completed_requests_dfs_list, ignore_index=True)
                cols = all_completed_df.columns.tolist()
                key_cols = [
                    "epoch",
                    "scheduler_name",
                    "id",
                    "arrival_time",
                    "start_time",
                    "completion_time",
                ]
                for col in key_cols:
                    if col in cols:
                        cols.remove(col)
                cols = key_cols + cols
                all_completed_df = all_completed_df[cols]

                numeric_cols_completed = all_completed_df.select_dtypes(include=np.number).columns
                float_cols_completed_to_round = numeric_cols_completed.drop(["epoch", "id"], errors="ignore")
                if not float_cols_completed_to_round.empty:
                    all_completed_df[float_cols_completed_to_round] = (
                        all_completed_df[float_cols_completed_to_round].round(4)
                    )

                completed_filename = (
                    f"exp_completed_requests_batch_addbaseline_{cc.CONFIG.get('llm_name', 'unknown_llm')}"
                    f"_epoch_{cc.CONFIG.get('cnducb_nu', '')}nu_{cc.CONFIG.get('clinucb_lambda_reg', '')}lam.csv"
                )
                all_completed_df.to_csv(completed_filename, index=False)
                logging.info("Completed requests data saved to %s", completed_filename)
                print(f"Completed requests data saved to {completed_filename}")
            else:
                logging.warning("No completed requests data collected across all schedulers and epochs.")
                print("\nNo completed requests data was collected.")

        except Exception as e:
            logging.error("Error saving results to CSV: %s", e, exc_info=True)
            print(f"\nError saving results to CSV: {e}")


    # Return the aggregated data (optional, mainly for direct execution/testing)
    final_summary_df = pd.DataFrame(all_epochs_sum_list) if all_epochs_sum_list else pd.DataFrame()
    final_ts_df = (
        pd.DataFrame(all_epochs_timeseries_records_list)
        if all_epochs_timeseries_records_list
        else pd.DataFrame()
    )
    final_completed_df_return = (
        pd.concat(all_epochs_completed_requests_dfs_list, ignore_index=True)
        if all_epochs_completed_requests_dfs_list
        else pd.DataFrame()
    )

    return final_summary_df, final_ts_df, final_completed_df_return


# In the __main__ block, update the call to run_experiment to specify the number of epochs.

if __name__ == "__main__":
    print("Executing schedulers_and_simulation_temp_epoch.py directly...")

    # Example configuration overrides for testing
    example_config_overrides = {
        "simulation_duration": 100, # Run a short simulation for testing
        # "request_rate_lambda": 1.0, # Lower rate for testing
        # "max_dataset_samples": 1000, # Limit dataset size for testing
        # "log_level": "DEBUG", # More verbose logging for testing
        # "edge_concurrency": 1,
        # "edge_batch_capacity_k": 4,
        # "cloud_concurrency": 2,
        # "nn_train_interval": 1, # Train more frequently for testing NN schedulers
        # "nn_training_buffer_maxlen": 100, # Smaller buffer for testing
        # "llm_name": "test_llm", # Add a test LLM name
    }

    NUM_EPOCHS_FOR_TEST = 10 # Example: Run for 2 epochs

    # Check if default placeholder paths are still being used and warn
    default_dataset_path = cc.DEFAULT_CONFIG["dataset_path"]
    default_bert_path = cc.DEFAULT_CONFIG["bert_model_name"]

    dataset_placeholder = "<DATASET_PATH_PLACEHOLDER>"
    bert_placeholder = "<BERT_MODEL_PATH_PLACEHOLDER>"

    if default_dataset_path == dataset_placeholder:
        logging.warning(
            "Using placeholder dataset path. Please update 'dataset_path' in DEFAULT_CONFIG or example_config_overrides for your environment."
        )
    if default_bert_path == bert_placeholder:
        logging.warning(
            "Using placeholder BERT model path. Please update 'bert_model_name' in DEFAULT_CONFIG or example_config_overrides for your environment."
        )

    # Run the experiment
    results_summary_df_main, results_timeseries_df_main, results_completed_requests_df_main = run_experiment(
        experiment_config_override=example_config_overrides,
        epochs=NUM_EPOCHS_FOR_TEST
    )

    # Print summary results (now using the returned DataFrames)
    print("\n--- Direct Run Simulation Summary ---")
    if not results_summary_df_main.empty:
        print("\nResults Table (Summary):")
        try:
            print(results_summary_df_main.to_string())
        except Exception as e_print:
             print(f"Error printing Summary DataFrame: {e_print}.")
             print("Raw summary data might be available in the saved CSV.")
    else:
        print("Summary DataFrame is empty.")


    # Print sample timeseries records
    if not results_timeseries_df_main.empty:
        print("\n--- Sample Timeseries Data ---")
        print(results_timeseries_df_main.head().to_string())
        if len(results_timeseries_df_main) > 5:
            print("...")
    else:
        print("Timeseries logging is disabled for the core simulation run.")


    # Print sample completed requests records
    if not results_completed_requests_df_main.empty:
        print("\n--- Sample Completed Requests Data ---")
        print(results_completed_requests_df_main.head().to_string())
        if len(results_completed_requests_df_main) > 5:
             print("...")
    else:
        print("Completed Requests DataFrame is empty.")

    print("\n--- Direct Run Finished ---")