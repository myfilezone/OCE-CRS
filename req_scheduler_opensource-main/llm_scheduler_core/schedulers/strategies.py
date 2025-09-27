# --- Imports ---
import os
import time
import random
import heapq
import logging
import random
import threading  # For locks in some schedulers
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import llm_scheduler_core as cc
from llm_scheduler_core import (
    AsyncTrainer,
    EdgeNodeBatching,
    Node,
    TimeEstimatorNN,
    TrainingBuffer,
    calculate_individual_processing_time,
    scale_time_for_training,
    unscale_time_from_prediction,
)
from llm_scheduler_core.request_model import Request


# --- Base Scheduler Class ---
class BaseScheduler:
    def __init__(self, scheduler_name, edge_node, cloud_node, metrics_collector):
        self.scheduler_name = scheduler_name
        self.edge_node = edge_node # Instance of EdgeNodeBatching
        self.cloud_node = cloud_node # Instance of Node (for individual cloud requests)
        self.metrics_collector = metrics_collector
        self.waiting_queue = deque() # Using deque for FCFS-like addition, can be overridden
        logging.info(f"Scheduler '{self.scheduler_name}' initialized with Edge: {edge_node.name}, Cloud: {cloud_node.name}")

    def add_request(self, request):
        """Adds a request to the scheduler's waiting queue."""
        # Use cc.current_time for global simulation time
        logging.debug(f"T={cc.current_time:.4f}: {self.scheduler_name} added Req {request.id} to queue. Queue size: {len(self.waiting_queue)}")
        self.waiting_queue.append(request)


    def schedule(self, current_simulation_time):
        """
        The core scheduling logic. Decides which requests go to edge (batched)
        and which go to cloud (individual).
        This method must be implemented by subclasses.
        It should attempt to schedule requests and then call self.metrics_collector.record_decision_point
        with the outcome and relevant queue state *before* removal.

        Returns:
             bool: True if any requests were successfully scheduled in this cycle, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement the schedule method.")

    def get_feedback(self, completed_request_or_batch_info, is_batch_feedback):
        """
        Processes feedback for completed requests/batches.
        Used by learning-based schedulers to update their models.
        Args:
            completed_request_or_batch_info: Either a single Request object (for cloud)
                                             or a batch_info dict (from EdgeNodeBatching.release_slot).
            is_batch_feedback (bool): True if feedback is for an edge batch, False for individual cloud request.
        """
        pass # Default implementation does nothing


    def shutdown(self):
        """Performs any cleanup operations for the scheduler (e.g., shutting down trainers)."""
        logging.info(f"Shutting down scheduler: {self.scheduler_name}")
        pass # Default implementation does nothing

    def _remove_requests_from_queue(self, requests_to_remove_ids):
        """Helper to remove requests from a deque or heapq based waiting_queue by ID."""
        if isinstance(self.waiting_queue, deque):
            new_queue = deque()
            for req in self.waiting_queue:
                if req.id not in requests_to_remove_ids:
                    new_queue.append(req)
            self.waiting_queue = new_queue
            # logging.debug(f"{self.scheduler_name}: Removed {len(requests_to_remove_ids)} requests from deque. New size: {len(self.waiting_queue)}")
        elif isinstance(self.waiting_queue, list): # Assume list means heapq base
             # Reconstruct the heap by keeping only requests not in the remove list
            new_queue = [req for req in self.waiting_queue if req.id not in requests_to_remove_ids]
            heapq.heapify(new_queue) # Re-heapify the list
            self.waiting_queue = new_queue
            # logging.debug(f"{self.scheduler_name}: Removed {len(requests_to_remove_ids)} requests from heapq. New size: {len(self.waiting_queue)}")
        else:
            logging.warning(f"{self.scheduler_name}: Unknown waiting_queue type {type(self.waiting_queue)}. Cannot remove requests by ID.")
            # If the type is unknown, a full queue rebuild or alternative removal might be needed.
            # For now, we just log and do nothing. This should not happen with the current scheduler types.


# --- Simple Schedulers (Non-Learning, FCFS-based for Edge/Cloud assignment) ---
class EdgeCloudFCFSScheduler(BaseScheduler):
    """B3: Edge-Cloud FCFS. Tries Edge (batch FCFS), then Cloud (individual FCFS)."""
    def __init__(self, edge_node, cloud_node, metrics_collector):
        super().__init__("B3_Edge_Cloud_FCFS", edge_node, cloud_node, metrics_collector)
        # Use deque as the waiting_queue initialized in BaseScheduler

    def schedule(self, current_simulation_time):
        scheduled_something = False

        scheduled_edge_batch_requests = []
        scheduled_cloud_requests = []
        requests_to_remove_ids = set()


        # 1. Try to schedule an edge batch (FCFS)
        if self.edge_node.is_available() and self.waiting_queue:
            num_to_take_for_batch = min(len(self.waiting_queue), self.edge_node.batch_capacity_k)
            if num_to_take_for_batch > 0:
                # Take requests from the front without removing yet, in case assignment fails
                edge_batch_to_attempt = [self.waiting_queue[i] for i in range(num_to_take_for_batch)]

                assign_result = self.edge_node.assign_batch(edge_batch_to_attempt, current_simulation_time)

                if assign_result:
                    # Edge batch assignment successful
                    batch_internal_id, batch_completion_time = assign_result
                    cc.add_event(batch_completion_time, "edge_batch_completion", batch_internal_id, self.scheduler_name)
                    scheduled_something = True
                    scheduled_edge_batch_requests = edge_batch_to_attempt
                    # Mark these requests for removal from the queue later
                    requests_to_remove_ids.update(req.id for req in edge_batch_to_attempt)


        # 2. Try to schedule cloud requests (FCFS) if no edge batch was scheduled or if cloud is available
        # and there are still requests in the queue not scheduled to edge.
        # We only consider requests *not* already assigned to the edge in this cycle.
        remaining_queue_for_cloud_consideration = [req for req in self.waiting_queue if req.id not in requests_to_remove_ids]

        while self.cloud_node.is_available() and remaining_queue_for_cloud_consideration:
             # Take the first available request for cloud consideration
             request_to_assign = remaining_queue_for_cloud_consideration[0]

             processing_time_on_cloud = calculate_individual_processing_time(
                 request_to_assign.base_edge_inference_time, 'cloud'
             )
             final_processing_time_with_rtt = processing_time_on_cloud + cc.CONFIG.get("network_rtt", 0.05) # Use cc.CONFIG

             # Attempt to assign the request to the cloud node
             try:
                 assigned_req_id, individual_completion_time = self.cloud_node.assign_request(
                     request_to_assign, current_simulation_time, final_processing_time_with_rtt
                 )
                 # If assignment is successful, add completion event
                 cc.add_event(individual_completion_time, "cloud_individual_completion", assigned_req_id, self.scheduler_name)
                 scheduled_something = True
                 scheduled_cloud_requests.append(request_to_assign) # Add to the list of scheduled requests
                 requests_to_remove_ids.add(assigned_req_id) # Mark for removal

                 # Remove the assigned request from the temporary consideration list for cloud
                 remaining_queue_for_cloud_consideration.pop(0)

             except RuntimeError:
                 # Cloud Node is suddenly not available - stop assigning more cloud requests
                 logging.warning(f"T={current_simulation_time:.4f}: CloudNode not available after check? Could not assign Req {request_to_assign.id} for {self.scheduler_name}. Stopping cloud assignments.")
                 break


        # Remove all successfully scheduled requests from the actual waiting queue
        if requests_to_remove_ids:
             self._remove_requests_from_queue(requests_to_remove_ids)


        # Record the decision point metrics *after* attempting scheduling and removing from queue
        self.metrics_collector.record_decision_point(
            timestamp=current_simulation_time,
            scheduler_name=self.scheduler_name,
            waiting_queue_len_after_scheduling=len(self.waiting_queue),
            edge_node=self.edge_node, # Pass even if not used by scheduler logic
            cloud_node=self.cloud_node,
            scheduled_edge_batch_requests=scheduled_edge_batch_requests,
            scheduled_cloud_requests=scheduled_cloud_requests
        )

        return scheduled_something # Indicate if any assignments were made


class RandomOffloadScheduler(BaseScheduler):
    """B4 (was B3): Randomly offloads to Edge (batched) or Cloud (individual)."""
    def __init__(self, edge_node, cloud_node, metrics_collector):
        super().__init__("B4_Random_Offload", edge_node, cloud_node, metrics_collector)
        # Use deque as the waiting_queue initialized in BaseScheduler

    def schedule(self, current_simulation_time):
        if not self.waiting_queue: return False # Nothing to schedule

        scheduled_something = False

        scheduled_edge_batch_requests = []
        scheduled_cloud_requests = []
        requests_to_remove_ids = set()

        # Decide randomly whether to attempt Edge or Cloud first in this cycle
        attempt_edge_first = random.random() < 0.5 # 50% chance for edge first

        decision_made_leading_to_assignment = False # Flag to check if any assignment happened in preferred path

        # Attempt scheduling based on random preference
        if attempt_edge_first:
            # Try Edge first
            if self.edge_node.is_available() and self.waiting_queue:
                num_to_consider = min(len(self.waiting_queue), self.edge_node.batch_capacity_k)
                if num_to_consider > 0:
                    # Select random requests from the queue for the batch
                    # Use list() to sample from deque
                    edge_batch_to_attempt = random.sample(list(self.waiting_queue), num_to_consider)

                    assign_result = self.edge_node.assign_batch(edge_batch_to_attempt, current_simulation_time)
                    if assign_result:
                        batch_id, batch_completion_time = assign_result
                        cc.add_event(batch_completion_time, "edge_batch_completion", batch_id, self.scheduler_name)
                        scheduled_something = True
                        scheduled_edge_batch_requests = edge_batch_to_attempt
                        requests_to_remove_ids.update(req.id for req in edge_batch_to_attempt)
                        decision_made_leading_to_assignment = True # Assignment happened

                    else:
                         # Edge was available but assign_batch returned None - unexpected, but log
                         logging.warning(f"T={current_simulation_time:.4f}: EdgeNode available but assign_batch failed for {self.scheduler_name}.")


            # If edge scheduling didn't happen in the preferred path, and Cloud is available, try Cloud (failover)
            # Note: We check self.waiting_queue again because edge scheduling might have removed items
            if not decision_made_leading_to_assignment and self.cloud_node.is_available(): # Check if cloud is available for failover
                # Consider requests not already selected for edge (empty in this branch)
                # Take from the current waiting queue, excluding those already marked for edge removal (empty)
                remaining_queue_for_cloud = [req for req in list(self.waiting_queue) if req.id not in requests_to_remove_ids] # Use list() for sampling
                if remaining_queue_for_cloud:
                    # Select a random request from the remaining queue for cloud
                    request_to_assign = random.choice(remaining_queue_for_cloud)

                    processing_time_on_cloud = calculate_individual_processing_time(
                        request_to_assign.base_edge_inference_time, 'cloud'
                    )
                    final_processing_time_with_rtt = processing_time_on_cloud + cc.CONFIG.get("network_rtt", 0.05) # Use cc.CONFIG

                    try:
                        assigned_id, completion_time = self.cloud_node.assign_request(request_to_assign, current_simulation_time, final_processing_time_with_rtt)
                        cc.add_event(completion_time, "cloud_individual_completion", assigned_id, self.scheduler_name)
                        scheduled_something = True
                        scheduled_cloud_requests.append(request_to_assign)
                        requests_to_remove_ids.add(assigned_id)
                        # decision_made_leading_to_assignment = True # Assignment happened in failover path

                    except RuntimeError:
                         logging.warning(f"T={current_simulation_time:.4f}: CloudNode not available after check? Could not assign Req {request_to_assign.id} for {self.scheduler_name}.")


        else: # Attempt Cloud first
             if self.cloud_node.is_available() and self.waiting_queue:
                # Select a random request from the queue for cloud
                request_to_assign = random.choice(list(self.waiting_queue)) # Use list() to sample from deque

                processing_time_on_cloud = calculate_individual_processing_time(
                    request_to_assign.base_edge_inference_time, 'cloud'
                )
                final_processing_time_with_rtt = processing_time_on_cloud + cc.CONFIG.get("network_rtt", 0.05) # Use cc.CONFIG

                try:
                    assigned_id, completion_time = self.cloud_node.assign_request(request_to_assign, current_simulation_time, final_processing_time_with_rtt)
                    cc.add_event(completion_time, "cloud_individual_completion", assigned_id, self.scheduler_name)
                    scheduled_something = True
                    scheduled_cloud_requests.append(request_to_assign)
                    requests_to_remove_ids.add(assigned_id)
                    decision_made_leading_to_assignment = True # Assignment happened

                except RuntimeError:
                    logging.warning(f"T={current_simulation_time:.4f}: CloudNode not available after check? Could not assign Req {request_to_assign.id} for {self.scheduler_name}.")

             # If cloud scheduling didn't happen in the preferred path, and Edge is available, try Edge (failover)
             # Note: We check self.waiting_queue again because cloud scheduling might have removed items
             if not decision_made_leading_to_assignment and self.edge_node.is_available(): # Check if edge is available for failover
                 # Consider requests not already selected for cloud
                 remaining_queue_for_edge = [req for req in list(self.waiting_queue) if req.id not in requests_to_remove_ids] # Use list() for sampling
                 if remaining_queue_for_edge:
                     num_to_consider = min(len(remaining_queue_for_edge), self.edge_node.batch_capacity_k)
                     if num_to_consider > 0:
                         # Select random requests from the remaining queue for the batch
                         edge_batch_to_attempt = random.sample(remaining_queue_for_edge, num_to_consider)

                         assign_result = self.edge_node.assign_batch(edge_batch_to_attempt, current_simulation_time)
                         if assign_result:
                             batch_id, batch_completion_time = assign_result
                             cc.add_event(batch_completion_time, "edge_batch_completion", batch_id, self.scheduler_name)
                             scheduled_something = True
                             scheduled_edge_batch_requests = edge_batch_to_attempt
                             requests_to_remove_ids.update(req.id for req in edge_batch_to_attempt)
                             # decision_made_leading_to_assignment = True # Assignment happened in failover path

                         else:
                            logging.warning(f"T={current_simulation_time:.4f}: EdgeNode available but assign_batch failed for {self.scheduler_name}.")


        # Remove all successfully scheduled requests from the actual waiting queue
        if requests_to_remove_ids:
             self._remove_requests_from_queue(requests_to_remove_ids)

        # Record the decision point metrics *after* attempting scheduling and removing from queue
        self.metrics_collector.record_decision_point(
            timestamp=current_simulation_time,
            scheduler_name=self.scheduler_name,
            waiting_queue_len_after_scheduling=len(self.waiting_queue),
            edge_node=self.edge_node,
            cloud_node=self.cloud_node,
            scheduled_edge_batch_requests=scheduled_edge_batch_requests,
            scheduled_cloud_requests=scheduled_cloud_requests
        )

        return scheduled_something # Indicate if any assignments were made


class RoundRobinOffloadScheduler(BaseScheduler):
    """B5 (was B4): Round-robin between Edge (batch) or Cloud (individual)."""
    def __init__(self, edge_node, cloud_node, metrics_collector):
        super().__init__("B5_Round_Robin_Offload", edge_node, cloud_node, metrics_collector)
        self.next_target_is_edge = True # State variable for round-robin

    def schedule(self, current_simulation_time):
        if not self.waiting_queue: return False # Nothing to schedule

        scheduled_something = False

        scheduled_edge_batch_requests = []
        scheduled_cloud_requests = []
        requests_to_remove_ids = set()

        decision_made_leading_to_assignment = False # Flag to check if any assignment happened in preferred path


        # Attempt scheduling based on round-robin state
        if self.next_target_is_edge:
            # Try Edge batch
            if self.edge_node.is_available() and self.waiting_queue:
                num_for_batch = min(len(self.waiting_queue), self.edge_node.batch_capacity_k)
                if num_for_batch > 0:
                    # Take requests from the front (FCFS-like within RR slot)
                    edge_batch_to_assign = [self.waiting_queue[i] for i in range(num_for_batch)] # Peek without popping

                    assign_result = self.edge_node.assign_batch(edge_batch_to_assign, current_simulation_time)
                    if assign_result:
                        batch_id, comp_time = assign_result
                        cc.add_event(comp_time, "edge_batch_completion", batch_id, self.scheduler_name)
                        scheduled_something = True
                        scheduled_edge_batch_requests = edge_batch_to_assign
                        requests_to_remove_ids.update(req.id for req in edge_batch_to_assign)
                        self.next_target_is_edge = False # Switch target for next cycle
                        decision_made_leading_to_assignment = True # Assignment happened
                    else:
                        logging.warning(f"T={current_simulation_time:.4f}: EdgeNode available but assign_batch failed for {self.scheduler_name}.")

            # If Edge couldn't be scheduled in preferred path, and Cloud is available, try Cloud (failover)
            if not decision_made_leading_to_assignment and self.cloud_node.is_available() and self.waiting_queue: # Check queue again
                 # Consider requests not already selected for edge (empty in this branch)
                 remaining_queue_for_cloud = [req for req in list(self.waiting_queue) if req.id not in requests_to_remove_ids] # Use list()
                 if remaining_queue_for_cloud:
                      # Take the first available request for cloud (FCFS)
                      request_to_assign = remaining_queue_for_cloud[0] # Peek at the front

                      processing_time_on_cloud = calculate_individual_processing_time(request_to_assign.base_edge_inference_time, 'cloud')
                      final_processing_time_with_rtt = processing_time_on_cloud + cc.CONFIG.get("network_rtt", 0.05) # Use cc.CONFIG

                      try:
                         assigned_id, completion_time = self.cloud_node.assign_request(request_to_assign, current_simulation_time, final_processing_time_with_rtt)
                         cc.add_event(completion_time, "cloud_individual_completion", assigned_id, self.scheduler_name)
                         scheduled_something = True
                         scheduled_cloud_requests.append(request_to_assign)
                         requests_to_remove_ids.add(assigned_id)
                         self.next_target_is_edge = True # Switch target for next cycle (cloud failover still switches)
                         # decision_made_leading_to_assignment = True # Assignment happened in failover path

                      except RuntimeError:
                          logging.warning(f"T={current_simulation_time:.4f}: CloudNode not available after check? Could not assign Req {request_to_assign.id} for {self.scheduler_name}. Stopping cloud assignments.")


        else: # self.next_target_is_cloud (attempt Cloud first)
            if self.cloud_node.is_available() and self.waiting_queue:
                # Take the first request from the queue (FCFS)
                request_to_assign = self.waiting_queue[0] # Peek at the front

                processing_time_on_cloud = calculate_individual_processing_time(request_to_assign.base_edge_inference_time, 'cloud')
                final_processing_time_with_rtt = processing_time_on_cloud + cc.CONFIG.get("network_rtt", 0.05) # Use cc.CONFIG

                try:
                    assigned_id, completion_time = self.cloud_node.assign_request(request_to_assign, current_simulation_time, final_processing_time_with_rtt)
                    cc.add_event(completion_time, "cloud_individual_completion", assigned_id, self.scheduler_name)
                    scheduled_something = True
                    scheduled_cloud_requests.append(request_to_assign)
                    requests_to_remove_ids.add(assigned_id)
                    self.next_target_is_edge = True # Switch target for next cycle
                    decision_made_leading_to_assignment = True # Assignment happened

                except RuntimeError:
                    logging.warning(f"T={current_simulation_time:.4f}: CloudNode not available after check? Could not assign Req {request_to_assign.id} for {self.scheduler_name}.")


            # If Cloud couldn't be scheduled in preferred path, and Edge is available, try Edge (failover)
            if not decision_made_leading_to_assignment and self.edge_node.is_available() and self.waiting_queue: # Check queue again
                # Consider requests not already selected for cloud
                 remaining_queue_for_edge = [req for req in list(self.waiting_queue) if req.id not in requests_to_remove_ids] # Use list()
                 if remaining_queue_for_edge:
                      num_for_batch = min(len(remaining_queue_for_edge), self.edge_node.batch_capacity_k)
                      if num_for_batch > 0:
                         # Take requests from the front (FCFS-like within RR slot)
                         edge_batch_to_assign = [remaining_queue_for_edge[i] for i in range(num_for_batch)] # Peek

                         assign_result = self.edge_node.assign_batch(edge_batch_to_assign, current_simulation_time)
                         if assign_result:
                             batch_id, comp_time = assign_result
                             cc.add_event(comp_time, "edge_batch_completion", batch_id, self.scheduler_name)
                             scheduled_something = True
                             scheduled_edge_batch_requests = edge_batch_to_assign
                             requests_to_remove_ids.update(req.id for req in edge_batch_to_assign)
                             self.next_target_is_edge = False # Switch target for next cycle (edge failover still switches)
                             # decision_made_leading_to_assignment = True # Assignment happened in failover path

                         else:
                            logging.warning(f"T={current_simulation_time:.4f}: EdgeNode available but assign_batch failed for {self.scheduler_name}.")


        # Remove all successfully scheduled requests from the actual waiting queue
        if requests_to_remove_ids:
             self._remove_requests_from_queue(requests_to_remove_ids)

        # Record the decision point metrics *after* attempting scheduling and removing from queue
        self.metrics_collector.record_decision_point(
            timestamp=current_simulation_time,
            scheduler_name=self.scheduler_name,
            waiting_queue_len_after_scheduling=len(self.waiting_queue),
            edge_node=self.edge_node,
            cloud_node=self.cloud_node,
            scheduled_edge_batch_requests=scheduled_edge_batch_requests,
            scheduled_cloud_requests=scheduled_cloud_requests
        )

        return scheduled_something # Indicate if any assignments were made

class PredictionBasedScheduler(BaseScheduler):
    def __init__(self, scheduler_name, edge_node, cloud_node, metrics_collector,
                 nn_model_class, nn_optimizer_class, nn_criterion_class, async_trainer_class,
                 embedding_dim_for_nn):
        super().__init__(scheduler_name, edge_node, cloud_node, metrics_collector)
        # Use heapq for the waiting queue for priority queue behavior (SJF/UCB)
        self.waiting_queue = [] # Initialize as a list, heapq functions will operate on it

        # Use cc.DEVICE and cc.CONFIG for NN components
        self.nn_model = nn_model_class(embedding_dim_for_nn, cc.CONFIG["nn_hidden_layers"]).to(cc.DEVICE)
        self.nn_optimizer = nn_optimizer_class(self.nn_model.parameters(), lr=cc.CONFIG["nn_learning_rate"])
        self.nn_criterion = nn_criterion_class()
        
        # Calculate num_nn_params directly from the model
        self.num_nn_params = sum(p.numel() for p in self.nn_model.parameters() if p.requires_grad)
        logging.info(f"NN Model has {self.num_nn_params} trainable parameters (calculated in scheduler init).")

        self.trainer = async_trainer_class(
            self.nn_model, self.nn_optimizer, self.nn_criterion,
            TrainingBuffer(maxlen=cc.CONFIG.get("nn_training_buffer_maxlen", 10000)),
            cc.CONFIG["nn_batch_size"], cc.CONFIG["nn_train_interval"]
        )

    def add_request(self, request):
        """Adds a request to the waiting queue, predicting its time first."""
        # Predict processing time using the trainer (prediction is in scaled/offset domain)
        predicted_time_scaled_offset = self.trainer.predict_time_scaled_domain(request.embedding)

        # Store the scaled prediction in the request object for training feedback
        request.base_predicted_processing_time_scaled = predicted_time_scaled_offset

        # Unscale the prediction back to the original time scale (seconds) for scheduling decisions
        predicted_time_original_scale = unscale_time_from_prediction(predicted_time_scaled_offset)

        # Store the prediction in the request object for use in scheduling logic (__lt__)
        request.predicted_processing_time = predicted_time_original_scale

        # Add the request to the priority queue (heapq)
        heapq.heappush(self.waiting_queue, request)

        # logging.debug(f"T={cc.current_time:.4f}: {self.scheduler_name} added Req {request.id} to queue with predicted time {predicted_time_original_scale:.4f}. Queue size: {len(self.waiting_queue)}")


    def get_feedback(self, completed_request_or_batch_info, is_batch_feedback):
        """
        Processes feedback from completed requests/batches to update the time estimator.
        """
        # The core logic for adding feedback to the buffer and triggering training
        # is handled within the trainer itself. We just need to provide the necessary data.
        if is_batch_feedback:
            batch_info = completed_request_or_batch_info
            # For a batch, the actual time experienced by all requests is the batch processing time
            actual_batch_processing_time = batch_info['requests'][0].simulated_processing_time # All in batch have the same sim time

            # Add feedback for each request in the batch using the batch time
            for req in batch_info['requests']:
                # Use the actual batch processing time as the 'actual_time' for training feedback
                scaled_time_for_nn_training = scale_time_for_training(actual_batch_processing_time)
                # Add embedding and scaled actual time to the trainer's buffer
                self.trainer.buffer.add(req.embedding, scaled_time_for_nn_training)

            # Trigger training after adding feedback for the batch
            self.trainer.trigger_train()

        else: # Individual cloud request feedback
            request = completed_request_or_batch_info
            # For cloud requests, the actual time experienced is its simulated processing time (inc. RTT)
            actual_individual_processing_time = request.simulated_processing_time

            # Use the actual individual processing time as the 'actual_time' for training feedback
            scaled_time_for_nn_training = scale_time_for_training(actual_individual_processing_time)
            # Add embedding and scaled actual time to the trainer's buffer
            self.trainer.buffer.add(request.embedding, scaled_time_for_nn_training)

            # Trigger training after adding feedback for the individual request
            self.trainer.trigger_train()


    def shutdown(self):
        """Shuts down the asynchronous trainer."""
        super().shutdown() # Call base shutdown (does nothing by default)
        logging.info(f"PredictionBasedScheduler {self.scheduler_name} shutting down trainer.")
        if self.trainer:
            self.trainer.shutdown()
        logging.info(f"PredictionBasedScheduler {self.scheduler_name} shutdown complete.")


    def _remove_requests_from_heap(self, requests_to_remove_ids):
        """Helper to remove requests from a heapq-based waiting_queue by ID."""
        # Create a new list containing requests whose IDs are NOT in the remove set
        new_queue = [req for req in self.waiting_queue if req.id not in requests_to_remove_ids]
        # Re-heapify the new list
        heapq.heapify(new_queue)
        # Replace the old queue with the new one
        self.waiting_queue = new_queue
        # logging.debug(f"{self.scheduler_name}: Removed {len(requests_to_remove_ids)} requests from heapq. New size: {len(self.waiting_queue)}")


class CooperativeSchedulerBase(PredictionBasedScheduler):
    """Base class for schedulers that can offload to either Edge (batched) or Cloud (individual)."""
    def __init__(self, scheduler_name, edge_node, cloud_node, metrics_collector,
                 nn_model_class, nn_optimizer_class, nn_criterion_class, async_trainer_class,
                 embedding_dim_for_nn):
        super().__init__(scheduler_name, edge_node, cloud_node, metrics_collector,
                         nn_model_class, nn_optimizer_class, nn_criterion_class, async_trainer_class, embedding_dim_for_nn)
        # Uses heapq from PredictionBasedScheduler

    def _schedule_cooperative(self, current_simulation_time,
                              select_edge_batch_fn, select_cloud_requests_fn,
                              candidate_pool_for_decision):
        """
        Helper method for cooperative scheduling logic.
        Args:
            current_simulation_time (float): Current simulation time.
            select_edge_batch_fn (callable): Function to select candidates for the edge batch from the candidate pool.
                                             Signature: select_edge_batch_fn(candidate_pool, batch_capacity_k) -> list[Request]
            select_cloud_requests_fn (callable): Function to select candidates for cloud processing from the remaining pool.
                                                Signature: select_cloud_requests_fn(remaining_candidates, cloud_concurrency) -> list[Request]
            candidate_pool_for_decision (list[Request]): The list of requests considered for scheduling in this cycle (e.g., top M from queue).
        Returns:
             bool: True if any requests were successfully scheduled, False otherwise.
        """
        scheduled_something = False

        scheduled_edge_batch_requests = []
        scheduled_cloud_requests = []
        requests_to_remove_ids = set()


        if not candidate_pool_for_decision:
            # If no candidates were selected from the queue, nothing can be scheduled
             # Record decision point with no scheduled tasks and current queue state (which might be empty or have items not in the empty candidate pool)
             self.metrics_collector.record_decision_point(
                 timestamp=current_simulation_time,
                 scheduler_name=self.scheduler_name,
                 waiting_queue_len_after_scheduling=len(self.waiting_queue), # Queue length is unchanged if candidates was empty
                 edge_node=self.edge_node,
                 cloud_node=self.cloud_node,
                 scheduled_edge_batch_requests=[],
                 scheduled_cloud_requests=[]
             )
             return False # Nothing to schedule


        # 1. Attempt to schedule an edge batch from the candidate pool
        edge_batch_to_attempt = []
        if self.edge_node.is_available():
            # Use the provided function to select edge batch candidates from the pool
            # Pass a copy of the candidate pool to the selection function if it modifies the list
            edge_batch_to_attempt = select_edge_batch_fn(list(candidate_pool_for_decision), self.edge_node.batch_capacity_k)

            if edge_batch_to_attempt:
                assign_result = self.edge_node.assign_batch(edge_batch_to_attempt, current_simulation_time)

                if assign_result:
                    # Edge batch assignment successful
                    batch_internal_id, batch_completion_time = assign_result
                    cc.add_event(batch_completion_time, "edge_batch_completion", batch_internal_id, self.scheduler_name)
                    scheduled_something = True
                    scheduled_edge_batch_requests = edge_batch_to_attempt
                    requests_to_remove_ids.update(req.id for req in edge_batch_to_attempt)

                else:
                    logging.warning(f"T={current_simulation_time:.4f}: EdgeNode available but assign_batch failed for {self.scheduler_name}.")


        # 2. Attempt to schedule cloud requests from the remaining candidates in the pool
        # Consider requests in the candidate pool that were NOT selected for the edge batch
        # Create a copy of the candidate pool and remove items assigned to edge
        remaining_cands_for_cloud = [req for req in candidate_pool_for_decision if req.id not in requests_to_remove_ids]

        if remaining_cands_for_cloud and self.cloud_node.is_available():
            # Use the provided function to select cloud requests from the remaining candidates
            # Pass a copy if select_cloud_requests_fn modifies the list
            cloud_reqs_to_attempt = select_cloud_requests_fn(list(remaining_cands_for_cloud), self.cloud_node.concurrency)


            for request_to_assign in cloud_reqs_to_attempt:
                # Check cloud availability again before assigning each request
                if self.cloud_node.is_available():
                    processing_time_on_cloud = calculate_individual_processing_time(
                        request_to_assign.base_edge_inference_time, 'cloud'
                    )
                    final_processing_time_with_rtt = processing_time_on_cloud + cc.CONFIG.get("network_rtt", 0.05) # Use cc.CONFIG

                    try:
                        assigned_id, completion_time = self.cloud_node.assign_request(request_to_assign, current_simulation_time, final_processing_time_with_rtt)
                        cc.add_event(completion_time, "cloud_individual_completion", assigned_id, self.scheduler_name)
                        scheduled_something = True
                        scheduled_cloud_requests.append(request_to_assign)
                        requests_to_remove_ids.add(assigned_id)
                    except RuntimeError:
                         # Cloud Node became unavailable - stop assigning more cloud requests
                         logging.warning(f"T={current_simulation_time:.4f}: CloudNode not available after check? Could not assign Req {request_to_assign.id} for {self.scheduler_name}. Stopping cloud assignments.")
                         break # Stop the loop over cloud candidates
                else:
                    # Cloud is not available, stop trying to schedule cloud requests
                    break # Stop the loop over cloud candidates


        # Remove all successfully scheduled requests from the actual waiting queue (heapq)
        if requests_to_remove_ids:
             self._remove_requests_from_heap(requests_to_remove_ids)


        # Record the decision point metrics *after* attempting scheduling and removing from queue
        self.metrics_collector.record_decision_point(
            timestamp=current_simulation_time,
            scheduler_name=self.scheduler_name,
            waiting_queue_len_after_scheduling=len(self.waiting_queue), # Queue length AFTER removal
            edge_node=self.edge_node,
            cloud_node=self.cloud_node,
            scheduled_edge_batch_requests=scheduled_edge_batch_requests,
            scheduled_cloud_requests=scheduled_cloud_requests
        )

        return scheduled_something

class CombinatorialLinUCBScheduler(BaseScheduler): # B8 (was B?)
    """B8: Combinatorial LinUCB with Diagonal Covariance for Edge batch selection.
       This scheduler uses LinUCB to score requests based on their embeddings
       and assigns them to Edge (batch) or Cloud (individual) based on these scores.
       It includes asynchronous parameter updates and detailed metrics recording."""
    def __init__(self, edge_node, cloud_node, metrics_collector, embedding_dim):
        # This scheduler inherits BaseScheduler, not PredictionBasedScheduler.
        # It implements its own scoring (LinUCB) and scheduling logic.
        super().__init__("B8_CLinUCB_Diag", edge_node, cloud_node, metrics_collector)
        # The waiting queue is managed as a heapq, sorted by UCB score (stored as -predicted_time).
        self.waiting_queue = [] # Initialize as list for heapq

        # LinUCB parameters and configuration
        self.alpha = cc.CONFIG.get("clinucb_alpha", 0.1) # Exploration parameter
        self.lambda_reg = cc.CONFIG.get("clinucb_lambda_reg", 0.1) # Regularization parameter
        self.d = embedding_dim # Embedding dimension

        # Validate embedding dimension
        if self.d is None or not isinstance(self.d, int) or self.d <= 0:
             raise ValueError(f"Embedding dimension (d) must be a positive integer for {self.__class__.__name__}. Got {self.d}.")


        # LinUCB parameters: A is d x d covariance matrix, b is d x 1 reward vector.
        # Using diagonal approximation: A is a diagonal matrix (store diagonal elements as a vector).
        self.A_diag = np.ones(self.d, dtype=np.float32) * self.lambda_reg # Initialize A as lambda_reg * Identity (diagonal)
        self.b = np.zeros(self.d, dtype=np.float32) # Initialize b as zero vector

        # Buffer to store feedback (context, reward) for asynchronous updates
        # Using deque as a thread-safe efficient buffer
        self.feedback_buffer = deque(maxlen=cc.CONFIG.get("linucb_feedback_buffer_max", 5000))

        # Lock for accessing LinUCB parameters (A_diag, b) during updates and predictions
        self.param_lock = threading.Lock()
        # Executor for asynchronous parameter updates. Using max_workers=1 ensures updates are sequential.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.update_futures = [] # To keep track of submitted update tasks


        logging.info(f"{self.scheduler_name} initialized: d={self.d}, alpha={self.alpha}, lambda_reg={self.lambda_reg}, feedback_buffer_max={self.feedback_buffer.maxlen}")


    def _update_linucb_params_async(self):
        """Asynchronous task to perform one batch update of LinUCB parameters from the feedback buffer."""
        pid = os.getpid() # Get process ID for logging in multiprocessing environments
        logging.debug(f"[PID:{pid}] Starting asynchronous LinUCB parameter update.")
        self._process_feedback_buffer_batch()
        logging.debug(f"[PID:{pid}] Finished asynchronous LinUCB parameter update.")

    def _process_feedback_buffer_batch(self):
        """Processes a batch of feedback from the buffer and applies updates."""
        feedback_batch_size = 100 # Define batch size for processing
        processed_count = 0
        # Initialize update vectors for the current batch
        A_d_upd_batch, b_upd_batch = np.zeros(self.d, dtype=np.float32), np.zeros(self.d, dtype=np.float32)
        batch_feedback = [] # Temporarily hold samples for batch processing

        # Extract a batch of feedback samples from the left of the deque
        while processed_count < feedback_batch_size:
            try:
                # Pop from buffer (deque.popleft() is thread-safe)
                batch_feedback.append(self.feedback_buffer.popleft())
                processed_count += 1 # Increment counter for this sample
            except IndexError:
                # The feedback buffer is empty.
                break # Exit the batch extraction loop


        # Process the extracted batch
        for ctx, rwd in batch_feedback:
            try:
                # Ensure context (embedding) is a flat numpy array of float32 with correct dimension
                if not isinstance(ctx, np.ndarray):
                    # Try converting list/tuple to numpy array
                    ctx_v = np.array(ctx, dtype=np.float32).flatten()
                elif ctx.dtype != np.float32:
                    # Convert numpy array to float32 if needed
                    ctx_v = ctx.astype(np.float32).flatten()
                else:
                    # Use the array directly if it's already float32
                    ctx_v = ctx.flatten()

                # Check if the flattened context dimension matches the expected dimension
                if ctx_v.shape[0] != self.d:
                    logging.warning(f"LinUCB feedback context dimension mismatch. Expected {self.d}, got {ctx_v.shape[0]}. Skipping sample.")
                    processed_count -= 1 # Decrement count as this sample is skipped
                    continue # Skip this sample

                # Apply the update rules for diagonal A: A_ii += x_i^2, b_i += r * x_i
                A_d_upd_batch += ctx_v * ctx_v
                b_upd_batch += rwd * ctx_v

            except Exception as e:
                # Catch any other unexpected errors during sample processing
                logging.error(f"Error processing feedback sample in LinUCB update batch: {e}", exc_info=True)
                processed_count -= 1 # Decrement count as this sample caused an error
                continue # Skip this sample and try the next one


        # Apply the accumulated batch updates to the main LinUCB parameters (self.A_diag, self.b) under a lock.
        # This is the critical section where parameters are modified.
        if processed_count > 0: # Only apply updates if at least one sample was processed successfully in this batch
            with self.param_lock:
                self.A_diag += A_d_upd_batch
                self.b += b_upd_batch
                logging.debug(f"Applied LinUCB parameter update batch. Processed {processed_count} samples. Buffer remaining: {len(self.feedback_buffer)}")


    def _trigger_linucb_update(self):
        """Submits an asynchronous update task to the thread executor if the buffer has data and the executor is not overloaded."""
        # Use the parameter lock (or a separate lock) when accessing the list of futures
        with self.param_lock:
             # Clean up completed futures from the list to avoid infinite growth
             self.update_futures = [f for f in self.update_futures if not f.done()]

             # Define a maximum number of pending update tasks to avoid overwhelming the executor
             max_allowed_futures = 2 # For example, allow current task + one pending task

             # Submit a new update task only if there's data in the buffer and we haven't exceeded the limit of pending tasks
             if len(self.feedback_buffer) > 0 and len(self.update_futures) < max_allowed_futures:
                 try:
                     # Submit the asynchronous update method to the thread executor
                     future = self.executor.submit(self._update_linucb_params_async)
                     self.update_futures.append(future) # Add the new future to the list
                     logging.debug(f"{self.scheduler_name} LinUCB update task submitted. Active futures: {len(self.update_futures)}")
                 except Exception as e:
                      # Log any errors that occur during task submission
                      logging.error(f"{self.scheduler_name} Failed to submit LinUCB update task: {e}", exc_info=True)
             # else:
                  # Log why an update wasn't triggered if helpful for debugging
                  # logging.debug(f"{self.scheduler_name} LinUCB update not triggered. Buffer size: {len(self.feedback_buffer)}, Active futures: {len(self.update_futures)}")


    def get_feedback(self, completed_request_or_batch_info, is_batch_feedback):
        """
        Processes feedback from completed requests/batches and adds it to the LinUCB feedback buffer.
        The 'reward' for LinUCB is defined as the negative of the processing time.
        """
        # Determine the reward based on processing time
        if is_batch_feedback:
            # Feedback is from an edge batch completion
            batch_info = completed_request_or_batch_info
            # The reward for the batch decision is the negative of the batch's total processing time
            # Using simulated_processing_time which includes potential queueing + processing
            batch_reward = -batch_info['requests'][0].simulated_processing_time # Assuming all requests in batch have same simulated_processing_time for the batch

            # For each request in the batch, add a feedback sample: (request_embedding, batch_reward)
            # In combinatorial UCB for batches, each item contributes its context (embedding)
            # with the reward received by the combination (the batch).
            for req in batch_info['requests']:
                if hasattr(req, 'embedding'):
                    self.feedback_buffer.append((req.embedding, batch_reward))
                else:
                    logging.warning(f"Request {req.id} in completed batch is missing embedding. Cannot add feedback sample.")

        else: # Feedback is from an individual cloud request completion
            request = completed_request_or_batch_info
            # The reward for the individual cloud request is the negative of its total time (processing + RTT)
            individual_reward = -request.simulated_processing_time # simulated_processing_time should include RTT for cloud

            # Add a feedback sample: (request_embedding, individual_reward)
            if hasattr(request, 'embedding'):
                self.feedback_buffer.append((request.embedding, individual_reward))
            else:
                 logging.warning(f"Completed individual request {request.id} is missing embedding. Cannot add feedback sample.")


        # After adding feedback to the buffer, trigger an asynchronous update task if needed.
        self._trigger_linucb_update()


    def _predict_ucb_reward(self, emb_np):
        """Calculates the UCB score (predicted reward + uncertainty) for a given request embedding."""
        # Ensure the input embedding is a flat numpy array of float32 with correct dimension.
        # This is the 'context' vector x for the LinUCB model.
        if not isinstance(emb_np, np.ndarray):
            ctx_v = np.array(emb_np, dtype=np.float32).flatten()
        elif emb_np.dtype != np.float32:
            ctx_v = emb_np.astype(np.float32).flatten()
        else:
            ctx_v = emb_np.flatten()

        # Check for dimension mismatch
        if ctx_v.shape[0] != self.d:
            logging.warning(f"{self.scheduler_name} Prediction context dimension mismatch. Expected {self.d}, got {ctx_v.shape[0]}. Returning -inf reward.")
            return -float('inf') # Return a very low reward for dimension mismatch

        # Calculate the mean reward (theta.T * x) and the uncertainty bound (alpha * sqrt(x.T * A_inv * x))
        # Access LinUCB parameters under a lock as they might be updated asynchronously.
        with self.param_lock:
            # LinUCB parameter theta = A_inv * b
            # With diagonal A, A_inv is a diagonal matrix with elements 1/A_ii.
            # theta_i = b_i / A_ii
            # Mean reward = dot(theta, x) = sum(theta_i * x_i) = sum((b_i / A_ii) * x_i)
            # Variance term = x.T * A_inv * x = sum(x_i^2 / A_ii)

            # Add a small value to A_diag elements for numerical stability before division
            A_diag_stable = self.A_diag + 1e-9
            # Defensive check: ensure denominators are positive even after adding epsilon
            A_diag_stable[A_diag_stable <= 0] = 1e-9

            theta = self.b / A_diag_stable # Element-wise division to get theta vector

            mean_reward = np.dot(theta, ctx_v) # Calculate the linear prediction (mean reward)
            variance_term = np.sum(ctx_v * ctx_v / A_diag_stable) # Calculate the variance term (x.T * A_inv * x)
            ucb_bound = self.alpha * np.sqrt(max(0.0, variance_term)) # Ensure variance is non-negative before sqrt


        # The UCB score is the sum of the mean reward and the uncertainty bound.
        # For maximizing reward (minimizing processing time, which is negative reward), we add the bound.
        predicted_ucb_reward = mean_reward + ucb_bound

        # logging.debug(f"T={cc.current_time:.4f}: UCB for Req with embedding: Mean={mean_reward:.4f}, Bound={ucb_bound:.4f}, UCB={predicted_ucb_reward:.4f}")

        return predicted_ucb_reward


    def add_request(self, request):
        """Adds a request to the waiting queue, calculating its initial UCB score."""
        # Calculate the initial UCB score (predicted reward) for this request based on current LinUCB parameters.
        # This score will be used to place it in the priority queue.
        ucb_reward = self._predict_ucb_reward(request.embedding)

        # Store the negative of the UCB reward in predicted_processing_time.
        # The waiting queue is a min-heap based on predicted_processing_time.
        # Storing -UCB_reward ensures that requests with higher UCB rewards (i.e., more desirable)
        # will have a lower negative value and thus bubble up to the top of the min-heap.
        request.predicted_processing_time = -ucb_reward

        # Add the request to the priority queue (heapq).
        heapq.heappush(self.waiting_queue, request)

        # logging.debug(f"T={cc.current_time:.4f}: {self.scheduler_name} added Req {request.id} to queue with initial UCB {ucb_reward:.4f} (predicted_time {-ucb_reward:.4f}). Queue size: {len(self.waiting_queue)}")

    def _schedule_from_candidate_pool(self, current_simulation_time,
                                       select_edge_batch_fn, select_cloud_requests_fn,
                                       candidate_pool_for_decision):
        """
        Helper method for cooperative scheduling logic from a candidate pool for CLinUCB.
        Adapted from CooperativeSchedulerBase._schedule_cooperative.
        This method handles assignment and removal from the main waiting queue.
        """
        scheduled_something = False
        scheduled_edge_batch_requests = []
        scheduled_cloud_requests = []
        requests_to_remove_ids = set()

        if not candidate_pool_for_decision:
             self.metrics_collector.record_decision_point(
                 timestamp=current_simulation_time,
                 scheduler_name=self.scheduler_name,
                 waiting_queue_len_after_scheduling=len(self.waiting_queue),
                 edge_node=self.edge_node,
                 cloud_node=self.cloud_node,
                 scheduled_edge_batch_requests=[],
                 scheduled_cloud_requests=[]
             )
             return False

        edge_batch_to_attempt = []
        if self.edge_node.is_available():
            # Use the provided selector to choose edge-batch candidates from a copy of the pool
            edge_batch_to_attempt = select_edge_batch_fn(list(candidate_pool_for_decision), self.edge_node.batch_capacity_k)

            if edge_batch_to_attempt:
                assign_result = self.edge_node.assign_batch(edge_batch_to_attempt, current_simulation_time)

                if assign_result:
                    batch_internal_id, batch_completion_time = assign_result
                    cc.add_event(batch_completion_time, "edge_batch_completion", batch_internal_id, self.scheduler_name)
                    scheduled_something = True
                    scheduled_edge_batch_requests = edge_batch_to_attempt
                    requests_to_remove_ids.update(req.id for req in edge_batch_to_attempt)
                else:
                    logging.warning(f"T={current_simulation_time:.4f}: EdgeNode available but assign_batch failed for {self.scheduler_name}. Attempted batch size: {len(edge_batch_to_attempt)}. Requests remain in candidate pool.")

        remaining_cands_for_cloud = [req for req in candidate_pool_for_decision if req.id not in requests_to_remove_ids]

        if remaining_cands_for_cloud and self.cloud_node.is_available():
            # Use the provided selector to choose cloud requests from a copy of the remaining candidates
            cloud_reqs_to_attempt = select_cloud_requests_fn(list(remaining_cands_for_cloud), self.cloud_node.concurrency)

            for request_to_assign in cloud_reqs_to_attempt:
                 if self.cloud_node.is_available():
                     processing_time_on_cloud = calculate_individual_processing_time(
                         request_to_assign.base_edge_inference_time, 'cloud'
                     )
                     final_processing_time_with_rtt = processing_time_on_cloud + cc.CONFIG.get("network_rtt", 0.05)

                     try:
                         assigned_id, completion_time = self.cloud_node.assign_request(request_to_assign, current_simulation_time, final_processing_time_with_rtt)
                         cc.add_event(completion_time, "cloud_individual_completion", assigned_id, self.scheduler_name)
                         scheduled_something = True
                         scheduled_cloud_requests.append(request_to_assign)
                         requests_to_remove_ids.add(assigned_id)
                     except RuntimeError:
                         logging.warning(f"T={current_simulation_time:.4f}: CloudNode not available after check? Could not assign Req {request_to_assign.id} for {self.scheduler_name}. Stopping cloud assignments in this cycle.")
                         break
                 else:
                    break


        if requests_to_remove_ids:
             self._remove_requests_from_queue(requests_to_remove_ids)


        self.metrics_collector.record_decision_point(
            timestamp=current_simulation_time,
            scheduler_name=self.scheduler_name,
            waiting_queue_len_after_scheduling=len(self.waiting_queue),
            edge_node=self.edge_node,
            cloud_node=self.cloud_node,
            scheduled_edge_batch_requests=scheduled_edge_batch_requests,
            scheduled_cloud_requests=scheduled_cloud_requests
        )

        return scheduled_something
    
    def _select_ucb_edge_batch(self, cands, k_cap):
        """Select the requests with the highest UCB scores for the edge batch."""
        return sorted(cands, key=lambda r: r.predicted_processing_time)[:k_cap]

    def _select_ucb_cloud_requests(self, rem_cands, c_cap):
        """Select the highest-scoring remaining requests for the cloud."""
        return sorted(rem_cands, key=lambda r: r.predicted_processing_time)[:c_cap]



    def schedule(self, current_simulation_time):
        """
        Scheduling logic for CLinUCB using a candidate pool.
        Re-calculates UCB scores for requests in the candidate pool, selects candidates
        for Edge (batch) and Cloud (individual) based on scores, and schedules them.
        """
        m_factor = cc.CONFIG.get("candidate_pool_size_m_factor", 2)
        cand_pool_size = min(int(m_factor * self.edge_node.batch_capacity_k), len(self.waiting_queue))

        curr_cands_decision = heapq.nsmallest(cand_pool_size, self.waiting_queue)

        # Update UCB scores for the selected candidates using the latest LinUCB parameters.
        for request in curr_cands_decision:
            if hasattr(request, 'embedding'):
                try:
                    ucb_reward = self._predict_ucb_reward(request.embedding)
                    request.predicted_processing_time = -ucb_reward
                except Exception as e:
                    logging.error(f"T={current_simulation_time:.4f}: Error recalculating UCB score for Req {request.id} in candidate pool: {e}", exc_info=True)

        return self._schedule_from_candidate_pool(
            current_simulation_time,
            self._select_ucb_edge_batch,
            self._select_ucb_cloud_requests,
            curr_cands_decision
        )

    def shutdown(self):
        """Performs cleanup, shutting down the thread executor and processing any remaining feedback."""
        super().shutdown() # Call the base class shutdown if needed
        logging.info(f"{self.scheduler_name} shutting down executor. Waiting for active update tasks ({len(self.update_futures)})...")

        # Wait for all submitted update tasks to complete
        for future in list(self.update_futures): # Iterate over a copy in case list is modified
             try:
                  future.result() # Wait for completion (and raise exceptions if any)
             except Exception as e:
                  logging.error(f"Error in completed LinUCB update future: {e}", exc_info=True)

        self.executor.shutdown(wait=True)
        logging.info(f"{self.scheduler_name} executor shut down. Processing remaining feedback buffer synchronously.")

        # Process any feedback that might still be in the buffer synchronously before exiting.
        # This should run in the main simulation thread (or the thread calling shutdown).
        while len(self.feedback_buffer) > 0:
             self._process_feedback_buffer_batch() # Process a batch synchronously

        logging.info(f"{self.scheduler_name} shutdown complete.")


class CNGreedyScheduler(CooperativeSchedulerBase): # B9
    """B9: Combinatorial Epsilon-Greedy Scheduler.
       Uses epsilon-greedy to choose between greedy SJF and random for edge batch.
       Always uses greedy SJF for cloud individuals from remaining candidates.
       Implements v1's strategy of updating predictions only for the candidate pool."""
    def __init__(self, edge_node, cloud_node, metrics_collector, embedding_dim):
        super().__init__("B9_CNGreedy", edge_node, cloud_node, metrics_collector,
                         TimeEstimatorNN, torch.optim.Adam, torch.nn.MSELoss, AsyncTrainer, embedding_dim)
        self.epsilon = cc.CONFIG.get("cngreedy_epsilon", 0.1) # Exploration parameter for epsilon-greedy
        logging.info(f"{self.scheduler_name} initialized with epsilon={self.epsilon}")
        # Uses heapq via PredictionBasedScheduler (base class), sorted by predicted_processing_time (for greedy part)

    def _select_greedy_edge_batch(self, cands, k_cap):
        """Selects the k candidates with the shortest predicted times for the edge batch (greedy)."""
        # This function receives a list copy of the candidate pool from _schedule_cooperative.
        # Sorting by predicted time ensures SJF selection among the provided candidates.
        # Although nsmallest provides a sorted list, explicitly sorting the copy is safe.
        return sorted(cands, key=lambda r: r.predicted_processing_time)[:k_cap]


    def _select_random_edge_batch(self, cands, k_cap):
        """Selects a random batch of size up to k from candidates (exploration)."""
        # Selects a random sample from the provided candidate list.
        if not cands:
            return []
        return random.sample(cands, min(len(cands), k_cap))


    def _select_cloud_requests_sjf(self, rem_cands, c_cap):
        """Selects the c_cap remaining candidates with the shortest predicted times for cloud (always greedy SJF for cloud)."""
        # This function receives a list of candidates remaining after edge selection.
        # It sorts these remaining candidates by predicted time and selects the top c_cap.
        return sorted(rem_cands, key=lambda r: r.predicted_processing_time)[:c_cap]


    def schedule(self, current_simulation_time):
        # Determine the size of the candidate pool (e.g., top M requests).
        # This is the set of requests considered for scheduling in this cycle.
        m_factor = cc.CONFIG.get("candidate_pool_size_m_factor", 2)
        # Calculate the pool size: m_factor * edge batch capacity, but ensure it does not exceed the current queue length.
        cand_pool_size = min(int(m_factor * self.edge_node.batch_capacity_k), len(self.waiting_queue))

        # If the calculated pool size is zero (because the queue is empty), get an empty list.
        # The base class helper _schedule_cooperative handles the empty candidate pool case correctly,
        # including recording metrics for that scenario as per the previous merge task (from v2's logic).
        curr_cands_decision = heapq.nsmallest(cand_pool_size, self.waiting_queue)

        # No explicit check here for empty curr_cands_decision and early return,
        # as the base class helper handles the empty candidate pool argument correctly,
        # including recording metrics for the "no candidates selected" scenario.


        # --- Start: v1's Specific Prediction Update Logic for the Candidate Pool ---
        # Update predictions *only* for the requests within the selected candidate pool.
        # This is the core unique scheduling step from v1 for this scheduler type.
        # It iterates through the curr_cands_decision list (a copy of requests from the heap)
        # and updates the 'predicted_processing_time' attribute of these specific Request objects.
        # Note: This does NOT re-heapify the main waiting queue based on these updates.
        # The re-sorting for greedy/SJF selection happens *within* the _select_..._fn functions
        # which receive a list copy of this updated pool from the base class helper.
        for request in curr_cands_decision:
            # Ensure request has an embedding attribute before attempting prediction and trainer is available.
            if hasattr(request, 'embedding') and self.trainer:
                try:
                    # Use the trainer's prediction method which handles scaled/unscaled conversion.
                    predicted_time_scaled_offset = self.trainer.predict_time_scaled_domain(request.embedding)
                    request.base_predicted_processing_time = predicted_time_scaled_offset # Store scaled value too if needed
                    # Assuming unscale_time_from_prediction is available and correctly handles scaled values
                    request.predicted_processing_time = unscale_time_from_prediction(predicted_time_scaled_offset)
                    # logging.debug(f"T={current_simulation_time:.4f}: Updated prediction for Req {request.id} in candidate pool: {request.predicted_processing_time:.4f}")
                except Exception as e:
                    logging.error(f"T={current_simulation_time:.4f}: Error updating prediction for request {request.id} in candidate pool: {e}", exc_info=True)
                    # Handle prediction failure, e.g., leave previous predicted time or set a default high value.
                    # Leaving the previous value seems reasonable here if prediction fails.
            # else:
                 # Request doesn't have embedding or trainer is not available.
                 # Its predicted_processing_time remains whatever it was (initial or from previous updates).
                 # The selection functions will use this value.
        # --- End: v1's Specific Prediction Update Logic ---


        # --- Start: Epsilon-Greedy Choice (from both v1 and v2) ---
        # Based on the epsilon probability, randomly choose which edge batch selection
        # function will be used by the base class helper in this scheduling cycle.
        if random.random() < self.epsilon:
            edge_sel_fn = self._select_random_edge_batch
            # logging.debug(f"T={current_simulation_time:.4f}: Epsilon-greedy chose RANDOM edge batch selection.")
        else:
            edge_sel_fn = self._select_greedy_edge_batch
            # logging.debug(f"T={current_simulation_time:.4f}: Epsilon-greedy chose GREEDY (SJF) edge batch selection.")
        # --- End: Epsilon-Greedy Choice ---


        # Call the core cooperative scheduling logic implemented in the base class helper method (_schedule_cooperative).
        # This helper method (from the previous task merge) is responsible for:
        # 1. Attempting to schedule an edge batch using the chosen edge_sel_fn and the candidate pool.
        # 2. Attempting to schedule individual cloud requests from the *remaining* candidates
        #    (those in the pool not assigned to edge) using the fixed _select_cloud_requests_sjf function.
        #    The selection functions will receive a list copy of curr_cands_decision (which has updated predictions)
        #    and perform their logic based on predicted_processing_time.
        # 3. Removing any successfully scheduled requests (from both edge and cloud) from the main waiting queue.
        # 4. Recording detailed metrics *after* all scheduling attempts and removal, using the v2-based logic
        #    integrated into the base class (always records, provides final queue length and lists of scheduled requests).
        return self._schedule_cooperative(
            current_simulation_time,
            edge_sel_fn, # Pass the chosen edge selection function (_select_greedy or _select_random)
            self._select_cloud_requests_sjf, # Always use the greedy SJF cloud selection function
            curr_cands_decision # Pass the candidate pool with updated predictions for its members
        )

class CNDUCBScheduler(CooperativeSchedulerBase): # Proposed (was B?)
    """Proposed: Combinatorial Neural DuCB. Selects edge batch and cloud requests based on DUCB scores.
       Inherits CooperativeSchedulerBase for assignment and metrics."""
    def __init__(self, edge_node, cloud_node, metrics_collector, embedding_dim):
        # Initialize the base class (CooperativeSchedulerBase), passing NN related components
        super().__init__("Proposed_CN_DUCB", edge_node, cloud_node, metrics_collector,
                         TimeEstimatorNN, torch.optim.Adam, torch.nn.MSELoss, AsyncTrainer, embedding_dim)
        # Uses heapq from PredictionBasedScheduler (base class), sorted by DUCB score (lowest predicted time including uncertainty)


        # DUCB specific parameters from configuration
        self.lambda_param_ucb = cc.CONFIG.get("clinucb_lambda_reg", 0.1) # Regularization parameter for U matrix update
        self.nu_ducb = cc.CONFIG.get("cnducb_nu", 0.01) # Weighting factor for the uncertainty term in DUCB score

        # Number of trainable parameters in the NN model. This is needed for the dimension of the U matrix.
        # Assume self.trainer (AsyncTrainer) calculates and stores this after model initialization.
        # self.num_nn_params is expected to be available from the base class/trainer.
        # Defensive check in case num_nn_params is not set or is zero.
        if not hasattr(self, 'num_nn_params') or self.num_nn_params is None or self.num_nn_params <= 0:
             # Attempt to calculate if not available, or set to 1 for dummy update if no params
             try:
                 if self.trainer and self.trainer.model:
                      self.num_nn_params = sum(p.numel() for p in self.trainer.model.parameters() if p.requires_grad)
                 else:
                      self.num_nn_params = 1 # Fallback
             except Exception:
                  self.num_nn_params = 1 # Fallback
             if self.num_nn_params <= 0: self.num_nn_params = 1


        # DUCB uses the sum of outer products of gradients to update a covariance matrix (U).
        # With a diagonal approximation, U is a diagonal matrix, and we store its diagonal elements in U_diag.
        # Initialize U_diag as lambda_param_ucb * Identity (diagonal) on the correct device.
        try:
             # Use self.num_nn_params for dimension
             self.U_diag = torch.ones(self.num_nn_params, device=cc.DEVICE) * self.lambda_param_ucb
        except Exception as e:
             logging.error(f"{self.scheduler_name} Error initializing U_diag with {self.num_nn_params} params on {cc.DEVICE}: {e}. Initializing with size 1 on CPU.", exc_info=True)
             self.num_nn_params = 1 # Fallback to size 1 if initialization fails
             self.U_diag = torch.ones(self.num_nn_params, device='cpu') * self.lambda_param_ucb


        self.U_diag_lock = threading.Lock() # Lock for safely accessing self.U_diag from different threads (e.g., trainer update)


        logging.info(f"{self.scheduler_name} initialized: nu_ducb={self.nu_ducb}, lambda_param_ucb={self.lambda_param_ucb}, num_nn_params={self.num_nn_params}, device={cc.DEVICE}")


    def _calculate_ducb_score(self, request):
        """
        Calculates the DUCB score for a given request based on the current NN prediction
        and the estimated uncertainty derived from the gradient and the U matrix.
        A lower DUCB score indicates higher priority (shorter predicted time + lower uncertainty penalty).
        """
        # Get the mean prediction from the NN for the request's embedding (in scaled/offset time domain).
        # The trainer's method handles getting the embedding and running the model.
        mu_s_off = self.trainer.predict_time_scaled_domain(request.embedding)

        # Store the scaled mean prediction (useful for debugging or other purposes)
        # Note: The actual value used for sorting in the heap is the full DUCB score stored in predicted_processing_time.
        request.base_predicted_processing_time_scaled = mu_s_off


        sigma_s_off_sq = 0.0 # Initialize the squared uncertainty term to zero


        # Calculate the gradient of the prediction with respect to the NN parameters.
        # This gradient 'g' is used to estimate the uncertainty for DUCB.
        # Gradient calculation is only relevant if the NN model has trainable parameters.
        if self.num_nn_params > 0 and self.trainer: # Ensure trainer is available and model has params
            try:
                 # get_gradient_of_prediction_scaled_domain returns a flattened tensor
                 # on the correct device (cc.DEVICE) and ensures requires_grad is handled temporarily.
                 g = self.trainer.get_gradient_of_prediction_scaled_domain(request.embedding)

                 # Ensure the obtained gradient is valid and has the expected dimension.
                 if g is not None and torch.is_tensor(g) and g.numel() == self.num_nn_params:
                     # Ensure g is on the correct device for calculations with self.U_diag
                     g = g.to(cc.DEVICE)

                     # Calculate the squared uncertainty term: lambda * nu * g^T * U_inv * g
                     # With diagonal U_diag, U_inv is diagonal with elements 1/U_diag[ii].
                     # g^T * U_inv * g = sum(g_i^2 / U_diag[ii]) - element-wise square and sum.

                     # Add a small value to the diagonal elements for numerical stability before division.
                     U_d_stable = self.U_diag + 1e-9
                     # Defensive check: ensure denominators are positive even after adding epsilon.
                     U_d_stable[U_d_stable <= 0] = 1e-9

                     # Element-wise square of gradient, then element-wise division by U_d_stable, and finally sum.
                     s_sq_elements = g * g / U_d_stable
                     s_sq_s_off_t = self.lambda_param_ucb * self.nu_ducb * torch.sum(s_sq_elements)

                     # Convert the resulting tensor scalar to a Python float item.
                     sigma_s_off_sq = s_sq_s_off_t.item()

                     # Ensure the squared uncertainty is non-negative (should theoretically be, but numerical stability).
                     sigma_s_off_sq = max(0.0, sigma_s_off_sq)

                     # logging.debug(f"Calculated DUCB uncertainty for Req {request.id}: {np.sqrt(sigma_s_off_sq):.4f}")

                 # else:
                     # Gradient calculation failed (e.g., embedding was None) or dimension mismatch.
                     # sigma_s_off_sq remains initialized at 0.0.
                     # logging.warning(f"Gradient calculation failed or dimension mismatch for Req {request.id}. Gradient shape: {g.shape if torch.is_tensor(g) else 'None'}. Expected {self.num_nn_params}. Uncertainty is 0.")

            except Exception as e:
                 # Catch any other unexpected errors during uncertainty calculation.
                 logging.error(f"{self.scheduler_name} Error calculating DUCB uncertainty term for Req {request.id}: {e}", exc_info=True)
                 sigma_s_off_sq = 0.0 # Default to zero uncertainty on error


        # Calculate the DUCB score in the scaled/offset domain.
        # The score combines the mean prediction and the uncertainty.
        # We want lower scores to represent higher priority (shorter predicted time + lower uncertainty).
        # DUCB Score = Predicted Mean Time - Uncertainty Term (Scaled/Offset)
        # sigma_s_off is sqrt(lambda * nu * g.T * U_inv * g).
        ducb_s_s_off = mu_s_off - np.sqrt(sigma_s_off_sq) # Take sqrt to get sigma and subtract


        # Unscale the final DUCB score back to the original time scale (seconds).
        ducb_score_original_scale = unscale_time_from_prediction(ducb_s_s_off)


        # Ensure the resulting score is positive (as it's a proxy for time).
        # Add a small minimum value to prevent zero or negative scores if unscaling results in that.
        # Lower DUCB score means higher priority (shorter predicted time + lower uncertainty = better).
        return max(0.001, ducb_score_original_scale)


    def add_request(self, request):
        """Adds a request to the waiting queue after calculating its DUCB score."""
        # Calculate the DUCB score for this new request using the current model parameters.
        # This score will determine its initial position in the priority queue.
        ducb_score = self._calculate_ducb_score(request)

        # Store the DUCB score in the request's predicted_processing_time attribute.
        # The waiting queue (heapq) is a min-heap, sorting by predicted_processing_time.
        # By storing the DUCB score directly, requests with lower DUCB scores (meaning
        # shorter predicted time combined with lower uncertainty penalty) will have higher
        # priority (bubble up to the top) in the heap.
        request.predicted_processing_time = ducb_score # Store the calculated DUCB score


        # Add the request to the priority queue (heapq).
        heapq.heappush(self.waiting_queue, request)

        # logging.debug(f"T={cc.current_time:.4f}: {self.scheduler_name} added Req {request.id} to queue with DUCB score {ducb_score:.4f}. Queue size: {len(self.waiting_queue)}")


    def _select_ducb_edge_batch(self, cands, k_cap):
        """
        Selects the k candidates with the lowest DUCB scores (highest priority) for the edge batch.
        Also updates the DUCB U matrix based on the gradients of the selected edge batch requests.
        This update step is specific to DUCB and happens upon selecting an action (the edge batch).
        """
        # The candidate pool passed to this function is a list copy from _schedule_cooperative.
        # The requests in this list already have their predicted_processing_time attribute
        # set to their latest DUCB score from the re-scoring loop in the schedule() method.

        # Sort the candidate list by the stored DUCB score (predicted_processing_time)
        # to ensure we select the k requests with the lowest scores (highest priority).
        selected_edge_batch = sorted(cands, key=lambda r: r.predicted_processing_time)[:k_cap]


        # Update the DUCB U matrix based on the gradients of the requests selected for the edge batch.
        # This update step is a core part of the DUCB algorithm's exploration mechanism.
        # This is only relevant if the NN model has trainable parameters.
        if self.num_nn_params > 0 and selected_edge_batch and self.trainer: # Ensure num_params > 0, batch is not empty, and trainer is available
            # Initialize a tensor to accumulate the sum of squared gradients for the batch.
            # Ensure it's on the correct device.
            sum_g_g_T_diag = torch.zeros(self.num_nn_params, device=cc.DEVICE)

            # Iterate through the requests selected for the edge batch.
            for req in selected_edge_batch:
                if hasattr(req, 'embedding'):
                    try:
                         # Get the gradient of the prediction for this request (in scaled/offset domain).
                         # Trainer's method handles device and requires_grad temporarily.
                         g = self.trainer.get_gradient_of_prediction_scaled_domain(req.embedding)

                         # If the gradient is valid and has the expected dimension, add its element-wise square to the sum.
                         if g is not None and torch.is_tensor(g) and g.numel() == self.num_nn_params:
                              # Ensure g is on the same device as sum_g_g_T_diag before the operation
                              g = g.to(cc.DEVICE)
                              sum_g_g_T_diag += g * g # Element-wise square and accumulate

                         # else:
                             # Gradient calculation failed or dimension mismatch for this request.
                             # logging.warning(f"Gradient calculation failed or dimension mismatch for Req {req.id} in selected edge batch. Shape: {g.shape if torch.is_tensor(g) else 'None'}. Expected {self.num_nn_params}.")
                    except Exception as e:
                         logging.error(f"{self.scheduler_name} Error calculating gradient for Req {req.id} in selected edge batch: {e}", exc_info=True)
                         # Continue to the next request even if one fails


            # Update the global U_diag matrix by adding the accumulated sum of squared gradients from the batch.
            # This update must be done under the U_diag_lock for thread safety.
            # Only perform the update if the accumulated sum has non-zero elements (meaning valid gradients were processed).
            if torch.sum(sum_g_g_T_diag) > 0:
                 with self.U_diag_lock:
                      self.U_diag += sum_g_g_T_diag
                      # logging.debug(f"T={cc.current_time:.4f}: Updated DUCB U_diag based on selected edge batch. Sum of sq gradients added: {torch.sum(sum_g_g_T_diag).item():.4f}")
             # else:
                  # logging.debug(f"No non-zero gradients for DUCB U_diag update from selected edge batch ({len(selected_edge_batch)} requests).")

        # else:
             # No trainable parameters, no edge batch selected, or trainer not available. No U_diag update needed/possible.


        # Return the list of requests selected for the edge batch.
        return selected_edge_batch


    def _select_cloud_requests_sjf_ducb(self, rem_cands, c_cap):
        """
        Selects the c_cap remaining candidates with the lowest predicted *mean* times (SJF-like) for cloud.
        Cloud selection in this scheduler is based on the mean prediction component of DUCB,
        not the full DUCB score including the uncertainty term. This is because the DUCB
        exploration mechanism (U matrix update) is primarily tied to the edge batch selection.
        """
        # The candidate list passed here contains requests from the pool that were not selected/assigned to the edge.

        # Re-calculate the mean prediction (scaled/offset, then unscaled) for these remaining candidates.
        # This ensures the prediction is based on the latest NN model parameters if the trainer has updated them.
        for req in rem_cands:
            if hasattr(req, 'embedding') and self.trainer:
                try:
                     # Get only the mean prediction component from the NN.
                     mu_s_off = self.trainer.predict_time_scaled_domain(req.embedding)
                     # Update predicted_processing_time with just the unscaled mean prediction for SJF sorting.
                     req.predicted_processing_time = unscale_time_from_prediction(mu_s_off)
                except Exception as e:
                     logging.error(f"{self.scheduler_name} Error re-calculating mean prediction for Req {req.id} for cloud selection: {e}", exc_info=True)
                     # Leave predicted_processing_time as is if calculation fails.
            # else:
                 # Request is missing embedding or trainer is unavailable.
                 # Predicted time remains as it was (likely its DUCB score from the last re-score).
                 # This might not be ideal for SJF but is a fallback.


        # Select the c_cap candidates with the lowest predicted mean times (standard SJF) from the remaining candidates.
        return sorted(rem_cands, key=lambda r: r.predicted_processing_time)[:c_cap]


    def schedule(self, current_simulation_time):
        """
        Main scheduling logic for CNDUCB using a candidate pool.
        Selects a candidate pool (top M), re-calculates DUCB scores for candidates,
        and calls the base class cooperative scheduling helper.
        """
        m_factor = cc.CONFIG.get("candidate_pool_size_m_factor", 2)
        cand_pool_size = min(int(m_factor * self.edge_node.batch_capacity_k), len(self.waiting_queue))

        # Extract the top-scoring requests from the priority queue for consideration this cycle.
        curr_cands_decision = heapq.nsmallest(cand_pool_size, self.waiting_queue)

        # Refresh DUCB scores for the candidate pool to account for updated model parameters.
        for request in curr_cands_decision:
            if hasattr(request, 'embedding'):
                 try:
                    self._calculate_ducb_score(request)
                 except Exception as e:
                    logging.error(f"T={current_simulation_time:.4f}: Error recalculating DUCB score for Req {request.id} in candidate pool: {e}", exc_info=True)

        return self._schedule_cooperative(
            current_simulation_time,
            self._select_ducb_edge_batch,
            self._select_cloud_requests_sjf_ducb,
            curr_cands_decision
        )

# --- NEW BASELINE SCHEDULERS (originating from the continuous-time scheduler package) ---

class LeastConnectionsScheduler(BaseScheduler): # B11
    """
    Assigns requests to the node (edge or cloud) with the fewest active connections (busy slots).
    For edge, this means the number of currently active *batches*.
    Prioritizes edge in case of a tie or if both are available and have 0 busy slots.
    """
    def __init__(self, edge_node, cloud_node, metrics_collector):
        super().__init__("B11_Least_Connections", edge_node, cloud_node, metrics_collector)
        # Uses the deque from BaseScheduler for FCFS from the global queue
        logging.info(f"{self.scheduler_name} initialized.")

    def schedule(self, current_simulation_time):
        scheduled_something = False
        scheduled_edge_batch_requests = []
        scheduled_cloud_requests = []
        requests_to_remove_ids = set()

        # Try to schedule an edge batch first if edge has fewer busy slots or equal and preferred
        if self.edge_node.is_available() and self.waiting_queue:
            edge_busy_slots = self.edge_node.concurrency - self.edge_node.available_slots
            cloud_busy_slots = self.cloud_node.concurrency - self.cloud_node.available_slots

            # Decide if edge is preferred based on busy slots
            # For Least Connections, we compare the number of *active batches* on edge
            # with the number of *active individual requests* on cloud.
            # If both are available, and edge has fewer or equal busy slots (batches) than cloud (individuals)
            # or if only edge is available: prioritize edge.
            if self.edge_node.is_available() and \
               (not self.cloud_node.is_available() or edge_busy_slots <= cloud_busy_slots): # Prefer edge on tie
                
                num_to_take_for_batch = min(len(self.waiting_queue), self.edge_node.batch_capacity_k)
                if num_to_take_for_batch > 0:
                    # Take requests from the front of the queue (FCFS from waiting_queue)
                    edge_batch_to_attempt = [self.waiting_queue[i] for i in range(num_to_take_for_batch)]

                    assign_result = self.edge_node.assign_batch(edge_batch_to_attempt, current_simulation_time)
                    if assign_result:
                        batch_internal_id, batch_completion_time = assign_result
                        cc.add_event(batch_completion_time, "edge_batch_completion", batch_internal_id, self.scheduler_name)
                        scheduled_something = True
                        scheduled_edge_batch_requests = edge_batch_to_attempt
                        requests_to_remove_ids.update(req.id for req in edge_batch_to_attempt)
                    else:
                        logging.warning(f"T={current_simulation_time:.4f}: EdgeNode available but assign_batch failed for {self.scheduler_name}.")
            
        # Try to schedule cloud requests from the remaining queue
        # This will run if edge was not scheduled or if cloud has fewer busy slots
        # We need to filter the waiting queue to exclude requests already assigned to edge in this cycle.
        remaining_queue_for_cloud_consideration = [req for req in self.waiting_queue if req.id not in requests_to_remove_ids]

        while self.cloud_node.is_available() and remaining_queue_for_cloud_consideration:
            # Re-evaluate node busy slots as they might have changed after edge assignment
            edge_busy_slots = self.edge_node.concurrency - self.edge_node.available_slots
            cloud_busy_slots = self.cloud_node.concurrency - self.cloud_node.available_slots

            # If edge is now less busy than cloud (and available), or cloud is the only option, schedule to cloud
            if self.cloud_node.is_available() and \
               (not self.edge_node.is_available() or cloud_busy_slots < edge_busy_slots): # Cloud preferred if less busy, otherwise edge takes priority
                
                request_to_assign = remaining_queue_for_cloud_consideration[0] # FCFS from remaining queue

                processing_time_on_cloud = calculate_individual_processing_time(
                    request_to_assign.base_edge_inference_time, 'cloud'
                )
                final_processing_time_with_rtt = processing_time_on_cloud + cc.CONFIG.get("network_rtt", 0.05)

                try:
                    assigned_req_id, individual_completion_time = self.cloud_node.assign_request(
                        request_to_assign, current_simulation_time, final_processing_time_with_rtt
                    )
                    cc.add_event(individual_completion_time, "cloud_individual_completion", assigned_req_id, self.scheduler_name)
                    scheduled_something = True
                    scheduled_cloud_requests.append(request_to_assign)
                    requests_to_remove_ids.add(assigned_req_id)
                    remaining_queue_for_cloud_consideration.pop(0) # Remove from temp list
                except RuntimeError:
                    logging.warning(f"T={current_simulation_time:.4f}: CloudNode not available after check? Could not assign Req {request_to_assign.id} for {self.scheduler_name}. Stopping cloud assignments.")
                    break # Stop trying to schedule cloud requests
            else:
                # No more suitable assignments in this cycle
                break

        # Remove all successfully scheduled requests from the actual waiting queue
        if requests_to_remove_ids:
             self._remove_requests_from_queue(requests_to_remove_ids)

        self.metrics_collector.record_decision_point(
            timestamp=current_simulation_time,
            scheduler_name=self.scheduler_name,
            waiting_queue_len_after_scheduling=len(self.waiting_queue),
            edge_node=self.edge_node,
            cloud_node=self.cloud_node,
            scheduled_edge_batch_requests=scheduled_edge_batch_requests,
            scheduled_cloud_requests=scheduled_cloud_requests
        )
        return scheduled_something

class HistoricalLRTScheduler(BaseScheduler): # B12
    """
    Schedules requests to the node (edge or cloud) that is predicted to yield the shortest
    overall job completion time (JCT) based on historical average processing times
    and estimated current queueing delay.
    """
    def __init__(self, edge_node, cloud_node, metrics_collector):
        super().__init__("B12_Historical_Least_Response_Time", edge_node, cloud_node, metrics_collector)
        # Use a min-heap for the waiting queue, requests ordered by projected JCT
        self.waiting_queue = [] # Override deque for min-heap behavior
        logging.info(f"{self.scheduler_name} initialized.")

    def add_request(self, request):
        # When a request arrives, calculate its projected JCT for both nodes
        # based on current historical averages and estimated queueing delays.

        # Use the current_time from the simulation global for queue estimation
        current_time_for_queue_estimate = cc.current_time

        # Estimate queueing delay for each node
        edge_queue_delay = self.edge_node.estimate_queueing_delay(current_time_for_queue_estimate)
        cloud_queue_delay = self.cloud_node.estimate_queueing_delay(current_time_for_queue_estimate)

        # Estimate processing time on edge (batch processing, so use base_edge_inference_time of this request or node's avg?)
        # For fairness, we should project the *batch* time if assigned to edge.
        # But for request-specific routing, we need an estimate for THIS request on that node.
        # Let's use the node's historical average *batch* processing time for edge,
        # and individual request's base_edge_inference_time * cloud_multiplier for cloud.
        # The prompt implies 'Historical Least Response Time' so we should probably use node's average.
        # Edge node: avg_processing_time is average *batch* processing time.
        projected_edge_proc_time = self.edge_node.avg_processing_time 
        
        # Cloud node: avg_processing_time is average *individual* processing time on cloud.
        projected_cloud_proc_time = self.cloud_node.avg_processing_time 


        # Calculate total projected JCT for each node
        # Projected JCT = (Estimated Queueing Delay) + (Node's Historical Average Processing Time) + (RTT if Cloud)
        # For simplicity, we assume an individual request on edge will take `projected_edge_proc_time` if part of an "average" batch.
        # This is a simplification as actual batch time depends on max of requests.
        projected_jct_edge = edge_queue_delay + projected_edge_proc_time
        projected_jct_cloud = cloud_queue_delay + projected_cloud_proc_time + cc.CONFIG.get("network_rtt", 0.05)

        # Store the minimum projected JCT as the 'predicted_processing_time' for heap ordering
        # And also store the preferred node for this request.
        if projected_jct_edge <= projected_jct_cloud:
            request.predicted_processing_time = projected_jct_edge # Re-using this field for the score
            request._preferred_node_for_lrt = 'edge'
        else:
            request.predicted_processing_time = projected_jct_cloud
            request._preferred_node_for_lrt = 'cloud'

        heapq.heappush(self.waiting_queue, request)
        logging.debug(f"Req {request.id} added to Historical LRT heap. Proj JCT Edge: {projected_jct_edge:.4f}, Proj JCT Cloud: {projected_jct_cloud:.4f}. Preferred: {getattr(request, '_preferred_node_for_lrt', 'N/A')}")

    def schedule(self, current_simulation_time):
        scheduled_something = False
        scheduled_edge_batch_requests = []
        scheduled_cloud_requests = []
        requests_to_remove_ids = set()

        # Need to handle edge batching for this scheduler.
        # We process requests from the heap one by one (based on LRT preference),
        # but if edge is chosen, we still need to form a batch.

        # First, try to schedule to the preferred edge node
        if self.edge_node.is_available() and self.waiting_queue:
            # Find requests in the queue that prefer edge and try to form a batch
            edge_preferred_candidates = [req for req in self.waiting_queue if getattr(req, '_preferred_node_for_lrt', 'edge') == 'edge']
            
            if edge_preferred_candidates:
                # Sort edge preferred candidates by their predicted JCT (which is their predicted_processing_time)
                # and take up to batch_capacity_k requests
                edge_batch_to_attempt = sorted(edge_preferred_candidates, key=lambda r: r.predicted_processing_time)[:self.edge_node.batch_capacity_k]

                if edge_batch_to_attempt:
                    assign_result = self.edge_node.assign_batch(edge_batch_to_attempt, current_simulation_time)

                    if assign_result:
                        batch_internal_id, batch_completion_time = assign_result
                        cc.add_event(batch_completion_time, "edge_batch_completion", batch_internal_id, self.scheduler_name)
                        scheduled_something = True
                        scheduled_edge_batch_requests = edge_batch_to_attempt
                        requests_to_remove_ids.update(req.id for req in edge_batch_to_attempt)
                    else:
                        logging.warning(f"T={current_simulation_time:.4f}: EdgeNode available but assign_batch failed for {self.scheduler_name}.")
        
        # Now, handle cloud requests or edge-preferred requests that couldn't be batched
        # or were not chosen for the initial edge batch.
        # We need to extract all requests that are *not* assigned yet (from the initial heap state)
        # and try to assign them to cloud if it's their preferred node and available.
        # To do this correctly, we must iterate the *original* waiting_queue (or its current state after edge assignment).
        
        # Create a temporary list of requests that are still in the queue and not yet scheduled
        remaining_requests_in_queue = [req for req in list(self.waiting_queue) if req.id not in requests_to_remove_ids]
        
        # Sort these remaining requests by their projected JCT to prioritize them for cloud
        remaining_requests_in_queue.sort(key=lambda r: r.predicted_processing_time)

        for request_to_assign in remaining_requests_in_queue:
            if not self.cloud_node.is_available(): # Break if cloud is full
                break

            # If this request was edge-preferred but couldn't be batched, or if it's cloud-preferred
            # Attempt to assign to cloud
            if request_to_assign.id not in requests_to_remove_ids: # Double check not already assigned
                processing_time_on_cloud = calculate_individual_processing_time(
                    request_to_assign.base_edge_inference_time, 'cloud'
                )
                final_processing_time_with_rtt = processing_time_on_cloud + cc.CONFIG.get("network_rtt", 0.05)

                try:
                    assigned_req_id, individual_completion_time = self.cloud_node.assign_request(
                        request_to_assign, current_simulation_time, final_processing_time_with_rtt
                    )
                    cc.add_event(individual_completion_time, "cloud_individual_completion", assigned_req_id, self.scheduler_name)
                    scheduled_something = True
                    scheduled_cloud_requests.append(request_to_assign)
                    requests_to_remove_ids.add(assigned_req_id)
                except RuntimeError:
                    logging.warning(f"T={current_simulation_time:.4f}: CloudNode not available after check? Could not assign Req {request_to_assign.id} for {self.scheduler_name}. Stopping cloud assignments.")
                    break # Stop trying to schedule cloud requests

        # Remove all successfully scheduled requests from the actual waiting queue (heapq)
        if requests_to_remove_ids:
             self._remove_requests_from_queue(requests_to_remove_ids)

        self.metrics_collector.record_decision_point(
            timestamp=current_simulation_time,
            scheduler_name=self.scheduler_name,
            waiting_queue_len_after_scheduling=len(self.waiting_queue),
            edge_node=self.edge_node,
            cloud_node=self.cloud_node,
            scheduled_edge_batch_requests=scheduled_edge_batch_requests,
            scheduled_cloud_requests=scheduled_cloud_requests
        )
        return scheduled_something

    def get_feedback(self, completed_request_or_batch_info, is_batch_feedback):
        # The Node and EdgeNodeBatching classes already update their historical_processing_times
        # when `release_slot` is called. So, no explicit feedback logic is needed here for LRT.
        pass

class QueueLengthBasedScheduler(BaseScheduler): # B13
    """
    Assigns requests to the node (edge or cloud) with the shortest 'queue length',
    where queue length is approximated by the number of currently busy processing slots.
    Prioritizes edge in case of a tie.
    """
    def __init__(self, edge_node, cloud_node, metrics_collector):
        super().__init__("B13_Queue_Length_Based", edge_node, cloud_node, metrics_collector)
        # Uses the deque from BaseScheduler for FCFS from the global queue
        logging.info(f"{self.scheduler_name} initialized.")

    def schedule(self, current_simulation_time):
        scheduled_something = False
        scheduled_edge_batch_requests = []
        scheduled_cloud_requests = []

        # First, try to schedule an edge batch based on the relative queue lengths (busy slots).
        if self.waiting_queue and self.edge_node.is_available():
            edge_busy_slots = self.edge_node.concurrency - self.edge_node.available_slots
            cloud_busy_slots = self.cloud_node.concurrency - self.cloud_node.available_slots

            if (not self.cloud_node.is_available()) or (edge_busy_slots <= cloud_busy_slots):
                num_to_take_for_batch = min(len(self.waiting_queue), self.edge_node.batch_capacity_k)
                if num_to_take_for_batch > 0:
                    # Pop requests from the front of the queue so that we only remove them if the assignment succeeds.
                    edge_batch_to_attempt = [self.waiting_queue.popleft() for _ in range(num_to_take_for_batch)]

                    assign_result = self.edge_node.assign_batch(edge_batch_to_attempt, current_simulation_time)
                    if assign_result:
                        batch_internal_id, batch_completion_time = assign_result
                        cc.add_event(batch_completion_time, "edge_batch_completion", batch_internal_id, self.scheduler_name)
                        scheduled_edge_batch_requests = edge_batch_to_attempt
                        scheduled_something = True
                    else:
                        # If assignment failed, restore the requests in their original order.
                        for req in reversed(edge_batch_to_attempt):
                            self.waiting_queue.appendleft(req)
                        logging.warning(
                            f"T={current_simulation_time:.4f}: EdgeNode available but assign_batch failed for {self.scheduler_name}."
                        )

        # Next, attempt to schedule individual requests on the cloud node if it is the less busy option.
        while self.waiting_queue and self.cloud_node.is_available():
            edge_busy_slots = self.edge_node.concurrency - self.edge_node.available_slots
            cloud_busy_slots = self.cloud_node.concurrency - self.cloud_node.available_slots

            # If edge is available and not busier than cloud, prefer keeping requests for edge in this cycle.
            if self.edge_node.is_available() and edge_busy_slots <= cloud_busy_slots:
                break

            request_to_assign = self.waiting_queue.popleft()

            processing_time_on_cloud = calculate_individual_processing_time(
                request_to_assign.base_edge_inference_time, "cloud"
            )
            final_processing_time_with_rtt = processing_time_on_cloud + cc.CONFIG.get("network_rtt", 0.05)

            try:
                assigned_req_id, individual_completion_time = self.cloud_node.assign_request(
                    request_to_assign, current_simulation_time, final_processing_time_with_rtt
                )
            except RuntimeError:
                # Put the request back and stop trying to schedule on the cloud this cycle.
                self.waiting_queue.appendleft(request_to_assign)
                logging.warning(
                    f"T={current_simulation_time:.4f}: CloudNode not available after check? Could not assign Req {request_to_assign.id} for {self.scheduler_name}. Stopping cloud assignments."
                )
                break

            cc.add_event(
                individual_completion_time,
                "cloud_individual_completion",
                assigned_req_id,
                self.scheduler_name,
            )
            scheduled_cloud_requests.append(request_to_assign)
            scheduled_something = True

        self.metrics_collector.record_decision_point(
            timestamp=current_simulation_time,
            scheduler_name=self.scheduler_name,
            waiting_queue_len_after_scheduling=len(self.waiting_queue),
            edge_node=self.edge_node,
            cloud_node=self.cloud_node,
            scheduled_edge_batch_requests=scheduled_edge_batch_requests,
            scheduled_cloud_requests=scheduled_cloud_requests
        )
        return scheduled_something



__all__ = [
    "BaseScheduler",
    "EdgeCloudFCFSScheduler",
    "RandomOffloadScheduler",
    "RoundRobinOffloadScheduler",
    "PredictionBasedScheduler",
    "CooperativeSchedulerBase",
    "CombinatorialLinUCBScheduler",
    "CNGreedyScheduler",
    "CNDUCBScheduler",
    "LeastConnectionsScheduler",
    "HistoricalLRTScheduler",
    "QueueLengthBasedScheduler",
]
