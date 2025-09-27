"""Training utilities for the LLM scheduler core simulations."""
from __future__ import annotations

import logging
import os
import random
import threading
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler

from . import config as config_module


class TimeEstimatorNN(nn.Module):
    def __init__(self, input_dim, hidden_layers_config):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers_config:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x_embeddings):
        return self.network(x_embeddings)


class TrainingBuffer:
    def __init__(self, maxlen=10000):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        logging.info("TrainingBuffer initialized with maxlen=%s", maxlen)

    def add(self, context_embedding, actual_time_scaled):
        with self.lock:
            if not isinstance(context_embedding, np.ndarray):
                context_embedding = np.array(context_embedding, dtype=np.float32)
            elif context_embedding.dtype != np.float32:
                context_embedding = context_embedding.astype(np.float32)
            self.buffer.append((context_embedding, float(actual_time_scaled)))

    def get_batch(self, batch_size) -> Tuple[torch.Tensor, torch.Tensor]:
        with self.lock:
            num_samples_to_draw = min(batch_size, len(self.buffer))
            if num_samples_to_draw == 0:
                return None, None
            indices = random.sample(range(len(self.buffer)), num_samples_to_draw)
            batch_data = [self.buffer[i] for i in indices]
        try:
            context_embeddings_batch = torch.tensor(np.array([item[0] for item in batch_data]), dtype=torch.float32)
            actual_times_batch = torch.tensor(np.array([item[1] for item in batch_data]), dtype=torch.float32).unsqueeze(1)
            return context_embeddings_batch, actual_times_batch
        except Exception:  # pylint: disable=broad-except
            return None, None

    def __len__(self):
        with self.lock:
            return len(self.buffer)


class AsyncTrainer:
    def __init__(self, nn_model, optimizer, criterion, training_buffer, batch_size_nn, train_interval_nn):
        pid = os.getpid()

        self.model = nn_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.buffer = training_buffer

        self.batch_size = batch_size_nn
        self.train_interval = train_interval_nn

        config_device = getattr(config_module, "DEVICE", None)
        if isinstance(config_device, torch.device):
            self.device = config_device
        elif config_device is not None:
            try:
                self.device = torch.device(config_device)
            except (TypeError, ValueError):
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        # Ensure the model resides on the same device that will be used for
        # training and inference tensors.
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        self.executor = ThreadPoolExecutor(
            max_workers=config_module.CONFIG.get("max_training_workers", 1),
            thread_name_prefix="NNTrainerThread",
        )

        self.train_trigger_counter = 0
        self.internal_state_lock = threading.Lock()
        self.model_access_lock = threading.Lock()

        self.target_scaler = StandardScaler()
        self.scaler_is_fitted = False

        self.active_training_futures = []
        self.training_loss_history = deque(maxlen=200)

        logging.info(
            "[PID:%s] AsyncTrainer initialized: Device=%s, BatchSize=%s, TrainInterval=%s",
            pid,
            self.device,
            self.batch_size,
            self.train_interval,
        )

    def _core_train_step(self):
        pid = os.getpid()
        current_buffer_len = len(self.buffer)
        if current_buffer_len < self.batch_size:
            return

        context_embeddings, actual_times_scaled_domain = self.buffer.get_batch(self.batch_size)
        if context_embeddings is None or actual_times_scaled_domain is None:
            return

        target_times_np = actual_times_scaled_domain.numpy().reshape(-1, 1)

        if not self.scaler_is_fitted and current_buffer_len >= self.batch_size:
            try:
                self.target_scaler.fit(target_times_np)
                self.scaler_is_fitted = True
            except Exception:  # pylint: disable=broad-except
                return

        if not self.scaler_is_fitted:
            return

        try:
            targets_for_nn = torch.tensor(
                self.target_scaler.transform(target_times_np),
                dtype=torch.float32,
            ).to(self.device)
        except Exception:  # pylint: disable=broad-except
            return

        context_embeddings = context_embeddings.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        predictions_nn_domain = self.model(context_embeddings)
        loss = self.criterion(predictions_nn_domain, targets_for_nn)

        if torch.isnan(loss) or torch.isinf(loss):
            self.optimizer.zero_grad()
            return

        loss.backward()
        self.optimizer.step()
        self.model.eval()
        self.training_loss_history.append(loss.item())
        logging.info(
            "[PID:%s] Trainer step completed: Loss=%0.6f, BufferSize=%s, ScalerMean=%0.4f",
            pid,
            loss.item(),
            current_buffer_len,
            self.target_scaler.mean_[0],
        )

    def train_step_if_needed_async(self):
        with self.model_access_lock:
            try:
                self._core_train_step()
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Exception in _core_train_step within async wrapper: %s", exc, exc_info=True)

    def trigger_train(self, force_train=False):
        should_submit_training_task = False
        with self.internal_state_lock:
            self.train_trigger_counter += 1
            if force_train or (
                self.train_interval > 0 and self.train_trigger_counter % self.train_interval == 0
            ):
                if len(self.buffer) >= self.batch_size:
                    should_submit_training_task = True

            self.active_training_futures = [f for f in self.active_training_futures if not f.done()]

            if should_submit_training_task and len(self.active_training_futures) < config_module.CONFIG.get("max_training_workers", 1) * 2:
                future = self.executor.submit(self.train_step_if_needed_async)
                self.active_training_futures.append(future)
                return True
        return False

    def predict_time_scaled_domain(self, context_embedding_np):
        pid = os.getpid()

        default_pred_scaled_offset = config_module.CONFIG.get("inference_time_base_offset", 0.1)
        if self.scaler_is_fitted and hasattr(self.target_scaler, "mean_") and self.target_scaler.mean_ is not None:
            try:
                mean_in_nn_domain = np.array([[self.target_scaler.mean_[0]]])
                default_pred_scaled_offset = self.target_scaler.inverse_transform(mean_in_nn_domain)[0][0]
            except Exception:  # pylint: disable=broad-except
                pass

        with self.model_access_lock:
            if not self.scaler_is_fitted:
                return max(0.001, default_pred_scaled_offset)

            self.model.eval()
            with torch.no_grad():
                try:
                    if not isinstance(context_embedding_np, np.ndarray):
                        context_embedding_np = np.array(context_embedding_np, dtype=np.float32)
                    elif context_embedding_np.dtype != np.float32:
                        context_embedding_np = context_embedding_np.astype(np.float32)

                    if context_embedding_np.ndim == 1:
                        context_embedding_np = context_embedding_np.reshape(1, -1)

                    context_tensor = torch.tensor(context_embedding_np, dtype=torch.float32).to(self.device)

                    prediction_nn_domain = self.model(context_tensor)
                    prediction_nn_domain_np = prediction_nn_domain.cpu().numpy()

                    prediction_scaled_offset_domain = self.target_scaler.inverse_transform(prediction_nn_domain_np)[0][0]

                except Exception as exc:  # pylint: disable=broad-except
                    logging.error(
                        "[PID:%s] Trainer prediction or inverse_transform error: %s",
                        pid,
                        exc,
                        exc_info=True,
                    )
                    return max(0.001, default_pred_scaled_offset)

            return max(0.001, prediction_scaled_offset_domain)

    def get_gradient_of_prediction_scaled_domain(self, context_embedding_np):
        pid = os.getpid()
        gradient_tensor = None

        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if num_trainable_params == 0:
            return torch.zeros(0, device=self.device)

        with self.model_access_lock:
            if (
                not self.scaler_is_fitted
                or not hasattr(self.target_scaler, "scale_")
                or self.target_scaler.scale_ is None
            ):
                return torch.zeros(num_trainable_params, device=self.device)

            self.model.eval()
            try:
                if not isinstance(context_embedding_np, np.ndarray):
                    context_embedding_np = np.array(context_embedding_np, dtype=np.float32)
                elif context_embedding_np.dtype != np.float32:
                    context_embedding_np = context_embedding_np.astype(np.float32)

                if context_embedding_np.ndim == 1:
                    context_embedding_np = context_embedding_np.reshape(1, -1)

                feature_tensor = torch.tensor(context_embedding_np, dtype=torch.float32).to(self.device)

                with torch.enable_grad():
                    self.model.zero_grad()
                    prediction_nn_domain = self.model(feature_tensor)
                    if prediction_nn_domain.requires_grad:
                        prediction_nn_domain.backward(retain_graph=False)
                        grad_list_for_params = []
                        for param in self.model.parameters():
                            if param.requires_grad and param.grad is not None:
                                grad_list_for_params.append(param.grad.flatten().detach().clone())
                            elif param.requires_grad:
                                grad_list_for_params.append(torch.zeros_like(param).flatten())

                        if grad_list_for_params:
                            gradient_nn_domain = torch.cat(grad_list_for_params)
                            scaler_std_dev = (
                                self.target_scaler.scale_[0] if self.target_scaler.scale_ is not None else 1.0
                            )
                            gradient_tensor = gradient_nn_domain * scaler_std_dev
                        else:
                            gradient_tensor = torch.zeros(num_trainable_params, device=self.device)
                        self.model.zero_grad()
                    else:
                        gradient_tensor = torch.zeros(num_trainable_params, device=self.device)
            except Exception as exc:  # pylint: disable=broad-except
                if self.model:
                    self.model.zero_grad()
                logging.error("[PID:%s] Gradient computation error: %s", pid, exc, exc_info=True)
                gradient_tensor = torch.zeros(num_trainable_params, device=self.device)

        if gradient_tensor is None or gradient_tensor.numel() != num_trainable_params:
            return torch.zeros(num_trainable_params, device=self.device)
        return gradient_tensor.to(self.device)

    def shutdown(self):
        logging.info(
            "AsyncTrainer shutting down executor. Waiting for active tasks (%s)...",
            len(self.active_training_futures),
        )
        self.executor.shutdown(wait=True)
        logging.info("AsyncTrainer shutdown complete.")


__all__ = ["TimeEstimatorNN", "TrainingBuffer", "AsyncTrainer"]
