"""Embedding backends for the LLM scheduler core simulations."""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from . import config as config_module


class BaseEmbedder:
    def __init__(self, model_name: str, embedding_dim: int):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        logging.info("Embedder base initialized: Model=%s, Dim=%s", model_name, embedding_dim)

    def get_embedding(self, text: str) -> np.ndarray:  # pragma: no cover - abstract method
        raise NotImplementedError


class BertEmbedder(BaseEmbedder):
    def __init__(self, model_path: Optional[str] = None, embedding_dim: Optional[int] = None):
        pid = os.getpid()

        model_path = model_path or config_module.CONFIG.get("bert_model_name")
        embedding_dim = embedding_dim or config_module.CONFIG.get("bert_embedding_dim")

        super().__init__(model_path, embedding_dim)
        if not model_path or not embedding_dim:
            raise ValueError("BERT model_path or embedding_dim not configured.")

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
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertModel.from_pretrained(model_path)
            logging.info("[PID:%s] Initializing BERT Embedder on device: %s", pid, self.device)
            self.model.to(self.device)
            self.model.eval()
            logging.info("[PID:%s] BERT embedder initialized: %s on %s", pid, model_path, self.device)
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("[PID:%s] BERT initialization failed for path '%s': %s", pid, model_path, exc, exc_info=True)
            raise

    @torch.no_grad()
    def get_embedding(self, text: str) -> np.ndarray:
        pid = os.getpid()
        try:
            if not isinstance(text, str) or not text.strip():
                return np.zeros(self.embedding_dim, dtype=np.float32)

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            if cls_embedding.ndim == 0:
                return np.zeros(self.embedding_dim, dtype=np.float32)
            if cls_embedding.shape[0] != self.embedding_dim:
                if cls_embedding.shape[0] > self.embedding_dim:
                    cls_embedding = cls_embedding[: self.embedding_dim]
                else:
                    padded_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
                    padded_embedding[: cls_embedding.shape[0]] = cls_embedding
                    cls_embedding = padded_embedding
            return cls_embedding.astype(np.float32)
        except Exception as exc:  # pylint: disable=broad-except
            logging.error(
                "[PID:%s] BERT embedding error for text '%s...': %s",
                pid,
                text[:50],
                exc,
                exc_info=True,
            )
            return np.zeros(self.embedding_dim, dtype=np.float32)


__all__ = ["BaseEmbedder", "BertEmbedder"]
