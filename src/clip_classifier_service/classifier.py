from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class ScoredLabel:
    id: str
    name: str
    score: float
    similarity: float = 0.0


@dataclass(frozen=True)
class ClassificationResult:
    model: str
    labels: list[ScoredLabel]
    elapsed_ms: float


class ClassifierBackend(Protocol):
    def encode_texts(self, prompts: list[str]) -> torch.Tensor:
        ...

    def encode_image_bytes(self, image_bytes: bytes) -> torch.Tensor:
        ...

    def encode_image_batch(self, image_batch: list[bytes]) -> torch.Tensor:
        ...


class ClipClassifier:
    def __init__(
        self,
        labels,
        backend: ClassifierBackend,
        model_name: str = "test-backend",
    ) -> None:
        self.labels = list(labels)
        self.backend = backend
        self.model_name = model_name
        self.prompt_label_indices: list[list[int]] = []
        prompts = []
        for label in self.labels:
            start = len(prompts)
            prompts.extend(label.prompts)
            self.prompt_label_indices.append(list(range(start, len(prompts))))
        self.text_features = self._normalize(self.backend.encode_texts(prompts))
        self.ready = True

    def classify(self, image_bytes: bytes, top_k: int, threshold: float) -> ClassificationResult:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        started = time.perf_counter()
        image_features = self._normalize(self.backend.encode_image_bytes(image_bytes))
        similarities = self._label_similarities(image_features).squeeze(0)
        scores = (100.0 * similarities).softmax(dim=-1)
        count = min(top_k, len(self.labels))
        values, indices = torch.topk(scores, k=count)

        ranked: list[ScoredLabel] = []
        for value, index in zip(values.tolist(), indices.tolist(), strict=True):
            score = float(value)
            if score < threshold:
                continue
            label = self.labels[int(index)]
            ranked.append(
                ScoredLabel(
                    id=label.id,
                    name=label.name,
                    score=score,
                    similarity=float(similarities[int(index)]),
                )
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        return ClassificationResult(model=self.model_name, labels=ranked, elapsed_ms=elapsed_ms)

    def classify_batch(
        self,
        image_batch: list[bytes],
        top_k: int,
        threshold: float,
    ) -> list[ClassificationResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if not image_batch:
            return []

        started = time.perf_counter()
        image_features = self._normalize(self.backend.encode_image_batch(image_batch))
        similarities = self._label_similarities(image_features)
        scores = (100.0 * similarities).softmax(dim=-1)
        count = min(top_k, len(self.labels))
        values, indices = torch.topk(scores, k=count, dim=-1)
        elapsed_ms = (time.perf_counter() - started) * 1000

        results: list[ClassificationResult] = []
        for row_index, (row_values, row_indices) in enumerate(
            zip(values.tolist(), indices.tolist(), strict=True)
        ):
            ranked: list[ScoredLabel] = []
            for value, index in zip(row_values, row_indices, strict=True):
                score = float(value)
                if score < threshold:
                    continue
                label = self.labels[int(index)]
                ranked.append(
                    ScoredLabel(
                        id=label.id,
                        name=label.name,
                        score=score,
                        similarity=float(similarities[row_index, int(index)]),
                    )
                )
            results.append(
                ClassificationResult(
                    model=self.model_name,
                    labels=ranked,
                    elapsed_ms=elapsed_ms / len(image_batch),
                )
            )
        return results

    def _label_similarities(self, image_features: torch.Tensor) -> torch.Tensor:
        prompt_similarities = image_features @ self.text_features.T
        return torch.stack(
            [
                prompt_similarities[:, prompt_indices].max(dim=1).values
                for prompt_indices in self.prompt_label_indices
            ],
            dim=1,
        )

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
