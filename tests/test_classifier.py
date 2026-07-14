from dataclasses import dataclass

import torch

from clip_classifier_service.classifier import ClipClassifier
from clip_classifier_service.labels import Label


@dataclass(frozen=True)
class FakeBackend:
    image_features: torch.Tensor
    text_features: torch.Tensor

    def encode_texts(self, prompts: list[str]) -> torch.Tensor:
        assert prompts == ["cat prompt", "dog prompt"]
        return self.text_features

    def encode_image_bytes(self, image_bytes: bytes) -> torch.Tensor:
        assert image_bytes == b"image"
        return self.image_features


def test_classifier_returns_ranked_scores() -> None:
    labels = [
        Label(id="cat", name="Cat", prompt="cat prompt", enabled=True),
        Label(id="dog", name="Dog", prompt="dog prompt", enabled=True),
    ]
    backend = FakeBackend(
        image_features=torch.tensor([[1.0, 0.0]]),
        text_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    classifier = ClipClassifier(labels=labels, backend=backend)

    result = classifier.classify(image_bytes=b"image", top_k=2, threshold=0.0)

    assert result.labels[0].id == "cat"
    assert result.labels[0].score > result.labels[1].score
    assert result.model == "test-backend"


def test_classifier_applies_threshold() -> None:
    labels = [
        Label(id="cat", name="Cat", prompt="cat prompt", enabled=True),
        Label(id="dog", name="Dog", prompt="dog prompt", enabled=True),
    ]
    backend = FakeBackend(
        image_features=torch.tensor([[1.0, 0.0]]),
        text_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    classifier = ClipClassifier(labels=labels, backend=backend)

    result = classifier.classify(image_bytes=b"image", top_k=2, threshold=0.9)

    assert [label.id for label in result.labels] == ["cat"]


def test_classifier_classifies_batch_with_single_backend_call() -> None:
    labels = [
        Label(id="cat", name="Cat", prompt="cat prompt", enabled=True),
        Label(id="dog", name="Dog", prompt="dog prompt", enabled=True),
    ]

    class BatchBackend(FakeBackend):
        calls = 0

        def encode_image_batch(self, image_batch: list[bytes]) -> torch.Tensor:
            self.calls += 1
            assert image_batch == [b"cat-image", b"dog-image"]
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    backend = BatchBackend(
        image_features=torch.tensor([[1.0, 0.0]]),
        text_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    classifier = ClipClassifier(labels=labels, backend=backend)

    results = classifier.classify_batch(
        image_batch=[b"cat-image", b"dog-image"],
        top_k=1,
        threshold=0.0,
    )

    assert [result.labels[0].id for result in results] == ["cat", "dog"]
    assert backend.calls == 1
