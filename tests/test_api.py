from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clip_classifier_service.api import create_app
from clip_classifier_service.classifier import ClassificationResult, ScoredLabel
from clip_classifier_service.encryption import encrypt_image_for_server, generate_key_pair
from clip_classifier_service.settings import Settings


@dataclass
class FakeClassifier:
    ready: bool = True

    def classify(self, image_bytes: bytes, top_k: int, threshold: float) -> ClassificationResult:
        assert image_bytes == b"fake-image"
        assert top_k == 1
        assert threshold == 0.25
        return ClassificationResult(
            model="fake-model",
            labels=[ScoredLabel(id="cat", name="Cat", score=0.99)],
            elapsed_ms=1.5,
        )

    def classify_batch(
        self,
        image_batch: list[bytes],
        top_k: int,
        threshold: float,
    ) -> list[ClassificationResult]:
        return [self.classify(image_bytes, top_k, threshold) for image_bytes in image_batch]


def test_health_and_ready_endpoints(tmp_path: Path) -> None:
    app = create_app(Settings(model_cache_dir=tmp_path), classifier=FakeClassifier())
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.json() == {"status": "ok"}
    assert health.headers["x-request-id"]
    assert client.get("/readyz").json() == {"status": "ready"}


def test_base_path_prefixes_all_routes(tmp_path: Path) -> None:
    app = create_app(
        Settings(model_cache_dir=tmp_path, base_path="/openai-clip"),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    assert client.get("/healthz").status_code == 404
    assert client.get("/openai-clip/healthz").json() == {"status": "ok"}
    assert client.get("/openai-clip/readyz").json() == {"status": "ready"}
    assert client.get("/openai-clip/metrics").status_code == 200


def test_base_path_classify_endpoint(tmp_path: Path) -> None:
    app = create_app(
        Settings(model_cache_dir=tmp_path, base_path="/openai-clip"),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    response = client.post(
        "/openai-clip/v1/classify",
        params={"top_k": 1, "threshold": 0.25},
        files={"file": ("image.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["labels"] == [{"id": "cat", "name": "Cat", "score": 0.99}]


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer primary-token"}


def test_auth_rejects_missing_token_on_all_routes(tmp_path: Path) -> None:
    key_pair = generate_key_pair()
    public_key_path = tmp_path / "server_public.key"
    private_key_path = tmp_path / "server_private.key"
    public_key_path.write_text(key_pair.public_key_b64, encoding="ascii")
    private_key_path.write_text(key_pair.private_key_b64, encoding="ascii")
    settings = Settings(
        model_cache_dir=tmp_path,
        auth_enabled=True,
        api_tokens=("primary-token",),
        encryption_enabled=True,
        encryption_public_key_path=public_key_path,
        encryption_private_key_path=private_key_path,
    )
    app = create_app(settings, classifier=FakeClassifier())
    client = TestClient(app)
    encrypted = encrypt_image_for_server(key_pair.public_key_b64, b"fake-image")

    requests = [
        ("get", "/healthz", {}),
        ("get", "/readyz", {}),
        ("get", "/metrics", {}),
        ("get", "/v1/encryption/public-key", {}),
        (
            "post",
            "/v1/classify",
            {"files": {"file": ("image.jpg", b"fake-image", "image/jpeg")}},
        ),
        (
            "post",
            "/v1/classify/encrypted",
            {"json": encrypted},
        ),
        (
            "post",
            "/v1/classify/batch",
            {"files": [("files", ("one.jpg", b"fake-image", "image/jpeg"))]},
        ),
        (
            "post",
            "/v1/classify/batch/encrypted",
            {"json": {"items": [{"filename": "one.jpg", **encrypted}]}},
        ),
    ]

    for method, path, kwargs in requests:
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["detail"] == "missing bearer token"


def test_auth_rejects_invalid_token(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            model_cache_dir=tmp_path,
            auth_enabled=True,
            api_tokens=("primary-token",),
        ),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    response = client.get("/healthz", headers={"Authorization": "Bearer wrong-token"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "invalid bearer token"


def test_auth_rejects_missing_token_before_invalid_content_length(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            model_cache_dir=tmp_path,
            auth_enabled=True,
            api_tokens=("primary-token",),
        ),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/classify",
        headers={"content-length": "not-int"},
        files={"file": ("image.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "missing bearer token"


def test_auth_rejects_missing_token_before_large_content_length(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            model_cache_dir=tmp_path,
            auth_enabled=True,
            api_tokens=("primary-token",),
            max_image_bytes=4,
            max_multipart_overhead_bytes=0,
        ),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/classify",
        headers={"content-length": "5"},
        files={"file": ("image.jpg", b"tiny", "image/jpeg")},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "missing bearer token"


def test_auth_accepts_any_configured_token(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            model_cache_dir=tmp_path,
            auth_enabled=True,
            api_tokens=("primary-token", "secondary-token"),
        ),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    first = client.get("/healthz", headers={"Authorization": "Bearer primary-token"})
    second = client.get("/readyz", headers={"Authorization": "Bearer secondary-token"})

    assert first.status_code == 200
    assert first.json() == {"status": "ok"}
    assert second.status_code == 200
    assert second.json() == {"status": "ready"}


def test_auth_applies_under_base_path(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            model_cache_dir=tmp_path,
            base_path="/openai-clip",
            auth_enabled=True,
            api_tokens=("primary-token",),
        ),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    unauthorized = client.get("/openai-clip/healthz")
    authorized = client.get(
        "/openai-clip/healthz",
        headers={"Authorization": "Bearer primary-token"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json() == {"status": "ok"}


def test_auth_disables_docs_when_configured(tmp_path: Path, auth_headers: dict[str, str]) -> None:
    app = create_app(
        Settings(
            model_cache_dir=tmp_path,
            auth_enabled=True,
            api_tokens=("primary-token",),
            auth_disable_docs=True,
        ),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    assert client.get("/docs", headers=auth_headers).status_code == 404
    assert client.get("/redoc", headers=auth_headers).status_code == 404
    assert client.get("/openapi.json", headers=auth_headers).status_code == 404


def test_classify_endpoint_returns_top_labels(tmp_path: Path) -> None:
    app = create_app(Settings(model_cache_dir=tmp_path), classifier=FakeClassifier())
    client = TestClient(app)

    response = client.post(
        "/v1/classify",
        params={"top_k": 1, "threshold": 0.25},
        files={"file": ("image.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["labels"] == [{"id": "cat", "name": "Cat", "score": 0.99}]


def test_public_key_endpoint_returns_encryption_metadata(tmp_path: Path) -> None:
    key_pair = generate_key_pair()
    public_key_path = tmp_path / "server_public.key"
    private_key_path = tmp_path / "server_private.key"
    public_key_path.write_text(key_pair.public_key_b64, encoding="ascii")
    private_key_path.write_text(key_pair.private_key_b64, encoding="ascii")
    settings = Settings(
        model_cache_dir=tmp_path,
        encryption_enabled=True,
        encryption_public_key_path=public_key_path,
        encryption_private_key_path=private_key_path,
    )
    app = create_app(settings, classifier=FakeClassifier())
    client = TestClient(app)

    response = client.get("/v1/encryption/public-key")

    assert response.status_code == 200
    assert response.json() == {
        "alg": "AES-256-GCM+SealedBox-v1",
        "public_key": key_pair.public_key_b64,
    }


def test_encrypted_classify_endpoint_returns_top_labels(tmp_path: Path) -> None:
    key_pair = generate_key_pair()
    private_key_path = tmp_path / "server_private.key"
    private_key_path.write_text(key_pair.private_key_b64, encoding="ascii")
    settings = Settings(
        model_cache_dir=tmp_path,
        encryption_enabled=True,
        encryption_private_key_path=private_key_path,
    )
    app = create_app(settings, classifier=FakeClassifier())
    client = TestClient(app)
    payload = encrypt_image_for_server(key_pair.public_key_b64, b"fake-image")

    response = client.post(
        "/v1/classify/encrypted",
        params={"top_k": 1, "threshold": 0.25},
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["labels"] == [{"id": "cat", "name": "Cat", "score": 0.99}]


def test_encrypted_classify_rejects_tampered_payload(tmp_path: Path) -> None:
    key_pair = generate_key_pair()
    private_key_path = tmp_path / "server_private.key"
    private_key_path.write_text(key_pair.private_key_b64, encoding="ascii")
    settings = Settings(
        model_cache_dir=tmp_path,
        encryption_enabled=True,
        encryption_private_key_path=private_key_path,
    )
    app = create_app(settings, classifier=FakeClassifier())
    client = TestClient(app)
    payload = encrypt_image_for_server(key_pair.public_key_b64, b"fake-image")
    payload["ciphertext"] = payload["ciphertext"][:-2] + "AA"

    response = client.post("/v1/classify/encrypted", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid encrypted payload"


def test_encrypted_batch_classify_returns_per_image_results(tmp_path: Path) -> None:
    key_pair = generate_key_pair()
    private_key_path = tmp_path / "server_private.key"
    private_key_path.write_text(key_pair.private_key_b64, encoding="ascii")
    settings = Settings(
        model_cache_dir=tmp_path,
        encryption_enabled=True,
        encryption_private_key_path=private_key_path,
        max_batch_size=2,
    )
    app = create_app(settings, classifier=FakeClassifier())
    client = TestClient(app)
    encrypted = encrypt_image_for_server(key_pair.public_key_b64, b"fake-image")

    response = client.post(
        "/v1/classify/batch/encrypted",
        params={"top_k": 1, "threshold": 0.25},
        json={
            "items": [
                {"filename": "one.jpg", **encrypted},
                {"filename": "two.jpg", **encrypted},
            ]
        },
    )

    assert response.status_code == 200
    assert [item["filename"] for item in response.json()["results"]] == ["one.jpg", "two.jpg"]
    assert response.json()["results"][0]["labels"] == [{"id": "cat", "name": "Cat", "score": 0.99}]


def test_require_encrypted_uploads_rejects_plaintext_classify(tmp_path: Path) -> None:
    settings = Settings(
        model_cache_dir=tmp_path,
        encryption_enabled=True,
        require_encrypted_uploads=True,
    )
    app = create_app(settings, classifier=FakeClassifier())
    client = TestClient(app)

    response = client.post(
        "/v1/classify",
        params={"top_k": 1, "threshold": 0.25},
        files={"file": ("image.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 403


def test_classify_rejects_oversized_images(tmp_path: Path) -> None:
    settings = Settings(model_cache_dir=tmp_path, max_image_bytes=4)
    app = create_app(settings, classifier=FakeClassifier())
    client = TestClient(app)

    response = client.post(
        "/v1/classify",
        files={"file": ("image.jpg", b"too-large", "image/jpeg")},
    )

    assert response.status_code == 413


def test_classify_rejects_large_content_length_before_reading(tmp_path: Path) -> None:
    settings = Settings(
        model_cache_dir=tmp_path,
        max_image_bytes=4,
        max_multipart_overhead_bytes=0,
    )
    app = create_app(settings, classifier=FakeClassifier())
    client = TestClient(app)

    response = client.post(
        "/v1/classify",
        headers={"content-length": "5"},
        files={"file": ("image.jpg", b"tiny", "image/jpeg")},
    )

    assert response.status_code == 413


def test_readyz_returns_503_when_model_not_ready(tmp_path: Path) -> None:
    app = create_app(Settings(model_cache_dir=tmp_path), classifier=None, load_on_startup=False)
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503


def test_query_validation_rejects_invalid_top_k(tmp_path: Path) -> None:
    app = create_app(Settings(model_cache_dir=tmp_path), classifier=FakeClassifier())
    client = TestClient(app)

    response = client.post(
        "/v1/classify",
        params={"top_k": 0},
        files={"file": ("image.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 422


def test_backend_value_error_returns_400(tmp_path: Path) -> None:
    class FailingClassifier(FakeClassifier):
        def classify(
            self,
            image_bytes: bytes,
            top_k: int,
            threshold: float,
        ) -> ClassificationResult:
            raise ValueError("invalid image")

    app = create_app(Settings(model_cache_dir=tmp_path), classifier=FailingClassifier())
    client = TestClient(app)

    response = client.post(
        "/v1/classify",
        files={"file": ("image.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid image"


def test_batch_classify_returns_per_image_results(tmp_path: Path) -> None:
    app = create_app(
        Settings(model_cache_dir=tmp_path, max_batch_size=2),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/classify/batch",
        params={"top_k": 1, "threshold": 0.25},
        files=[
            ("files", ("one.jpg", b"fake-image", "image/jpeg")),
            ("files", ("two.jpg", b"fake-image", "image/jpeg")),
        ],
    )

    assert response.status_code == 200
    assert [item["filename"] for item in response.json()["results"]] == ["one.jpg", "two.jpg"]
    assert response.json()["results"][0]["labels"] == [{"id": "cat", "name": "Cat", "score": 0.99}]


def test_batch_classify_rejects_too_many_files(tmp_path: Path) -> None:
    app = create_app(
        Settings(model_cache_dir=tmp_path, max_batch_size=1),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/classify/batch",
        files=[
            ("files", ("one.jpg", b"fake-image", "image/jpeg")),
            ("files", ("two.jpg", b"fake-image", "image/jpeg")),
        ],
    )

    assert response.status_code == 413


def test_batch_classify_rejects_oversized_member(tmp_path: Path) -> None:
    app = create_app(
        Settings(model_cache_dir=tmp_path, max_image_bytes=4, max_batch_size=2),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/classify/batch",
        files=[
            ("files", ("one.jpg", b"fake", "image/jpeg")),
            ("files", ("two.jpg", b"too-large", "image/jpeg")),
        ],
    )

    assert response.status_code == 413


@pytest.mark.parametrize("endpoint", ["/v1/classify", "/v1/classify/batch"])
def test_metrics_endpoint_is_available_after_errors(tmp_path: Path, endpoint: str) -> None:
    app = create_app(
        Settings(model_cache_dir=tmp_path, max_image_bytes=4),
        classifier=FakeClassifier(),
    )
    client = TestClient(app)

    if endpoint.endswith("batch"):
        client.post(endpoint, files=[("files", ("one.jpg", b"too-large", "image/jpeg"))])
    else:
        client.post(endpoint, files={"file": ("one.jpg", b"too-large", "image/jpeg")})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "clip_classifier_requests_total" in response.text
