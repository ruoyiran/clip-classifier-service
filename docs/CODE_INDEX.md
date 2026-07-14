# Code Index

## Package: `src/clip_classifier_service/`

- `main.py`: ASGI entrypoint. Creates settings and FastAPI app.
- `api.py`: HTTP app factory, lifespan model loading, upload limits, request ID middleware, GPU concurrency guard, classification routes, metrics endpoint.
- `settings.py`: `pydantic-settings` configuration loaded from defaults, `.env`, and `CLIP_` environment variables.
- `labels.py`: YAML label loader and validation for fixed-label classifier configuration.
- `classifier.py`: Model-agnostic classification logic, text feature cache, single-image and batch ranking.
- `openclip_backend.py`: OpenCLIP/PyTorch backend, model loading, image decoding, pixel safety limits, single and batch image encoding.
- `encryption.py`: AES-256-GCM image encryption helpers, libsodium SealedBox key wrapping, and key generation.
- `metrics.py`: Prometheus counters and inference latency histogram.
- `__init__.py`: Package marker.

## Tests: `tests/`

- `test_api.py`: FastAPI endpoints, upload limits, readiness, query validation, error handling, metrics availability, batch behavior.
- `test_classifier.py`: Ranking, threshold filtering, and true batch classifier behavior with a fake backend.
- `test_labels.py`: Label YAML parsing, disabled labels, duplicate IDs, and empty enabled set validation.
- `test_settings.py`: Settings defaults and validation.
- `test_encryption.py`: Hybrid encryption round trip and tamper rejection.

## Configuration

- `config/labels.yaml`: Default labels. Replace with production labels before exposing the service.
- `.env.example`: Documented runtime configuration defaults.
- `requirements.txt`: Runtime dependencies.
- `requirements-dev.txt`: Runtime dependencies plus test/lint tools.
- `pyproject.toml`: Package metadata, editable install setup, pytest config, and ruff config.

## Scripts

- `scripts/check_cuda.py`: Verifies PyTorch CUDA access and runs a small GPU tensor operation.
- `scripts/preload_model.py`: Downloads/loads model and labels, then reports readiness.
- `scripts/generate_encryption_keys.py`: Generates service public/private encryption key files.
- `scripts/encrypt_sample_image.py`: Encrypts a local image and calls the encrypted classification API.
- `scripts/start.ps1`: Installs runtime dependencies and starts Uvicorn with one worker.
- `scripts/install_windows_service.ps1`: Installs the service through NSSM and configures log files.

## Generated/Ignored Paths

- `.venv/`: Local virtual environment.
- `models/`: Local model cache.
- `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`: Tool caches.
- `*.egg-info/`: Editable package metadata.
