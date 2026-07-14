# AGENTS.md

Guidance for agents and engineers working in this repository.

## Project Summary

This project is a local FastAPI service for zero-shot image classification with
`hf-hub:apple/DFN5B-CLIP-ViT-H-14-378` through OpenCLIP/PyTorch.

The intended production shape is a fixed-label classifier:

- Load the model and configured labels at service startup.
- Precompute label text features once.
- For each request, decode images, encode image features on the GPU, and rank labels by similarity.
- Run one Uvicorn worker per GPU to avoid loading duplicate model copies into VRAM.

## Repository Layout

- `src/clip_classifier_service/`: Service package.
- `tests/`: Unit and API tests.
- `config/labels.yaml`: Default label configuration.
- `scripts/`: Local setup, CUDA, preload, and Windows service helper scripts.
- `docs/`: Project documentation and code indexes.
- `models/`: Local Hugging Face/OpenCLIP cache. This is generated and ignored by git.

## Required Environment

- Windows PowerShell.
- Python 3.12.
- NVIDIA GPU with CUDA-capable PyTorch. This machine was validated with RTX 4090 and `torch==2.2.2+cu121`.
- Network access or pre-seeded model cache for first model load.

## Setup Commands

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
Copy-Item .env.example .env
```

## Verification Commands

Use these before claiming work is complete:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts\check_cuda.py
```

For model-path verification:

```powershell
.\.venv\Scripts\python.exe scripts\preload_model.py
```

The preload command may download model weights. Set `HF_TOKEN` for better Hugging Face limits.

## Run Commands

```powershell
.\.venv\Scripts\python.exe -m uvicorn clip_classifier_service.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Do not increase `--workers` on a single GPU unless the memory impact has been measured. Each worker loads its own model copy.

## Coding Rules

- Keep source code under `src/clip_classifier_service/`.
- Keep tests under `tests/`.
- Preserve the `src/` package layout and editable install flow.
- Prefer focused modules with clear responsibilities.
- Do not commit `.venv/`, `models/`, caches, logs, or `*.egg-info/`.
- Avoid broad refactors unless they directly support the task.
- If changing API behavior, update tests and `docs/`.
- If changing runtime settings, update `.env.example`, `README.md`, and `docs/OPERATIONS.md`.
- If `CLIP_BASE_PATH` is configured, all route examples and tests must include that prefix.
- Never commit service encryption private keys or generated key files.

## Production Safety Constraints

- Enforce upload byte limits with both request-level and per-file checks.
- Preserve `CLIP_MAX_IMAGE_PIXELS` protection to avoid decompression abuse.
- Keep invalid image payloads as 4xx responses, not 500s.
- Keep invalid encrypted payloads as 4xx responses, not 500s.
- Preserve single-GPU backpressure through the semaphore in `api.py`.
- Keep model downloads out of first live traffic by using `scripts/preload_model.py`.
- Expose the service behind an internal gateway or reverse proxy for TLS, auth, and rate limits.

## Documentation Index

- `README.md`: Quick setup and user-facing usage.
- `docs/INDEX.md`: Documentation entrypoint.
- `docs/CODE_INDEX.md`: Source and test map.
- `docs/ARCHITECTURE.md`: Runtime architecture and data flow.
- `docs/OPERATIONS.md`: Deployment, configuration, and verification notes.
