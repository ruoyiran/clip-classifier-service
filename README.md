# CLIP Classifier Service

FastAPI service for zero-shot image classification with `hf-hub:apple/DFN5B-CLIP-ViT-H-14-378`.

## Hardware status on this machine

- GPU: NVIDIA GeForce RTX 4090, 24564 MiB VRAM
- RAM: about 128 GB
- Disk: about 1.1 TB free on `C:`
- CUDA through PyTorch: verified with `torch 2.2.2+cu121`

## Setup

```powershell
cd C:\Users\Admin\workspace\AI\clip-classifier-service
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
Copy-Item .env.example .env
```

Edit `config/labels.yaml` for the production label set before exposing the service.
Each enabled label can define either `prompt` or `prompts`; when `prompts` is used,
the classifier takes the best cosine similarity across that label's prompts.

## Verify CUDA

```powershell
.\.venv\Scripts\python.exe scripts\check_cuda.py
```

## Preload model

This downloads the model to `models/` on first run and initializes text features for the configured labels.
Set `HF_TOKEN` before this step if you need higher Hugging Face download limits.

```powershell
.\.venv\Scripts\python.exe scripts\preload_model.py
```

## Encryption keys

Encrypted upload endpoints use `AES-256-GCM+SealedBox-v1`. Generate service keys before enabling encrypted uploads:

```powershell
.\.venv\Scripts\python.exe scripts\generate_encryption_keys.py
```

Then set:

```powershell
$env:CLIP_ENCRYPTION_ENABLED="true"
```

## API authentication

Production deployments should enable static Bearer Token authentication:

```powershell
$env:CLIP_AUTH_ENABLED="true"
$env:CLIP_API_TOKENS="replace-with-a-long-random-token"
```

When authentication is enabled, every API route requires:

```text
Authorization: Bearer <token>
```

`CLIP_API_TOKENS` accepts multiple comma-separated tokens for rotation. If
`CLIP_AUTH_ENABLED=true` and no token is configured, service startup fails.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn clip_classifier_service.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Use one worker per GPU. Multiple Uvicorn workers will load multiple model copies and can exhaust VRAM.

To serve every API under a prefix such as `/openai-clip`, set:

```powershell
$env:CLIP_BASE_PATH="/openai-clip"
```

## Install as Windows service

Install NSSM first, then run:

```powershell
New-Item -ItemType Directory -Force logs
.\scripts\install_windows_service.ps1 -ServiceName ClipClassifierService -Port 8000
Start-Service ClipClassifierService
```

## API

- Single-image payload limits:
  - Plain/decrypted image bytes: `31457280` bytes, 30 MiB.
  - Encrypted JSON request body: `44040192` bytes, 42 MiB.

Supported input formats include JPEG/JPG, PNG, BMP, HEIC, and HEIF. The
service validates the decoded image bytes rather than trusting file extensions
or MIME types.

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /v1/encryption/public-key`
- `POST /v1/classify` with `multipart/form-data` field `file`, optional `top_k`, optional `threshold`
- `POST /v1/classify/encrypted` with JSON encrypted payload, optional `top_k`, optional `threshold`
- `POST /v1/classify/batch` with repeated `multipart/form-data` field `files`, optional `top_k`, optional `threshold`
- `POST /v1/classify/batch/encrypted` with JSON encrypted payloads, optional `top_k`, optional `threshold`

Each returned label includes `score` (softmax over configured labels) and
`similarity` (raw image-text cosine similarity for the best prompt in that label).
The request `threshold` filters by `score`, not `similarity`. Clients that need
raw cosine thresholds should request `threshold=0.0` with enough `top_k` labels
and apply their own thresholding to `similarity`.

If `CLIP_AUTH_ENABLED=true`, include `Authorization: Bearer <token>` on every request.

If `CLIP_BASE_PATH=/openai-clip`, every route receives that prefix, for example `/openai-clip/healthz`, `/openai-clip/v1/encryption/public-key`, `/openai-clip/v1/classify`, and `/openai-clip/v1/classify/encrypted`.

Example:

```powershell
curl.exe -H "Authorization: Bearer <token>" -F "file=@C:\path\image.heic" "http://127.0.0.1:8000/v1/classify?top_k=3"
```

Batch example:

```powershell
curl.exe -H "Authorization: Bearer <token>" -F "files=@C:\path\one.png" -F "files=@C:\path\two.bmp" "http://127.0.0.1:8000/v1/classify/batch?top_k=3"
```

Encrypted example:

```powershell
.\.venv\Scripts\python.exe scripts\encrypt_sample_image.py C:\path\image.heic --base-url http://127.0.0.1:8000 --token <token>
```

External encrypted batch smoke test:

```powershell
python test_external_api.py C:\path\one.png C:\path\two.heic --base-url https://example.internal/openai-clip --batch-size 8 --top-k 3
```

## Production notes

- Put this service behind an internal gateway or reverse proxy for TLS and rate limiting.
- Enable `CLIP_AUTH_ENABLED=true` and keep `CLIP_API_TOKENS` private.
- Keep `CLIP_MAX_IMAGE_BYTES`, `CLIP_MAX_ENCRYPTED_IMAGE_BYTES`, `CLIP_MAX_MULTIPART_OVERHEAD_BYTES`, and `CLIP_MAX_IMAGE_PIXELS` conservative to prevent upload, decompression, and memory abuse.
- Keep `config/server_private.key` private and out of git.
- Pre-download model weights during deployment, not on first live request.
- Monitor `/metrics` for latency, request status, and error rates using a client that sends the Bearer token.
