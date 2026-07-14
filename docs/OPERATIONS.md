# Operations

## Local Setup

```powershell
cd C:\Users\Admin\workspace\AI\clip-classifier-service
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
Copy-Item .env.example .env
```

Edit `.env` and `config/labels.yaml` before production use.

## Configuration

All service settings can be provided with `CLIP_` environment variables.

- `CLIP_MODEL_NAME`: OpenCLIP model name. Default: `hf-hub:apple/DFN5B-CLIP-ViT-H-14-378`.
- `CLIP_BASE_PATH`: Optional API path prefix such as `/openai-clip`. Default: empty.
- `CLIP_MODEL_CACHE_DIR`: Model cache path. Default: `models`.
- `CLIP_LABELS_PATH`: Label YAML path. Default: `config/labels.yaml`.
- `CLIP_DEVICE`: Runtime device. Default: `cuda`.
- `CLIP_PRECISION`: `fp16` or `fp32`. Default: `fp16`.
- `CLIP_MAX_IMAGE_BYTES`: Per-image upload byte limit. Default: `31457280` bytes, 30 MiB.
- `CLIP_MAX_ENCRYPTED_IMAGE_BYTES`: Per-image encrypted JSON payload limit. Default: `44040192` bytes, 42 MiB.
- `CLIP_MAX_MULTIPART_OVERHEAD_BYTES`: Request body overhead allowance.
- `CLIP_MAX_IMAGE_PIXELS`: Decoded image pixel limit.
- `CLIP_MAX_BATCH_SIZE`: Max images per batch request.
- `CLIP_DEFAULT_TOP_K`: Default number of labels returned.
- `CLIP_DEFAULT_THRESHOLD`: Default softmax `score` threshold. It does not filter raw `similarity`.
- `CLIP_AUTH_ENABLED`: Enables static Bearer Token authentication for every API route.
- `CLIP_API_TOKENS`: Comma-separated valid Bearer tokens. Required when auth is enabled.
- `CLIP_AUTH_DISABLE_DOCS`: Disables `/docs`, `/redoc`, and `/openapi.json` when auth is enabled.
- `CLIP_ENCRYPTION_ENABLED`: Enables encrypted upload endpoints.
- `CLIP_REQUIRE_ENCRYPTED_UPLOADS`: Rejects plaintext classify endpoints when enabled.
- `CLIP_ENCRYPTION_PRIVATE_KEY_PATH`: Service private key path.
- `CLIP_ENCRYPTION_PUBLIC_KEY_PATH`: Service public key path.

## API Tokens

Generate a long random token outside the repository and configure it in `.env`:

```powershell
CLIP_AUTH_ENABLED=true
CLIP_API_TOKENS=<long-random-token>
CLIP_AUTH_DISABLE_DOCS=true
```

Use comma-separated tokens during rotation:

```powershell
CLIP_API_TOKENS=<old-token>,<new-token>
```

## Encryption Keys

Generate keys once per deployment:

```powershell
.\.venv\Scripts\python.exe scripts\generate_encryption_keys.py
```

Do not commit `config/server_private.key` or `config/server_public.key`.

## Verification

Run before deployment:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts\check_cuda.py
.\.venv\Scripts\python.exe scripts\preload_model.py
```

`scripts/preload_model.py` downloads and initializes the model. Set `HF_TOKEN` to improve Hugging Face rate limits.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn clip_classifier_service.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Use one worker per GPU.

## Windows Service

Install NSSM first, then:

```powershell
.\scripts\install_windows_service.ps1 -ServiceName ClipClassifierService -Port 8000
Start-Service ClipClassifierService
```

The script writes service logs under `logs/`.

## API Smoke Tests

```powershell
$token = "<token>"
curl.exe -H "Authorization: Bearer $token" http://127.0.0.1:8000/healthz
curl.exe -H "Authorization: Bearer $token" http://127.0.0.1:8000/readyz
curl.exe -H "Authorization: Bearer $token" http://127.0.0.1:8000/metrics
curl.exe -H "Authorization: Bearer $token" -F "file=@C:\path\image.jpg" "http://127.0.0.1:8000/v1/classify?top_k=3"
curl.exe -H "Authorization: Bearer $token" -F "files=@C:\path\one.jpg" -F "files=@C:\path\two.jpg" "http://127.0.0.1:8000/v1/classify/batch?top_k=3"
.\.venv\Scripts\python.exe scripts\encrypt_sample_image.py C:\path\image.jpg --base-url http://127.0.0.1:8000 --token $token
```

The `threshold` query parameter filters by returned softmax `score`. If a client
needs raw cosine similarity thresholds, request `threshold=0.0` with enough
`top_k` labels and apply thresholds to each label's `similarity` field.

If `CLIP_BASE_PATH=/openai-clip`, include that prefix in every URL:

```powershell
curl.exe -H "Authorization: Bearer $token" http://127.0.0.1:8000/openai-clip/healthz
curl.exe -H "Authorization: Bearer $token" -F "file=@C:\path\image.jpg" "http://127.0.0.1:8000/openai-clip/v1/classify?top_k=3"
```

## Production Checklist

- Replace default labels with the real production label set.
- Preload model weights during deployment.
- Enable `CLIP_AUTH_ENABLED=true` and configure a private `CLIP_API_TOKENS` value.
- Generate encryption keys before setting `CLIP_ENCRYPTION_ENABLED=true`.
- Put the service behind an internal gateway or reverse proxy.
- Enforce TLS and rate limiting upstream.
- Keep `--workers 1` per GPU unless measured otherwise.
- Monitor `/metrics`, GPU memory, process memory, and request latency with authenticated clients.
- Keep `.env`, model cache, logs, and local virtual environments out of git.
