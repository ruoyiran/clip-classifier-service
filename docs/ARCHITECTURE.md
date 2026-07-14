# Architecture

## Goal

Serve `hf-hub:apple/DFN5B-CLIP-ViT-H-14-378` as a fixed-label image classification API on one local NVIDIA GPU.

## Runtime Components

1. `Settings` loads configuration from defaults, `.env`, and `CLIP_` environment variables.
2. `load_labels()` reads `config/labels.yaml` and returns enabled labels only. Labels may define one `prompt` or multiple `prompts`.
3. `OpenClipBackend` loads OpenCLIP model weights and preprocessing transforms.
4. `ClipClassifier` encodes all label prompts once and keeps normalized text features in memory.
   A label's similarity is the maximum cosine similarity across its prompts.
5. FastAPI serves classification, health, readiness, encryption metadata, and Prometheus metrics endpoints.
   If `CLIP_BASE_PATH` is configured, every route is mounted under that prefix.
6. Optional static Bearer Token authentication protects every API route when `CLIP_AUTH_ENABLED=true`.

## Startup Flow

```text
uvicorn -> clip_classifier_service.main -> create_app()
  -> lifespan startup
  -> create_classifier()
  -> ensure model cache dir
  -> load labels
  -> load OpenCLIP model
  -> precompute text features
  -> readyz returns ready
```

If model loading fails, the process startup fails. This is intentional for production because an unhealthy model service should not accept traffic.

## Single Image Request Flow

```text
POST /v1/classify, or POST <base_path>/v1/classify when CLIP_BASE_PATH is set
  -> request ID middleware
  -> content-length guard
  -> Bearer token dependency when auth is enabled
  -> FastAPI multipart parser
  -> per-file byte limit
  -> capped upload read
  -> single-GPU semaphore
  -> asyncio.to_thread(classifier.classify)
  -> Pillow decode with HEIC/HEIF plugin and pixel guard
  -> OpenCLIP image encode
  -> similarity ranking with softmax scores and raw best-prompt similarity
  -> JSON response
```

## Batch Request Flow

```text
POST /v1/classify/batch, or POST <base_path>/v1/classify/batch when CLIP_BASE_PATH is set
  -> request ID middleware
  -> content-length guard using max image bytes * max batch size
  -> Bearer token dependency when auth is enabled
  -> max batch size check
  -> per-file byte limit
  -> capped upload reads
  -> single-GPU semaphore
  -> asyncio.to_thread(classifier.classify_batch)
  -> stacked image tensor
  -> one model forward pass
  -> per-image JSON results
```

## Encrypted Request Flow

```text
GET /v1/encryption/public-key
  -> Bearer token dependency when auth is enabled
  -> returns AES-256-GCM+SealedBox-v1 and service public key

POST /v1/classify/encrypted
  -> content-length guard
  -> Bearer token dependency when auth is enabled
  -> JSON encrypted payload validation
  -> SealedBox unwraps one-time AES key with service private key
  -> AES-256-GCM decrypts image bytes
  -> existing image safety checks and CLIP classification
```

Encrypted batch follows the same flow per item, then uses the existing batch classifier path.

## Concurrency Model

The service uses one Uvicorn worker and a process-local `asyncio.Semaphore(1)` around GPU inference. Blocking PyTorch work is moved through `asyncio.to_thread()` so health, readiness, and metrics are less likely to be delayed by long inference.

Use one process per GPU unless memory and latency have been measured. Multiple Uvicorn workers on one GPU load multiple model copies.

## Safety Boundaries

- `CLIP_AUTH_ENABLED`: Requires `Authorization: Bearer <token>` on every API route.
- `CLIP_API_TOKENS`: Comma-separated valid tokens; startup fails if auth is enabled and empty.
- `CLIP_MAX_IMAGE_BYTES`: Per-image upload byte limit. Default: `31457280` bytes, 30 MiB.
- `CLIP_MAX_ENCRYPTED_IMAGE_BYTES`: Per-image encrypted payload byte limit. Default: `44040192` bytes, 42 MiB.
- `CLIP_MAX_MULTIPART_OVERHEAD_BYTES`: Request body allowance for multipart overhead.
- `CLIP_MAX_IMAGE_PIXELS`: Decoded pixel limit for Pillow.
- Supported decoded image inputs include JPEG/JPG, PNG, BMP, HEIC, and HEIF.
- Invalid image payloads are converted to `400`.
- Oversized payloads are converted to `413`.
- Query params are validated by FastAPI.

TLS, global rate limiting, and public internet exposure are intentionally out of process and should be handled by an upstream gateway.
Application-layer encryption protects uploaded images from accidental plaintext exposure inside intermediate components. Bearer Token authentication identifies authorized API clients but does not replace TLS.

## Observability

- `X-Request-ID` is propagated or generated for each response.
- `/metrics` exposes Prometheus counters and inference latency histogram and requires the Bearer token when auth is enabled.
- Current metrics focus on classifier routes. Gateway logs should provide request-level audit context.
