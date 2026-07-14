from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from clip_classifier_service.classifier import ClipClassifier
from clip_classifier_service.encryption import (
    ENCRYPTION_ALGORITHM,
    decrypt_encrypted_image_payload,
    read_key_b64,
)
from clip_classifier_service.labels import load_labels
from clip_classifier_service.metrics import INFERENCE_SECONDS, REQUESTS
from clip_classifier_service.openclip_backend import OpenClipBackend, ensure_model_cache_dir
from clip_classifier_service.settings import Settings


def create_classifier(settings: Settings) -> ClipClassifier:
    ensure_model_cache_dir(settings.model_cache_dir)
    labels = load_labels(settings.labels_path)
    backend = OpenClipBackend(settings)
    return ClipClassifier(labels=labels, backend=backend, model_name=settings.model_name)


class EncryptedImagePayload(BaseModel):
    alg: str
    encrypted_key: str
    nonce: str
    ciphertext: str
    filename: str | None = None


class EncryptedBatchPayload(BaseModel):
    items: list[EncryptedImagePayload]


async def read_upload_limited(file: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="image exceeds max_image_bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def serialize_result(result):
    return {
        "model": result.model,
        "elapsed_ms": result.elapsed_ms,
        "labels": [
            {"id": label.id, "name": label.name, "score": label.score}
            for label in result.labels
        ],
    }


def encrypted_payload_to_dict(payload: EncryptedImagePayload) -> dict[str, str]:
    return {
        "alg": payload.alg,
        "encrypted_key": payload.encrypted_key,
        "nonce": payload.nonce,
        "ciphertext": payload.ciphertext,
    }


def get_encryption_private_key(settings: Settings) -> str:
    if not settings.encryption_enabled:
        raise HTTPException(status_code=404, detail="encrypted uploads are not enabled")
    try:
        return read_key_b64(settings.encryption_private_key_path)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def create_app(
    settings: Settings | None = None,
    classifier=None,
    *,
    load_on_startup: bool = True,
) -> FastAPI:
    settings = settings or Settings()
    state = {"classifier": classifier}
    gpu_semaphore = asyncio.Semaphore(1)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["classifier"] is None and load_on_startup:
            state["classifier"] = create_classifier(settings)
        app.state.classifier = state["classifier"]
        yield

    app = FastAPI(title=settings.service_name, version="0.1.0", lifespan=lifespan)
    router = APIRouter()
    classify_route_prefix = f"{settings.base_path}/v1/classify"

    @app.middleware("http")
    async def request_id_middleware(request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.middleware("http")
    async def content_length_limit_middleware(request: Request, call_next):
        if request.url.path.startswith(classify_route_prefix):
            content_length = request.headers.get("content-length")
            try:
                parsed_content_length = int(content_length) if content_length else None
            except ValueError:
                return Response(content="invalid content-length", status_code=400)
            encrypted_request = request.url.path.endswith("/encrypted")
            batch_request = request.url.path.endswith("/batch") or request.url.path.endswith(
                "/batch/encrypted"
            )
            per_image_limit = (
                settings.max_encrypted_image_bytes
                if encrypted_request
                else settings.max_image_bytes + settings.max_multipart_overhead_bytes
            )
            max_body_bytes = (
                per_image_limit * settings.max_batch_size if batch_request else per_image_limit
            )
            if parsed_content_length and parsed_content_length > max_body_bytes:
                return Response(content="request body too large", status_code=413)
        return await call_next(request)

    @router.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    def readyz() -> dict[str, str]:
        current = state.get("classifier")
        if current is None or not getattr(current, "ready", False):
            raise HTTPException(status_code=503, detail="model is not ready")
        return {"status": "ready"}

    @router.get("/metrics")
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @router.get("/v1/encryption/public-key")
    def encryption_public_key() -> dict[str, str]:
        if not settings.encryption_enabled:
            raise HTTPException(status_code=404, detail="encrypted uploads are not enabled")
        try:
            public_key = read_key_b64(settings.encryption_public_key_path)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"alg": ENCRYPTION_ALGORITHM, "public_key": public_key}

    @router.post("/v1/classify")
    async def classify(
        file: Annotated[UploadFile, File()],
        top_k: Annotated[int, Query(ge=1, le=100)] = settings.default_top_k,
        threshold: Annotated[float, Query(ge=0.0, le=1.0)] = settings.default_threshold,
    ):
        if settings.require_encrypted_uploads:
            raise HTTPException(status_code=403, detail="encrypted uploads are required")
        if file.size is not None and file.size > settings.max_image_bytes:
            REQUESTS.labels(endpoint="/v1/classify", status="too_large").inc()
            raise HTTPException(status_code=413, detail="image exceeds max_image_bytes")
        image_bytes = await read_upload_limited(file, settings.max_image_bytes)

        current = state.get("classifier")
        if current is None or not getattr(current, "ready", False):
            REQUESTS.labels(endpoint="/v1/classify", status="not_ready").inc()
            raise HTTPException(status_code=503, detail="model is not ready")

        try:
            async with gpu_semaphore:
                with INFERENCE_SECONDS.time():
                    result = await asyncio.to_thread(
                        current.classify,
                        image_bytes=image_bytes,
                        top_k=top_k,
                        threshold=threshold,
                    )
        except ValueError as exc:
            REQUESTS.labels(endpoint="/v1/classify", status="bad_request").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        REQUESTS.labels(endpoint="/v1/classify", status="ok").inc()
        return serialize_result(result)

    @router.post("/v1/classify/encrypted")
    async def classify_encrypted(
        payload: EncryptedImagePayload,
        top_k: Annotated[int, Query(ge=1, le=100)] = settings.default_top_k,
        threshold: Annotated[float, Query(ge=0.0, le=1.0)] = settings.default_threshold,
    ):
        current = state.get("classifier")
        if current is None or not getattr(current, "ready", False):
            REQUESTS.labels(endpoint="/v1/classify/encrypted", status="not_ready").inc()
            raise HTTPException(status_code=503, detail="model is not ready")

        try:
            image_bytes = decrypt_encrypted_image_payload(
                encrypted_payload_to_dict(payload),
                get_encryption_private_key(settings),
            )
        except ValueError as exc:
            REQUESTS.labels(endpoint="/v1/classify/encrypted", status="bad_request").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if len(image_bytes) > settings.max_image_bytes:
            REQUESTS.labels(endpoint="/v1/classify/encrypted", status="too_large").inc()
            raise HTTPException(status_code=413, detail="image exceeds max_image_bytes")

        try:
            async with gpu_semaphore:
                with INFERENCE_SECONDS.time():
                    result = await asyncio.to_thread(
                        current.classify,
                        image_bytes=image_bytes,
                        top_k=top_k,
                        threshold=threshold,
                    )
        except ValueError as exc:
            REQUESTS.labels(endpoint="/v1/classify/encrypted", status="bad_request").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        REQUESTS.labels(endpoint="/v1/classify/encrypted", status="ok").inc()
        return serialize_result(result)

    @router.post("/v1/classify/batch")
    async def classify_batch(
        files: Annotated[list[UploadFile], File()],
        top_k: Annotated[int, Query(ge=1, le=100)] = settings.default_top_k,
        threshold: Annotated[float, Query(ge=0.0, le=1.0)] = settings.default_threshold,
    ):
        if settings.require_encrypted_uploads:
            raise HTTPException(status_code=403, detail="encrypted uploads are required")
        if len(files) > settings.max_batch_size:
            REQUESTS.labels(endpoint="/v1/classify/batch", status="too_many").inc()
            raise HTTPException(status_code=413, detail="batch exceeds max_batch_size")

        current = state.get("classifier")
        if current is None or not getattr(current, "ready", False):
            REQUESTS.labels(endpoint="/v1/classify/batch", status="not_ready").inc()
            raise HTTPException(status_code=503, detail="model is not ready")

        results = []
        image_batch: list[bytes] = []
        for file in files:
            if file.size is not None and file.size > settings.max_image_bytes:
                REQUESTS.labels(endpoint="/v1/classify/batch", status="too_large").inc()
                raise HTTPException(
                    status_code=413,
                    detail=f"{file.filename} exceeds max_image_bytes",
                )
            try:
                image_bytes = await read_upload_limited(file, settings.max_image_bytes)
            except HTTPException as exc:
                REQUESTS.labels(endpoint="/v1/classify/batch", status="too_large").inc()
                raise HTTPException(
                    status_code=413,
                    detail=f"{file.filename} exceeds max_image_bytes",
                ) from exc
            image_batch.append(image_bytes)

        try:
            with INFERENCE_SECONDS.time():
                async with gpu_semaphore:
                    batch_results = await asyncio.to_thread(
                        current.classify_batch,
                        image_batch=image_batch,
                        top_k=top_k,
                        threshold=threshold,
                    )
        except ValueError as exc:
            REQUESTS.labels(endpoint="/v1/classify/batch", status="bad_request").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for file, result in zip(files, batch_results, strict=True):
            item = serialize_result(result)
            item["filename"] = file.filename
            results.append(item)

        REQUESTS.labels(endpoint="/v1/classify/batch", status="ok").inc()
        return {"results": results}

    @router.post("/v1/classify/batch/encrypted")
    async def classify_batch_encrypted(
        payload: EncryptedBatchPayload,
        top_k: Annotated[int, Query(ge=1, le=100)] = settings.default_top_k,
        threshold: Annotated[float, Query(ge=0.0, le=1.0)] = settings.default_threshold,
    ):
        if len(payload.items) > settings.max_batch_size:
            REQUESTS.labels(endpoint="/v1/classify/batch/encrypted", status="too_many").inc()
            raise HTTPException(status_code=413, detail="batch exceeds max_batch_size")

        current = state.get("classifier")
        if current is None or not getattr(current, "ready", False):
            REQUESTS.labels(endpoint="/v1/classify/batch/encrypted", status="not_ready").inc()
            raise HTTPException(status_code=503, detail="model is not ready")

        private_key_b64 = get_encryption_private_key(settings)
        image_batch: list[bytes] = []
        for item in payload.items:
            try:
                image_bytes = decrypt_encrypted_image_payload(
                    encrypted_payload_to_dict(item),
                    private_key_b64,
                )
            except ValueError as exc:
                REQUESTS.labels(endpoint="/v1/classify/batch/encrypted", status="bad_request").inc()
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if len(image_bytes) > settings.max_image_bytes:
                REQUESTS.labels(endpoint="/v1/classify/batch/encrypted", status="too_large").inc()
                raise HTTPException(status_code=413, detail="image exceeds max_image_bytes")
            image_batch.append(image_bytes)

        try:
            with INFERENCE_SECONDS.time():
                async with gpu_semaphore:
                    batch_results = await asyncio.to_thread(
                        current.classify_batch,
                        image_batch=image_batch,
                        top_k=top_k,
                        threshold=threshold,
                    )
        except ValueError as exc:
            REQUESTS.labels(endpoint="/v1/classify/batch/encrypted", status="bad_request").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        results = []
        for item, result in zip(payload.items, batch_results, strict=True):
            serialized = serialize_result(result)
            if item.filename is not None:
                serialized["filename"] = item.filename
            results.append(serialized)

        REQUESTS.labels(endpoint="/v1/classify/batch/encrypted", status="ok").inc()
        return {"results": results}

    app.include_router(router, prefix=settings.base_path)
    return app
