import pytest

from clip_classifier_service.settings import Settings


@pytest.fixture(autouse=True)
def isolate_settings_environment(monkeypatch):
    """Keep repository-local production .env values from changing unit-test defaults."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in (
        "CLIP_API_TOKENS",
        "CLIP_AUTH_DISABLE_DOCS",
        "CLIP_AUTH_ENABLED",
        "CLIP_BASE_PATH",
        "CLIP_DEFAULT_THRESHOLD",
        "CLIP_DEFAULT_TOP_K",
        "CLIP_DEVICE",
        "CLIP_ENCRYPTION_ENABLED",
        "CLIP_ENCRYPTION_PRIVATE_KEY_PATH",
        "CLIP_ENCRYPTION_PUBLIC_KEY_PATH",
        "CLIP_HOST",
        "CLIP_LABELS_PATH",
        "CLIP_LOG_LEVEL",
        "CLIP_MAX_BATCH_SIZE",
        "CLIP_MAX_ENCRYPTED_IMAGE_BYTES",
        "CLIP_MAX_IMAGE_BYTES",
        "CLIP_MAX_IMAGE_PIXELS",
        "CLIP_MAX_MULTIPART_OVERHEAD_BYTES",
        "CLIP_MODEL_CACHE_DIR",
        "CLIP_MODEL_NAME",
        "CLIP_PORT",
        "CLIP_PRECISION",
        "CLIP_REQUIRE_ENCRYPTED_UPLOADS",
        "CLIP_SERVICE_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
