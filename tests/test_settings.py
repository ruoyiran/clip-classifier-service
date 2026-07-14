from pathlib import Path

from clip_classifier_service.settings import Settings


def test_settings_uses_safe_defaults(tmp_path: Path) -> None:
    settings = Settings(model_cache_dir=tmp_path / "models")

    assert settings.model_name == "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
    assert settings.base_path == ""
    assert settings.auth_enabled is False
    assert settings.api_tokens == ()
    assert settings.auth_disable_docs is True
    assert settings.device == "cuda"
    assert settings.precision == "fp16"
    assert settings.max_image_bytes == 8 * 1024 * 1024
    assert settings.model_cache_dir == tmp_path / "models"


def test_settings_rejects_invalid_limits(tmp_path: Path) -> None:
    try:
        Settings(model_cache_dir=tmp_path, max_batch_size=0)
    except ValueError as exc:
        assert "max_batch_size" in str(exc)
    else:
        raise AssertionError("max_batch_size=0 should be rejected")


def test_settings_normalizes_base_path(tmp_path: Path) -> None:
    settings = Settings(model_cache_dir=tmp_path, base_path="openai-clip/")

    assert settings.base_path == "/openai-clip"


def test_settings_rejects_invalid_base_path(tmp_path: Path) -> None:
    try:
        Settings(model_cache_dir=tmp_path, base_path="/openai clip")
    except ValueError as exc:
        assert "base_path" in str(exc)
    else:
        raise AssertionError("base_path with spaces should be rejected")


def test_settings_parses_multiple_api_tokens(tmp_path: Path) -> None:
    settings = Settings(
        model_cache_dir=tmp_path,
        api_tokens=" primary-token, secondary-token ,, ",
    )

    assert settings.api_tokens == ("primary-token", "secondary-token")


def test_settings_parses_api_tokens_from_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLIP_AUTH_ENABLED", "true")
    monkeypatch.setenv("CLIP_API_TOKENS", "primary-token,secondary-token")

    settings = Settings(model_cache_dir=tmp_path)

    assert settings.auth_enabled is True
    assert settings.api_tokens == ("primary-token", "secondary-token")


def test_settings_rejects_enabled_auth_without_tokens(tmp_path: Path) -> None:
    try:
        Settings(model_cache_dir=tmp_path, auth_enabled=True, api_tokens="")
    except ValueError as exc:
        assert "api_tokens" in str(exc)
    else:
        raise AssertionError("auth_enabled=True without api_tokens should be rejected")
