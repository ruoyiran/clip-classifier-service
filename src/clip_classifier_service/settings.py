from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLIP_",
        extra="ignore",
    )

    model_name: str = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
    base_path: str = ""
    model_cache_dir: Path = Path("models")
    labels_path: Path = Path("config/labels.yaml")
    device: str = "cuda"
    precision: Literal["fp16", "fp32"] = "fp16"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    max_image_bytes: int = 30 * 1024 * 1024
    max_encrypted_image_bytes: int = 42 * 1024 * 1024
    max_multipart_overhead_bytes: int = 64 * 1024
    max_image_pixels: int = 16_000_000
    max_batch_size: int = 8
    default_top_k: int = 5
    default_threshold: float = 0.0
    service_name: str = "clip-classifier-service"
    auth_enabled: bool = False
    api_tokens: Annotated[tuple[str, ...], NoDecode] = ()
    auth_disable_docs: bool = True
    encryption_enabled: bool = False
    require_encrypted_uploads: bool = False
    encryption_private_key_path: Path = Path("config/server_private.key")
    encryption_public_key_path: Path = Path("config/server_public.key")

    @field_validator(
        "max_image_bytes",
        "max_encrypted_image_bytes",
        "max_image_pixels",
        "max_batch_size",
        "default_top_k",
    )
    @classmethod
    def validate_positive_ints(cls, value: int, info):
        if value < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return value

    @field_validator("max_multipart_overhead_bytes")
    @classmethod
    def validate_non_negative_ints(cls, value: int, info):
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return value

    @field_validator("default_threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("default_threshold must be between 0.0 and 1.0")
        return value

    @field_validator("api_tokens", mode="before")
    @classmethod
    def parse_api_tokens(cls, value):
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(token.strip() for token in value.split(",") if token.strip())
        return tuple(token.strip() for token in value if str(token).strip())

    @field_validator("base_path")
    @classmethod
    def normalize_base_path(cls, value: str) -> str:
        value = value.strip()
        if value in {"", "/"}:
            return ""
        if " " in value or "?" in value or "#" in value:
            raise ValueError(
                "base_path must be a URL path prefix without spaces, query, or fragment"
            )
        if not value.startswith("/"):
            value = f"/{value}"
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_auth_configuration(self):
        if self.auth_enabled and not self.api_tokens:
            raise ValueError("api_tokens must contain at least one token when auth_enabled is true")
        return self
