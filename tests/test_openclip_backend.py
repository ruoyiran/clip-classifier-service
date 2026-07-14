from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from clip_classifier_service.openclip_backend import OpenClipBackend


def make_image_bytes(format_name: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format=format_name)
    return buffer.getvalue()


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    backend = SimpleNamespace(
        settings=SimpleNamespace(max_image_pixels=1_000),
        preprocess=lambda image: torch.tensor([image.mode == "RGB", image.size == (4, 4)]),
    )
    return OpenClipBackend._preprocess_image(backend, image_bytes)


@pytest.mark.parametrize("format_name", ["PNG", "BMP"])
def test_preprocess_image_accepts_pillow_image_formats(format_name: str) -> None:
    result = preprocess_image(make_image_bytes(format_name))

    assert result.tolist() == [True, True]


def test_preprocess_image_accepts_heic() -> None:
    from pillow_heif import from_pillow

    image = Image.new("RGB", (4, 4), "red")
    heif_file = from_pillow(image)
    buffer = io.BytesIO()
    heif_file.save(buffer)

    result = preprocess_image(buffer.getvalue())

    assert result.tolist() == [True, True]


def test_preprocess_image_rejects_invalid_image_bytes() -> None:
    with pytest.raises(ValueError, match="invalid image"):
        preprocess_image(b"not an image")
