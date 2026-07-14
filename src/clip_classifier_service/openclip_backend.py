from __future__ import annotations

import io
import os
from pathlib import Path

import open_clip
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from clip_classifier_service.settings import Settings


class OpenClipBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.device)
        self.dtype = (
            torch.float16
            if settings.precision == "fp16" and self.device.type == "cuda"
            else torch.float32
        )

        os.environ.setdefault("HF_HOME", str(settings.model_cache_dir.resolve()))
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.model_name,
            device=self.device,
            precision=settings.precision if self.device.type == "cuda" else "fp32",
        )
        self.tokenizer = open_clip.get_tokenizer(settings.model_name)
        self.model.eval()

    def encode_texts(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens)
        return features.float()

    def encode_image_bytes(self, image_bytes: bytes) -> torch.Tensor:
        tensor = self._preprocess_image(image_bytes).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            features = self.model.encode_image(tensor)
        return features.float()

    def encode_image_batch(self, image_batch: list[bytes]) -> torch.Tensor:
        tensors = [self._preprocess_image(image_bytes) for image_bytes in image_batch]
        batch = torch.stack(tensors).to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            features = self.model.encode_image(batch)
        return features.float()

    def _preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        Image.MAX_IMAGE_PIXELS = self.settings.max_image_pixels
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            if image.width * image.height > self.settings.max_image_pixels:
                raise ValueError("image exceeds max_image_pixels")
            image = ImageOps.exif_transpose(image).convert("RGB")
        except ValueError:
            raise
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
            raise ValueError("invalid image") from exc
        return self.preprocess(image)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(device)


def ensure_model_cache_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
