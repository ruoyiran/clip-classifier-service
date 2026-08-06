from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import open_clip
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from clip_classifier_service.settings import Settings

register_heif_opener(thumbnails=False)

# Route operators that Metal does not implement to the CPU instead of raising.
# Must be set before the first torch MPS call.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class OpenClipBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.device)
        self.dtype = (
            torch.float16
            if settings.precision == "fp16" and self.device.type == "cuda"
            else torch.float32
        )

        self._apply_model_cache_dir(settings.model_cache_dir)
        self.model, _, self.preprocess = self._create_model()
        self.tokenizer = open_clip.get_tokenizer(settings.model_name)
        self.model.eval()

    @staticmethod
    def _apply_model_cache_dir(model_cache_dir: Path) -> None:
        """Redirect Hugging Face downloads into the configured cache directory.

        Setting HF_HOME here is not enough on its own: huggingface_hub reads it
        into module constants at import time, and open_clip has already imported
        it by then. Without also rewriting those constants, CLIP_MODEL_CACHE_DIR
        is silently ignored and weights land in ~/.cache/huggingface.
        """
        resolved = model_cache_dir.resolve()
        os.environ["HF_HOME"] = str(resolved)
        hub_cache = resolved / "hub"
        os.environ["HF_HUB_CACHE"] = str(hub_cache)
        hub_cache.mkdir(parents=True, exist_ok=True)

        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HOME = str(resolved)
        hf_constants.HF_HUB_CACHE = str(hub_cache)
        # open_clip and huggingface_hub.file_download bind the cache path at
        # import time, so rewrite it wherever it was already captured.
        for module_name in ("huggingface_hub.file_download", "huggingface_hub._snapshot_download"):
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, "HF_HUB_CACHE"):
                module.HF_HUB_CACHE = str(hub_cache)

    def _create_model(self):
        try:
            return open_clip.create_model_and_transforms(
                self.settings.model_name,
                device=self.device,
                precision=self._precision_for(self.device),
            )
        except (RuntimeError, NotImplementedError):
            # Metal can fail to place a model this large. CPU always works, so
            # degrade rather than leave the service unable to start.
            if self.device.type != "mps":
                raise
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            return open_clip.create_model_and_transforms(
                self.settings.model_name,
                device=self.device,
                precision=self._precision_for(self.device),
            )

    def _precision_for(self, device: torch.device) -> str:
        return self.settings.precision if device.type == "cuda" else "fp32"

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
        if device == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        return torch.device(device)


def ensure_model_cache_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
