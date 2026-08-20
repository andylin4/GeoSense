"""Frozen CLIP backbone.

This module is the only place that decides which encoder is used and how images
are preprocessed. That matters because of the invalidation rule in the design:
changing the backbone or the crop makes every cached embedding meaningless,
while changing labels costs nothing. Everything that can invalidate the
embedding cache lives here and gets recorded in :attr:`Backbone.fingerprint`,
so a stale ``embeddings.npy`` can be detected instead of silently misused.

The default is StreetCLIP, domain-matched to street-level imagery. The
backbone is always frozen; nothing in this project fine-tunes it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = ["Backbone", "pick_device", "STREETCLIP", "CLIP_L14", "CLIP_B32"]

STREETCLIP = "geolocal/StreetCLIP"
CLIP_L14 = "openai/clip-vit-large-patch14"
CLIP_B32 = "openai/clip-vit-base-patch32"


def _as_embedding(output: Any) -> torch.Tensor:
    """Pull the projected embedding out of a CLIP feature call.

    transformers <5 returned a bare tensor from ``get_image_features``;
    transformers >=5 returns a ``BaseModelOutputWithPooling`` whose
    ``pooler_output`` holds the projected embedding. Support both so a library
    upgrade cannot silently change what gets cached.
    """
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(
        f"cannot extract an embedding tensor from {type(output).__name__}"
    )


def pick_device(prefer: str | None = None) -> torch.device:
    """MPS on Apple Silicon, CUDA on Colab, CPU otherwise."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class Fingerprint:
    """Identity of everything that affects an embedding's value."""

    model_id: str
    embed_dim: int
    image_size: int

    @property
    def key(self) -> str:
        raw = f"{self.model_id}|{self.embed_dim}|{self.image_size}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "embed_dim": self.embed_dim,
            "image_size": self.image_size,
            "key": self.key,
        }


class Backbone:
    """A frozen CLIP encoder plus its preprocessing, loaded once and reused."""

    def __init__(
        self,
        model_id: str = STREETCLIP,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.model_id = model_id
        self.device = pick_device(device)
        # fp16 on MPS is a real speedup; CPU stays fp32 for numerical sanity.
        self.dtype = dtype or (
            torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        )

        self.model = CLIPModel.from_pretrained(model_id, dtype=self.dtype)
        self.model.eval().to(self.device)
        self.model.requires_grad_(False)  # frozen, always

        self.processor = CLIPProcessor.from_pretrained(model_id)

    @property
    def embed_dim(self) -> int:
        return int(self.model.config.projection_dim)

    @property
    def image_size(self) -> int:
        # transformers returns a plain dict, a SizeDict, or a bare int here
        # depending on version, so probe rather than assume.
        size = self.processor.image_processor.crop_size
        if isinstance(size, int):
            return size
        height = size["height"] if hasattr(size, "__getitem__") else size.height
        return int(height)

    @property
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(self.model_id, self.embed_dim, self.image_size)

    @torch.inference_mode()
    def encode_images(self, images: list[Image], *, normalize: bool = True) -> np.ndarray:
        """Encode a batch of PIL images to ``(B, embed_dim)`` float32.

        Images arrive already cropped -- this method does not crop. Cropping is
        the caller's job precisely so training and inference can be verified to
        use the same rule.
        """
        if not images:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        features = _as_embedding(
            self.model.get_image_features(pixel_values=pixel_values)
        )
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], *, normalize: bool = True) -> np.ndarray:
        """Encode text prompts to ``(N, embed_dim)`` float32.

        Only the zero-shot baseline needs this; the trained heads never do.
        """
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        features = _as_embedding(
            self.model.get_text_features(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
            )
        )
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    @property
    def logit_scale(self) -> float:
        """CLIP's learned temperature, needed to turn similarities into probs."""
        return float(self.model.logit_scale.exp().item())

    def __repr__(self) -> str:
        fp = self.fingerprint
        return (
            f"Backbone({self.model_id!r}, dim={fp.embed_dim}, "
            f"image_size={fp.image_size}, device={self.device.type}, "
            f"dtype={str(self.dtype).removeprefix('torch.')})"
        )
