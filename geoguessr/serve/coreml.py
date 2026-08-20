"""CoreML conversion for the vision encoder.

Encoding one image is the whole inference cost -- the heads are a matrix
multiply. Measured on this machine the PyTorch/MPS path runs ~1.2s per image
against a ~50ms target, so this is the difference between a tool that feels
instant and one you stop reaching for.

Only the **vision tower** is converted. Text encoding is used exactly once, by
the zero-shot baseline, and never on the live path. The heads stay in numpy.

Conversion is slow (minutes) and produces a ~1.5GB ``.mlpackage``, so it is a
build step, not something the live path ever does. :class:`CoreMLBackbone`
mirrors :class:`~geoguessr.embed.backbone.Backbone`'s ``encode_images`` so it
drops into the predictor unchanged.

Important: a CoreML backbone must **not** be used to build training embeddings
unless the head was trained on CoreML-produced vectors. Conversion is lossy
(fp16, fused ops); mixing the two silently shifts the embedding space. Use it
for serving only, and check :func:`compare_backends` before trusting it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..embed.backbone import Backbone

__all__ = ["convert_vision_encoder", "CoreMLBackbone", "compare_backends"]


class _VisionWrapper(torch.nn.Module):
    """Traceable module returning L2-normalized image embeddings.

    Normalization is folded in so the CoreML graph produces exactly what
    ``Backbone.encode_images`` produces, rather than something the caller has
    to remember to post-process.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.clip.get_image_features(pixel_values=pixel_values)
        features = out if isinstance(out, torch.Tensor) else out.pooler_output
        return features / features.norm(dim=-1, keepdim=True)


def convert_vision_encoder(
    backbone: Backbone,
    out_path: str | Path = "artifacts/streetclip_vision.mlpackage",
    *,
    compute_units: str = "ALL",
) -> Path:
    """Trace the vision tower and convert it to a CoreML package.

    Args:
        backbone: a loaded :class:`Backbone`. It is moved to CPU/fp32 for
            tracing -- MPS tensors cannot be traced reliably.
        compute_units: ``"ALL"`` lets CoreML use the Neural Engine, which is
            the entire point. ``"CPU_ONLY"`` is useful for debugging a
            numerical mismatch.
    """
    import coremltools as ct

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size = backbone.image_size
    model = backbone.model.to("cpu").float().eval()
    wrapper = _VisionWrapper(model).eval()

    example = torch.rand(1, 3, size, size)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=example.shape)],
        outputs=[ct.TensorType(name="embedding")],
        compute_units=getattr(ct.ComputeUnit, compute_units),
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.short_description = (
        f"{backbone.model_id} vision encoder, L2-normalized "
        f"{backbone.embed_dim}-d output"
    )
    mlmodel.save(str(out_path))
    return out_path


class CoreMLBackbone:
    """Serving-time stand-in for :class:`Backbone`, backed by CoreML.

    Exposes ``encode_images`` with identical semantics, so
    :class:`~geoguessr.serve.predictor.Predictor` needs no changes. Reuses the
    original processor so preprocessing is bit-identical to training.
    """

    def __init__(
        self,
        package: str | Path,
        *,
        model_id: str = "geolocal/StreetCLIP",
        processor=None,
    ):
        import coremltools as ct

        self.model_id = model_id
        self.package = Path(package)
        if not self.package.exists():
            raise FileNotFoundError(
                f"no CoreML package at {self.package}. "
                "Run scripts/convert_coreml.py first."
            )
        self.model = ct.models.MLModel(str(self.package))

        if processor is None:
            from transformers import CLIPProcessor

            processor = CLIPProcessor.from_pretrained(model_id)
        self.processor = processor

    @property
    def embed_dim(self) -> int:
        spec = self.model.get_spec()
        return int(spec.description.output[0].type.multiArrayType.shape[-1])

    def encode_images(self, images: list[Image], *, normalize: bool = True) -> np.ndarray:
        """Encode images to ``(B, dim)``. Normalization is already baked in.

        CoreML models are traced at a fixed batch size of 1, so a batch is run
        as a loop. That is fine here -- the live path encodes one screenshot.
        """
        if not images:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        inputs = self.processor(images=images, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)

        rows = [
            self.model.predict({"pixel_values": pixel_values[i : i + 1]})["embedding"]
            for i in range(pixel_values.shape[0])
        ]
        return np.vstack(rows).astype(np.float32)

    def __repr__(self) -> str:
        return f"CoreMLBackbone({self.package.name}, dim={self.embed_dim})"


def compare_backends(
    torch_backbone: Backbone, coreml_backbone: CoreMLBackbone, images: list[Image]
) -> dict[str, float]:
    """Quantify how much conversion changed the embeddings.

    Cosine similarity should sit above ~0.99. Anything lower means the CoreML
    encoder is describing images differently from the one the head was trained
    on, and predictions will drift in ways no test would otherwise catch.
    """
    a = torch_backbone.encode_images(images)
    b = coreml_backbone.encode_images(images)

    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    cosine = (a * b).sum(axis=1)

    return {
        "mean_cosine": float(cosine.mean()),
        "min_cosine": float(cosine.min()),
        "max_abs_diff": float(np.abs(a - b).max()),
    }
