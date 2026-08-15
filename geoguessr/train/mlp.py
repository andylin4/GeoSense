"""Phase 3b: the MLP country head.

The linear probe answers "do these embeddings carry geography at all". This
answers "how much is left on the table". It trains on the same cached
embeddings, so a run is minutes of CPU -- no images are decoded.

Imbalance is still handled by reweighting, not resampling (decision #5); here
that is ``CrossEntropyLoss(weight=...)`` with inverse-frequency weights, the
direct equivalent of sklearn's ``class_weight="balanced"``.

Subclasses :class:`~geoguessr.train.head.BaseHead`, so it drops into the same
harness and the same serving path as the linear probe with no other changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .head import BaseHead

__all__ = ["MLPHead", "train_mlp_head", "balanced_class_weights"]


def balanced_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights, matching sklearn's "balanced" formula.

    ``w_c = n_samples / (n_present_classes * count_c)``. Classes with no rows
    get weight 0 -- they contribute no loss rather than an infinite one.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    present = counts > 0

    weights = np.zeros(n_classes, dtype=np.float64)
    weights[present] = len(labels) / (present.sum() * counts[present])
    return weights


class _MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: Sequence[int],
                 dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous = in_dim
        for width in hidden:
            layers += [nn.Linear(previous, width), nn.GELU(), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead(BaseHead):
    """A small torch MLP over frozen CLIP embeddings."""

    def __init__(self, model: _MLP, class_names: Sequence[str], *,
                 meta: dict | None = None):
        super().__init__(class_names, meta=meta)
        self.model = model.eval()

    @torch.inference_mode()
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(embeddings, dtype=np.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return torch.softmax(self.model(x), dim=-1).numpy().astype(np.float64)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "class_names": self.class_names,
                "meta": self.meta,
                "arch": {
                    "in_dim": self.meta["embed_dim"],
                    "hidden": self.meta["hidden"],
                    "dropout": self.meta["dropout"],
                },
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> MLPHead:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        arch = payload["arch"]
        model = _MLP(arch["in_dim"], len(payload["class_names"]),
                     arch["hidden"], arch["dropout"])
        model.load_state_dict(payload["state_dict"])
        return cls(model, payload["class_names"], meta=payload["meta"])

    def __repr__(self) -> str:
        return (f"MLPHead(hidden={self.meta.get('hidden')}, "
                f"classes={self.n_classes})")


def train_mlp_head(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    val_embeddings: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    hidden: Sequence[int] = (512,),
    dropout: float = 0.3,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    balanced: bool = True,
    patience: int = 8,
    seed: int = 0,
    verbose: bool = False,
) -> MLPHead:
    """Train an MLP on cached embeddings, keeping the best-validation weights.

    Early stopping is on validation loss when a validation set is given.
    Without one the final epoch's weights are kept, which is fine for a smoke
    test but not for anything you intend to report.
    """
    torch.manual_seed(seed)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(f"{embeddings.shape[0]} embeddings but {labels.shape[0]} labels")

    n_classes = len(class_names)
    model = _MLP(embeddings.shape[1], n_classes, hidden, dropout)

    weight = None
    if balanced:
        weight = torch.as_tensor(
            balanced_class_weights(labels, n_classes), dtype=torch.float32
        )
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x = torch.as_tensor(embeddings)
    y = torch.as_tensor(labels)
    has_val = val_embeddings is not None and val_labels is not None
    if has_val:
        vx = torch.as_tensor(np.asarray(val_embeddings, dtype=np.float32))
        vy = torch.as_tensor(np.asarray(val_labels, dtype=np.int64))

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(x), batch_size):
            rows = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x[rows]), y[rows])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(rows)

        train_loss = total / len(x)

        if has_val:
            model.eval()
            with torch.inference_mode():
                val_loss = float(criterion(model(vx), vy))
            improved = val_loss < best_loss - 1e-5
            if improved:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if verbose:
                print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}"
                      + ("  *" if improved else ""))
            if stale >= patience:
                break
        elif verbose:
            print(f"  epoch {epoch:3d}  train {train_loss:.4f}")

    if has_val:
        model.load_state_dict(best_state)

    meta = {
        "kind": "mlp",
        "n_train": int(embeddings.shape[0]),
        "embed_dim": int(embeddings.shape[1]),
        "hidden": list(hidden),
        "dropout": dropout,
        "balanced": balanced,
        "epochs_run": epoch,
        "best_epoch": best_epoch if has_val else epoch,
        "best_val_loss": best_loss if has_val else None,
    }
    return MLPHead(model, class_names, meta=meta)
