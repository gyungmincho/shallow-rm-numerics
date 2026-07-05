"""Random matrix utilities."""

from __future__ import annotations

import numpy as np


def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a Haar-random unitary using QR decomposition."""
    if dim <= 0:
        raise ValueError("dim must be positive.")
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    diagonal = np.diag(r)
    phases = diagonal / np.where(np.abs(diagonal) == 0, 1.0, np.abs(diagonal))
    return q * phases.conj()
