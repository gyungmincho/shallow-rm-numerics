"""MPS utilities used by the numerical workflows."""

from __future__ import annotations

import numpy as np


def mps_chunk_contraction(mps_list: list[np.ndarray]) -> np.ndarray:
    """Contract consecutive site tensors into one block tensor."""
    if not mps_list:
        raise ValueError("mps_list must be non-empty.")
    tensor = np.asarray(mps_list[0], dtype=complex).copy()
    for site_tensor in mps_list[1:]:
        site_tensor = np.asarray(site_tensor, dtype=complex)
        tensor = np.tensordot(tensor, site_tensor, axes=[[1], [0]]).transpose(0, 2, 1, 3)
        shape = tensor.shape
        tensor = tensor.reshape(shape[0], shape[1], shape[2] * shape[3])
    return tensor


def block_mps_tensors(mps_list: list[np.ndarray], block_size: int) -> list[np.ndarray]:
    """Group site-level MPS tensors into block-level MPS tensors."""
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if len(mps_list) % block_size != 0:
        raise ValueError("The number of MPS tensors must be divisible by block_size.")
    return [mps_chunk_contraction(mps_list[i : i + block_size]) for i in range(0, len(mps_list), block_size)]


def check_right_normalized(mps_list: list[np.ndarray], atol: float = 1e-8) -> bool:
    """Check the right-normalization condition."""
    for tensor in mps_list:
        left_dim, _right_dim, physical_dim = tensor.shape
        gram = np.zeros((left_dim, left_dim), dtype=complex)
        for physical in range(physical_dim):
            mat = tensor[:, :, physical]
            gram += mat @ mat.conj().T
        if not np.allclose(gram, np.eye(left_dim), atol=atol):
            return False
    return True


def right_canonicalize(mps_list: list[np.ndarray], max_bond_dim: int | None = None) -> list[np.ndarray]:
    """Return a right-canonical approximation of an MPS."""
    tensors = [np.asarray(tensor, dtype=complex).copy() for tensor in mps_list]
    for site in range(len(tensors) - 1, 0, -1):
        tensor = tensors[site]
        left_dim, right_dim, physical_dim = tensor.shape
        mat = tensor.reshape(left_dim, right_dim * physical_dim)
        q_t, r_t = np.linalg.qr(mat.T)
        q = q_t.T
        r = r_t.T
        if max_bond_dim is not None and q.shape[0] > max_bond_dim:
            q = q[:max_bond_dim]
            r = r[:, :max_bond_dim]
        tensors[site] = q.reshape(q.shape[0], right_dim, physical_dim)
        tensors[site - 1] = np.tensordot(tensors[site - 1], r, axes=[[1], [0]]).transpose(0, 2, 1)
    norm = np.linalg.norm(tensors[0].reshape(-1))
    if norm > 0:
        tensors[0] /= norm
    return tensors


def _sample_from_right_canonical(mps_list: list[np.ndarray], rng: np.random.Generator) -> list[int]:
    vec = np.array([1.0 + 0j])
    probability_prefix = 1.0
    samples: list[int] = []
    for tensor in mps_list:
        contracted = np.tensordot(vec, tensor, axes=[[0], [0]])
        probabilities = np.sum(np.abs(contracted) ** 2, axis=0).real / probability_prefix
        probabilities = np.maximum(probabilities, 0.0)
        probabilities = probabilities / np.sum(probabilities)
        outcome = int(rng.choice(np.arange(tensor.shape[-1]), p=probabilities))
        probability_prefix *= probabilities[outcome]
        vec = contracted[:, outcome]
        samples.append(outcome)
    return samples


def sample_mps(mps_list: list[np.ndarray], rng: np.random.Generator, num_samples: int = 1) -> list[list[int]]:
    """Sample computational-basis outcomes from an MPS."""
    canonical = right_canonicalize(mps_list)
    return [_sample_from_right_canonical(canonical, rng) for _ in range(num_samples)]


def apply_block_unitaries_to_mps_blocks(blocks: list[np.ndarray], unitary_blocks: np.ndarray) -> list[np.ndarray]:
    """Apply local block unitaries to block-MPS tensors."""
    if len(blocks) != len(unitary_blocks):
        raise ValueError("blocks and unitary_blocks must have the same length.")
    return [np.tensordot(tensor, unitary, axes=[[2], [1]]) for tensor, unitary in zip(blocks, unitary_blocks)]
