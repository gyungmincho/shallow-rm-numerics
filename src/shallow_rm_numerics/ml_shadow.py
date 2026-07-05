"""DMRG-based shadow-kernel simulations for bond-alternating XXZ states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dmrg import dmrg_bond_alternating_xxz_state
from .mps import apply_block_unitaries_to_mps_blocks, block_mps_tensors, sample_mps
from .random_utils import haar_unitary


def measurement_vectors_from_mps(
    mps_list: list[np.ndarray],
    *,
    block_size: int,
    num_unitaries: int,
    shots_per_unitary: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate block-shadow measurement vectors from an MPS state.

    The returned array has shape
    ``(num_unitaries, shots_per_unitary, num_blocks, 2**block_size)``.
    """
    blocks = block_mps_tensors(mps_list, block_size)
    num_blocks = len(blocks)
    block_dim = 2**block_size
    data = np.empty((num_unitaries, shots_per_unitary, num_blocks, block_dim), dtype=complex)
    for unitary_index in range(num_unitaries):
        unitary_blocks = np.asarray([haar_unitary(block_dim, rng) for _ in range(num_blocks)])
        rotated_blocks = apply_block_unitaries_to_mps_blocks(blocks, unitary_blocks)
        outcomes = sample_mps(rotated_blocks, rng, num_samples=shots_per_unitary)
        for shot_index, shot in enumerate(outcomes):
            for block_index, outcome in enumerate(shot):
                data[unitary_index, shot_index, block_index] = unitary_blocks[block_index].T.conj()[:, int(outcome)]
    return data


def shadow_kernel_inter(
    data_left_unitary: np.ndarray,
    data_right_unitary: np.ndarray,
    *,
    gamma: float | np.ndarray,
) -> float:
    """Compute the per-unitary-pair kernel factor."""
    if data_left_unitary.shape != data_right_unitary.shape:
        raise ValueError("data_left_unitary and data_right_unitary must have the same shape.")
    block_dim = data_left_unitary.shape[-1]
    num_blocks = data_left_unitary.shape[-2]
    if np.isscalar(gamma):
        gamma_array = np.full(num_blocks, float(gamma), dtype=float)
    else:
        gamma_array = np.asarray(gamma, dtype=float)
        if gamma_array.shape != (num_blocks,):
            raise ValueError("gamma array must have shape (num_blocks,).")

    overlaps = np.sum(data_left_unitary * data_right_unitary.conj(), axis=-1)
    scores = np.abs(overlaps) ** 2 * (block_dim + 1) ** 2 - (block_dim + 2)
    per_shot = np.mean(scores * gamma_array[None, :], axis=-1)
    return float(np.exp(np.mean(per_shot)))


def shadow_kernel_entry(
    shadows_left: np.ndarray,
    shadows_right: np.ndarray,
    *,
    gamma: float | np.ndarray,
    tau: float = 1.0,
    exclude_identical_unitaries: bool = False,
) -> float:
    """Compute the exponential shadow kernel between two shadow datasets."""
    if shadows_left.shape != shadows_right.shape:
        raise ValueError("Both shadow datasets must have the same shape.")
    values: list[float] = []
    identical = np.array_equal(shadows_left, shadows_right)
    for left_index, left_unitary_data in enumerate(shadows_left):
        for right_index, right_unitary_data in enumerate(shadows_right):
            if exclude_identical_unitaries and identical and left_index == right_index:
                continue
            values.append(shadow_kernel_inter(left_unitary_data, right_unitary_data, gamma=gamma))
    if not values:
        raise ValueError("No unitary pairs remain after applying the exclusion rule.")
    return float(np.exp(tau * np.mean(values)))


def shadow_kernel_matrix(
    shadows: np.ndarray,
    *,
    gamma: float | np.ndarray,
    tau: float = 1.0,
    exclude_identical_unitaries: bool = False,
) -> np.ndarray:
    """Compute a symmetric shadow-kernel matrix."""
    num_data = shadows.shape[0]
    matrix = np.empty((num_data, num_data), dtype=float)
    for i in range(num_data):
        for j in range(i, num_data):
            value = shadow_kernel_entry(
                shadows[i],
                shadows[j],
                gamma=gamma,
                tau=tau,
                exclude_identical_unitaries=exclude_identical_unitaries,
            )
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def normalize_and_center_kernel(kernel_matrix: np.ndarray) -> np.ndarray:
    """Normalize diagonal entries and center the kernel matrix."""
    kernel_matrix = np.asarray(kernel_matrix, dtype=float)
    diagonal = np.sqrt(np.maximum(np.diag(kernel_matrix), 1e-300))
    normalized = kernel_matrix / diagonal[:, None] / diagonal[None, :]
    num_data = normalized.shape[0]
    centering = np.ones((num_data, num_data), dtype=float) / num_data
    return normalized - centering @ normalized - normalized @ centering + centering @ normalized @ centering


def kernel_pca_projection(kernel_matrix: np.ndarray, num_components: int = 2) -> np.ndarray:
    """Return kernel PCA coordinates with shape ``(num_components, n_data)``."""
    centered = normalize_and_center_kernel(kernel_matrix)
    eigvals, eigvecs = np.linalg.eigh(centered)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-300)
    eigvecs = eigvecs[:, order]
    components = np.diag(1.0 / np.sqrt(eigvals[:num_components])) @ eigvecs[:, :num_components].T @ centered
    if components[0, 0] > 0:
        components = -components
    return components


def generate_bond_alternating_xxz_shadow_data(
    *,
    num_sites: int,
    block_size: int,
    parameter_values: np.ndarray,
    coupling_even: float = 1.0,
    delta: float = 0.0,
    boundary_penalty: float = 0.1,
    num_unitaries: int = 30,
    shots_per_unitary: int = 1,
    max_bond_dim: int = 30,
    dmrg_sweeps: int = 5,
    dmrg_krylov: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Generate shadow-vector data for a DMRG parameter sweep."""
    if num_sites % block_size != 0:
        raise ValueError("num_sites must be divisible by block_size.")
    rng = np.random.default_rng(seed)
    data = []
    for coupling_odd in tqdm(parameter_values, desc=f"ML data L={num_sites}, k={block_size}", leave=False):
        result = dmrg_bond_alternating_xxz_state(
            num_sites,
            coupling_even=coupling_even,
            coupling_odd=float(coupling_odd),
            delta=delta,
            boundary_penalty=boundary_penalty,
            max_bond_dim=max_bond_dim,
            num_sweeps=dmrg_sweeps,
            n_krylov=dmrg_krylov,
        )
        data.append(
            measurement_vectors_from_mps(
                result.mps,
                block_size=block_size,
                num_unitaries=num_unitaries,
                shots_per_unitary=shots_per_unitary,
                rng=rng,
            )
        )
    return np.asarray(data)


def run_ml_kernel_simulation(
    *,
    system_sizes: list[int] | None = None,
    block_sizes: list[int] | None = None,
    num_data: int = 40,
    parameter_min: float = 0.0,
    parameter_max: float = 2.0,
    num_unitaries: int = 30,
    shots_per_unitary: int = 1,
    coupling_even: float = 1.0,
    delta: float = 0.0,
    boundary_penalty: float = 0.1,
    max_bond_dim: int = 30,
    dmrg_sweeps: int = 5,
    dmrg_krylov: int = 5,
    exclude_identical_unitaries: bool = False,
    seed: int = 0,
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Generate DMRG shadow-kernel data and PCA projections."""
    system_sizes = system_sizes or [12]
    block_sizes = block_sizes or [1, 2, 3]
    parameters = np.linspace(parameter_min, parameter_max, num_data)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gamma_default = {1: 1.0, 2: 0.25, 3: 1.0 / 16.0}
    rows: list[dict[str, int | float | str | bool]] = []
    for num_sites in system_sizes:
        for block_size in block_sizes:
            gamma = gamma_default.get(block_size, 1.0 / (4 ** max(block_size - 1, 0)))
            shadows = generate_bond_alternating_xxz_shadow_data(
                num_sites=num_sites,
                block_size=block_size,
                parameter_values=parameters,
                coupling_even=coupling_even,
                delta=delta,
                boundary_penalty=boundary_penalty,
                num_unitaries=num_unitaries,
                shots_per_unitary=shots_per_unitary,
                max_bond_dim=max_bond_dim,
                dmrg_sweeps=dmrg_sweeps,
                dmrg_krylov=dmrg_krylov,
                seed=seed + 10_003 * num_sites + 719 * block_size,
            )
            kernel = shadow_kernel_matrix(
                shadows,
                gamma=gamma,
                exclude_identical_unitaries=exclude_identical_unitaries,
            )
            projection = kernel_pca_projection(kernel, num_components=2)
            np.savez_compressed(
                output_path / f"fig5_ml_shadow_kernel_L{num_sites}_k{block_size}.npz",
                parameters=parameters,
                kernel_matrix=kernel,
                pca_projection=projection,
                shadow_data=shadows,
            )
            for idx, parameter in enumerate(parameters):
                rows.append(
                    {
                        "num_sites": num_sites,
                        "block_size": block_size,
                        "num_unitaries": num_unitaries,
                        "shots_per_unitary": shots_per_unitary,
                        "parameter": float(parameter),
                        "pca1": float(projection[0, idx]),
                        "pca2": float(projection[1, idx]),
                        "solver": "dmrg",
                        "boundary_penalty": float(boundary_penalty),
                        "exclude_identical_unitaries": bool(exclude_identical_unitaries),
                    }
                )
            pd.DataFrame(rows).to_csv(output_path / "fig5_ml_kernel_pca_summary.csv", index=False)
    return pd.DataFrame(rows)
