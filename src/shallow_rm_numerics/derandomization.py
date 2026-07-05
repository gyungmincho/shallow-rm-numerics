"""Derandomized single-qubit and two-qubit block measurement selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .pauli import (
    cluster_heisenberg_h2_cross_terms_open,
    pauli_strings_to_int_array,
    roll_two_qubit_blocks,
    two_qubit_block_strings,
)


@dataclass(slots=True)
class DerandomizationResult:
    n_qubits: int
    method: str
    target_hits: int
    num_observables: int
    num_bases: int
    min_hits: int
    mean_hits: float
    max_hits: int
    bases: np.ndarray
    hit_counts: np.ndarray

    def summary(self) -> dict[str, int | float | str]:
        return {
            "n_qubits": self.n_qubits,
            "method": self.method,
            "target_hits": self.target_hits,
            "num_observables": self.num_observables,
            "num_bases": self.num_bases,
            "min_hits": self.min_hits,
            "mean_hits": self.mean_hits,
            "max_hits": self.max_hits,
        }


TWO_QUBIT_MUB_BASES: dict[int, list[str]] = {
    0: ["01", "10", "11"],
    1: ["02", "20", "22"],
    2: ["03", "30", "33"],
    3: ["12", "21", "33"],
    4: ["23", "32", "11"],
    5: ["13", "31", "22"],
    6: ["11", "22", "33"],
    7: ["01", "20", "21"],
    8: ["01", "30", "31"],
    9: ["02", "10", "12"],
    10: ["02", "30", "32"],
    11: ["03", "10", "13"],
    12: ["03", "20", "23"],
}

_ODD_BOUNDARY_SINGLE_KEYS = np.asarray([0, 1, 2, 7, 8, 9, 10, 11, 12], dtype=np.int16)


def _random_argmin(values: np.ndarray | list[float], rng: np.random.Generator) -> int:
    values = np.asarray(values)
    minimum = np.min(values)
    candidates = np.flatnonzero(values == minimum)
    return int(rng.choice(candidates))


def _single_suffix_weights(observables: np.ndarray) -> np.ndarray:
    n_qubits = observables.shape[1]
    suffix = np.empty((n_qubits, observables.shape[0]), dtype=np.int16)
    for site in range(n_qubits):
        suffix[site] = np.count_nonzero(observables[:, site + 1 :] != 0, axis=1)
    return suffix


def _single_measurement_masks(observables: np.ndarray) -> np.ndarray:
    masks = np.zeros((observables.shape[0], observables.shape[1], 3), dtype=np.int8)
    for pauli_idx in range(3):
        masks[:, :, pauli_idx] = ((observables == 0) | (observables == pauli_idx + 1)).astype(np.int8)
    return masks


def select_single_qubit_bases(
    observables: np.ndarray,
    *,
    target_hits: int = 100,
    error: float = 0.9,
    seed: int | None = 0,
    max_bases: int | None = None,
) -> DerandomizationResult:
    """Select derandomized single-qubit Pauli measurement bases.

    Basis entries are encoded as 1, 2, 3 for X, Y, Z.
    """
    observables = np.asarray(observables, dtype=np.int8)
    if observables.ndim != 2:
        raise ValueError("observables must have shape (num_observables, n_qubits).")
    if target_hits <= 0:
        raise ValueError("target_hits must be positive.")

    rng = np.random.default_rng(seed)
    n_observables, n_qubits = observables.shape
    suffix_weights = _single_suffix_weights(observables)
    masks = _single_measurement_masks(observables)
    mu = 1.0 - np.exp(-(error**2) / 2.0)
    hit_counts = np.zeros(n_observables, dtype=np.int32)
    bases: list[np.ndarray] = []

    while int(np.min(hit_counts)) < target_hits:
        if max_bases is not None and len(bases) >= max_bases:
            break
        compatible = np.ones(n_observables, dtype=np.int8)
        basis = np.empty(n_qubits, dtype=np.int8)
        exp_weight = np.exp(-(error**2) * hit_counts / 2.0)
        for site in range(n_qubits):
            costs = []
            denominator = np.power(3.0, suffix_weights[site])
            for pauli_idx in range(3):
                term = exp_weight * (1.0 - mu * compatible * masks[:, site, pauli_idx] / denominator)
                costs.append(float(np.sum(term)))
            choice = _random_argmin(costs, rng) + 1
            basis[site] = choice
            compatible *= ((observables[:, site] == 0) | (observables[:, site] == choice)).astype(np.int8)
        hit_counts += compatible.astype(np.int32)
        bases.append(basis)

    basis_array = np.asarray(bases, dtype=np.int8)
    return DerandomizationResult(
        n_qubits=n_qubits,
        method="single_qubit",
        target_hits=target_hits,
        num_observables=n_observables,
        num_bases=len(bases),
        min_hits=int(np.min(hit_counts)),
        mean_hits=float(np.mean(hit_counts)),
        max_hits=int(np.max(hit_counts)),
        bases=basis_array,
        hit_counts=hit_counts,
    )


def _two_qubit_basis_reverse_map() -> dict[str, list[int]]:
    paulis = [i + j for i in "0123" for j in "0123"][1:]
    out = {pauli: [] for pauli in paulis}
    for basis_index, paulis_in_basis in TWO_QUBIT_MUB_BASES.items():
        for pauli in paulis_in_basis:
            out[pauli].append(basis_index)
    return out


def _two_qubit_masks(block_observables: np.ndarray, reverse_map: dict[str, list[int]]) -> np.ndarray:
    n_observables, n_blocks = block_observables.shape
    num_bases = len(TWO_QUBIT_MUB_BASES)
    masks = np.zeros((n_observables, n_blocks, num_bases), dtype=np.int8)
    for obs_idx in range(n_observables):
        for block_idx in range(n_blocks):
            pauli = str(block_observables[obs_idx, block_idx])
            if pauli == "00":
                masks[obs_idx, block_idx, :] = 1
            else:
                for basis_index in reverse_map[pauli]:
                    masks[obs_idx, block_idx, basis_index] = 1
    return masks


def _two_qubit_suffix_weights(block_observables: np.ndarray) -> np.ndarray:
    n_blocks = block_observables.shape[1]
    suffix = np.empty((n_blocks, block_observables.shape[0]), dtype=np.int16)
    for block_idx in range(n_blocks):
        suffix[block_idx] = np.count_nonzero(block_observables[:, block_idx + 1 :] != "00", axis=1)
    return suffix


def select_two_qubit_alternating_bases(
    pauli_strings: list[str],
    *,
    target_hits: int = 100,
    error: float = 0.9,
    seed: int | None = 0,
    max_bases: int | None = None,
) -> DerandomizationResult:
    """Select alternating even/odd two-qubit block measurement bases.

    Basis entries are indices of ``TWO_QUBIT_MUB_BASES``. Even-numbered rows use
    the even partition. Odd-numbered rows use the one-site-shifted partition.
    """
    if not pauli_strings:
        raise ValueError("pauli_strings must be non-empty.")
    n_qubits = len(pauli_strings[0])
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even for two-qubit block measurements.")
    if target_hits <= 0:
        raise ValueError("target_hits must be positive.")

    rng = np.random.default_rng(seed)
    even_observables = two_qubit_block_strings(pauli_strings)
    odd_observables = roll_two_qubit_blocks(even_observables)
    n_observables, n_blocks = even_observables.shape
    reverse_map = _two_qubit_basis_reverse_map()

    even_masks = _two_qubit_masks(even_observables, reverse_map)
    odd_masks = _two_qubit_masks(odd_observables, reverse_map)
    even_suffix = _two_qubit_suffix_weights(even_observables)
    odd_suffix = _two_qubit_suffix_weights(odd_observables)

    mu = 1.0 - np.exp(-(error**2) / 2.0)
    hit_counts = np.zeros(n_observables, dtype=np.int32)
    bases: list[np.ndarray] = []

    while int(np.min(hit_counts)) < target_hits:
        if max_bases is not None and len(bases) >= max_bases:
            break
        measurement_index = len(bases)
        use_even = measurement_index % 2 == 0
        observables = even_observables if use_even else odd_observables
        masks = even_masks if use_even else odd_masks
        suffix = even_suffix if use_even else odd_suffix

        compatible = np.ones(n_observables, dtype=np.int8)
        basis = np.empty(n_blocks, dtype=np.int16)
        exp_weight = np.exp(-(error**2) * hit_counts / 2.0)
        for block_idx in range(n_blocks):
            candidate_keys = np.arange(len(TWO_QUBIT_MUB_BASES), dtype=np.int16)
            if (not use_even) and block_idx == n_blocks - 1:
                candidate_keys = _ODD_BOUNDARY_SINGLE_KEYS
            costs = []
            denominator = np.power(5.0, suffix[block_idx])
            for basis_index in candidate_keys:
                term = exp_weight * (1.0 - mu * compatible * masks[:, block_idx, basis_index] / denominator)
                costs.append(float(np.sum(term)))
            choice = int(candidate_keys[_random_argmin(costs, rng)])
            basis[block_idx] = choice
            allowed = ["00"] + TWO_QUBIT_MUB_BASES[choice]
            compatible *= np.isin(observables[:, block_idx], allowed).astype(np.int8)
        hit_counts += compatible.astype(np.int32)
        bases.append(basis)

    basis_array = np.asarray(bases, dtype=np.int16)
    return DerandomizationResult(
        n_qubits=n_qubits,
        method="two_qubit_alternating",
        target_hits=target_hits,
        num_observables=n_observables,
        num_bases=len(bases),
        min_hits=int(np.min(hit_counts)),
        mean_hits=float(np.mean(hit_counts)),
        max_hits=int(np.max(hit_counts)),
        bases=basis_array,
        hit_counts=hit_counts,
    )


def run_cluster_heisenberg_derandomization(
    n_values: list[int],
    *,
    methods: tuple[Literal["single_qubit", "two_qubit_alternating"], ...] = (
        "single_qubit",
        "two_qubit_alternating",
    ),
    target_hits: int = 100,
    error: float = 0.9,
    seed: int = 0,
    output_dir: str | Path | None = None,
    save_bases: bool = True,
) -> pd.DataFrame:
    """Run derandomized measurement selection for a list of system sizes."""
    rows: list[dict[str, int | float | str]] = []
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for n_qubits in n_values:
        pauli_strings = cluster_heisenberg_h2_cross_terms_open(n_qubits, unique=False)
        observables = pauli_strings_to_int_array(pauli_strings)
        for method_index, method in enumerate(methods):
            method_seed = seed + 1009 * n_qubits + 9173 * method_index
            if method == "single_qubit":
                result = select_single_qubit_bases(
                    observables,
                    target_hits=target_hits,
                    error=error,
                    seed=method_seed,
                )
            elif method == "two_qubit_alternating":
                result = select_two_qubit_alternating_bases(
                    pauli_strings,
                    target_hits=target_hits,
                    error=error,
                    seed=method_seed,
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            rows.append(result.summary())
            if output_path is not None and save_bases:
                np.savez_compressed(
                    output_path / f"derandomization_{method}_n{n_qubits}.npz",
                    bases=result.bases,
                    hit_counts=result.hit_counts,
                )

    df = pd.DataFrame(rows)
    if output_path is not None:
        df.to_csv(output_path / "fig2_derandomization_summary.csv", index=False)
    return df
