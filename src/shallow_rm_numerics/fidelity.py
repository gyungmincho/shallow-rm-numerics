"""Multi-shot fidelity-estimation simulations for block shadows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .mps import block_mps_tensors
from .qiskit_helpers import (
    append_block_clifford_measurement,
    bitstring_to_block_indices,
    build_random_brickwork_circuit,
    qiskit_mps_to_tensors,
    run_counts_in_batches,
    save_matrix_product_state,
)


def qubit_to_ptm_basis_change() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1j, -1j, 0], [1, 0, 0, -1]],
        dtype=complex,
    ).reshape((4, 2, 2)) / np.sqrt(2.0)


def kron_power_tensor(base: np.ndarray, power: int) -> np.ndarray:
    out = np.asarray(base)
    for _ in range(power - 1):
        out = np.kron(out, base)
    return out


def split_same_index_tensor(block_size: int) -> np.ndarray:
    base = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=complex).reshape(2, 2, 2)
    return kron_power_tensor(base, block_size)


def block_shadow_inverse_diagonal(block_size: int) -> np.ndarray:
    values = np.full(4**block_size, 2**block_size + 1, dtype=float)
    values[0] = 1.0
    return values


def fidelity_tensor_train_for_unitary(
    target_block_mps: list[np.ndarray],
    unitary_blocks: list[np.ndarray],
    block_size: int,
) -> list[np.ndarray]:
    """Build the MPS transfer tensors used for fidelity estimation."""
    basis_change = kron_power_tensor(qubit_to_ptm_basis_change(), block_size)
    inverse_diag = block_shadow_inverse_diagonal(block_size)
    split_tensor = split_same_index_tensor(block_size)

    tensor_train: list[np.ndarray] = []
    for mps_tensor, unitary in zip(target_block_mps, unitary_blocks):
        unitary_dagger = unitary.T.conj()
        out = np.tensordot(unitary_dagger, split_tensor, axes=[[1], [1]])
        out = np.tensordot(unitary.T, out, axes=[[1], [2]])
        out = np.tensordot(basis_change, out, axes=[[1, 2], [1, 0]])
        out = np.tensordot(np.diag(inverse_diag), out, axes=[[1], [0]])
        out = np.tensordot(basis_change.conj(), out, axes=[[0], [0]])
        out = np.tensordot(mps_tensor.conj(), out, axes=[[2], [0]])
        out = np.tensordot(mps_tensor, out, axes=[[2], [2]]).transpose(2, 0, 3, 1, 4)
        d0, d1, d2, d3, d4 = out.shape
        out = out.reshape(d0 * d1, d2 * d3, d4)
        tensor_train.append(out)
    return tensor_train


def evaluate_fidelity_tensor_train(tensor_train: list[np.ndarray], block_outcomes: list[int]) -> float:
    if len(tensor_train) != len(block_outcomes):
        raise ValueError("tensor_train and block_outcomes must have the same length.")
    out = np.array([[1.0 + 0j]])
    for tensor, outcome in zip(tensor_train, block_outcomes):
        out = out @ tensor[:, :, int(outcome)]
    if out.shape != (1, 1):
        raise ValueError(f"Unexpected contraction shape: {out.shape}")
    return float(np.real_if_close(out[0, 0]).real)


def estimate_fidelity_from_counts(
    tensor_train: list[np.ndarray],
    counts: dict[str, int],
    *,
    num_qubits: int,
    block_size: int,
    reverse_bits: bool = True,
) -> float:
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("counts must contain at least one shot.")
    value = 0.0
    for bitstring, count in counts.items():
        block_outcomes = bitstring_to_block_indices(bitstring, num_qubits, block_size, reverse=reverse_bits)
        value += count * evaluate_fidelity_tensor_train(tensor_train, block_outcomes)
    return value / shots


def run_fidelity_simulation_qiskit(
    *,
    num_qubits: int = 48,
    depth: int = 8,
    block_sizes: list[int] | None = None,
    shots_list: list[int] | None = None,
    num_unitaries: int = 800,
    seed: int = 0,
    max_circuits_per_job: int = 50,
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Run the Qiskit/Aer MPS simulation for Fig. 3(a)-type data."""
    block_sizes = block_sizes or [1, 2, 3]
    shots_list = shots_list or [1, 5, 10, 20, 40, 60, 80, 100, 200, 500, 1000]
    if any(num_qubits % block_size != 0 for block_size in block_sizes):
        raise ValueError("num_qubits must be divisible by every block size.")

    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    target_circuit = build_random_brickwork_circuit(num_qubits, depth, seed=seed)
    mps_data = save_matrix_product_state(target_circuit)
    target_mps = qiskit_mps_to_tensors(mps_data)

    rows: list[dict[str, int | float]] = []
    for block_size in block_sizes:
        target_block_mps = block_mps_tensors(target_mps, block_size)
        for shots in shots_list:
            circuits = []
            unitary_blocks_all: list[list[np.ndarray]] = []
            for _ in range(num_unitaries):
                circuit_seed = int(rng.integers(2**31 - 1))
                circuit, unitary_blocks = append_block_clifford_measurement(
                    target_circuit,
                    block_size,
                    seed=circuit_seed,
                )
                circuits.append(circuit)
                unitary_blocks_all.append(unitary_blocks)

            counts_list = run_counts_in_batches(
                circuits,
                shots=shots,
                method="matrix_product_state",
                max_circuits_per_job=max_circuits_per_job,
            )
            estimates: list[float] = []
            for unitary_blocks, counts in tqdm(
                list(zip(unitary_blocks_all, counts_list)),
                desc=f"fidelity k={block_size}, shots={shots}",
                leave=False,
            ):
                tensor_train = fidelity_tensor_train_for_unitary(target_block_mps, unitary_blocks, block_size)
                estimates.append(
                    estimate_fidelity_from_counts(
                        tensor_train,
                        counts,
                        num_qubits=num_qubits,
                        block_size=block_size,
                        reverse_bits=True,
                    )
                )
            estimates_array = np.asarray(estimates, dtype=float)
            row = {
                "num_qubits": num_qubits,
                "depth": depth,
                "block_size": block_size,
                "num_unitaries": num_unitaries,
                "shots_per_unitary": shots,
                "mean_estimate": float(np.mean(estimates_array)),
                "std_over_unitaries": float(np.std(estimates_array, ddof=0)),
                "standard_error": float(np.std(estimates_array, ddof=0) / np.sqrt(num_unitaries)),
                "log10_standard_error": float(np.log10(np.std(estimates_array, ddof=0) / np.sqrt(num_unitaries))),
            }
            rows.append(row)
            np.savez_compressed(
                output_path / f"fig3_fidelity_estimates_n{num_qubits}_d{depth}_k{block_size}_shots{shots}.npz",
                estimates=estimates_array,
            )
            pd.DataFrame(rows).to_csv(output_path / "fig3_fidelity_summary.csv", index=False)
    return pd.DataFrame(rows)
