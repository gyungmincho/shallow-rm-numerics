"""Multi-shot purity-estimation simulations for block shadows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .qiskit_helpers import (
    append_block_clifford_measurement,
    bitstring_to_block_indices,
    build_random_brickwork_circuit,
    run_counts_in_batches,
)


def expand_counts_to_blocks(
    counts: dict[str, int],
    *,
    num_qubits: int,
    block_size: int,
    reverse_bits: bool = True,
) -> np.ndarray:
    """Expand a count dictionary into a shot-by-block integer array."""
    rows: list[list[int]] = []
    for bitstring, count in counts.items():
        blocks = bitstring_to_block_indices(bitstring, num_qubits, block_size, reverse=reverse_bits)
        rows.extend([blocks] * int(count))
    return np.asarray(rows, dtype=np.int16)


def pair_kernel_values(blocks_left: np.ndarray, blocks_right: np.ndarray, block_dim: int) -> np.ndarray:
    """Return purity-kernel values for aligned arrays of block outcomes."""
    same_counts = np.count_nonzero(blocks_left == blocks_right, axis=1)
    n_blocks = blocks_left.shape[1]
    return ((-block_dim) ** same_counts) * ((-1) ** n_blocks)


def purity_estimate_from_counts(
    counts: dict[str, int],
    *,
    num_qubits: int,
    block_size: int,
    reverse_bits: bool = True,
    exact_pair_limit: int = 2_000_000,
    pair_samples: int = 500_000,
    seed: int | None = None,
) -> float:
    """Estimate purity from one reused measurement basis.

    For small shot counts this evaluates all unordered pairs exactly. For large
    shot counts it uses an unbiased random-pair estimate to avoid the quadratic
    memory and runtime cost of the original notebook implementation.
    """
    rng = np.random.default_rng(seed)
    blocks = expand_counts_to_blocks(
        counts,
        num_qubits=num_qubits,
        block_size=block_size,
        reverse_bits=reverse_bits,
    )
    shots = blocks.shape[0]
    if shots < 2:
        raise ValueError("At least two shots are required for purity estimation.")
    total_pairs = shots * (shots - 1) // 2
    block_dim = 2**block_size

    if total_pairs <= exact_pair_limit:
        accumulator = 0.0
        for start in range(shots - 1):
            left = np.repeat(blocks[start : start + 1], shots - start - 1, axis=0)
            right = blocks[start + 1 :]
            accumulator += float(np.sum(pair_kernel_values(left, right, block_dim)))
        return 2.0 * accumulator / (shots * (shots - 1))

    samples = min(pair_samples, total_pairs)
    first = rng.integers(0, shots, size=samples)
    second = rng.integers(0, shots - 1, size=samples)
    second = second + (second >= first)
    values = pair_kernel_values(blocks[first], blocks[second], block_dim)
    return float(np.mean(values))


def run_purity_simulation_qiskit(
    *,
    num_qubits: int = 24,
    depth: int = 12,
    block_sizes: list[int] | None = None,
    shots_list: list[int] | None = None,
    num_unitaries: int = 500,
    seed: int = 0,
    max_circuits_per_job: int = 50,
    exact_pair_limit: int = 2_000_000,
    pair_samples: int = 500_000,
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Run the Qiskit/Aer MPS simulation for Fig. 3(b)-type purity data."""
    block_sizes = block_sizes or [1, 2, 3]
    shots_list = shots_list or [2000, 3000, 4000, 6000, 8000, 10000]
    shots_list = sorted(set(int(value) for value in shots_list))
    if any(num_qubits % block_size != 0 for block_size in block_sizes):
        raise ValueError("num_qubits must be divisible by every block size.")

    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_circuit = build_random_brickwork_circuit(num_qubits, depth, seed=seed)

    rows: list[dict[str, int | float]] = []
    for block_size in block_sizes:
        for shots in shots_list:
            circuits = []
            for _ in range(num_unitaries):
                circuit, _ = append_block_clifford_measurement(
                    target_circuit,
                    block_size,
                    seed=int(rng.integers(2**31 - 1)),
                )
                circuits.append(circuit)
            counts_list = run_counts_in_batches(
                circuits,
                shots=shots,
                method="matrix_product_state",
                max_circuits_per_job=max_circuits_per_job,
            )
            estimates = []
            for idx, counts in enumerate(tqdm(counts_list, desc=f"purity k={block_size}, shots={shots}", leave=False)):
                estimates.append(
                    purity_estimate_from_counts(
                        counts,
                        num_qubits=num_qubits,
                        block_size=block_size,
                        reverse_bits=True,
                        exact_pair_limit=exact_pair_limit,
                        pair_samples=pair_samples,
                        seed=seed + 1_000_003 * idx + 10_007 * block_size + shots,
                    )
                )
            estimates_array = np.asarray(estimates, dtype=float)
            rmse = float(np.sqrt(np.mean((estimates_array - 1.0) ** 2) / num_unitaries))
            row = {
                "num_qubits": num_qubits,
                "depth": depth,
                "block_size": block_size,
                "num_unitaries": num_unitaries,
                "shots_per_unitary": shots,
                "mean_estimate": float(np.mean(estimates_array)),
                "rmse_of_mean": rmse,
                "log10_rmse_of_mean": float(np.log10(rmse)),
                "pair_mode": "exact" if shots * (shots - 1) // 2 <= exact_pair_limit else "sampled",
                "pair_samples": int(pair_samples),
            }
            rows.append(row)
            np.savez_compressed(
                output_path / f"fig3_purity_estimates_n{num_qubits}_d{depth}_k{block_size}_shots{shots}.npz",
                estimates=estimates_array,
            )
            pd.DataFrame(rows).to_csv(output_path / "fig3_purity_summary.csv", index=False)
    return pd.DataFrame(rows)
