"""Optional Qiskit helpers used by the numerical scripts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def require_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator, random_clifford, random_unitary
        from qiskit_aer import Aer
    except Exception as exc:  # pragma: no cover - depends on optional packages
        raise ImportError(
            "Qiskit and qiskit-aer are required for this script. Install them with "
            "`pip install -e .[qiskit]` or use the provided conda environment."
        ) from exc
    return QuantumCircuit, Operator, random_clifford, random_unitary, Aer


def build_random_brickwork_circuit(num_qubits: int, depth: int, seed: int | None = None):
    """Build the brickwork random circuit used to generate a target state."""
    QuantumCircuit, _, _, random_unitary, _ = require_qiskit()
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for layer in range(depth):
        indices = range(0, num_qubits, 2) if layer % 2 == 0 else range(1, num_qubits - 1, 2)
        for qubit in indices:
            u1 = random_unitary(2, seed=int(rng.integers(2**31 - 1)))
            u2 = random_unitary(2, seed=int(rng.integers(2**31 - 1)))
            qc.unitary(u1, qubit)
            qc.unitary(u2, qubit + 1)
            qc.cx(qubit, qubit + 1)
    for qubit in range(num_qubits):
        u = random_unitary(2, seed=int(rng.integers(2**31 - 1)))
        qc.unitary(u, qubit)
    return qc.decompose()


def append_block_clifford_measurement(base_circuit, block_size: int, seed: int | None = None):
    """Append independent block Clifford measurement unitaries and measurements."""
    QuantumCircuit, Operator, random_clifford, _, _ = require_qiskit()
    num_qubits = base_circuit.num_qubits
    if num_qubits % block_size != 0:
        raise ValueError("num_qubits must be divisible by block_size.")
    rng = np.random.default_rng(seed)
    qc = base_circuit.copy()
    unitary_circuit = QuantumCircuit(num_qubits)
    unitary_blocks: list[np.ndarray] = []
    for start in range(0, num_qubits, block_size):
        clifford = random_clifford(block_size, seed=int(rng.integers(2**31 - 1)))
        clifford_circuit = clifford.to_circuit()
        unitary_circuit.compose(clifford_circuit, qubits=range(start, start + block_size), inplace=True)
        unitary_blocks.append(Operator(clifford_circuit.reverse_bits()).data)
    qc.compose(unitary_circuit, qubits=range(num_qubits), inplace=True)
    qc.measure_all()
    return qc, unitary_blocks


def run_counts_in_batches(
    circuits: Sequence,
    *,
    shots: int,
    method: str = "matrix_product_state",
    max_circuits_per_job: int = 50,
) -> list[dict[str, int]]:
    """Run circuits on Aer in batches and return count dictionaries."""
    _, _, _, _, Aer = require_qiskit()
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if max_circuits_per_job <= 0:
        raise ValueError("max_circuits_per_job must be positive.")
    backend_name = "aer_simulator_matrix_product_state" if method == "matrix_product_state" else "aer_simulator"
    backend = Aer.get_backend(backend_name)
    out: list[dict[str, int]] = []
    for start in range(0, len(circuits), max_circuits_per_job):
        batch = list(circuits[start : start + max_circuits_per_job])
        result = backend.run(batch, shots=shots).result()
        counts = result.get_counts()
        if isinstance(counts, dict):
            counts = [counts]
        out.extend(counts)
    return out


def save_matrix_product_state(circuit):
    """Return Aer matrix-product-state data for a circuit."""
    _, _, _, _, Aer = require_qiskit()
    backend = Aer.get_backend("aer_simulator_matrix_product_state")
    qc = circuit.copy()
    qc.save_matrix_product_state()
    data = backend.run(qc).result().data()
    return data["matrix_product_state"]


def qiskit_mps_to_tensors(mps_data) -> list[np.ndarray]:
    """Convert Aer MPS data to tensors with shape (left, right, physical).

    The converter accepts the common Aer format ``(gammas, lambdas)`` where each
    gamma stores two physical matrices. Schmidt coefficients are absorbed into
    the right bond of each tensor except for the last site.
    """
    if isinstance(mps_data, tuple) and len(mps_data) == 2:
        gammas, lambdas = mps_data
    elif isinstance(mps_data, list) and len(mps_data) == 2 and isinstance(mps_data[1], list):
        gammas, lambdas = mps_data
    else:
        gammas, lambdas = mps_data, []

    tensors: list[np.ndarray] = []
    for site, gamma in enumerate(gammas):
        if isinstance(gamma, np.ndarray) and gamma.ndim == 3:
            tensor = np.asarray(gamma, dtype=complex)
            if tensor.shape[0] == 2:
                tensor = np.moveaxis(tensor, 0, 2)
        elif isinstance(gamma, (tuple, list)) and len(gamma) == 2:
            tensor = np.stack([np.asarray(gamma[0], dtype=complex), np.asarray(gamma[1], dtype=complex)], axis=2)
        else:
            raise ValueError("Unsupported Qiskit MPS tensor format.")

        if site < len(lambdas):
            lam = np.asarray(lambdas[site], dtype=complex)
            if lam.ndim == 1 and tensor.shape[1] == lam.shape[0]:
                tensor = tensor * lam[None, :, None]
        tensors.append(tensor)
    return tensors


def bitstring_to_block_indices(bitstring: str, num_qubits: int, block_size: int, *, reverse: bool = True) -> list[int]:
    """Convert a Qiskit count key to block outcome integers."""
    clean = bitstring.replace(" ", "")
    if len(clean) != num_qubits:
        raise ValueError(f"Expected {num_qubits} bits, got {len(clean)} bits.")
    if reverse:
        clean = clean[::-1]
    return [int(clean[i : i + block_size], 2) for i in range(0, num_qubits, block_size)]
