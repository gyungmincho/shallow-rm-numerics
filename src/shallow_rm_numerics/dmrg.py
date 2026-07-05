"""MPO/MPS and DMRG routines used for the Fig. 5 numerical workflow.

The implementation is a cleaned, self-contained version of the helper DMRG code
used in the original notebooks. Tensors use the convention
``(left_bond, right_bond, physical_index)`` for MPS tensors and
``(physical_out, physical_in, left_bond, right_bond)`` for MPO tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


Array = np.ndarray


@dataclass(slots=True)
class DMRGResult:
    """Container returned by the DMRG convenience wrappers."""

    energies: list[list[float]]
    mps: list[Array]


def _spin_operators() -> tuple[Array, Array, Array, Array, Array]:
    zero = np.zeros((2, 2), dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return zero, identity, sx, sy, sz


def mpo_bond_alternating_xxz(
    coupling_even: float,
    coupling_odd: float,
    delta: float,
    num_sites: int,
    boundary_penalty: float = 0.1,
) -> list[Array]:
    """Return an open-boundary MPO for an alternating XXZ chain.

    The Hamiltonian is
    ``sum_i J_i (X_i X_{i+1} + Y_i Y_{i+1} + delta Z_i Z_{i+1})``
    with ``J_i = coupling_even`` on even bonds and ``coupling_odd`` on odd
    bonds. A small first-site ``Z`` field can be added via ``boundary_penalty``
    to pick one state in nearly degenerate regimes, matching the numerical
    workflow used in the notebooks.
    """
    if num_sites < 2:
        raise ValueError("num_sites must be at least 2.")

    zero, identity, sx, sy, sz = _spin_operators()

    x1 = np.array([identity, zero, zero, zero, zero])
    x2 = np.array([sx, zero, zero, zero, zero])
    x3 = np.array([sy, zero, zero, zero, zero])
    x4 = np.array([sz, zero, zero, zero, zero])

    x5_even = np.array([zero, sx * coupling_even, sy * coupling_even, sz * coupling_even * delta, identity])
    x5_odd = np.array([zero, sx * coupling_odd, sy * coupling_odd, sz * coupling_odd * delta, identity])
    x5_first = np.array(
        [
            boundary_penalty * coupling_even * sz,
            sx * coupling_even,
            sy * coupling_even,
            sz * coupling_even * delta,
            identity,
        ]
    )

    mpo_even = np.stack([x1, x2, x3, x4, x5_even]).transpose([2, 3, 0, 1])
    mpo_odd = np.stack([x1, x2, x3, x4, x5_odd]).transpose([2, 3, 0, 1])
    mpo_first = np.stack([x1, x2, x3, x4, x5_first]).transpose([2, 3, 0, 1])

    mpo_list: list[Array] = []
    for site in range(num_sites):
        if site == 0:
            mpo_list.append(mpo_first)
        elif site % 2 == 0:
            mpo_list.append(mpo_even)
        else:
            mpo_list.append(mpo_odd)

    mpo_list[0] = mpo_list[0][:, :, -1, :][:, :, None, :]
    mpo_list[-1] = mpo_list[-1][:, :, :, 0][:, :, :, None]
    return mpo_list


def mpo_xx_yy(coupling_even: float, coupling_odd: float, num_sites: int) -> list[Array]:
    """Return the alternating ``XX + YY`` MPO used for the SSH-limit tests."""
    return mpo_bond_alternating_xxz(
        coupling_even=coupling_even / 2.0,
        coupling_odd=coupling_odd / 2.0,
        delta=0.0,
        num_sites=num_sites,
        boundary_penalty=0.0,
    )


def check_right_normalized(mps_list: list[Array], atol: float = 1e-8) -> bool:
    """Return whether all MPS tensors are right-normalized."""
    for tensor in mps_list:
        gram = np.tensordot(tensor, tensor.conj(), axes=[[1, 2], [1, 2]])
        if not np.allclose(gram, np.eye(tensor.shape[0]), atol=atol):
            return False
    return True


def check_left_normalized(mps_list: list[Array], atol: float = 1e-8) -> bool:
    """Return whether all MPS tensors are left-normalized."""
    for tensor in mps_list:
        gram = np.tensordot(tensor, tensor.conj(), axes=[[0, 2], [0, 2]])
        if not np.allclose(gram, np.eye(tensor.shape[1]), atol=atol):
            return False
    return True


def update_left(environment: Array, upper_mps: Array, lower_mps: Array, mpo: Array) -> Array:
    """Add one site to a left environment."""
    out = np.tensordot(environment, upper_mps, axes=[[1], [0]])
    out = np.tensordot(out, mpo, axes=[[1, 3], [2, 1]])
    out = np.tensordot(out, lower_mps, axes=[[0, 2], [0, 2]]).transpose([2, 0, 1])
    return out


def update_right(environment: Array, upper_mps: Array, lower_mps: Array, mpo: Array) -> Array:
    """Add one site to a right environment."""
    return update_left(
        environment,
        upper_mps.transpose([1, 0, 2]),
        lower_mps.transpose([1, 0, 2]),
        mpo.transpose([0, 1, 3, 2]),
    )


def _sub_mps_init(previous_environment: Array, mpo: Array, max_bond_dim: int) -> tuple[Array, Array]:
    rank4 = np.tensordot(previous_environment, mpo, axes=[[2], [2]])[:, :, :, :, 0]
    rank4 = np.transpose(rank4, [0, 2, 1, 3])
    shape = rank4.shape
    keep = min(shape[2] * shape[3], max_bond_dim)
    rank2 = np.reshape(rank4, [shape[0] * shape[1], shape[2] * shape[3]])
    _eigvals, eigvecs = np.linalg.eigh(rank2)
    tensor = np.reshape(eigvecs[:, :keep], [shape[2], shape[3], keep]).transpose([0, 2, 1])
    next_environment = update_left(previous_environment, tensor, tensor.conj(), mpo)
    return tensor, next_environment


def mps_init_list_from_mpo_list(mpo_list: list[Array], max_bond_dim: int = 30) -> list[Array]:
    """Build a deterministic left-normalized initial MPS from an MPO list."""
    mps_list: list[Array] = []
    environment = np.array([1], dtype=np.complex128).reshape([1, 1, 1])
    for mpo in mpo_list:
        tensor, environment = _sub_mps_init(environment, mpo, max_bond_dim=max_bond_dim)
        mps_list.append(tensor)
    mps_list[-1] = mps_list[-1][:, :1, :]
    return mps_list


def lanczos_project(vector: Array, basis_vectors: list[Array]) -> Array:
    """Orthogonalize ``vector`` against the tensor Krylov basis."""
    basis = np.asarray(basis_vectors)
    coefficients = np.tensordot(basis.conj(), vector, axes=[[1, 2, 3], [0, 1, 2]])
    return vector - np.tensordot(basis, coefficients, axes=[[0], [0]])


def lanczos_method(
    left_environment: Array,
    center_mpo: Array,
    right_environment: Array,
    initial_tensor: Array,
    n_krylov: int = 15,
    tol: float = 1e-8,
) -> tuple[float, Array]:
    """Solve a one-site effective Hamiltonian using a small Lanczos basis."""
    norm = np.sqrt(np.tensordot(initial_tensor.conj(), initial_tensor, axes=[[0, 1, 2], [0, 1, 2]]).real)
    if norm <= 0:
        raise ValueError("initial_tensor has zero norm.")

    vectors = [initial_tensor / norm]
    alpha_values: list[float] = []
    beta_values: list[float] = []

    for step in range(n_krylov):
        working = np.tensordot(left_environment, vectors[step], axes=[[1], [0]])
        working = np.tensordot(working, center_mpo, axes=[[1, 3], [2, 1]])
        working = np.tensordot(working, right_environment, axes=[[1, 3], [1, 2]]).transpose([0, 2, 1])
        alpha = np.tensordot(working, vectors[step].conj(), axes=[[0, 1, 2], [0, 1, 2]]).item().real
        alpha_values.append(float(alpha))

        if step < n_krylov - 1:
            residual = lanczos_project(working, vectors)
            residual = lanczos_project(residual, vectors)
            beta = np.sqrt(np.tensordot(residual.conj(), residual, axes=[[0, 1, 2], [0, 1, 2]]).item().real)
            if beta < tol:
                break
            beta_values.append(float(beta))
            vectors.append(residual / beta)

    eigenvalues, eigenvectors = scipy.linalg.eigh_tridiagonal(alpha_values, beta_values)
    eigen_tensor = np.tensordot(eigenvectors[:, 0], np.asarray(vectors), axes=[[0], [0]])

    energy_tensor = np.tensordot(left_environment, eigen_tensor, axes=[[1], [0]])
    energy_tensor = np.tensordot(energy_tensor, center_mpo, axes=[[1, 3], [2, 1]])
    energy_tensor = np.tensordot(energy_tensor, right_environment, axes=[[1, 3], [1, 2]]).transpose([0, 2, 1])
    energy = np.tensordot(energy_tensor, eigen_tensor.conj(), axes=[[0, 1, 2], [0, 1, 2]]).item().real
    return float(energy), eigen_tensor


# Backward-compatible alias for notebooks that used the misspelled name.
Lanzos_method = lanczos_method


def dmrg_1site_right_to_left_sweep(
    environments: dict[int, Array],
    mps_list: list[Array],
    mpo_list: list[Array],
    n_krylov: int = 5,
) -> list[float]:
    """Perform one one-site DMRG sweep from right to left in place."""
    energies: list[float] = []
    num_sites = len(mps_list)
    for site in range(num_sites - 1, -1, -1):
        energy, tensor = lanczos_method(environments[site - 1], mpo_list[site], environments[site + 1], mps_list[site], n_krylov=n_krylov)
        shape = tensor.shape
        matrix = tensor.reshape([shape[0], shape[1] * shape[2]])
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        mps_list[site] = np.reshape(vh, [vh.shape[0], shape[1], shape[2]])

        if site == 0:
            phase = u[0, 0] * singular_values[0]
            if abs(phase) > 0:
                mps_list[site] = mps_list[site] * phase / abs(phase)
        else:
            mps_list[site - 1] = np.tensordot(mps_list[site - 1], u @ np.diag(singular_values), axes=[[1], [0]]).transpose([0, 2, 1])
        environments[site] = update_right(environments[site + 1], mps_list[site], mps_list[site].conj(), mpo_list[site])
        energies.append(energy)
    return energies


def dmrg_1site_left_to_right_sweep(
    environments: dict[int, Array],
    mps_list: list[Array],
    mpo_list: list[Array],
    n_krylov: int = 5,
) -> list[float]:
    """Perform one one-site DMRG sweep from left to right in place."""
    energies: list[float] = []
    num_sites = len(mps_list)
    for site in range(num_sites):
        energy, tensor = lanczos_method(environments[site - 1], mpo_list[site], environments[site + 1], mps_list[site], n_krylov=n_krylov)
        tensor = tensor.transpose([0, 2, 1])
        shape = tensor.shape
        matrix = tensor.reshape([shape[0] * shape[1], shape[2]])
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        mps_list[site] = np.reshape(u, [shape[0], shape[1], u.shape[1]]).transpose([0, 2, 1])

        if site == num_sites - 1:
            phase = singular_values[0] * vh[0, 0]
            if abs(phase) > 0:
                mps_list[site] = mps_list[site] * phase / abs(phase)
        else:
            mps_list[site + 1] = np.tensordot(np.diag(singular_values) @ vh, mps_list[site + 1], axes=[[1], [0]])
        environments[site] = update_left(environments[site - 1], mps_list[site], mps_list[site].conj(), mpo_list[site])
        energies.append(energy)
    return energies


def dmrg_1site(
    mps_init_list: list[Array],
    mpo_list: list[Array],
    num_sweeps: int,
    n_krylov: int = 5,
) -> tuple[list[list[float]], list[Array]]:
    """Run one-site DMRG and return sweep energies and the optimized MPS."""
    if len(mps_init_list) != len(mpo_list):
        raise ValueError("mps_init_list and mpo_list must have the same length.")
    mps_list = [tensor.copy() for tensor in mps_init_list]
    num_sites = len(mps_list)

    environments: dict[int, Array] = {
        -1: np.array([1], dtype=np.complex128).reshape([1, 1, 1]),
        num_sites: np.array([1], dtype=np.complex128).reshape([1, 1, 1]),
    }
    for site in range(num_sites):
        environments[site] = update_left(environments[site - 1], mps_list[site], mps_list[site].conj(), mpo_list[site])

    energy_history: list[list[float]] = []
    sweep_count = 0
    while sweep_count < num_sweeps:
        energy_history.append(dmrg_1site_right_to_left_sweep(environments, mps_list, mpo_list, n_krylov=n_krylov))
        sweep_count += 1
        if sweep_count >= num_sweeps:
            break
        energy_history.append(dmrg_1site_left_to_right_sweep(environments, mps_list, mpo_list, n_krylov=n_krylov))
        sweep_count += 1

    if not check_right_normalized(mps_list):
        mps_list = normal_to_canonical(mps_list, mode="right")
    return energy_history, mps_list


def dmrg_2site_right_to_left_sweep(
    environments: dict[int, Array],
    mps_list: list[Array],
    mpo_list: list[Array],
    max_bond_dim: int,
    n_krylov: int = 5,
) -> list[float]:
    """Perform one two-site DMRG sweep from right to left in place."""
    num_sites = len(mps_list)
    energies: list[float] = []
    isometry = np.eye(4, dtype=np.complex128).reshape(2, 2, 4)
    for site in range(num_sites - 1, 0, -1):
        merged_mpo = np.tensordot(mpo_list[site - 1], mpo_list[site], axes=[[3], [2]])
        merged_mpo = np.tensordot(isometry, merged_mpo, axes=[[0, 1], [1, 4]])
        merged_mpo = np.tensordot(isometry, merged_mpo, axes=[[0, 1], [1, 3]])

        merged_mps = np.tensordot(mps_list[site - 1], mps_list[site], axes=[[1], [0]])
        merged_mps = np.tensordot(merged_mps, isometry, axes=[[1, 3], [0, 1]])
        energy, tensor = lanczos_method(environments[site - 2], merged_mpo, environments[site + 1], merged_mps, n_krylov=n_krylov)
        shape = tensor.shape
        matrix = tensor.reshape(shape[0], shape[1], 2, 2).transpose([0, 2, 1, 3]).reshape(2 * shape[0], 2 * shape[1])
        keep = min(max_bond_dim, matrix.shape[0], matrix.shape[1])
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        vh_keep = vh[:keep]
        us_keep = (u @ np.diag(singular_values))[:, :keep]
        mps_list[site] = vh_keep.reshape(keep, shape[1], 2)
        mps_list[site - 1] = us_keep.reshape(shape[0], 2, keep).transpose([0, 2, 1])
        environments[site] = update_right(environments[site + 1], mps_list[site], mps_list[site].conj(), mpo_list[site])
        energies.append(energy)

    tensor = mps_list[0]
    matrix = tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    phase = u[0, 0] * singular_values[0]
    mps_list[0] = vh.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2])
    if abs(phase) > 0:
        mps_list[0] = mps_list[0] * phase / abs(phase)
    environments[0] = update_right(environments[1], mps_list[0], mps_list[0].conj(), mpo_list[0])
    return energies


def dmrg_2site_left_to_right_sweep(
    environments: dict[int, Array],
    mps_list: list[Array],
    mpo_list: list[Array],
    max_bond_dim: int,
    n_krylov: int = 5,
) -> list[float]:
    """Perform one two-site DMRG sweep from left to right in place."""
    num_sites = len(mps_list)
    energies: list[float] = []
    isometry = np.eye(4, dtype=np.complex128).reshape(2, 2, 4)
    for site in range(num_sites - 1):
        merged_mpo = np.tensordot(mpo_list[site], mpo_list[site + 1], axes=[[3], [2]])
        merged_mpo = np.tensordot(isometry, merged_mpo, axes=[[0, 1], [1, 4]])
        merged_mpo = np.tensordot(isometry, merged_mpo, axes=[[0, 1], [1, 3]])

        merged_mps = np.tensordot(mps_list[site], mps_list[site + 1], axes=[[1], [0]])
        merged_mps = np.tensordot(merged_mps, isometry, axes=[[1, 3], [0, 1]])
        energy, tensor = lanczos_method(environments[site - 1], merged_mpo, environments[site + 2], merged_mps, n_krylov=n_krylov)
        shape = tensor.shape
        matrix = tensor.reshape(shape[0], shape[1], 2, 2).transpose([0, 2, 1, 3]).reshape(2 * shape[0], 2 * shape[1])
        keep = min(max_bond_dim, matrix.shape[0], matrix.shape[1])
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        u_keep = u[:, :keep]
        sv_keep = (np.diag(singular_values) @ vh)[:keep]
        mps_list[site] = u_keep.reshape(shape[0], 2, keep).transpose([0, 2, 1])
        mps_list[site + 1] = sv_keep.reshape(keep, shape[1], 2)
        environments[site] = update_left(environments[site - 1], mps_list[site], mps_list[site].conj(), mpo_list[site])
        energies.append(energy)

    tensor = mps_list[-1]
    matrix = tensor.transpose([0, 2, 1]).reshape(tensor.shape[0] * tensor.shape[2], tensor.shape[1])
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    phase = singular_values[0] * vh[0, 0]
    mps_list[-1] = u.reshape(tensor.shape[0], tensor.shape[2], 1).transpose([0, 2, 1])
    if abs(phase) > 0:
        mps_list[-1] = mps_list[-1] * phase / abs(phase)
    environments[num_sites - 1] = update_left(environments[num_sites - 2], mps_list[-1], mps_list[-1].conj(), mpo_list[-1])
    return energies


def dmrg_2site(
    mps_init_list: list[Array],
    mpo_list: list[Array],
    num_sweeps: int,
    max_bond_dim: int,
    n_krylov: int = 5,
) -> tuple[list[list[float]], list[Array]]:
    """Run two-site DMRG and return sweep energies and the optimized MPS."""
    if len(mps_init_list) != len(mpo_list):
        raise ValueError("mps_init_list and mpo_list must have the same length.")
    mps_list = [tensor.copy() for tensor in mps_init_list]
    num_sites = len(mps_list)

    environments: dict[int, Array] = {
        -1: np.array([1], dtype=np.complex128).reshape([1, 1, 1]),
        num_sites: np.array([1], dtype=np.complex128).reshape([1, 1, 1]),
    }
    for site in range(num_sites):
        environments[site] = update_left(environments[site - 1], mps_list[site], mps_list[site].conj(), mpo_list[site])

    energy_history: list[list[float]] = []
    sweep_count = 0
    while sweep_count < num_sweeps:
        energy_history.append(
            dmrg_2site_right_to_left_sweep(environments, mps_list, mpo_list, max_bond_dim=max_bond_dim, n_krylov=n_krylov)
        )
        sweep_count += 1
        if sweep_count >= num_sweeps:
            break
        energy_history.append(
            dmrg_2site_left_to_right_sweep(environments, mps_list, mpo_list, max_bond_dim=max_bond_dim, n_krylov=n_krylov)
        )
        sweep_count += 1

    if not check_right_normalized(mps_list):
        mps_list = normal_to_canonical(mps_list, mode="right")
    return energy_history, mps_list


def normal_to_canonical(mps_list: list[Array], mode: str = "right") -> list[Array]:
    """Canonicalize an MPS using sequential SVDs."""
    if mode != "right":
        raise NotImplementedError("Only right canonicalization is implemented.")
    inter = np.array([1], dtype=np.complex128).reshape(1, 1)
    out: list[Array] = []
    for tensor in reversed(mps_list):
        contracted = np.tensordot(tensor, inter, axes=[1, 0]).transpose([0, 2, 1])
        d0, d1, d2 = contracted.shape
        matrix = contracted.reshape(d0, d1 * d2)
        u, singular_values, vh = scipy.linalg.svd(matrix, full_matrices=False)
        singular_values = np.where(singular_values < 1e-12, 0.0, singular_values)
        inter = u @ np.diag(singular_values)
        out.append(vh.reshape(vh.shape[0], d1, d2))
    phase = inter[0, 0] if inter.shape == (1, 1) else 1.0
    if abs(phase) > 0:
        out[-1] = out[-1] * phase / abs(phase)
    return list(reversed(out))


def sampling_contraction(prefix: Array, tensor: Array, rng: np.random.Generator) -> tuple[int, Array]:
    """Sample one physical index from a right-normalized MPS tensor."""
    branch_0 = prefix @ tensor[:, :, 0]
    branch_1 = prefix @ tensor[:, :, 1]
    p0 = np.sum(branch_0 * branch_0.conj()).real
    p1 = np.sum(branch_1 * branch_1.conj()).real
    probs = np.array([p0, p1], dtype=float)
    probs = probs / probs.sum()
    bit = int(rng.choice([0, 1], p=probs))
    return bit, [branch_0, branch_1][bit]


def mps_sampling(mps_list: list[Array], rng: np.random.Generator | None = None) -> list[str]:
    """Sample one computational-basis bit string from a right-normalized MPS."""
    if not check_right_normalized(mps_list):
        mps_list = normal_to_canonical(mps_list, mode="right")
    rng = np.random.default_rng() if rng is None else rng
    bits: list[str] = []
    prefix = np.array([1], dtype=np.complex128).reshape(1, 1)
    for tensor in mps_list:
        bit, prefix = sampling_contraction(prefix, tensor, rng)
        bits.append(str(bit))
    return bits


def mps_chunk_contraction(mps_list: list[Array]) -> Array:
    """Contract consecutive MPS tensors into a block tensor."""
    if not mps_list:
        raise ValueError("mps_list must be non-empty.")
    tensor = mps_list[0].copy()
    for site_tensor in mps_list[1:]:
        tensor = np.tensordot(tensor, site_tensor, axes=[[-2], [0]])
    block_size = len(mps_list)
    basis = np.eye(2**block_size, dtype=np.complex128).reshape([2**block_size] + [2] * block_size)
    tensor = np.tensordot(tensor, basis, axes=[list(range(1, block_size)) + [block_size + 1], range(1, block_size + 1)])
    return tensor


def block_mps_tensors(mps_list: list[Array], block_size: int) -> list[Array]:
    """Group an MPS into block tensors of physical dimension ``2**block_size``."""
    if len(mps_list) % block_size != 0:
        raise ValueError("The number of sites must be divisible by block_size.")
    blocks = []
    for start in range(0, len(mps_list), block_size):
        blocks.append(mps_chunk_contraction(mps_list[start : start + block_size]))
    return blocks


def dmrg_bond_alternating_xxz_state(
    num_sites: int,
    *,
    coupling_even: float = 1.0,
    coupling_odd: float = 1.0,
    delta: float = 0.0,
    max_bond_dim: int = 30,
    num_sweeps: int = 5,
    n_krylov: int = 5,
    boundary_penalty: float = 0.1,
    two_site_warmup: bool = False,
) -> DMRGResult:
    """Compute a ground-state MPS for the alternating XXZ chain."""
    mpo_list = mpo_bond_alternating_xxz(
        coupling_even=coupling_even,
        coupling_odd=coupling_odd,
        delta=delta,
        num_sites=num_sites,
        boundary_penalty=boundary_penalty,
    )
    init = mps_init_list_from_mpo_list(mpo_list, max_bond_dim=max_bond_dim)
    if two_site_warmup:
        energies, mps = dmrg_2site(init, mpo_list, num_sweeps=max(1, num_sweeps // 2), max_bond_dim=max_bond_dim, n_krylov=n_krylov)
        energies_1, mps = dmrg_1site(mps, mpo_list, num_sweeps=num_sweeps, n_krylov=n_krylov)
        energies.extend(energies_1)
    else:
        energies, mps = dmrg_1site(init, mpo_list, num_sweeps=num_sweeps, n_krylov=n_krylov)
    if not check_right_normalized(mps):
        mps = normal_to_canonical(mps, mode="right")
    return DMRGResult(energies=energies, mps=mps)


def dmrg_ssh_state(
    coupling_even: float,
    coupling_odd: float,
    num_sites: int,
    max_bond_dim: int = 30,
    num_sweeps: int = 5,
    n_krylov: int = 5,
) -> DMRGResult:
    """Compute an SSH-limit MPS with only ``XX + YY`` couplings."""
    return dmrg_bond_alternating_xxz_state(
        num_sites=num_sites,
        coupling_even=coupling_even / 2.0,
        coupling_odd=coupling_odd / 2.0,
        delta=0.0,
        max_bond_dim=max_bond_dim,
        num_sweeps=num_sweeps,
        n_krylov=n_krylov,
        boundary_penalty=0.0,
    )


def pauli_product(op_str1: str, op_str2: str) -> str:
    """Multiply Pauli strings while dropping the global phase."""
    if len(op_str1) != len(op_str2):
        raise ValueError("Pauli strings must have the same length.")
    out = []
    for first, second in zip(op_str1, op_str2):
        if first == "I":
            out.append(second)
        elif second == "I":
            out.append(first)
        elif first == second:
            out.append("I")
        else:
            remaining = {"X", "Y", "Z"} - {first, second}
            out.append(remaining.pop())
    return "".join(out)


def cluster_heisenberg_terms_open(num_sites: int) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return Pauli-string terms of the open-boundary cluster-Heisenberg model."""
    cluster_terms: list[str] = []
    xx_terms: list[str] = []
    yy_terms: list[str] = []
    zz_terms: list[str] = []

    for site in range(num_sites - 2):
        term = ["I"] * num_sites
        term[site] = "Z"
        term[site + 1] = "X"
        term[site + 2] = "Z"
        cluster_terms.append("".join(term))

    for site in range(num_sites - 1):
        for op, target in [("X", xx_terms), ("Y", yy_terms), ("Z", zz_terms)]:
            term = ["I"] * num_sites
            term[site] = op
            term[site + 1] = op
            target.append("".join(term))
    return cluster_terms, xx_terms, yy_terms, zz_terms
