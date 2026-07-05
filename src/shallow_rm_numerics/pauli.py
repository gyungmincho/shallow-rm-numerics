"""Utilities for phase-free Pauli-string manipulations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

PAULI_TO_INT = {"I": 0, "X": 1, "Y": 2, "Z": 3}
INT_TO_PAULI = {value: key for key, value in PAULI_TO_INT.items()}


def validate_pauli_string(pauli: str) -> None:
    invalid = set(pauli) - set(PAULI_TO_INT)
    if invalid:
        raise ValueError(f"Invalid Pauli characters: {sorted(invalid)}")


def pauli_product_phase_free(left: str, right: str) -> str:
    """Return the phase-free product of two Pauli strings.

    The output ignores the global phase. This is sufficient for measurement-basis
    hitting tests and for constructing the support of H^2 observables.
    """
    validate_pauli_string(left)
    validate_pauli_string(right)
    if len(left) != len(right):
        raise ValueError("Pauli strings must have the same length.")

    out: list[str] = []
    for p_left, p_right in zip(left, right):
        if p_left == "I":
            out.append(p_right)
        elif p_right == "I":
            out.append(p_left)
        elif p_left == p_right:
            out.append("I")
        else:
            axes = {"X", "Y", "Z"}
            axes.remove(p_left)
            axes.remove(p_right)
            out.append(axes.pop())
    return "".join(out)


def pauli_commutes(left: str, right: str) -> bool:
    """Return True if two phase-free Pauli strings commute."""
    validate_pauli_string(left)
    validate_pauli_string(right)
    if len(left) != len(right):
        raise ValueError("Pauli strings must have the same length.")
    anticommute_count = 0
    for p_left, p_right in zip(left, right):
        if p_left == "I" or p_right == "I" or p_left == p_right:
            continue
        anticommute_count += 1
    return anticommute_count % 2 == 0


def pauli_weight(pauli: str | Sequence[int]) -> int:
    if isinstance(pauli, str):
        return sum(char != "I" for char in pauli)
    array = np.asarray(pauli)
    return int(np.count_nonzero(array != 0))


def block_weight(pauli: str, block_size: int) -> int:
    if len(pauli) % block_size != 0:
        raise ValueError("The Pauli-string length must be divisible by block_size.")
    return sum(pauli[i : i + block_size] != "I" * block_size for i in range(0, len(pauli), block_size))


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def pauli_set_product(left: Sequence[str], right: Sequence[str], *, unique: bool = False) -> list[str]:
    products = [pauli_product_phase_free(p_left, p_right) for p_left in left for p_right in right]
    return unique_preserve_order(products) if unique else products


def pauli_strings_to_int_array(strings: Sequence[str]) -> np.ndarray:
    if not strings:
        raise ValueError("At least one Pauli string is required.")
    length = len(strings[0])
    out = np.empty((len(strings), length), dtype=np.int8)
    for row, string in enumerate(strings):
        validate_pauli_string(string)
        if len(string) != length:
            raise ValueError("All Pauli strings must have the same length.")
        out[row] = [PAULI_TO_INT[char] for char in string]
    return out


def int_array_to_pauli_strings(array: np.ndarray) -> list[str]:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional array.")
    return ["".join(INT_TO_PAULI[int(value)] for value in row) for row in array]


def two_qubit_block_strings(strings: Sequence[str]) -> np.ndarray:
    """Return two-qubit block codes using 0,1,2,3 for I,X,Y,Z."""
    if not strings:
        raise ValueError("At least one Pauli string is required.")
    n = len(strings[0])
    if n % 2 != 0:
        raise ValueError("Only even system sizes are supported for two-qubit blocks.")
    out: list[list[str]] = []
    for string in strings:
        validate_pauli_string(string)
        if len(string) != n:
            raise ValueError("All Pauli strings must have the same length.")
        out.append([
            f"{PAULI_TO_INT[string[i]]}{PAULI_TO_INT[string[i + 1]]}"
            for i in range(0, n, 2)
        ])
    return np.asarray(out, dtype="U2")


def roll_two_qubit_blocks(block_array: np.ndarray) -> np.ndarray:
    """Convert even two-qubit blocks to the odd-block partition used in the notebooks."""
    block_array = np.asarray(block_array)
    if block_array.ndim != 2:
        raise ValueError("Expected a two-dimensional block array.")
    out: list[list[str]] = []
    for row in block_array:
        full = "".join(str(item) for item in row)
        rolled = full[1:] + full[0]
        out.append([rolled[i : i + 2] for i in range(0, len(rolled), 2)])
    return np.asarray(out, dtype="U2")


def cluster_heisenberg_terms_open(n_qubits: int) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return the four open-boundary term groups in the cluster-Heisenberg model."""
    if n_qubits < 3:
        raise ValueError("n_qubits must be at least 3.")

    cluster_terms: list[str] = []
    xx_terms: list[str] = []
    yy_terms: list[str] = []
    zz_terms: list[str] = []

    for site in range(n_qubits - 2):
        chars = ["I"] * n_qubits
        chars[site] = "Z"
        chars[site + 1] = "X"
        chars[site + 2] = "Z"
        cluster_terms.append("".join(chars))

    for site in range(n_qubits - 1):
        chars = ["I"] * n_qubits
        chars[site] = "X"
        chars[site + 1] = "X"
        xx_terms.append("".join(chars))

        chars = ["I"] * n_qubits
        chars[site] = "Y"
        chars[site + 1] = "Y"
        yy_terms.append("".join(chars))

        chars = ["I"] * n_qubits
        chars[site] = "Z"
        chars[site + 1] = "Z"
        zz_terms.append("".join(chars))

    return cluster_terms, xx_terms, yy_terms, zz_terms


def cluster_heisenberg_h2_cross_terms_open(n_qubits: int, *, unique: bool = False) -> list[str]:
    """Return phase-free Pauli strings from cross-products of the four term groups.

    This matches the numerical notebooks used for the derandomization comparison:
    products are taken between different term groups, not within the same group.
    """
    groups = cluster_heisenberg_terms_open(n_qubits)
    out: list[str] = []
    for first in range(len(groups)):
        for second in range(first + 1, len(groups)):
            out.extend(pauli_set_product(groups[first], groups[second], unique=False))
    return unique_preserve_order(out) if unique else out
