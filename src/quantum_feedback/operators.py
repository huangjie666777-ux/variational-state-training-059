"""Density-matrix quantum channels and gate operations."""

from __future__ import annotations

import numpy as np

I_MATRIX = np.eye(2, dtype=complex)
X_MATRIX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Z_MATRIX = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
H_MATRIX = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
Y_MATRIX = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)

PAULI_MATRICES = {"I": I_MATRIX, "X": X_MATRIX, "Y": Y_MATRIX, "Z": Z_MATRIX}


def pauli_string_matrix(pauli_string: str, num_qubits: int) -> np.ndarray:
    """Build a Pauli-string operator; the rightmost character addresses qubit 0."""
    factors = [
        PAULI_MATRICES[pauli_string[index]]
        for index in range(num_qubits - 1, -1, -1)
    ]
    operator = factors[0]
    for factor in factors[1:]:
        operator = np.kron(operator, factor)
    return operator


def rz_matrix(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)]).astype(complex)


def single_qubit_operator(matrix: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    factors = [matrix if factor_qubit == qubit else I_MATRIX for factor_qubit in range(num_qubits - 1, -1, -1)]
    operator = factors[0]
    for factor in factors[1:]:
        operator = np.kron(operator, factor)
    return operator


def apply_single_gate(rho: np.ndarray, matrix: np.ndarray, qubit: int) -> np.ndarray:
    operator = single_qubit_operator(matrix, qubit, int(np.log2(rho.shape[0])))
    return operator @ rho @ operator.conj().T


def apply_cx(rho: np.ndarray, control: int, target: int) -> np.ndarray:
    num_qubits = int(np.log2(rho.shape[0]))
    dimension = 1 << num_qubits
    operator = np.zeros((dimension, dimension), dtype=complex)
    for input_index in range(dimension):
        output_index = input_index
        if (input_index >> control) & 1:
            output_index ^= 1 << target
        operator[output_index, input_index] = 1.0
    return operator @ rho @ operator.conj().T


def apply_phase_flip(rho: np.ndarray, qubit: int, probability: float) -> np.ndarray:
    flipped = apply_single_gate(rho, Z_MATRIX, qubit)
    return (1.0 - probability) * rho + probability * flipped


def reset_to_zero(rho: np.ndarray, qubit: int) -> np.ndarray:
    num_qubits = int(np.log2(rho.shape[0]))
    p0 = projector(0, qubit, num_qubits)
    p1 = projector(1, qubit, num_qubits)
    selected = p0 @ rho @ p0
    operator = single_qubit_operator(X_MATRIX, qubit, num_qubits)
    flipped_state = p1 @ rho @ p1
    return selected + operator @ flipped_state @ operator.conj().T


def projector(value: int, qubit: int, num_qubits: int) -> np.ndarray:
    vector = np.zeros(2, dtype=complex)
    vector[value] = 1.0
    return single_qubit_operator(np.diag(vector).astype(complex), qubit, num_qubits)


def measure_project(rho: np.ndarray, qubit: int) -> tuple[float, np.ndarray, float, np.ndarray]:
    p0_matrix = projector(0, qubit, int(np.log2(rho.shape[0])))
    p1_matrix = projector(1, qubit, int(np.log2(rho.shape[0])))
    prob0 = float(np.real(np.trace(p0_matrix @ rho)))
    prob1 = float(np.real(np.trace(p1_matrix @ rho)))
    state0 = p0_matrix @ rho @ p0_matrix
    state1 = p1_matrix @ rho @ p1_matrix
    if prob0 > 0.0:
        state0 /= prob0
    if prob1 > 0.0:
        state1 /= prob1
    return prob0, state0, prob1, state1
