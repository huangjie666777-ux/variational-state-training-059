"""Density-matrix branch execution including measurement feedback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import operators as ops
from .circuit import Operation, bind_parameters, parameter_names, validate_circuit
from .errors import CircuitValidationError
from .results import ExecutionResult


@dataclass
class Branch:
    probability: float
    classical_bits: tuple[int, ...]
    rho: np.ndarray


def run(
    num_qubits: int,
    num_clbits: int,
    operations: list | tuple,
    parameters: dict | None = None,
) -> ExecutionResult:
    final_branches = execute_branches(
        num_qubits, num_clbits, operations, parameters=parameters
    )
    dimension = 1 << num_qubits
    rho = np.zeros((dimension, dimension), dtype=complex)
    quantum_rho = sum(
        (branch.probability * branch.rho for branch in final_branches),
        np.zeros_like(rho),
    )
    probabilities = _merge_probabilities(final_branches, num_clbits)
    return ExecutionResult(quantum_rho, probabilities, num_qubits, num_clbits)


def execute_branches(
    num_qubits: int,
    num_clbits: int,
    operations: list | tuple,
    parameters: dict | None = None,
) -> list[Branch]:
    """Run a (possibly parameterized) circuit and return weighted classical branches.

    Each branch carries its unnormalized joint probability and its normalized
    conditional density matrix. Parameter binding happens before any state is
    allocated and never mutates the template or the caller mapping.
    """
    normalized_operations = validate_circuit(num_qubits, num_clbits, operations)
    if parameters is not None:
        if not parameter_names(normalized_operations):
            raise CircuitValidationError("circuit has no named parameters")
        normalized_operations = bind_parameters(normalized_operations, parameters)
    elif parameter_names(normalized_operations):
        raise CircuitValidationError("parameterized circuit requires a parameter value mapping")
    dimension = 1 << num_qubits
    rho = np.zeros((dimension, dimension), dtype=complex)
    rho[0, 0] = 1.0
    branches = [Branch(1.0, tuple(0 for _ in range(num_clbits)), rho)]

    for operation in normalized_operations:
        branches = _execute(branches, operation)
    return branches


def execute_bound_operations(
    num_qubits: int, num_clbits: int, bound_operations: list[Operation]
) -> list[Branch]:
    """Execute an already validated and parameter-bound operation list."""
    dimension = 1 << num_qubits
    rho = np.zeros((dimension, dimension), dtype=complex)
    rho[0, 0] = 1.0
    branches = [Branch(1.0, tuple(0 for _ in range(num_clbits)), rho)]
    for operation in bound_operations:
        branches = _execute(branches, operation)
    return branches


def _execute(branches: list[Branch], operation: Operation) -> list[Branch]:
    if operation.name == "MEASURE":
        return _measure(branches, operation)
    if operation.name == "RESET":
        return _apply_each(branches, lambda state: ops.reset_to_zero(state, operation.qubit))
    if operation.name == "PHASE_FLIP":
        return _apply_each(
            branches,
            lambda state: ops.apply_phase_flip(state, operation.qubit, operation.probability),
        )
    return _apply_gate(branches, operation)


def _apply_gate(branches: list[Branch], operation: Operation) -> list[Branch]:
    next_branches: list[Branch] = []
    for branch in branches:
        active = (
            operation.condition_clbit is None
            or branch.classical_bits[operation.condition_clbit] == operation.condition_value
        )
        if not active:
            next_branches.append(branch)
            continue
        next_state = _gate_state(branch.rho, operation)
        next_branches.append(Branch(branch.probability, branch.classical_bits, next_state))
    return _merge_by_classical_bits(next_branches)


def _gate_state(rho: np.ndarray, operation: Operation) -> np.ndarray:
    if operation.name == "H":
        return ops.apply_single_gate(rho, ops.H_MATRIX, operation.qubit)
    if operation.name == "X":
        return ops.apply_single_gate(rho, ops.X_MATRIX, operation.qubit)
    if operation.name == "Z":
        return ops.apply_single_gate(rho, ops.Z_MATRIX, operation.qubit)
    if operation.name == "RZ":
        return ops.apply_single_gate(rho, ops.rz_matrix(operation.angle), operation.qubit)
    return ops.apply_cx(rho, operation.control, operation.target)


def _measure(branches: list[Branch], operation: Operation) -> list[Branch]:
    next_branches: list[Branch] = []
    for branch in branches:
        prob0, state0, prob1, state1 = ops.measure_project(branch.rho, operation.qubit)
        bits0 = _overwrite_bit(branch.classical_bits, operation.clbit, 0)
        bits1 = _overwrite_bit(branch.classical_bits, operation.clbit, 1)
        if branch.probability * prob0 > 0.0:
            next_branches.append(Branch(branch.probability * prob0, bits0, state0))
        if branch.probability * prob1 > 0.0:
            next_branches.append(Branch(branch.probability * prob1, bits1, state1))
    return _merge_by_classical_bits(next_branches)


def _apply_each(branches: list[Branch], channel) -> list[Branch]:
    next_branches = [
        Branch(branch.probability, branch.classical_bits, channel(branch.rho))
        for branch in branches
    ]
    return _merge_by_classical_bits(next_branches)


def _merge_by_classical_bits(branches: list[Branch]) -> list[Branch]:
    merged: dict[tuple[int, ...], Branch] = {}
    for branch in branches:
        if branch.classical_bits in merged:
            target = merged[branch.classical_bits]
            combined_weight = target.probability + branch.probability
            target.rho = (
                target.probability * target.rho + branch.probability * branch.rho
            ) / combined_weight
            target.probability = combined_weight
        else:
            merged[branch.classical_bits] = Branch(branch.probability, branch.classical_bits, branch.rho)
    return list(merged.values())


def _merge_probabilities(branches: list[Branch], num_clbits: int) -> dict[tuple[int, ...], float]:
    probabilities = {tuple(0 for _ in range(num_clbits)): 0.0}
    for branch in branches:
        probabilities[branch.classical_bits] = probabilities.get(branch.classical_bits, 0.0) + branch.probability
    return probabilities


def _overwrite_bit(bits: tuple[int, ...], index: int, value: int) -> tuple[int, ...]:
    values = list(bits)
    values[index] = value
    return tuple(values)
