"""Deterministic Pauli-sum energies, postselection, and exact gradients."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Mapping

import numpy as np

from . import operators as ops
from .circuit import Operation, bind_parameters, parameter_names, validate_circuit
from .engine import Branch, execute_bound_operations
from .errors import (
    CircuitValidationError,
    HamiltonianValidationError,
    ParameterBindingError,
    PostselectionError,
)

POSTSELECTION_FLOOR = 1e-12


@dataclass(frozen=True)
class ObjectiveEvaluation:
    parameters: dict[str, float]
    energy: float
    success_probability: float
    postselected: bool


@dataclass(frozen=True)
class GradientEvaluation:
    parameters: dict[str, float]
    energy: float
    success_probability: float
    gradient: dict[str, float]


def validate_hamiltonian(hamiltonian: Any, num_qubits: int) -> list[tuple[float, str]]:
    """Validate a real-coefficient Pauli sum covering every qubit position."""
    if isinstance(hamiltonian, Mapping):
        terms = list(hamiltonian.items())
    elif isinstance(hamiltonian, (list, tuple)) and all(
        isinstance(term, (list, tuple)) and len(term) == 2 for term in hamiltonian
    ):
        terms = list(hamiltonian)
    else:
        raise HamiltonianValidationError(
            "hamiltonian must be a mapping {pauli_string: coefficient} or a list of pairs"
        )
    if not terms:
        raise HamiltonianValidationError("hamiltonian must contain at least one term")

    normalized: list[tuple[float, str]] = []
    for term_index, (pauli_string, coefficient) in enumerate(terms):
        where = f"hamiltonian term {term_index}"
        if not isinstance(pauli_string, str):
            raise HamiltonianValidationError(f"{where}: Pauli string must be a string")
        if len(pauli_string) != num_qubits or any(char not in "IXYZ" for char in pauli_string):
            raise HamiltonianValidationError(
                f"{where}: Pauli string {pauli_string!r} must have length {num_qubits} "
                "and contain only I, X, Y, Z"
            )
        if (
            not isinstance(coefficient, (int, float, np.floating, np.integer))
            or isinstance(coefficient, bool)
        ):
            raise HamiltonianValidationError(f"{where}: coefficient must be a real number")
        coefficient = float(coefficient)
        if not isfinite(coefficient):
            raise HamiltonianValidationError(f"{where}: coefficient must be finite")
        normalized.append((coefficient, pauli_string))
    return normalized


def validate_postselection(
    postselection: Any, num_clbits: int
) -> dict[int, int] | None:
    """Validate a partial assignment of terminal classical register bits."""
    if postselection is None:
        return None
    if isinstance(postselection, (list, tuple)):
        items = list(postselection)
        if not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in items):
            raise PostselectionError("postselection must map clbit to 0 or 1")
    elif isinstance(postselection, Mapping):
        items = list(postselection.items())
    else:
        raise PostselectionError("postselection must map clbit to 0 or 1")

    result: dict[int, int] = {}
    for clbit, value in items:
        if not isinstance(clbit, (int, np.integer)) or isinstance(clbit, bool):
            raise PostselectionError("postselection classical bit must be an integer")
        if not 0 <= int(clbit) < num_clbits:
            raise PostselectionError(f"postselection classical bit {clbit} is out of range")
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or int(value) not in (0, 1):
            raise PostselectionError("postselection value must be 0 or 1")
        if int(clbit) in result and result[int(clbit)] != int(value):
            raise PostselectionError(f"postselection classical bit {clbit} assigned twice")
        result[int(clbit)] = int(value)
    return result


def prepare_operations(
    num_qubits: int,
    num_clbits: int,
    operations: Any,
    parameters: Mapping[str, float] | None = None,
) -> list[Operation]:
    """Validate a template and bind a complete parameter mapping without mutation."""
    normalized = validate_circuit(num_qubits, num_clbits, operations)
    names = parameter_names(normalized)
    if parameters is None:
        if names:
            raise CircuitValidationError("parameterized circuit requires a parameter value mapping")
        return normalized
    if not names:
        raise ParameterBindingError("circuit has no named parameters")
    return bind_parameters(normalized, parameters)


def selected_branches(
    branches: list[Branch], postselection: dict[int, int] | None
) -> list[Branch]:
    if postselection is None:
        return list(branches)
    return [
        branch
        for branch in branches
        if all(branch.classical_bits[clbit] == value for clbit, value in postselection.items())
    ]


def branch_statistics(
    branches: list[Branch], terms: list[tuple[float, str]], num_qubits: int
) -> tuple[float, float, np.ndarray | None]:
    """Return success probability, unnormalized energy numerator, and conditional rho."""
    success_probability = sum(branch.probability for branch in branches)
    success_probability = max(0.0, float(np.real(success_probability)))
    if success_probability <= POSTSELECTION_FLOOR:
        return success_probability, 0.0, None

    unnormalized_rho = np.zeros_like(branches[0].rho)
    for branch in branches:
        unnormalized_rho += branch.probability * branch.rho
    numerator = 0.0
    for coefficient, pauli_string in terms:
        operator = ops.pauli_string_matrix(pauli_string, num_qubits)
        expectation = float(np.real(np.trace(operator @ unnormalized_rho)))
        numerator += coefficient * expectation
    return success_probability, numerator, unnormalized_rho


def _run_statistics(
    bound_operations: list[Operation],
    num_qubits: int,
    num_clbits: int,
    terms: list[tuple[float, str]],
    postselection: dict[int, int] | None,
    *,
    allow_failure: bool = False,
) -> tuple[float, float]:
    branches = execute_bound(num_qubits, num_clbits, bound_operations)
    chosen = selected_branches(branches, postselection)
    success_probability, numerator, _ = branch_statistics(chosen, terms, num_qubits)
    if not allow_failure and success_probability <= POSTSELECTION_FLOOR:
        raise PostselectionError(
            f"postselection success probability {success_probability:.3e} is at or below "
            f"{POSTSELECTION_FLOOR:.0e}; conditional energy is not defined"
        )
    return success_probability, numerator


def execute_bound(
    num_qubits: int, num_clbits: int, bound_operations: list[Operation]
) -> list[Branch]:
    return execute_bound_operations(num_qubits, num_clbits, bound_operations)


def evaluate_energy(
    num_qubits: int,
    num_clbits: int,
    operations: Any,
    parameters: Mapping[str, float] | None,
    hamiltonian: Any,
    postselection: Any = None,
) -> ObjectiveEvaluation:
    """Compute the deterministic conditional energy and postselection probability."""
    terms = validate_hamiltonian(hamiltonian, num_qubits)
    selection = validate_postselection(postselection, num_clbits)
    bound_operations = prepare_operations(num_qubits, num_clbits, operations, parameters)
    success_probability, numerator = _run_statistics(
        bound_operations, num_qubits, num_clbits, terms, selection
    )
    names = parameter_names(validate_circuit(num_qubits, num_clbits, operations))
    return ObjectiveEvaluation(
        parameters={name: float(parameters[name]) for name in names} if parameters is not None else {},
        energy=numerator / success_probability,
        success_probability=success_probability,
        postselected=selection is not None,
    )


def energy_gradient(
    num_qubits: int,
    num_clbits: int,
    operations: Any,
    parameters: Mapping[str, float],
    hamiltonian: Any,
    postselection: Any = None,
) -> GradientEvaluation:
    """Return the conditional energy, success probability, and exact parameter gradient.

    Each RZ gate obeys the two-term parameter-shift rule. Gates sharing a named
    parameter contribute additively, including negative scales. Shifted circuits
    whose postselection probability is exactly zero contribute zero numerator and
    probability, so they cannot invalidate an otherwise well-defined derivative.
    """
    terms = validate_hamiltonian(hamiltonian, num_qubits)
    selection = validate_postselection(postselection, num_clbits)
    normalized = validate_circuit(num_qubits, num_clbits, operations)
    names = parameter_names(normalized)
    if not names:
        raise ParameterBindingError("energy_gradient requires a circuit with named parameters")
    bound_operations = bind_parameters(normalized, parameters)

    probability_base, numerator_base = _run_statistics(
        bound_operations, num_qubits, num_clbits, terms, selection
    )
    energy_base = numerator_base / probability_base
    accumulated = {name: 0.0 for name in names}

    rz_indices = [
        index for index, operation in enumerate(bound_operations) if operation.name == "RZ"
    ]
    for rz_index in rz_indices:
        base_gate = bound_operations[rz_index]
        scale = normalized[rz_index].scale
        for sign, shifted_angle in ((1.0, base_gate.angle + np.pi / 2.0),
                                    (-1.0, base_gate.angle - np.pi / 2.0)):
            shifted_gate = replace(base_gate, angle=shifted_angle)
            shifted_operations = list(bound_operations)
            shifted_operations[rz_index] = shifted_gate
            shifted_probability, shifted_numerator = _run_statistics(
                shifted_operations,
                num_qubits,
                num_clbits,
                terms,
                selection,
                allow_failure=True,
            )
            if sign == 1.0:
                plus_probability, plus_numerator = shifted_probability, shifted_numerator
            else:
                minus_probability, minus_numerator = shifted_probability, shifted_numerator

        probability_derivative = scale * (plus_probability - minus_probability) / 2.0
        numerator_derivative = scale * (plus_numerator - minus_numerator) / 2.0
        energy_derivative = (
            numerator_derivative / probability_base
            - energy_base * probability_derivative / probability_base
        )
        accumulated[normalized[rz_index].parameter_name] += energy_derivative

    return GradientEvaluation(
        parameters={name: float(parameters[name]) for name in names},
        energy=energy_base,
        success_probability=probability_base,
        gradient=accumulated,
    )
