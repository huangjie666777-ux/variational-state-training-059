"""Deterministic local variational training against an energy objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import Any, Mapping

import numpy as np

from .circuit import parameter_names, validate_circuit
from .errors import OptimizationError, PostselectionError
from .objective import (
    GradientEvaluation,
    energy_gradient,
    validate_hamiltonian,
    validate_postselection,
)

REASON_GRADIENT_TOLERANCE = "gradient_tolerance_reached"
REASON_MAX_ITERATIONS = "max_iterations_reached"
REASON_STEP_UNDERFLOW = "step_size_underflow"
REASON_INVALID_INITIAL_POINT = "invalid_initial_point"


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    parameters: dict[str, float]
    energy: float
    success_probability: float
    gradient_norm: float
    step_size: float
    accepted: bool


@dataclass(frozen=True)
class OptimizationResult:
    parameters: dict[str, float]
    energy: float
    success_probability: float
    gradient: dict[str, float]
    iterations: int
    stop_reason: str
    history: tuple[IterationRecord, ...] = field(default_factory=tuple)


def minimize_energy(
    num_qubits: int,
    num_clbits: int,
    operations: Any,
    initial_parameters: Mapping[str, float],
    hamiltonian: Any,
    *,
    postselection: Any = None,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-6,
    initial_step_size: float = 0.5,
    step_shrink: float = 0.5,
    minimum_step_size: float = 1e-12,
) -> OptimizationResult:
    """Run deterministic gradient descent with backtracking line search.

    A proposed step is accepted only when it stays postselection-valid and
    strictly lowers the conditional energy. Otherwise the step size is shrunk
    and the last valid point is retained. Reaching the iteration budget or
    stalling on a too-small step is reported as such, never as convergence.
    """
    names = parameter_names(validate_circuit(num_qubits, num_clbits, operations))
    validate_hamiltonian(hamiltonian, num_qubits)
    validate_postselection(postselection, num_clbits)
    _validate_configuration(
        names,
        initial_parameters,
        max_iterations,
        gradient_tolerance,
        initial_step_size,
        step_shrink,
        minimum_step_size,
    )

    current = {name: float(initial_parameters[name]) for name in names}
    try:
        current_eval = energy_gradient(
            num_qubits, num_clbits, operations, current, hamiltonian, postselection
        )
    except PostselectionError:
        return OptimizationResult(
            parameters=current,
            energy=float("nan"),
            success_probability=0.0,
            gradient={name: float("nan") for name in names},
            iterations=0,
            stop_reason=REASON_INVALID_INITIAL_POINT,
            history=(),
        )

    step_size = float(initial_step_size)
    history: list[IterationRecord] = []
    stop_reason = REASON_MAX_ITERATIONS

    for iteration in range(1, max_iterations + 1):
        gradient_norm = _norm(current_eval.gradient)
        accepted = False
        trial_step_size = step_size
        while True:
            trial = {
                name: current[name] - trial_step_size * current_eval.gradient[name]
                for name in names
            }
            trial_eval: GradientEvaluation | None
            try:
                trial_eval = energy_gradient(
                    num_qubits, num_clbits, operations, trial, hamiltonian, postselection
                )
                if isfinite(trial_eval.energy) and trial_eval.energy < current_eval.energy:
                    accepted = True
            except PostselectionError:
                trial_eval = None

            if accepted:
                current = trial
                current_eval = trial_eval
                step_size = float(initial_step_size)
                break
            trial_step_size *= step_shrink
            if trial_step_size < minimum_step_size:
                break

        history.append(
            _record(iteration, current_eval, gradient_norm, trial_step_size, accepted=accepted)
        )
        if not accepted:
            stop_reason = REASON_STEP_UNDERFLOW
            break
        if _norm(current_eval.gradient) <= gradient_tolerance:
            stop_reason = REASON_GRADIENT_TOLERANCE
            break

    return OptimizationResult(
        parameters=dict(current),
        energy=current_eval.energy,
        success_probability=current_eval.success_probability,
        gradient=dict(current_eval.gradient),
        iterations=len(history),
        stop_reason=stop_reason,
        history=tuple(history),
    )


def _record(
    iteration: int,
    evaluation: GradientEvaluation,
    gradient_norm: float,
    step_size: float,
    accepted: bool,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        parameters=dict(evaluation.parameters),
        energy=evaluation.energy,
        success_probability=evaluation.success_probability,
        gradient_norm=gradient_norm,
        step_size=step_size,
        accepted=accepted,
    )


def _norm(values: Mapping[str, float]) -> float:
    return sqrt(sum(value * value for value in values.values()))


def _validate_configuration(
    names,
    initial_parameters,
    max_iterations,
    gradient_tolerance,
    initial_step_size,
    step_shrink,
    minimum_step_size,
) -> None:
    if not names:
        raise OptimizationError("training requires a circuit with named parameters")
    if not isinstance(initial_parameters, Mapping):
        raise OptimizationError("initial_parameters must be a mapping")
    if set(initial_parameters) != set(names):
        raise OptimizationError(
            f"initial_parameters keys must exactly match circuit parameters {list(names)}"
        )
    for name, value in initial_parameters.items():
        if not isinstance(value, (int, float, np.floating, np.integer)) or isinstance(value, bool):
            raise OptimizationError(f"initial parameter {name!r} must be a real number")
        if not isfinite(float(value)):
            raise OptimizationError(f"initial parameter {name!r} must be finite")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations <= 0:
        raise OptimizationError("max_iterations must be a positive integer")
    _require_non_negative_float("gradient_tolerance", gradient_tolerance)
    _require_positive_float("initial_step_size", initial_step_size)
    if not isinstance(step_shrink, (int, float, np.floating, np.integer)) or isinstance(step_shrink, bool):
        raise OptimizationError("step_shrink must be a number between 0 and 1 (exclusive)")
    if not 0.0 < float(step_shrink) < 1.0:
        raise OptimizationError("step_shrink must be between 0 and 1 (exclusive)")
    _require_non_negative_float("minimum_step_size", minimum_step_size)
    if minimum_step_size > initial_step_size:
        raise OptimizationError("minimum_step_size must not exceed initial_step_size")


def _require_positive_float(field_name: str, value: Any) -> None:
    if (
        not isinstance(value, (int, float, np.floating, np.integer))
        or isinstance(value, bool)
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise OptimizationError(f"{field_name} must be a positive finite number")


def _require_non_negative_float(field_name: str, value: Any) -> None:
    if (
        not isinstance(value, (int, float, np.floating, np.integer))
        or isinstance(value, bool)
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        raise OptimizationError(f"{field_name} must be a non-negative finite number")
