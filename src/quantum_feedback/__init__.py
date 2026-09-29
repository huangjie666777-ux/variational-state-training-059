"""Small embeddable density-matrix quantum-circuit SDK."""

from .circuit import Operation, bind_parameters, validate_circuit
from .engine import run
from .errors import (
    CircuitValidationError,
    HamiltonianValidationError,
    OptimizationError,
    ParameterBindingError,
    PostselectionError,
    SamplingError,
)
from .objective import (
    GradientEvaluation,
    ObjectiveEvaluation,
    energy_gradient,
    evaluate_energy,
)
from .results import ExecutionResult
from .training import (
    IterationRecord,
    OptimizationResult,
    minimize_energy,
)

__all__ = [
    "CircuitValidationError",
    "ExecutionResult",
    "GradientEvaluation",
    "HamiltonianValidationError",
    "IterationRecord",
    "ObjectiveEvaluation",
    "Operation",
    "OptimizationError",
    "OptimizationResult",
    "ParameterBindingError",
    "PostselectionError",
    "SamplingError",
    "bind_parameters",
    "energy_gradient",
    "evaluate_energy",
    "minimize_energy",
    "run",
    "validate_circuit",
]
