"""Exception types raised by the quantum feedback SDK."""


class CircuitValidationError(ValueError):
    """Raised when a circuit or an operation is invalid."""


class SamplingError(ValueError):
    """Raised when sampling arguments are invalid."""


class ParameterBindingError(ValueError):
    """Raised when named parameter values are missing, extra, or invalid."""


class HamiltonianValidationError(ValueError):
    """Raised when an energy objective is not a valid weighted Pauli sum."""


class PostselectionError(ValueError):
    """Raised when postselection succeeds with probability at or below 1e-12."""


class OptimizationError(ValueError):
    """Raised when the deterministic minimizer is configured or started incorrectly."""
