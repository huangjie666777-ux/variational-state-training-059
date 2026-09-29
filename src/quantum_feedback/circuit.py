"""Circuit validation and normalized operation descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np

from .errors import CircuitValidationError
from .errors import ParameterBindingError

GATE_OPERATIONS = frozenset({"H", "X", "Z", "RZ", "CX"})
SINGLE_QUBIT_GATES = frozenset({"H", "X", "Z", "RZ"})


@dataclass(frozen=True)
class Operation:
    name: str
    qubit: int | None = None
    control: int | None = None
    target: int | None = None
    clbit: int | None = None
    angle: float | None = None
    probability: float | None = None
    condition_clbit: int | None = None
    condition_value: int | None = None
    parameter_name: str | None = None
    scale: float = 1.0
    offset: float = 0.0


def validate_circuit(
    num_qubits: Any,
    num_clbits: Any,
    operations: Any,
    parameters: Mapping[str, float] | None = None,
) -> list[Operation]:
    """Validate a circuit without retaining any caller-owned mutable object."""
    if not _is_plain_int(num_qubits) or not 1 <= num_qubits <= 6:
        raise CircuitValidationError("num_qubits must be an integer from 1 to 6")
    if not _is_plain_int(num_clbits) or not 1 <= num_clbits <= 6:
        raise CircuitValidationError("num_clbits must be an integer from 1 to 6")
    if not isinstance(operations, (list, tuple)):
        raise CircuitValidationError("operations must be a list or tuple")

    normalized: list[Operation] = []
    for position, raw_operation in enumerate(operations):
        normalized.append(_validate_operation(raw_operation, position, num_qubits, num_clbits))
    if parameters is not None:
        normalized = bind_parameters(normalized, parameters)
    return normalized


def parameter_names(operations: list[Operation]) -> tuple[str, ...]:
    """Return the tuple of distinct parameter names, in first-use order."""
    names: list[str] = []
    for operation in operations:
        name = getattr(operation, "parameter_name", None)
        if name is not None and name not in names:
            names.append(name)
    return tuple(names)


def bind_parameters(operations: Any, values: Any) -> list[Operation]:
    """Validate a complete value mapping and return a bound, caller-independent copy.

    The mapping must provide exactly one finite real value per referenced parameter
    name. Neither the template operations nor the caller mapping are mutated.
    """
    if not isinstance(values, Mapping):
        raise ParameterBindingError("parameter values must be a mapping of name to number")
    if not isinstance(operations, (list, tuple)) or not all(
        isinstance(operation, Operation) for operation in operations
    ):
        raise ParameterBindingError("operations must be validated Operation objects")

    required = list(dict.fromkeys(
        operation.parameter_name
        for operation in operations
        if operation.parameter_name is not None
    ))
    missing = [name for name in required if name not in values]
    extra = [key for key in values if key not in required]
    if missing:
        raise ParameterBindingError(f"missing parameter value(s): {missing}")
    if extra:
        raise ParameterBindingError(f"unexpected parameter value(s): {sorted(extra)}")

    bound_values: dict[str, float] = {}
    for name in required:
        value = values[name]
        if not isinstance(value, (int, float, np.floating, np.integer)) or isinstance(value, bool):
            raise ParameterBindingError(f"parameter {name!r} must be a real number")
        value = float(value)
        if not isfinite(value):
            raise ParameterBindingError(f"parameter {name!r} must be finite")
        bound_values[name] = value

    bound: list[Operation] = []
    for operation in operations:
        if operation.parameter_name is None:
            bound.append(operation)
        else:
            angle = operation.scale * bound_values[operation.parameter_name] + operation.offset
            bound.append(
                Operation(
                    operation.name,
                    qubit=operation.qubit,
                    control=operation.control,
                    target=operation.target,
                    clbit=operation.clbit,
                    angle=angle,
                    probability=operation.probability,
                    condition_clbit=operation.condition_clbit,
                    condition_value=operation.condition_value,
                )
            )
    return bound


def _validate_operation(
    raw_operation: Any, position: int, num_qubits: int, num_clbits: int
) -> Operation:
    where = f"operation {position}"
    if not isinstance(raw_operation, Mapping):
        raise CircuitValidationError(f"{where}: operation must be a mapping")

    operation = dict(raw_operation)
    name = operation.get("op")
    if not isinstance(name, str):
        raise CircuitValidationError(f"{where}: missing or non-string 'op'")

    condition = _validate_condition(operation.get("condition"), position, num_clbits)
    allowed = GATE_OPERATIONS | {"MEASURE", "RESET", "PHASE_FLIP"}
    if name not in allowed:
        raise CircuitValidationError(f"{where}: unknown operation {name!r}")
    if condition is not None and name not in GATE_OPERATIONS:
        raise CircuitValidationError(f"{where}: only gates may be conditional")

    if name in SINGLE_QUBIT_GATES:
        qubit = _required_qubit(operation, "qubit", position, num_qubits)
        angle = None
        parameter_name = None
        scale = 1.0
        offset = 0.0
        if name == "RZ":
            angle, parameter_name, scale, offset = _required_angle(operation, position)
        _reject_extra_fields(operation, position, {"op", "qubit", "angle", "condition"})
        return Operation(name, qubit=qubit, angle=angle,
                         condition_clbit=condition[0] if condition else None,
                         condition_value=condition[1] if condition else None,
                         parameter_name=parameter_name, scale=scale, offset=offset)

    if name == "CX":
        control = _required_qubit(operation, "control", position, num_qubits)
        target = _required_qubit(operation, "target", position, num_qubits)
        if control == target:
            raise CircuitValidationError(f"{where}: CX control and target must be different")
        _reject_extra_fields(operation, position, {"op", "control", "target", "condition"})
        return Operation(name, control=control, target=target,
                         condition_clbit=condition[0] if condition else None,
                         condition_value=condition[1] if condition else None)

    if name == "MEASURE":
        qubit = _required_qubit(operation, "qubit", position, num_qubits)
        clbit = _required_clbit(operation, position, num_clbits)
        _reject_extra_fields(operation, position, {"op", "qubit", "clbit"})
        return Operation(name, qubit=qubit, clbit=clbit)

    if name == "RESET":
        qubit = _required_qubit(operation, "qubit", position, num_qubits)
        _reject_extra_fields(operation, position, {"op", "qubit"})
        return Operation(name, qubit=qubit)

    qubit = _required_qubit(operation, "qubit", position, num_qubits)
    probability = _required_probability(operation, position)
    _reject_extra_fields(operation, position, {"op", "qubit", "p"})
    return Operation(name, qubit=qubit, probability=probability)


def _validate_condition(condition: Any, position: int, num_clbits: int) -> tuple[int, int] | None:
    if condition is None:
        return None
    if not isinstance(condition, (list, tuple)) or len(condition) != 2:
        raise CircuitValidationError(f"operation {position}: condition must be [clbit, value]")
    clbit, value = condition
    if not _is_plain_int(clbit) or not 0 <= clbit < num_clbits:
        raise CircuitValidationError(f"operation {position}: condition classical bit is out of range")
    if not _is_plain_int(value) or value not in (0, 1):
        raise CircuitValidationError(f"operation {position}: condition value must be 0 or 1")
    return clbit, value


def _required_qubit(operation: Mapping, field: str, position: int, num_qubits: int) -> int:
    value = operation.get(field)
    if not _is_plain_int(value) or not 0 <= value < num_qubits:
        raise CircuitValidationError(f"operation {position}: {field} is out of range")
    return value


def _required_clbit(operation: Mapping, position: int, num_clbits: int) -> int:
    value = operation.get("clbit")
    if not _is_plain_int(value) or not 0 <= value < num_clbits:
        raise CircuitValidationError(f"operation {position}: clbit is out of range")
    return value


def _required_angle(
    operation: Mapping, position: int
) -> tuple[float | None, str | None, float, float]:
    angle = operation.get("angle")
    if isinstance(angle, Mapping):
        where = f"operation {position}"
        name = angle.get("parameter")
        if not isinstance(name, str) or not name:
            raise CircuitValidationError(f"{where}: RZ parameter name must be a non-empty string")
        scale = angle.get("scale", 1.0)
        offset = angle.get("offset", 0.0)
        for field, value in (("scale", scale), ("offset", offset)):
            if not isinstance(value, (int, float, np.floating, np.integer)) or isinstance(value, bool):
                raise CircuitValidationError(f"{where}: RZ angle {field} must be a real number")
            if not isfinite(float(value)):
                raise CircuitValidationError(f"{where}: RZ angle {field} must be finite")
        extra = set(angle) - {"parameter", "scale", "offset"}
        if extra:
            raise CircuitValidationError(f"{where}: unexpected RZ angle field(s): {sorted(extra)}")
        return None, name, float(scale), float(offset)
    if not isinstance(angle, (int, float, np.floating, np.integer)) or isinstance(angle, bool):
        raise CircuitValidationError(f"operation {position}: RZ angle must be numeric")
    angle = float(angle)
    if not isfinite(angle):
        raise CircuitValidationError(f"operation {position}: RZ angle must be finite")
    return angle, None, 1.0, 0.0


def _required_probability(operation: Mapping, position: int) -> float:
    probability = operation.get("p")
    if not isinstance(probability, (int, float, np.floating, np.integer)) or isinstance(probability, bool):
        raise CircuitValidationError(f"operation {position}: phase-flip probability must be numeric")
    probability = float(probability)
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise CircuitValidationError(f"operation {position}: phase-flip probability must be between 0 and 1")
    return probability


def _reject_extra_fields(operation: Mapping, position: int, allowed: set[str]) -> None:
    extra = set(operation) - allowed
    if extra:
        raise CircuitValidationError(f"operation {position}: unexpected field(s): {sorted(extra)}")


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
