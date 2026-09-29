import copy
import unittest

import numpy as np

from quantum_feedback import (
    CircuitValidationError,
    HamiltonianValidationError,
    ParameterBindingError,
    PostselectionError,
    bind_parameters,
    energy_gradient,
    evaluate_energy,
    minimize_energy,
    run,
    validate_circuit,
)


OPERATIONS = [
    {"op": "H", "qubit": 0},
    {"op": "RZ", "qubit": 0, "angle": {"parameter": "theta", "offset": 0.1}},
    {"op": "CX", "control": 0, "target": 1},
    {"op": "MEASURE", "qubit": 0, "clbit": 0},
    {"op": "H", "qubit": 1, "condition": [0, 1]},
    {"op": "RZ", "qubit": 1,
     "angle": {"parameter": "theta", "scale": -1.5},
     "condition": [0, 1]},
    {"op": "H", "qubit": 1, "condition": [0, 1]},
    {"op": "PHASE_FLIP", "qubit": 1, "p": 0.1},
]
HAMILTONIAN = {"IZ": 0.7, "ZI": -0.4, "ZZ": 0.3}


class ParameterBindingTests(unittest.TestCase):
    def test_constant_and_parameterized_rz_produce_identical_states(self):
        theta = 0.63
        constant = copy.deepcopy(OPERATIONS)
        constant[1] = {"op": "RZ", "qubit": 0, "angle": theta + 0.1}
        constant[5] = {"op": "RZ", "qubit": 1,
                       "angle": -1.5 * theta, "condition": [0, 1]}
        parameterized = run(2, 1, OPERATIONS, parameters={"theta": theta})
        bound = run(2, 1, constant)
        np.testing.assert_allclose(
            parameterized.density_matrix, bound.density_matrix, atol=1e-12
        )

    def test_binding_is_complete_and_does_not_mutate_inputs(self):
        values = {"theta": 0.4}
        snapshot_ops = copy.deepcopy(OPERATIONS)
        snapshot_values = dict(values)
        bound = bind_parameters(validate_circuit(2, 1, OPERATIONS), values)
        self.assertEqual(OPERATIONS, snapshot_ops)
        self.assertEqual(values, snapshot_values)
        for operation in bound:
            self.assertIsNone(operation.parameter_name)

    def test_missing_extra_and_unbound_parameters_fail(self):
        with self.assertRaises(ParameterBindingError):
            bind_parameters(validate_circuit(2, 1, OPERATIONS), {})
        with self.assertRaises(ParameterBindingError):
            bind_parameters(validate_circuit(2, 1, OPERATIONS), {"theta": 0.0, "x": 1.0})
        with self.assertRaises(CircuitValidationError):
            run(2, 1, OPERATIONS)
        with self.assertRaises(ParameterBindingError):
            bind_parameters(
                validate_circuit(2, 1, [{"op": "H", "qubit": 0}]),
                {"x": 1.0},
            )

    def test_invalid_angle_descriptions_fail_validation(self):
        bad_angles = [
            {"parameter": ""},
            {"parameter": "a", "scale": "big"},
            {"parameter": "a", "offset": float("inf")},
            {"parameter": "a", "unexpected": 1.0},
        ]
        for angle in bad_angles:
            with self.assertRaises(CircuitValidationError):
                validate_circuit(1, 1, [{"op": "RZ", "qubit": 0, "angle": angle}])


class ObjectiveTests(unittest.TestCase):
    def test_exact_gradient_matches_parameter_shift_reference(self):
        for postselection in (None, [(0, 0)], [(0, 1)]):
            evaluation = energy_gradient(
                2, 1, OPERATIONS, {"theta": 0.7}, HAMILTONIAN,
                postselection=postselection,
            )
            reference = self._finite_difference_gradient(
                0.7, postselection, step=1e-7
            )
            self.assertAlmostEqual(
                evaluation.gradient["theta"], reference, delta=1e-7
            )

    def _finite_difference_gradient(self, theta, postselection, step):
        plus = evaluate_energy(
            2, 1, OPERATIONS, {"theta": theta + step}, HAMILTONIAN,
            postselection=postselection,
        )
        minus = evaluate_energy(
            2, 1, OPERATIONS, {"theta": theta - step}, HAMILTONIAN,
            postselection=postselection,
        )
        return (plus.energy - minus.energy) / (2.0 * step)

    def test_shared_parameter_negative_scale_and_unconditional_energy(self):
        evaluation = evaluate_energy(
            2, 1, OPERATIONS, {"theta": 0.7}, HAMILTONIAN
        )
        self.assertAlmostEqual(evaluation.success_probability, 1.0)
        conditional = evaluate_energy(
            2, 1, OPERATIONS, {"theta": 0.7}, HAMILTONIAN,
            postselection=[(0, 0)],
        )
        other = evaluate_energy(
            2, 1, OPERATIONS, {"theta": 0.7}, HAMILTONIAN,
            postselection=[(0, 1)],
        )
        mixed = 0.5 * conditional.energy + 0.5 * other.energy
        self.assertAlmostEqual(evaluation.success_probability, 1.0)
        self.assertAlmostEqual(evaluation.energy, mixed, places=10)

    def test_zero_probability_shift_does_not_break_valid_gradient(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": {"parameter": "a"}},
            {"op": "H", "qubit": 0},
            {"op": "H", "qubit": 1},
            {"op": "RZ", "qubit": 1,
             "angle": {"parameter": "a", "scale": 0.3}},
            {"op": "H", "qubit": 1},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
        ]
        hamiltonian = {"IZ": 1.0, "ZI": 0.5}
        theta = np.pi / 2.0
        evaluation = energy_gradient(
            2, 1, operations, {"a": theta}, hamiltonian,
            postselection=[(0, 1)],
        )
        self.assertGreater(evaluation.success_probability, 0.4)
        step = 1e-7
        reference = (
            evaluate_energy(
                2, 1, operations, {"a": theta + step}, hamiltonian,
                postselection=[(0, 1)],
            ).energy
            - evaluation.energy
        ) / step
        self.assertAlmostEqual(evaluation.gradient["a"], reference, delta=1e-6)

    def test_postselection_floor_is_enforced(self):
        operations = [
            {"op": "X", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
        ]
        with self.assertRaises(PostselectionError):
            evaluate_energy(
                1, 1, operations, None, {"Z": 1.0},
                postselection=[(0, 0)],
            )
        with self.assertRaises(PostselectionError):
            energy_gradient(
                1, 1,
                [{"op": "H", "qubit": 0},
                 {"op": "RZ", "qubit": 0, "angle": {"parameter": "a"}},
                 {"op": "H", "qubit": 0},
                 {"op": "MEASURE", "qubit": 0, "clbit": 0}],
                {"a": 0.0}, {"Z": 1.0}, postselection=[(0, 1)],
            )

    def test_hamiltonian_validation(self):
        for bad in ("X", {"Q": 1.0}, {"XX": 1j}, {}, [("X",)]):
            with self.assertRaises(HamiltonianValidationError):
                evaluate_energy(1, 1, [{"op": "H", "qubit": 0}], None, bad)


class OptimizerTests(unittest.TestCase):
    def test_training_reduces_energy_and_reports_convergence(self):
        result = minimize_energy(
            2, 1, OPERATIONS, {"theta": 0.2}, HAMILTONIAN,
            postselection=[(0, 1)],
            max_iterations=500,
            gradient_tolerance=1e-6,
        )
        self.assertEqual(result.stop_reason, "gradient_tolerance_reached")
        self.assertGreater(result.iterations, 0)
        initial_energy = evaluate_energy(
            2, 1, OPERATIONS, {"theta": 0.2}, HAMILTONIAN,
            postselection=[(0, 1)],
        ).energy
        self.assertLessEqual(result.energy, initial_energy)
        for earlier, later in zip(result.history, result.history[1:]):
            self.assertLessEqual(later.energy, earlier.energy + 1e-12)
        self.assertLessEqual(np.linalg.norm(list(result.gradient.values())), 1e-6)

    def test_budget_exhaustion_is_not_called_convergence(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
            {"op": "H", "qubit": 1, "condition": [0, 1]},
            {"op": "RZ", "qubit": 1, "angle": {"parameter": "a"},
             "condition": [0, 1]},
            {"op": "H", "qubit": 1, "condition": [0, 1]},
        ]
        result = minimize_energy(
            2, 1, operations, {"a": 0.3}, {"IZ": 1.0},
            postselection=[(0, 1)],
            max_iterations=3,
            gradient_tolerance=0.0,
            initial_step_size=0.1,
            step_shrink=0.5,
            minimum_step_size=1e-11,
        )
        self.assertEqual(result.stop_reason, "max_iterations_reached")
        self.assertEqual(result.iterations, 3)

    def test_failed_trial_retains_last_valid_point(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": {"parameter": "a"}},
            {"op": "H", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
        ]
        # Near the conditional minimum every allowed step raises the energy or
        # enters the forbidden postselection region; underflow must retain the
        # starting valid point, energy, and probability.
        result = minimize_energy(
            1, 1, operations, {"a": 1e-6}, {"Z": 1.0},
            postselection=[(0, 0)],
            max_iterations=3,
            gradient_tolerance=0.0,
            initial_step_size=1.0,
            step_shrink=0.5,
            minimum_step_size=0.25,
        )
        self.assertEqual(result.stop_reason, "step_size_underflow")
        self.assertEqual(result.parameters, {"a": 1e-6})
        self.assertTrue(np.isfinite(result.energy))
        self.assertGreater(result.success_probability, 1e-12)

    def test_invalid_initial_point_is_reported(self):
        operations = [
            {"op": "X", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
        ]
        template = operations + [
            {"op": "RZ", "qubit": 0, "angle": {"parameter": "a"},
             "condition": [0, 0]},
        ]
        result = minimize_energy(
            1, 1, template, {"a": 0.0}, {"Z": 1.0},
            postselection=[(0, 0)],
            max_iterations=2,
            gradient_tolerance=1e-9,
        )
        self.assertEqual(result.stop_reason, "invalid_initial_point")


if __name__ == "__main__":
    unittest.main()
