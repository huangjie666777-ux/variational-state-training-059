"""Train a shared-parameter circuit with measurement feedback and postselection."""

from quantum_feedback import evaluate_energy, energy_gradient, minimize_energy

# Two qubits and one classical bit. The same parameter theta appears in two RZ
# gates; the feedback branch uses a negative scale plus a constant offset.
operations = [
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

# Real Pauli coefficients; strings have one letter per qubit, rightmost is qubit 0.
hamiltonian = {"IZ": 0.7, "ZI": -0.4, "ZZ": 0.3}
postselection = [(0, 1)]

initial = {"theta": 0.2}
before = evaluate_energy(
    2, 1, operations, initial, hamiltonian, postselection=postselection
)
gradient = energy_gradient(
    2, 1, operations, initial, hamiltonian, postselection=postselection
)
print("initial energy:", before.energy)
print("initial success probability:", before.success_probability)
print("exact gradient:", gradient.gradient)

result = minimize_energy(
    2,
    1,
    operations,
    initial,
    hamiltonian,
    postselection=postselection,
    max_iterations=500,
    gradient_tolerance=1e-8,
)

print("final parameters:", result.parameters)
print("final energy:", result.energy)
print("final success probability:", result.success_probability)
print("iterations:", result.iterations)
print("stop reason:", result.stop_reason)
print("first history entry:", result.history[0])
print("last history entry:", result.history[-1])
