# Quantum Feedback SDK

一个可嵌入的小型量子电路模拟 SDK，使用密度矩阵确定性处理测量、经典反馈、reset 和相位翻转噪声，不使用随机轨迹近似。

## 安装与接口

开发环境使用：

```bash
.venv/bin/python -m pip wheel --no-build-isolation --no-deps . -w dist
```

主入口是 `run(num_qubits, num_clbits, operations)`，其中量子位数和经典位数都必须是 1 到 6 的整数，所有量子位和经典位初始为 0。`operations` 为按时间顺序执行的映射列表：

- `{"op": "H" | "X" | "Z", "qubit": q}`：单量子位门。
- `{"op": "RZ", "qubit": q, "angle": theta}`：角度单位为弧度。
- `{"op": "CX", "control": c, "target": t}`：控制位与目标位不能相同。
- `{"op": "MEASURE", "qubit": q, "clbit": c}`：计算基测量并覆写经典位。
- `{"op": "RESET", "qubit": q}`：将该量子位非破坏性地重置为 0，保留其他量子位的约化状态，不修改经典位。
- `{"op": "PHASE_FLIP", "qubit": q, "p": p}`：确定性混合信道 `(1-p)ρ + p ZρZ`，`0 <= p <= 1`。
- 任意门可加 `"condition": [c, v]`：只有经典位 `c` 等于 `v`（0 或 1）时执行。

返回的 `ExecutionResult` 提供：

- `density_matrix`：末态量子密度矩阵副本。
- `classical_probabilities`：经典位串到概率的映射。
- `samples(shots, seed)`：使用局部 `numpy.random.Generator` 采样，不污染全局 NumPy 随机状态。
- `sample_counts(shots, seed)`：返回位串计数。

## 位序

量子位 0 对应计算基整数索引的最低位。例如 2 量子位 Bell 态是 `|00> + |11>`，其基态索引为 0 和 3。经典输出位串高位在左：经典位 0 的值显示在最右侧，未测量经典位保持 0，因此 3 个经典位中只有 `c0=1` 时输出 `001`。

测量后，模拟器保留带概率的归一化条件密度矩阵分支。后续条件门只作用于匹配的经典分支；不同读数分支不发生相干干涉。只有经典记录完全相同的分支才按概率混合。经典位被覆写后，相同新读数的分支也会按概率混合。

## 校验

越界量子位或经典位、相同的 CX 控制位和目标位、未知操作、非有限 RZ 角度、非法概率和非法反馈条件都会抛出 `CircuitValidationError`。错误消息包含从 0 开始的操作位置，例如 `operation 3`。校验在分配量子态前完成；SDK 不会返回部分执行结果，也不会修改调用方传入的数据。

## 数值容差

所有矩阵使用 NumPy `complex128`。内部比较用于剪除严格为 0 的概率分支；测试默认使用 `1e-10` 的绝对和相对容差。最多 6 个量子位时密度矩阵为 64×64，确定性演化开销可控。

## 测试与示例

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python examples/feedback_noise.py
```

示例包含 Bell 纠缠、相位翻转噪声、测量以及基于测量结果的条件 X 门。

## 变分参数与能量训练

`RZ` 的角度除数值常量外，还可以写成命名参数的“缩放 + 偏移”：

```python
{"op": "RZ", "qubit": q,
 "angle": {"parameter": "theta", "scale": -1.5, "offset": 0.1}}
```

实际角度为 `scale * theta + offset`；`scale` 默认为 `1.0`，`offset` 默认为 `0.0`，允许负缩放。同一个参数名可以出现在任意多个门中。参数化电路调用 `run` 时必须提供 `parameters` 映射；映射必须恰好包含模板引用的全部参数名，所有值必须是有限实数。绑定先完整校验，再生成独立的已绑定操作副本，不修改模板和调用方映射。也可以直接使用 `bind_parameters(operations, values)`。

### 能量目标与后选

`evaluate_energy(num_qubits, num_clbits, operations, parameters, hamiltonian, postselection=None)` 计算密度矩阵的确定性期望，不进行抽样。`hamiltonian` 是实系数 Pauli 串之和，可写作 `{"IZ": 0.7, "ZI": -0.4}` 或 `[("IZ", 0.7), ...]`：每个串长度等于量子位数，只允许 `I/X/Y/Z`，右端字符对应量子位 0。

`postselection` 是对末尾经典寄存器的部分赋值，可写作 `[(clbit, value), ...]` 或 `{clbit: value}`；未指定时使用全部分支。返回 `ObjectiveEvaluation(parameters, energy, success_probability, postselected)`，其中能量是成功分支上的条件期望，`success_probability` 是后选成功概率。成功概率不超过 `1e-12` 时抛出 `PostselectionError`，不会返回没有数学定义的伪造能量。

### 精确梯度

`energy_gradient(...)` 返回 `GradientEvaluation(parameters, energy, success_probability, gradient)`。每个 RZ 门使用精确的两项参数移位规则（移位 ±π/2）；共享参数的各门贡献相加，缩放（包括负数）直接乘入。条件门只在其活跃的经典分支上产生贡献。后选能量的分子和成功概率都按同一规则求导，并通过

```
d(N/p) = dN/p - N dp/p^2
```

保持归一化求导的正确性。移位后若某个电路的后选概率恰好为零，该移位电路的分子和概率都按零处理，不会使基点有效的导数失败。该结果是解析精确规则，不使用有限差分或抽样估计。

### 确定性局部最小化

```python
result = minimize_energy(
    num_qubits, num_clbits, operations, initial_parameters, hamiltonian,
    postselection=None,
    max_iterations=100,
    gradient_tolerance=1e-6,
    initial_step_size=0.5,
    step_shrink=0.5,
    minimum_step_size=1e-12,
)
```

优化器从给定初值执行确定性梯度下降。每次试探只在后选仍有效且能量严格下降时接受；试探无效（包括后选失败）或能量上升时按 `step_shrink` 缩小步长并重试。失败时保留最后一个有效点、能量和成功概率。成功接受后步长恢复为 `initial_step_size`，以便后续迭代继续使用完整步长。

`OptimizationResult` 提供 `parameters`、`energy`、`success_probability`、`gradient`、`iterations`、`history` 和 `stop_reason`。每条 `IterationRecord` 记录该轮参数、能量、成功概率、梯度范数、试探步长以及是否接受。`stop_reason` 只可能是：

- `gradient_tolerance_reached`：接受一次更新后梯度范数不超过容差，这是唯一的收敛原因。
- `max_iterations_reached`：迭代预算耗尽；这不是收敛声明。
- `step_size_underflow`：步长已缩小到 `minimum_step_size` 以下仍无法下降，优化停滞；这不是收敛声明。
- `invalid_initial_point`：初始点后选失败，未执行任何迭代。

该方法只承诺找到给定初值附近的局部结果，不承诺全局最优。初值参数名必须与模板完全一致；非法配置抛出 `OptimizationError`。

### 变分示例

```bash
PYTHONPATH=src .venv/bin/python examples/variational_training.py
```

示例包含两个共享 `theta` 的 RZ（其中一个带负缩放和反馈条件）、测量后选以及相位翻转噪声，并打印初始能量、精确梯度、训练历史和真实停止原因。
