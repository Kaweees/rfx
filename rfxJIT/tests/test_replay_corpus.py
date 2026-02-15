"""Deterministic replay corpus tests for interpreter/lowered CPU equivalence."""

from __future__ import annotations

import numpy as np
import pytest

from rfxJIT.kernels.ir import DType, KernelIR, KernelOp, OpCode, TensorSpec
from rfxJIT.kernels.lowering import lower_kernel_ir
from rfxJIT.runtime.executor import execute_lowered_kernel
from rfxJIT.runtime.interpreter import execute_kernel

_UNARY_OPS = (OpCode.NEG, OpCode.RELU, OpCode.STEP, OpCode.EXP, OpCode.LOG)
_BINARY_OPS = (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV)


def _pick_op(rng: np.random.Generator, ops: tuple[OpCode, ...]) -> OpCode:
    return ops[int(rng.integers(0, len(ops)))]


def _kernel_signature(kernel: KernelIR) -> tuple[object, ...]:
    return (
        kernel.name,
        kernel.shape,
        tuple((spec.name, spec.shape, spec.dtype.value) for spec in kernel.inputs),
        (kernel.output.name, kernel.output.shape, kernel.output.dtype.value),
        tuple((op.op.value, op.out, op.inputs, op.const_value) for op in kernel.ops),
    )


def _random_kernel(seed: int) -> KernelIR:
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(1, 65)),)
    dtype = DType.F64 if bool(rng.integers(0, 2)) else DType.F32
    num_inputs = int(rng.integers(1, 4))
    inputs = tuple(TensorSpec(f"in{i}", shape, dtype=dtype) for i in range(num_inputs))

    available = [spec.name for spec in inputs]
    ops: list[KernelOp] = []
    num_ops = int(rng.integers(1, 24))

    for idx in range(num_ops - 1):
        out = f"t{idx}"
        use_const = bool(rng.integers(0, 5) == 0)
        if use_const:
            value = float(rng.normal())
            if bool(rng.integers(0, 7) == 0):
                value = 0.0
            ops.append(KernelOp(op=OpCode.CONST, out=out, const_value=value))
            available.append(out)
            continue

        if len(available) == 1 or bool(rng.integers(0, 3) == 0):
            op = _pick_op(rng, _UNARY_OPS)
            src = available[int(rng.integers(0, len(available)))]
            ops.append(KernelOp(op=op, out=out, inputs=(src,)))
        else:
            op = _pick_op(rng, _BINARY_OPS)
            lhs = available[int(rng.integers(0, len(available)))]
            rhs = available[int(rng.integers(0, len(available)))]
            ops.append(KernelOp(op=op, out=out, inputs=(lhs, rhs)))
        available.append(out)

    if bool(rng.integers(0, 6) == 0):
        ops.append(KernelOp(op=OpCode.CONST, out="y", const_value=float(rng.normal())))
    elif len(available) == 1 or bool(rng.integers(0, 3) == 0):
        op = _pick_op(rng, _UNARY_OPS)
        src = available[int(rng.integers(0, len(available)))]
        ops.append(KernelOp(op=op, out="y", inputs=(src,)))
    else:
        op = _pick_op(rng, _BINARY_OPS)
        lhs = available[int(rng.integers(0, len(available)))]
        rhs = available[int(rng.integers(0, len(available)))]
        ops.append(KernelOp(op=op, out="y", inputs=(lhs, rhs)))

    kernel = KernelIR(
        name=f"replay_seed_{seed}",
        shape=shape,
        inputs=inputs,
        output=TensorSpec("y", shape, dtype=dtype),
        ops=ops,
    )
    kernel.validate()
    return kernel


def _sample_inputs(kernel: KernelIR, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 10_000)
    inputs: dict[str, np.ndarray] = {}
    for spec in kernel.inputs:
        arr = rng.standard_normal(size=spec.shape).astype(spec.dtype.value)
        inputs[spec.name] = arr
    return inputs


@pytest.mark.parametrize("seed", list(range(64)))
def test_seeded_replay_matches_lowered_cpu(seed: int) -> None:
    kernel = _random_kernel(seed)
    lowered = lower_kernel_ir(kernel)
    inputs = _sample_inputs(kernel, seed)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        expected = execute_kernel(kernel, inputs)
        result = execute_lowered_kernel(lowered, inputs, backend="cpu")

    atol = 1e-12 if kernel.output.dtype == DType.F64 else 1e-6
    np.testing.assert_allclose(result, expected, atol=atol, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("seed", [0, 1, 5, 11, 23, 42, 63])
def test_seed_replays_identical_kernel(seed: int) -> None:
    first = _random_kernel(seed)
    second = _random_kernel(seed)
    assert _kernel_signature(first) == _kernel_signature(second)


def test_input_contract_errors_match_between_interpreter_and_lowered() -> None:
    kernel = _random_kernel(1234)
    lowered = lower_kernel_ir(kernel)
    inputs = _sample_inputs(kernel, 1234)
    missing = {k: v for k, v in inputs.items() if k != kernel.inputs[-1].name}
    extra = dict(inputs)
    extra["__extra"] = next(iter(inputs.values()))

    with pytest.raises(ValueError) as interp_missing:
        execute_kernel(kernel, missing)
    with pytest.raises(ValueError) as lowered_missing:
        execute_lowered_kernel(lowered, missing, backend="cpu")

    with pytest.raises(ValueError) as interp_extra:
        execute_kernel(kernel, extra)
    with pytest.raises(ValueError) as lowered_extra:
        execute_lowered_kernel(lowered, extra, backend="cpu")

    assert "Missing required inputs" in str(interp_missing.value)
    assert str(interp_missing.value) == str(lowered_missing.value)
    assert "Unexpected inputs provided" in str(interp_extra.value)
    assert str(interp_extra.value) == str(lowered_extra.value)
