"""Shared runtime execution helpers for interpreter and lowered executors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from rfxJIT.kernels.ir import OpCode, TensorSpec


def coerce_input_array(value: np.ndarray, spec: TensorSpec) -> np.ndarray:
    """Coerce input to the declared tensor dtype/shape."""
    arr = np.asarray(value, dtype=spec.dtype.value)
    if arr.shape != spec.shape:
        raise ValueError(f"Input {spec.name!r} has shape {arr.shape}, expected {spec.shape}")
    return arr


def validate_named_input_contract(
    expected_names: set[str],
    named_inputs: Mapping[str, np.ndarray],
) -> None:
    """Validate exact input-name contract (no missing/extra names)."""
    provided_names = set(named_inputs.keys())
    missing = expected_names - provided_names
    extra = provided_names - expected_names
    if missing:
        raise ValueError(f"Missing required inputs: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected inputs provided: {sorted(extra)}")


def coerce_named_inputs(
    input_specs: tuple[TensorSpec, ...],
    named_inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate input-name contract and coerce all named inputs."""
    validate_named_input_contract({spec.name for spec in input_specs}, named_inputs)
    return {spec.name: coerce_input_array(named_inputs[spec.name], spec) for spec in input_specs}


def execute_numpy_op(
    *,
    op: OpCode,
    args: Sequence[np.ndarray],
    shape: tuple[int, ...],
    dtype: str,
    const_value: float | None = None,
) -> np.ndarray:
    """Execute a single op with NumPy semantics used across CPU paths."""
    if op == OpCode.CONST:
        if const_value is None:
            raise ValueError("CONST op requires const_value")
        return np.full(shape, float(const_value), dtype=dtype)

    if op in {OpCode.NEG, OpCode.RELU, OpCode.STEP, OpCode.EXP, OpCode.LOG}:
        if len(args) != 1:
            raise ValueError(f"{op.value} expects 1 input, got {len(args)}")
        a0 = args[0]
        if op == OpCode.NEG:
            out = -a0
        elif op == OpCode.RELU:
            out = np.maximum(a0, 0.0)
        elif op == OpCode.STEP:
            out = (a0 > 0).astype(dtype)
        elif op == OpCode.EXP:
            out = np.exp(a0)
        else:
            out = np.log(a0)
        return out.astype(dtype, copy=False)

    if op in {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV}:
        if len(args) != 2:
            raise ValueError(f"{op.value} expects 2 inputs, got {len(args)}")
        lhs, rhs = args
        if op == OpCode.ADD:
            out = lhs + rhs
        elif op == OpCode.SUB:
            out = lhs - rhs
        elif op == OpCode.MUL:
            out = lhs * rhs
        else:
            out = lhs / rhs
        return out.astype(dtype, copy=False)

    raise ValueError(f"Unsupported op: {op}")
