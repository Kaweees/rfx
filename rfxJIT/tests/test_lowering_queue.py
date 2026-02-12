"""Tests for phase 1 lowering and dispatch runtime."""

from __future__ import annotations

import numpy as np
import pytest

from rfxJIT.kernels.ir import make_affine_relu_kernel
from rfxJIT.kernels.lowering import lower_kernel_ir
from rfxJIT.runtime.executor import execute_lowered_kernel
from rfxJIT.runtime.interpreter import execute_kernel
from rfxJIT.runtime.queue import KernelDispatchQueue


def _sample_inputs(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    return {
        "x": rng.standard_normal(size=shape, dtype=np.float32),
        "scale": rng.standard_normal(size=shape, dtype=np.float32),
        "bias": rng.standard_normal(size=shape, dtype=np.float32),
    }


def test_lowering_matches_reference_interpreter() -> None:
    kernel = make_affine_relu_kernel(shape=(32,))
    lowered = lower_kernel_ir(kernel)
    inputs = _sample_inputs((32,))

    expected = execute_kernel(kernel, inputs)
    result = execute_lowered_kernel(lowered, inputs)

    assert np.allclose(result, expected, atol=1e-6)
    assert lowered.output_name == "y"


def test_dispatch_queue_executes_kernel() -> None:
    kernel = make_affine_relu_kernel(shape=(64,))
    lowered = lower_kernel_ir(kernel)
    inputs = _sample_inputs((64,))
    expected = execute_lowered_kernel(lowered, inputs)

    with KernelDispatchQueue() as dispatch:
        future = dispatch.submit(lowered, inputs)
        result = future.result(timeout=2.0)

    assert np.allclose(result, expected, atol=1e-6)


def test_dispatch_queue_propagates_execution_errors() -> None:
    kernel = make_affine_relu_kernel(shape=(8,))
    lowered = lower_kernel_ir(kernel)
    bad_inputs = {"x": np.zeros(8, dtype=np.float32)}

    with KernelDispatchQueue() as dispatch:
        future = dispatch.submit(lowered, bad_inputs)
        with pytest.raises(ValueError, match="Missing required inputs"):
            future.result(timeout=2.0)


def test_dispatch_queue_rejects_submit_after_stop() -> None:
    kernel = make_affine_relu_kernel(shape=(4,))
    lowered = lower_kernel_ir(kernel)
    inputs = _sample_inputs((4,))

    dispatch = KernelDispatchQueue()
    dispatch.stop()

    with pytest.raises(RuntimeError, match="closed"):
        dispatch.submit(lowered, inputs)
