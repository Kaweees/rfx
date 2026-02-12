"""Lowering pass from high-level IR into execution-friendly instructions."""

from __future__ import annotations

from dataclasses import dataclass

from rfxJIT.kernels.ir import DType, KernelIR, KernelOp, OpCode, TensorSpec


@dataclass(frozen=True)
class LoweredOp:
    """Lowered instruction using slot indices instead of SSA names."""

    op: OpCode
    out_slot: int
    input_slots: tuple[int, ...] = ()
    const_value: float | None = None


@dataclass(frozen=True)
class LoweredKernel:
    """Executable representation of a kernel for the phase 1 runtime."""

    name: str
    shape: tuple[int, ...]
    dtype: DType
    input_specs: tuple[TensorSpec, ...]
    output_slot: int
    value_names: tuple[str, ...]
    ops: tuple[LoweredOp, ...]

    @property
    def output_name(self) -> str:
        return self.value_names[self.output_slot]

    def input_name_to_slot(self) -> dict[str, int]:
        return {spec.name: idx for idx, spec in enumerate(self.input_specs)}


def _lower_op(op: KernelOp, name_to_slot: dict[str, int]) -> LoweredOp:
    return LoweredOp(
        op=op.op,
        out_slot=name_to_slot[op.out],
        input_slots=tuple(name_to_slot[name] for name in op.inputs),
        const_value=op.const_value,
    )


def lower_kernel_ir(kernel: KernelIR) -> LoweredKernel:
    """Lower a validated kernel into slot-based executable form."""

    kernel.validate()

    value_names = [spec.name for spec in kernel.inputs]
    value_names.extend(op.out for op in kernel.ops)

    name_to_slot = {name: slot for slot, name in enumerate(value_names)}
    lowered_ops = tuple(_lower_op(op, name_to_slot) for op in kernel.ops)

    return LoweredKernel(
        name=kernel.name,
        shape=kernel.shape,
        dtype=kernel.output.dtype,
        input_specs=kernel.inputs,
        output_slot=name_to_slot[kernel.output.name],
        value_names=tuple(value_names),
        ops=lowered_ops,
    )
