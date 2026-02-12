# rfxJIT

Workstream for JIT-related systems in rfx:

- kernel lowering ideas
- runtime execution plans
- performance experiments

## Current Focus

- define a small, explicit IR for kernels
- prototype scheduling + fusion passes
- benchmark generated kernels against baseline paths

## Layout

- `notes/`: architecture notes and design records
- `runtime/`: runtime execution plumbing prototypes
- `kernels/`: kernel generation and lowering experiments
- `ROADMAP.md`: milestone plan

## Phase 0 Status

- typed elementwise kernel IR: `rfxJIT/kernels/ir.py`
- reference interpreter: `rfxJIT/runtime/interpreter.py`
- benchmark harness: `rfxJIT/runtime/benchmark.py`
- tests: `rfxJIT/tests/test_ir.py`

## Phase 1 Status

- IR lowering to slot-based executable form: `rfxJIT/kernels/lowering.py`
- lowered-kernel executor: `rfxJIT/runtime/executor.py`
- single-worker dispatch queue: `rfxJIT/runtime/queue.py`
- lowering/queue tests: `rfxJIT/tests/test_lowering_queue.py`

Run the benchmark:

```bash
python -m rfxJIT.runtime.benchmark --size 65536 --iterations 200
```
