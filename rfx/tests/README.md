# rfx Tests

Test suite for the rfx Python package.

## Coverage

- **Unit tests** -- Robot protocol, skills, agent, decorators, JIT runtime, teleop (config, session, transport, recorder, benchmark, LeRobot writer)
- **Model hub tests** -- `test_hub.py` covers policy save/load round-trip, config preservation, policy registry, legacy safetensors fallback, `inspect_policy`, normalizer serialization, torch/tinygrad bridge
- **Integration tests** -- `integration/test_rust_bindings.py` validates PyO3 bindings for Quaternion, Transform, LowPassFilter, Pid, Go2, and transport primitives
- **Fixtures** -- `conftest.py` provides shared test fixtures and config helpers

## Running

```bash
# Full suite
pytest rfx/tests/

# Or via moon
moon run :test
```
