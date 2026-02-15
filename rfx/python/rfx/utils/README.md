# rfx.utils

Utility modules for observation and action processing.

## Modules

- **`padding.py`** -- `pad_state`, `pad_action`, `unpad_action`, PaddingConfig. All tensors are padded to `max_*_dim=64` for consistent multi-embodiment training across different robot morphologies.
- **`transforms.py`** -- ObservationNormalizer (Welford online statistics), ActionChunker (temporal ensembling with exponential, average, and first modes)
