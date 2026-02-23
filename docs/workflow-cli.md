# Workflow CLI

`rfx` now provides one lifecycle-first workflow:

`collect -> validate -> train -> eval -> shadow -> deploy`

Every stage writes an immutable run record under `.rfx/runs/` and captures:

- config snapshot + hash
- git commit + dirty state
- environment summary
- input/output reference hashes
- stage reports and artifact references

## Stage Commands

```bash
rfx collect --repo-id my-org/so101-demos --robot-type so101 --episodes 5 --duration 30 --collection-root datasets
rfx validate --dataset data/session_001.jsonl --input data/session_001.jsonl
rfx train --config train.yaml --input data/session_001.jsonl --safety-profile safe-default
rfx eval --artifact-ref artifact://policy/sha256:... --metrics-json reports/eval_metrics.json
rfx shadow --artifact-ref artifact://policy/sha256:... --shadow-json reports/shadow_metrics.json
rfx deploy --artifact-ref artifact://policy/sha256:... --require-shadow --safety-profile safe-default
```

Using `uv` directly (same command surface as `cli/rfx.sh`):

```bash
uv run --python 3.13 python -m rfx.runtime.cli collect --repo-id my-org/so101-demos --robot-type so101 --episodes 5 --duration 30
```

## Run Registry

Inspect runs and lineage:

```bash
rfx runs list --limit 20
rfx runs show <run_id>
rfx lineage <run_id>
rfx reproduce <run_id>
```

`rfx reproduce` prints the recorded command and identifies missing dependencies (inputs, upstream runs, snapshots).

## Plane Routing Policy

In hybrid transport mode:

- `data/*` stays on local shared-memory/inproc.
- `control/*` routes through Zenoh.
- `rfx/*` routes through Zenoh.

When a required control-plane route is unavailable, rfx fails fast instead of silently falling back.
