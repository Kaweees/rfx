# Operator Checklist

Use this checklist before every production deploy.

## Preconditions

- Zenoh control plane reachable for your deployment topology.
- Robot config selected and safety profile defined.
- Dataset collected with the same robot contract version.

## Required Gates

1. `collect` run exists and dataset inputs are stored.
2. `validate` report passed (`timestamp`, `alignment`, `schema`, `missing data` checks).
3. `train` produced a policy artifact manifest with compatibility metadata.
4. `eval` report passed for the same artifact.
5. Optional but recommended: `shadow` report passed.
6. `deploy` preflight allowed the release.

## Commands

```bash
rfx runs list
rfx runs show <run_id>
rfx lineage <deploy_run_id>
```

## Blockers

Deploy must be blocked when any of the following occurs:

- Artifact manifest missing or schema incompatible.
- No successful eval report for the artifact.
- Shadow is required but no passing shadow report exists.
- Robot config hash mismatch.
- Safety profile mismatch.

## Incident Readiness

- Keep the `run_id` for rollback and audits.
- Verify `.rfx/runs/`, `.rfx/reports/`, and `.rfx/artifacts/` are persisted.
- Use `rfx reproduce <run_id>` to inspect exact replay prerequisites.
