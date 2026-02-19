# Repository Structure

Top-level layout:

- `rfx/`: production and test source code (includes `python/rfx/hub.py` for model management and `python/rfx/nn.py` for policy definitions)
- `packages/`: extension Python packages (`rfx-sdk-sim`, `rfx-sdk-go2`, `rfx-sdk-lerobot`)
- `docs/`: architecture, workflows, and contributor guides
  includes perf baselines in `docs/perf/baselines/`
- `rfxJIT/`: ultra-performance kernel and compiler engine
- `cli/`: command-line interfaces and helpers
- `.claude/skills/`: Claude skills used by agents

Quality/tooling:

- Hook logic: `.githooks/`
- Hook installer/config: `scripts/setup-git-hooks.sh`, `.pre-commit-config.yaml`
- Moon workspace: `.moon/workspace.yml`
- Python check runner: `scripts/python-checks.sh`
- Source setup runner: `scripts/setup-from-source.sh`
- CI workflow: `.github/workflows/ci.yml`
- Workflow guide: `docs/workflow.md`
