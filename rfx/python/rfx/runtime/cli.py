from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rfx.workflow.registry import (
    build_lineage,
    build_reproduce_context,
    create_run_record,
    generate_run_id,
    list_runs,
    load_run,
    materialize_refs,
    snapshot_config,
)
from rfx.workflow.stages import execute_stage

from .dora_bridge import DoraCliError, build_dataflow, run_dataflow
from .packages import discover_packages
from .registry import load_registry
from .runner import launch, run_node

TEMPLATE_MANIFEST = """\
[package]
name = "{name}"
version = "0.2.0"
python_module = "src"

[nodes]
{name}_node = "{module}:{symbol}"
"""

TEMPLATE_NODE = """\
from __future__ import annotations

from rfx.runtime.node import Node, NodeContext


class {symbol}(Node):
    publish_topics = ("{name}/state",)
    subscribe_topics = ()

    def __init__(self, context: NodeContext):
        super().__init__(context)
        self.counter = 0

    def tick(self) -> bool:
        self.counter += 1
        self.publish("{name}/state", {{"counter": self.counter, "backend": self.ctx.backend}})
        return True
"""

TEMPLATE_LAUNCH = """\
name: demo
backend: mock
profile: default
profiles:
  default:
    RFX_BACKEND: mock
nodes:
  - package: {name}
    node: {name}_node
    name: {name}.main
    rate_hz: 20
    max_steps: 200
    params: {{}}
"""


def cmd_pkg_create(args: argparse.Namespace) -> int:
    root = Path.cwd() / "packages" / args.name
    src_mod = root / "src" / args.name.replace("-", "_")
    src_mod.mkdir(parents=True, exist_ok=True)

    module = f"{args.name.replace('-', '_')}.nodes"
    symbol = "MainNode"
    (root / "rfx_pkg.toml").write_text(
        TEMPLATE_MANIFEST.format(name=args.name, module=module, symbol=symbol)
    )
    (root / "src" / args.name.replace("-", "_") / "__init__.py").write_text("")
    (root / "src" / args.name.replace("-", "_") / "nodes.py").write_text(
        TEMPLATE_NODE.format(name=args.name, symbol=symbol)
    )
    (root / "launch.yaml").write_text(TEMPLATE_LAUNCH.format(name=args.name))
    print(f"[rfx] created package: {root}")
    print(f"[rfx] run with: rfx run {args.name} {args.name}_node")
    print(f"[rfx] launch with: rfx launch packages/{args.name}/launch.yaml")
    return 0


def cmd_pkg_list(_args: argparse.Namespace) -> int:
    pkgs = discover_packages()
    if not pkgs:
        print("No rfx packages found under ./packages")
        return 0
    for name, pkg in sorted(pkgs.items()):
        print(f"{name}\t{pkg.version}\t{pkg.root}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    params = json.loads(args.params_json or "{}")
    steps = run_node(
        package=args.package,
        node=args.node,
        name=args.name,
        backend=args.backend,
        params=params,
        rate_hz=args.rate_hz,
        max_steps=args.max_steps,
    )
    print(f"[rfx] node completed steps={steps}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    return launch(args.file)


def cmd_graph(_args: argparse.Namespace) -> int:
    reg = load_registry()
    print(f"launch: {reg.get('launch')}")
    for n in reg.get("nodes", []):
        print(
            f"- {n['name']} (pkg={n['package']} node={n['node']} pid={n['pid']} backend={n.get('backend', '')})"
        )
    return 0


def cmd_topic_list(_args: argparse.Namespace) -> int:
    reg = load_registry()
    pubs = set(reg.get("topics", {}).get("publish", []))
    subs = set(reg.get("topics", {}).get("subscribe", []))
    topics = sorted(pubs | subs)
    if not topics:
        print("No topics registered in runtime graph yet.")
        return 0
    for t in topics:
        print(t)
    return 0


def cmd_dora_build(args: argparse.Namespace) -> int:
    try:
        build_dataflow(args.file, uv=not args.no_uv)
    except DoraCliError as exc:
        print(f"[rfx] {exc}")
        return 1
    print(f"[rfx] dora build succeeded: {args.file}")
    return 0


def cmd_dora_run(args: argparse.Namespace) -> int:
    env = {}
    for item in args.env:
        if "=" not in item:
            print(f"[rfx] invalid --env '{item}', expected KEY=VALUE")
            return 1
        key, value = item.split("=", 1)
        env[key] = value

    try:
        run_dataflow(args.file, uv=not args.no_uv, env=env)
    except DoraCliError as exc:
        print(f"[rfx] {exc}")
        return 1
    print(f"[rfx] dora run finished: {args.file}")
    return 0


def _parse_meta(items: list[str]) -> tuple[dict[str, Any], list[str]]:
    parsed: dict[str, Any] = {}
    errors: list[str] = []
    for item in items:
        if "=" not in item:
            errors.append(item)
            continue
        key, raw = item.split("=", 1)
        if not key:
            errors.append(item)
            continue
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return parsed, errors


def _invocation_argv() -> list[str]:
    if sys.argv:
        return list(sys.argv)
    return ["rfx"]


def cmd_stage(args: argparse.Namespace) -> int:
    metadata, meta_errors = _parse_meta(list(args.meta))
    if meta_errors:
        print(f"[rfx] invalid --meta entries (expected KEY=VALUE): {', '.join(meta_errors)}")
        return 2

    stage = str(args.stage)
    if stage == "collect":
        if args.repo_id:
            metadata["repo_id"] = args.repo_id
        metadata["episodes"] = int(args.episodes)
        if args.duration is not None:
            metadata["duration"] = float(args.duration)
        metadata["task"] = str(args.task)
        metadata["fps"] = int(args.fps)
        metadata["push_to_hub"] = bool(args.push)
        metadata["mcap"] = bool(args.mcap)
        metadata["state_dim"] = int(args.state_dim)
        metadata["output"] = str(args.collection_root)
    if args.dataset:
        metadata["dataset"] = args.dataset
    if args.metrics_json:
        metadata["metrics_json"] = args.metrics_json
    if args.shadow_json:
        metadata["shadow_json"] = args.shadow_json
    if args.artifact_ref:
        metadata["artifact_ref"] = args.artifact_ref
    if args.artifact_run_id:
        metadata["artifact_run_id"] = args.artifact_run_id
    if args.robot_type:
        metadata["robot_type"] = args.robot_type
    if args.safety_profile:
        metadata["safety_profile"] = args.safety_profile
    if args.robot_config_hash:
        metadata["robot_config_hash"] = args.robot_config_hash
    if args.require_shadow:
        metadata["require_shadow"] = True
    if args.min_success_rate is not None:
        metadata["min_success_rate"] = args.min_success_rate
    if args.max_policy_delta is not None:
        metadata["max_policy_delta"] = args.max_policy_delta

    run_id = generate_run_id(stage)
    config_data = snapshot_config(args.config)
    input_refs = materialize_refs(list(args.input))
    output_refs = materialize_refs(list(args.output))
    try:
        result = execute_stage(
            stage=stage,
            run_id=run_id,
            root=Path.cwd(),
            config_snapshot_data=config_data,
            input_refs=input_refs,
            output_refs=output_refs,
            metadata=metadata,
        )
        if result.generated_outputs:
            output_refs = output_refs + materialize_refs(result.generated_outputs)

        create_run_record(
            run_id=run_id,
            stage=stage,
            status=result.status,  # type: ignore[arg-type]
            invocation_argv=_invocation_argv(),
            config_snapshot_data=config_data,
            input_refs=input_refs,
            output_refs=output_refs,
            metadata=result.metadata,
            reports=result.reports,
            artifacts=result.artifacts,
        )
    except Exception as exc:
        create_run_record(
            run_id=run_id,
            stage=stage,
            status="failed",
            invocation_argv=_invocation_argv(),
            config_snapshot_data=config_data,
            input_refs=input_refs,
            output_refs=output_refs,
            metadata={**metadata, "error": str(exc), "exception_type": type(exc).__name__},
            reports=[],
            artifacts=[],
        )
        print(f"[rfx] {stage} run_id={run_id} status=failed")
        print(f"[rfx] {type(exc).__name__}: {exc}")
        return 2

    print(f"[rfx] {stage} run_id={run_id} status={result.status}")
    if result.message:
        print(f"[rfx] {result.message}")
    for report in result.reports:
        print(f"[rfx] report: {report}")
    for artifact in result.artifacts:
        print(f"[rfx] artifact: {artifact.get('ref')} ({artifact.get('path')})")

    return 0 if result.status == "succeeded" else 2


def cmd_runs_list(args: argparse.Namespace) -> int:
    runs = list_runs(stage=args.stage, status=args.status, limit=args.limit)
    if not runs:
        print("No workflow runs found.")
        return 0
    for run in runs:
        print(
            f"{run.get('run_id')}  {run.get('stage')}  {run.get('status')}  {run.get('finished_at')}"
        )
    return 0


def cmd_runs_show(args: argparse.Namespace) -> int:
    run = load_run(args.run_id)
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    lineage = build_lineage(args.run_id)
    for run in lineage:
        print(f"{run.get('run_id')}  {run.get('stage')}  {run.get('status')}")
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    context = build_reproduce_context(args.run_id)
    print(f"run_id: {context['run_id']}")
    print(f"stage: {context['stage']}")
    print(f"command: {context['command']}")
    missing = context.get("missing_dependencies", [])
    if missing:
        print("missing_dependencies:")
        for item in missing:
            print(f"- {item}")
        return 2
    return 0


def _add_stage_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="config file path or inline config text")
    parser.add_argument("--input", action="append", default=[], help="input refs (repeatable)")
    parser.add_argument("--output", action="append", default=[], help="output refs (repeatable)")
    parser.add_argument("--meta", action="append", default=[], help="metadata KEY=VALUE")
    parser.add_argument("--robot-type", default=None)
    parser.add_argument("--safety-profile", default=None)


def _add_collect_stage_args(parser: argparse.ArgumentParser) -> None:
    """Add collection-specific args to the collect subcommand."""
    parser.add_argument("--repo-id", default=None, help="HuggingFace dataset repo ID")
    parser.add_argument("--episodes", "-n", type=int, default=1, help="number of episodes")
    parser.add_argument(
        "--duration", "-d", type=float, default=None, help="duration per episode (s)"
    )
    parser.add_argument("--task", default="default", help="task label")
    parser.add_argument("--fps", type=int, default=30, help="recording frame rate")
    parser.add_argument("--push", action="store_true", help="push to Hub after collection")
    parser.add_argument("--mcap", action="store_true", help="also log MCAP sidecar")
    parser.add_argument("--state-dim", type=int, default=6, help="state dimension")
    parser.add_argument(
        "--collection-root",
        default="datasets",
        help="root directory for collected dataset output",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="rfx runtime CLI")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("pkg-create", help="create a new runtime package")
    s.add_argument("name")
    s.set_defaults(fn=cmd_pkg_create)

    s = sp.add_parser("pkg-list", help="list discovered runtime packages")
    s.set_defaults(fn=cmd_pkg_list)

    s = sp.add_parser("run", help="run a package node")
    s.add_argument("package")
    s.add_argument("node")
    s.add_argument("--name")
    s.add_argument("--backend", default="mock")
    s.add_argument("--rate-hz", type=float, default=50.0)
    s.add_argument("--max-steps", type=int, default=None)
    s.add_argument("--params-json", default="{}")
    s.set_defaults(fn=cmd_run)

    s = sp.add_parser("launch", help="run a launch file")
    s.add_argument("file")
    s.set_defaults(fn=cmd_launch)

    s = sp.add_parser("graph", help="show active launch graph")
    s.set_defaults(fn=cmd_graph)

    s = sp.add_parser("topic-list", help="list runtime topics")
    s.set_defaults(fn=cmd_topic_list)

    s = sp.add_parser("dora-build", help="build a Dora dataflow")
    s.add_argument("file")
    s.add_argument("--no-uv", action="store_true", help="disable --uv passthrough")
    s.set_defaults(fn=cmd_dora_build)

    s = sp.add_parser("dora-run", help="run a Dora dataflow")
    s.add_argument("file")
    s.add_argument("--no-uv", action="store_true", help="disable --uv passthrough")
    s.add_argument(
        "--env",
        action="append",
        default=[],
        help="extra env var for Dora process in KEY=VALUE form (repeatable)",
    )
    s.set_defaults(fn=cmd_dora_run)

    for stage in ("collect", "validate", "train", "eval", "shadow", "deploy"):
        s = sp.add_parser(stage, help=f"workflow stage: {stage}")
        _add_stage_common_args(s)
        if stage == "collect":
            _add_collect_stage_args(s)
        s.add_argument("--dataset", default=None, help="dataset path for validate stage")
        s.add_argument("--artifact-ref", default=None, help="artifact ref for eval/shadow/deploy")
        s.add_argument(
            "--artifact-run-id",
            default=None,
            help="upstream run_id containing artifact metadata",
        )
        s.add_argument("--metrics-json", default=None, help="metrics JSON file for eval stage")
        s.add_argument("--shadow-json", default=None, help="shadow JSON file for shadow stage")
        s.add_argument("--min-success-rate", type=float, default=None)
        s.add_argument("--max-policy-delta", type=float, default=None)
        s.add_argument(
            "--require-shadow",
            action="store_true",
            help="deploy gate: require a successful shadow report",
        )
        s.add_argument("--robot-config-hash", default=None)
        s.set_defaults(fn=cmd_stage, stage=stage)

    s = sp.add_parser("runs", help="workflow run registry commands")
    runs = s.add_subparsers(dest="runs_cmd", required=True)

    rs = runs.add_parser("list", help="list workflow runs")
    rs.add_argument("--stage", default=None)
    rs.add_argument("--status", default=None)
    rs.add_argument("--limit", type=int, default=20)
    rs.set_defaults(fn=cmd_runs_list)

    rs = runs.add_parser("show", help="show a workflow run record")
    rs.add_argument("run_id")
    rs.set_defaults(fn=cmd_runs_show)

    s = sp.add_parser("lineage", help="show upstream lineage for a run")
    s.add_argument("run_id")
    s.set_defaults(fn=cmd_lineage)

    s = sp.add_parser("reproduce", help="reconstruct command context for a run")
    s.add_argument("run_id")
    s.set_defaults(fn=cmd_reproduce)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.fn(args))


if __name__ == "__main__":
    main()
