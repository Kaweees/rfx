from __future__ import annotations

import argparse

import pytest

from rfx.runtime import cli
from rfx.runtime.dora_bridge import DoraCliError, build_dataflow, run_dataflow


def test_build_dataflow_raises_when_dora_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(DoraCliError, match="Dora CLI not found"):
        build_dataflow("graph.yml")


def test_run_dataflow_raises_when_dora_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(DoraCliError, match="Dora CLI not found"):
        run_dataflow("graph.yml")


def test_cli_dora_build_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "build_dataflow", lambda *_args, **_kwargs: 0)
    code = cli.cmd_dora_build(argparse.Namespace(file="graph.yml", no_uv=False))
    assert code == 0


def test_cli_dora_run_env_parse_error() -> None:
    code = cli.cmd_dora_run(argparse.Namespace(file="graph.yml", no_uv=False, env=["BAD_ENV"]))
    assert code == 1


def test_cli_dora_run_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_dataflow", lambda *_args, **_kwargs: 0)
    code = cli.cmd_dora_run(
        argparse.Namespace(file="graph.yml", no_uv=False, env=["RFX_BACKEND=dora"])
    )
    assert code == 0
