"""Tests for rfx.runtime.launch - Launch system enhancements."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rfx.runtime.launch import (
    LaunchNode,
    _substitute_env,
    evaluate_condition,
    load_launch_file,
)

# ---------------------------------------------------------------------------
# LaunchNode with remap, condition, managed fields
# ---------------------------------------------------------------------------


def test_launch_node_defaults() -> None:
    node = LaunchNode(package="my_pkg", node="my_node")
    assert node.remap == {}
    assert node.condition is None
    assert node.managed is False
    assert node.rate_hz == 50.0
    assert node.max_steps is None


def test_launch_node_with_remap() -> None:
    node = LaunchNode(
        package="p",
        node="n",
        remap={"input_topic": "sensor/lidar", "output_topic": "nav/cmd_vel"},
    )
    assert node.remap["input_topic"] == "sensor/lidar"
    assert node.remap["output_topic"] == "nav/cmd_vel"


def test_launch_node_with_condition_and_managed() -> None:
    node = LaunchNode(
        package="p",
        node="n",
        condition="${USE_SIM}",
        managed=True,
    )
    assert node.condition == "${USE_SIM}"
    assert node.managed is True


# ---------------------------------------------------------------------------
# evaluate_condition
# ---------------------------------------------------------------------------


def test_evaluate_condition_none_returns_true() -> None:
    assert evaluate_condition(None) is True


def test_evaluate_condition_true_values() -> None:
    for val in ("true", "True", "TRUE", "1", "yes", "Yes"):
        assert evaluate_condition(val) is True, f"Expected True for {val!r}"


def test_evaluate_condition_false_values() -> None:
    for val in ("false", "False", "0", "no", "No", ""):
        assert evaluate_condition(val) is False, f"Expected False for {val!r}"


def test_evaluate_condition_env_substitution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RFX_SIM", "true")
    assert evaluate_condition("${RFX_SIM}") is True

    monkeypatch.setenv("RFX_SIM", "false")
    assert evaluate_condition("${RFX_SIM}") is False


# ---------------------------------------------------------------------------
# _substitute_env
# ---------------------------------------------------------------------------


def test_substitute_env_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_VAR", "hello")
    assert _substitute_env("${MY_VAR}") == "hello"


def test_substitute_env_with_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_VAR", raising=False)
    assert _substitute_env("${MISSING_VAR:fallback}") == "fallback"


def test_substitute_env_no_default_keeps_original(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOEXIST", raising=False)
    assert _substitute_env("${NOEXIST}") == "${NOEXIST}"


def test_substitute_env_multiple(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A", "x")
    monkeypatch.setenv("B", "y")
    assert _substitute_env("${A}-${B}") == "x-y"


# ---------------------------------------------------------------------------
# load_launch_file parses nodes and includes
# ---------------------------------------------------------------------------


def test_load_launch_file_basic(tmp_path: Path) -> None:
    launch_yaml = textwrap.dedent("""\
        name: test_launch
        backend: inproc
        nodes:
          - package: perception
            node: CameraNode
            name: cam0
            params:
              fps: 30
            rate_hz: 30.0
            remap:
              raw: /camera/raw
            managed: true
    """)
    f = tmp_path / "launch.yaml"
    f.write_text(launch_yaml)

    spec = load_launch_file(f)
    assert spec.name == "test_launch"
    assert spec.backend == "inproc"
    assert len(spec.nodes) == 1

    node = spec.nodes[0]
    assert node.package == "perception"
    assert node.node == "CameraNode"
    assert node.name == "cam0"
    assert node.params["fps"] == 30
    assert node.rate_hz == 30.0
    assert node.remap == {"raw": "/camera/raw"}
    assert node.managed is True


def test_load_launch_file_with_includes(tmp_path: Path) -> None:
    launch_yaml = textwrap.dedent("""\
        name: main
        includes:
          - file: base.yaml
            condition: "true"
          - other.yaml
        nodes: []
    """)
    f = tmp_path / "main.yaml"
    f.write_text(launch_yaml)

    spec = load_launch_file(f)
    assert len(spec.includes) == 2
    assert spec.includes[0].file == "base.yaml"
    assert spec.includes[0].condition == "true"
    assert spec.includes[1].file == "other.yaml"
    assert spec.includes[1].condition is None


def test_load_launch_file_with_condition_on_node(tmp_path: Path) -> None:
    launch_yaml = textwrap.dedent("""\
        name: cond_launch
        nodes:
          - package: sim
            node: SimNode
            condition: "${USE_SIM:false}"
    """)
    f = tmp_path / "cond.yaml"
    f.write_text(launch_yaml)

    spec = load_launch_file(f)
    assert len(spec.nodes) == 1
    assert spec.nodes[0].condition == "${USE_SIM:false}"
