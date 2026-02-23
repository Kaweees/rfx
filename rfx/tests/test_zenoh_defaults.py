"""Tests that Zenoh is the default and only distributed transport backend."""

from __future__ import annotations

import pytest


def test_transport_config_defaults_to_zenoh() -> None:
    from rfx.teleop.config import TransportConfig

    config = TransportConfig()
    assert config.backend == "zenoh"


def test_transport_backend_type_has_no_dds() -> None:
    """DDS should not be an accepted backend value."""
    # "dds" is not in TransportBackend Literal, but we test the factory
    # rejects it at runtime even if passed dynamically.
    import rfx.teleop.transport as transport_mod
    from rfx.teleop.config import TransportConfig

    with pytest.raises(ValueError, match="Unsupported transport backend"):
        transport_mod.create_transport(TransportConfig(backend="dds"))  # type: ignore[arg-type]


def test_zenoh_compiled_in_default_build() -> None:
    """Zenoh support must be compiled in by default."""
    try:
        from rfx._rfx import Transport as _RustTransport
    except ImportError:
        pytest.skip("Rust extension not built")

    assert hasattr(_RustTransport, "zenoh_available"), "zenoh_available method missing"
    assert _RustTransport.zenoh_available(), (
        "Zenoh is not compiled in. Ensure pyproject.toml has "
        'features = ["extension-module", "zenoh"]'
    )


def test_auto_transport_raises_without_zenoh(monkeypatch) -> None:
    """auto_transport must fail loudly when Zenoh is unavailable."""
    import rfx.node as node_mod

    # Simulate missing Rust bindings
    monkeypatch.setattr(
        node_mod, "__name__", node_mod.__name__
    )  # no-op to trigger fresh import
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def fake_import(name, *args, **kwargs):
        if name == "rfx._rfx":
            raise ImportError("mocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="Rust bindings"):
        node_mod.auto_transport()


def test_node_create_validates_publish_rate() -> None:
    """node.create must reject invalid publish_rate_hz."""
    from rfx.node import create

    with pytest.raises(ValueError, match="publish_rate_hz"):
        create("so101", publish_rate_hz=0)

    with pytest.raises(ValueError, match="publish_rate_hz"):
        create("so101", publish_rate_hz=-10)


def test_node_create_rejects_unsupported_robot() -> None:
    """node.create must reject unknown robot types."""
    from rfx.node import create

    with pytest.raises(ValueError, match="Unsupported robot_type"):
        create("does_not_exist")


def test_teleop_config_zenoh_default_in_session() -> None:
    """TeleopSessionConfig should default to zenoh transport."""
    from rfx.teleop.config import TeleopSessionConfig

    config = TeleopSessionConfig.single_arm_pair()
    assert config.transport.backend == "zenoh"


def test_auto_transport_reads_env_connect(monkeypatch) -> None:
    """auto_transport should parse RFX_ZENOH_CONNECT env var."""
    import rfx.node as node_mod

    monkeypatch.setenv("RFX_ZENOH_CONNECT", "tcp/10.0.0.1:7447,tcp/10.0.0.2:7447")
    monkeypatch.delenv("RFX_ZENOH_SHARED_MEMORY", raising=False)

    # We can't actually create a Zenoh transport without infra, so we
    # patch the Rust call to capture the args.
    captured = {}

    class _FakeTransport:
        @staticmethod
        def zenoh_available():
            return True

        @staticmethod
        def zenoh(connect, listen, shared_memory, key_prefix):
            captured["connect"] = connect
            captured["listen"] = listen
            captured["shared_memory"] = shared_memory
            return object()

    monkeypatch.setattr("rfx._rfx.Transport", _FakeTransport)

    node_mod.auto_transport()
    assert captured["connect"] == ["tcp/10.0.0.1:7447", "tcp/10.0.0.2:7447"]
    assert captured["listen"] == []
    assert captured["shared_memory"] is True


def test_auto_transport_explicit_overrides_env(monkeypatch) -> None:
    """Explicit connect= arg must override RFX_ZENOH_CONNECT."""
    import rfx.node as node_mod

    monkeypatch.setenv("RFX_ZENOH_CONNECT", "tcp/env:7447")

    captured = {}

    class _FakeTransport:
        @staticmethod
        def zenoh_available():
            return True

        @staticmethod
        def zenoh(connect, listen, shared_memory, key_prefix):
            captured["connect"] = connect
            return object()

    monkeypatch.setattr("rfx._rfx.Transport", _FakeTransport)

    node_mod.auto_transport(connect=["tcp/explicit:7447"])
    assert captured["connect"] == ["tcp/explicit:7447"]


def test_auto_transport_shared_memory_env(monkeypatch) -> None:
    """RFX_ZENOH_SHARED_MEMORY=0 should disable shared memory."""
    import rfx.node as node_mod

    monkeypatch.setenv("RFX_ZENOH_SHARED_MEMORY", "0")

    captured = {}

    class _FakeTransport:
        @staticmethod
        def zenoh_available():
            return True

        @staticmethod
        def zenoh(connect, listen, shared_memory, key_prefix):
            captured["shared_memory"] = shared_memory
            return object()

    monkeypatch.setattr("rfx._rfx.Transport", _FakeTransport)

    node_mod.auto_transport()
    assert captured["shared_memory"] is False


def test_go2_dds_backend_deprecation_warning() -> None:
    """Using dds_backend= should emit FutureWarning."""
    import warnings

    from rfx.config import GO2_CONFIG
    from rfx.real.go2 import Go2Backend

    # We only need to test the warning path, not actual connection.
    # Go2Backend.__init__ will emit the warning before trying to connect.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            Go2Backend(config=GO2_CONFIG, dds_backend="rust")
        except Exception:
            pass  # Connection failure expected — we only care about the warning

        deprecation_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(deprecation_warnings) >= 1
        assert "dds_backend" in str(deprecation_warnings[0].message)
