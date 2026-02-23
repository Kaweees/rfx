# Runtime (ROS-like)

This runtime adds ROS-style workflow primitives with a Python-first API and Zenoh-backed transport.

## What It Includes

- Package model (`pkg-create`, `pkg-list`)
- Launch files (`launch`)
- Node lifecycle contract (`setup` / `tick` / `shutdown`)
- Graph and topic introspection (`graph`, `topic-list`)
- Backend profile abstraction (`mock` / `genesis` / `real`) via launch config
- Zenoh transport (Rust-backed, compiled in by default)

## Commands

Use `rfx` (installed entrypoint) or `cli/rfx.sh` from source checkout:

```bash
rfx pkg-create my_pkg
rfx pkg-list
rfx run my_pkg my_pkg_node --backend mock
rfx launch packages/my_pkg/launch.yaml
rfx graph
rfx topic-list
```

## Package Manifest

Each package uses `rfx_pkg.toml`:

```toml
[package]
name = "my_pkg"
version = "0.2.0"
python_module = "src"

[nodes]
my_pkg_node = "my_pkg.nodes:MainNode"
```

## Node Contract

Nodes should inherit `rfx.runtime.node.Node`:

- `setup(self)`: initialization
- `tick(self) -> bool`: one loop iteration (return `False` to stop)
- `shutdown(self)`: cleanup

Declare introspection topics with class attributes:

- `publish_topics = ("my_pkg/state",)`
- `subscribe_topics = ("my_pkg/cmd",)`

## Launch File

```yaml
name: demo
backend: genesis
profile: default
profiles:
  default:
    RFX_BACKEND: genesis
nodes:
  - package: my_pkg
    node: my_pkg_node
    name: my_pkg.main
    rate_hz: 20
    params: {}
```

`backend` and profile env values are propagated to nodes.

## Zenoh Transport

All nodes use the Zenoh transport by default (Rust-backed, compiled into the native extension). This provides cross-process pub/sub, shared-memory zero-copy on the same machine, and network-transparent communication between distributed robots.

If the native extension is not built, node creation will raise a clear error with build instructions.
