# Runtime (ROS-like)

This runtime adds ROS-style workflow primitives with a Python-first API and Rust-backed transport options.

## What It Includes

- Package model (`pkg-create`, `pkg-list`)
- Launch files (`launch`)
- Node lifecycle contract (`setup` / `tick` / `shutdown`)
- Graph and topic introspection (`graph`, `topic-list`)
- Backend profile abstraction (`mock` / `genesis` / `real`) via launch config
- Rust-backed transport path available through `rfx._rfx` transport bindings

## Commands

Use `cli/rfx.sh`:

```bash
cli/rfx.sh pkg-create my_pkg
cli/rfx.sh pkg-list
cli/rfx.sh run my_pkg my_pkg_node --backend mock
cli/rfx.sh launch packages/my_pkg/launch.yaml
cli/rfx.sh graph
cli/rfx.sh topic-list
```

## Package Manifest

Each package uses `rfx_pkg.toml`:

```toml
[package]
name = "my_pkg"
version = "0.1.0"
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

## Rust Advantage

`Node` transport defaults to:

- `RustTransport` when native bindings are available
- fallback to `InprocTransport` otherwise

This keeps the same node contract while letting performance-critical messaging move to Rust.
