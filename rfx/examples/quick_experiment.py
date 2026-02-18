#!/usr/bin/env python3
"""
Quick architecture experiments with rfx + tinygrad.

Shows how to prototype different policy architectures — bigger nets,
custom activations, residual blocks, self-attention, JIT compilation —
all in plain Python using tinygrad's transparent tensor API.

Key concepts:
    rfx.nn.Policy        — base class for tinygrad policies
    rfx.nn.MLP           — standard MLP with .load()/.save()
    rfx.nn.JitPolicy     — @TinyJit wrapper for compiled inference
    rfx.nn.go2_mlp       — default Go2 policy (48→256→256→12)

Usage:
    uv run rfx/examples/quick_experiment.py

Requirements:
    pip install tinygrad
"""

import rfx
from rfx.nn import Policy, MLP, go2_mlp

try:
    from tinygrad import Tensor
    from tinygrad.nn import Linear

    TINYGRAD = True
except ImportError:
    TINYGRAD = False
    print("tinygrad not installed. Install with: pip install tinygrad")
    print("Showing code structure only.\n")


# -------------------------------------------------------------------
# Experiment 1: Bigger network
# -------------------------------------------------------------------

def exp_bigger_network():
    print("1. Bigger Network")
    print("-" * 40)

    standard = go2_mlp(hidden=[256, 256])
    bigger = MLP(48, 12, hidden=[512, 512, 512])
    print(f"  Standard: {standard}")
    print(f"  Bigger:   {bigger}")

    if TINYGRAD:
        s_params = sum(p.numel() for p in standard.parameters())
        b_params = sum(p.numel() for p in bigger.parameters())
        print(f"  Params: {s_params:,} → {b_params:,} ({b_params / s_params:.1f}x)")
    print()


# -------------------------------------------------------------------
# Experiment 2: GELU activations
# -------------------------------------------------------------------

def exp_gelu():
    print("2. GELU Activations")
    print("-" * 40)

    if not TINYGRAD:
        print("  (requires tinygrad)\n")
        return

    class GELUPolicy(Policy):
        def __init__(self, obs_dim=48, act_dim=12):
            self.l1 = Linear(obs_dim, 256)
            self.l2 = Linear(256, 256)
            self.l3 = Linear(256, act_dim)

        def forward(self, obs: Tensor) -> Tensor:
            x = self.l1(obs).gelu()
            x = self.l2(x).gelu()
            return self.l3(x).tanh()

    policy = GELUPolicy()
    action = policy(Tensor.randn(1, 48))
    print(f"  Output: shape={action.shape}, range=[{action.min().numpy():.3f}, {action.max().numpy():.3f}]")
    print()


# -------------------------------------------------------------------
# Experiment 3: Residual connections
# -------------------------------------------------------------------

def exp_residual():
    print("3. Residual Connections")
    print("-" * 40)

    if not TINYGRAD:
        print("  (requires tinygrad)\n")
        return

    class ResidualPolicy(Policy):
        def __init__(self, obs_dim=48, act_dim=12, hidden=256):
            self.proj = Linear(obs_dim, hidden)
            self.b1_l1, self.b1_l2 = Linear(hidden, hidden), Linear(hidden, hidden)
            self.b2_l1, self.b2_l2 = Linear(hidden, hidden), Linear(hidden, hidden)
            self.out = Linear(hidden, act_dim)

        def _block(self, x, l1, l2):
            return (l2(l1(x).relu()) + x).relu()

        def forward(self, obs: Tensor) -> Tensor:
            x = self.proj(obs).relu()
            x = self._block(x, self.b1_l1, self.b1_l2)
            x = self._block(x, self.b2_l1, self.b2_l2)
            return self.out(x).tanh()

    action = ResidualPolicy()(Tensor.randn(1, 48))
    print(f"  2 residual blocks → output shape={action.shape}")
    print()


# -------------------------------------------------------------------
# Experiment 4: Self-attention over observation groups
# -------------------------------------------------------------------

def exp_attention():
    print("4. Self-Attention")
    print("-" * 40)

    if not TINYGRAD:
        print("  (requires tinygrad)\n")
        return

    class AttentionPolicy(Policy):
        """Treats 48-dim obs as 12 groups of 4, applies self-attention."""

        def __init__(self):
            self.num_groups, self.group_dim, self.hidden = 12, 4, 64
            self.group_proj = Linear(self.group_dim, self.hidden)
            self.q = Linear(self.hidden, self.hidden)
            self.k = Linear(self.hidden, self.hidden)
            self.v = Linear(self.hidden, self.hidden)
            self.out1 = Linear(self.hidden * self.num_groups, 256)
            self.out2 = Linear(256, 12)

        def forward(self, obs: Tensor) -> Tensor:
            B = obs.shape[0]
            x = self.group_proj(obs.reshape(B, self.num_groups, self.group_dim))
            scores = (self.q(x) @ self.k(x).transpose(-2, -1)) / self.hidden**0.5
            x = (scores.softmax(axis=-1) @ self.v(x)).reshape(B, -1)
            return self.out2(self.out1(x).relu()).tanh()

    action = AttentionPolicy()(Tensor.randn(1, 48))
    print(f"  12 groups x 4 features → output shape={action.shape}")
    print()


# -------------------------------------------------------------------
# Experiment 5: JIT compilation speedup
# -------------------------------------------------------------------

def exp_jit():
    print("5. JIT Compilation")
    print("-" * 40)

    if not TINYGRAD:
        print("  (requires tinygrad)\n")
        return

    import time
    from rfx.nn import JitPolicy

    policy = go2_mlp()
    jit_policy = JitPolicy(policy)
    obs = Tensor.randn(1, 48)

    # Warmup
    _ = policy(obs)
    _ = jit_policy(obs)

    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        _ = policy(obs)
    base_ms = (time.perf_counter() - t0) / N * 1000

    _ = jit_policy(obs)  # compile
    t0 = time.perf_counter()
    for _ in range(N):
        _ = jit_policy(obs)
    jit_ms = (time.perf_counter() - t0) / N * 1000

    print(f"  Base: {base_ms:.3f} ms/call")
    print(f"  JIT:  {jit_ms:.3f} ms/call  ({base_ms / jit_ms:.1f}x speedup)")
    print()


if __name__ == "__main__":
    print("rfx Architecture Experiments")
    print("=" * 50)
    print()
    exp_bigger_network()
    exp_gelu()
    exp_residual()
    exp_attention()
    exp_jit()
    print("=" * 50)
    print("Done!")
