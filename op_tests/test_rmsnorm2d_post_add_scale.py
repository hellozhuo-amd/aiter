# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""rmsnorm2d_fwd_post_add_scale: out = (rms_norm(x, w) + residual) * scalar in one HIP launch.

Must be bitwise equal to the unfused pair it replaces in Gemma4 decoder layers: aiter `rms_norm` (HIP) followed by
the Inductor elementwise kernel for `(h + residual) * layer_scalar` (fp32 math, one rounding).
"""

import pytest
import torch

import aiter

for _name in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _name):
        setattr(torch._dynamo.config, _name, 64)

_add_mul = torch.compile(lambda h, r, s: (h + r) * s, dynamic=True)


def reference(x, w, r, s, eps, compiled=True):
    h = aiter.rms_norm(x, w, eps)
    if compiled:
        return _add_mul(h, r, s)
    return ((h.float() + r.float()) * s.float()).to(x.dtype)


def fused(x, w, r, s, eps):
    out = torch.empty_like(x)
    return aiter.rmsnorm2d_fwd_post_add_scale(out, x, r, w, s, eps)


def inputs(M, N, dtype, x_scale=3.0, r_scale=10.0, scalar=0.8789, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(M, N, device="cuda") * x_scale).to(dtype)
    r = (torch.randn(M, N, device="cuda") * r_scale).to(dtype)
    w = (1.0 + 0.2 * torch.randn(N, device="cuda")).to(dtype)
    s = torch.tensor([scalar], device="cuda", dtype=dtype)
    return x, w, r, s


def assert_bitwise(a, b):
    nd = (a.view(torch.int16) != b.view(torch.int16)).sum().item()
    assert nd == 0, f"{nd} of {a.numel()} values differ"


@pytest.mark.parametrize("M", [1, 16, 31, 32, 64, 256, 1024, 4336, 16384])
@pytest.mark.parametrize("N", [5376])
def test_bitwise_vs_rmsnorm_and_inductor(M, N):
    x, w, r, s = inputs(M, N, torch.bfloat16)
    ref = reference(x, w, r, s, 1e-6)
    assert_bitwise(fused(x, w, r, s, 1e-6), ref)
    # Inductor's add*mul is fp32 math rounded once
    assert_bitwise(ref, reference(x, w, r, s, 1e-6, compiled=False))


@pytest.mark.parametrize("N", [512, 1024, 2048, 4096, 6144, 8192])
def test_bitwise_all_buckets(N):
    x, w, r, s = inputs(37, N, torch.bfloat16)
    assert_bitwise(fused(x, w, r, s, 1e-6), reference(x, w, r, s, 1e-6))


@pytest.mark.parametrize("scalar", [0.0361, 0.5, 0.99, 1.0, 3.0])
def test_bitwise_layer_scalars(scalar):
    """Gemma4-31B layer_scalar spans 0.036 to 0.99."""
    x, w, r, s = inputs(64, 5376, torch.bfloat16, scalar=scalar)
    assert_bitwise(fused(x, w, r, s, 1e-6), reference(x, w, r, s, 1e-6))


def test_extremes():
    """Zero rows, huge residuals (bf16 overflow to inf), tiny inputs."""
    x, w, r, s = inputs(8, 5376, torch.bfloat16)
    x[0] = 0
    r[1] = 3e38
    x[2] *= 1e-30
    r[3] = 0
    ref = reference(x, w, r, s, 1e-6)
    out = fused(x, w, r, s, 1e-6)
    assert torch.equal(out.isnan(), ref.isnan())
    keep = ~ref.isnan()
    assert_bitwise(out[keep], ref[keep])


def test_strided_rows():
    """Row strides of views (e.g. a slice of a padded buffer)."""
    x, w, r, s = inputs(48, 5376, torch.bfloat16)
    xb = torch.zeros(48, 6000, device="cuda", dtype=torch.bfloat16)
    xb[:, :5376] = x
    xv = xb[:, :5376]
    assert_bitwise(fused(xv, w, r, s, 1e-6), reference(x, w, r, s, 1e-6))


def test_determinism_and_batch_invariance():
    x, w, r, s = inputs(256, 5376, torch.bfloat16)
    first = fused(x, w, r, s, 1e-6)
    for _ in range(19):
        assert_bitwise(fused(x, w, r, s, 1e-6), first)
    for i in (0, 77, 255):
        assert_bitwise(fused(x[i : i + 1], w, r[i : i + 1], s, 1e-6), first[i : i + 1])


def test_graph_safe_scalar():
    """The scalar is read on the device: a CUDA graph sees later updates of the scalar tensor."""
    x, w, r, s = inputs(16, 5376, torch.bfloat16)
    out = torch.empty_like(x)
    aiter.rmsnorm2d_fwd_post_add_scale(out, x, r, w, s, 1e-6)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        aiter.rmsnorm2d_fwd_post_add_scale(out, x, r, w, s, 1e-6)
    s.fill_(0.25)
    g.replay()
    torch.cuda.synchronize()
    assert_bitwise(out, reference(x, w, r, s, 1e-6))
