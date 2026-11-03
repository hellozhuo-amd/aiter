# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""unified_attention with the fused dynamic 1 x 128 fp8 group quant epilogue (out_block_scale).

The fused output must be bitwise equal to the unfused pair: bf16 attention output followed by the HIP
`dynamic_per_group_scaled_quant_kernel` (per_group_quant_hip), for both scale layouts, on the 2d kernel (prefill,
mixed, decode), the 3d split-KV kernel + reduce_segments, and the 3d kernel with a single segment.
"""

import pytest
import torch

import aiter.ops.triton.attention.unified_attention as ua_mod
from aiter import QuantType, get_hip_quant
from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype

GROUP = 128


def make_inputs(
    seq_lens,  # [(query_len, kv_len)]
    num_heads,
    head_size,
    block_size,
    v_scale=1.0,
    fp8_q=True,
    seed=0,
    device="cuda",
):
    torch.manual_seed(seed)
    nq, nkv = num_heads
    q_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    num_tokens = sum(q_lens)
    max_blocks = max((k + block_size - 1) // block_size for k in kv_lens)
    num_blocks = max_blocks * len(seq_lens) + 8
    q = torch.randn(num_tokens, nq, head_size, device=device)
    q = q.to(e4m3_dtype) if fp8_q else q.to(torch.bfloat16)
    kv = torch.randn(num_blocks, block_size, 2 * nkv, head_size, device=device) * 2
    k_cache = kv[:, :, :nkv].to(e4m3_dtype)
    v_cache = kv[:, :, nkv:].to(e4m3_dtype)
    block_table = torch.randperm(num_blocks, device=device, dtype=torch.int32)[
        : len(seq_lens) * max_blocks
    ].view(len(seq_lens), max_blocks)
    cu_q = torch.tensor([0] + q_lens, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    seqused_k = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    one = torch.ones(1, device=device)
    return dict(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_q,
        max_seqlen_q=max(q_lens),
        seqused_k=seqused_k,
        max_seqlen_k=max(kv_lens),
        softmax_scale=head_size**-0.5,
        causal=True,
        block_table=block_table,
        softcap=0,
        q_descale=one * 0.5 if fp8_q else None,
        k_descale=one * 0.75,
        v_descale=one * v_scale,
    )


def run_pair(args, window, transpose_scale, rows=None):
    """(unfused fp8, unfused scales, fused fp8, fused scales), scales as (rows, G) views."""
    q = args["q"]
    T, H, D = q.shape
    rows = rows or T
    G = H * D // GROUP
    ref = torch.empty(T, H, D, device=q.device, dtype=torch.bfloat16)
    unified_attention(out=ref, window_size=window, **args)
    hip_quant = get_hip_quant(QuantType.per_1x128)
    ref_pad = torch.zeros(rows, H * D, device=q.device, dtype=torch.bfloat16)
    ref_pad[:T] = ref.view(T, H * D)
    q_ref, s_ref = hip_quant(ref_pad, quant_dtype=e4m3_dtype, transpose_scale=transpose_scale)

    out = torch.full((rows, H, D), 0, device=q.device, dtype=e4m3_dtype)
    scale = torch.full((rows, G), -1.0, device=q.device, dtype=torch.float32)
    unified_attention(
        out=out[:T],
        window_size=window,
        out_block_scale=scale,
        transpose_scale=transpose_scale,
        group_size=GROUP,
        **args,
    )
    if transpose_scale:  # (G, rows) bytes; compare the T written columns
        return q_ref[:T], s_ref.view(G, rows)[:, :T], out.view(rows, H * D)[:T], scale.view(G, rows)[:, :T]
    return q_ref[:T], s_ref[:T], out.view(rows, H * D)[:T], scale[:T]


def assert_bitwise(r):
    q_ref, s_ref, q_f, s_f = r
    assert torch.equal(s_f.contiguous().view(torch.int32), s_ref.contiguous().view(torch.int32)), (
        f"scales differ: {(s_f != s_ref).sum().item()} of {s_f.numel()}"
    )
    nd = (q_f.view(torch.uint8) != q_ref.view(torch.uint8)).sum().item()
    assert nd == 0, f"fp8 bytes differ: {nd} of {q_f.numel()}"


# Gemma4-31B: sliding layers head 256, 32 q / 16 kv heads, page 64, window 1024;
# full-attention layers head 512, 32 q / 4 kv heads, page 128.
LAYERS = {
    "sliding": dict(num_heads=(32, 16), head_size=256, block_size=64, window=(1023, 0)),
    "full": dict(num_heads=(32, 4), head_size=512, block_size=128, window=(-1, -1)),
}


def workload(M, kind):
    if kind == "decode":
        return [(1, 600 + 997 * i % 7000) for i in range(M)]
    if kind == "prefill":  # one chunk with prior context
        return [(M, M + 300)]
    # mixed: decodes plus one prefill
    return [(1, 800 + 131 * i % 3000) for i in range(max(M - 31, 1))] + [(min(M, 31), 1500)]


@pytest.mark.parametrize("transpose_scale", [False, True])
@pytest.mark.parametrize("layer", ["sliding", "full"])
@pytest.mark.parametrize(
    "M,kind",
    [(1, "decode"), (16, "decode"), (31, "decode"), (32, "decode"), (257, "decode"),
     (1, "prefill"), (31, "prefill"), (257, "prefill"), (4336, "prefill"), (32, "mixed"), (257, "mixed")],
)
def test_group_quant_bitwise(M, kind, layer, transpose_scale):
    L = LAYERS[layer]
    args = make_inputs(workload(M, kind), L["num_heads"], L["head_size"], L["block_size"])
    assert_bitwise(run_pair(args, L["window"], transpose_scale))


@pytest.mark.parametrize("transpose_scale", [False, True])
def test_group_quant_padded_rows(transpose_scale):
    """CUDA-graph padding: the scale tensor has more rows than the attention writes."""
    L = LAYERS["sliding"]
    args = make_inputs(workload(13, "decode"), L["num_heads"], L["head_size"], L["block_size"])
    assert_bitwise(run_pair(args, L["window"], transpose_scale, rows=16))


@pytest.mark.parametrize("v_scale", [0.0, 1e-30, 300.0])
def test_group_quant_extremes(v_scale):
    """All-zero groups (scale floor 1e-10), tiny values, and |x| >> 240 (fp8 saturation boundary)."""
    L = LAYERS["full"]
    args = make_inputs(workload(16, "decode"), L["num_heads"], L["head_size"], L["block_size"], v_scale=v_scale)
    assert_bitwise(run_pair(args, L["window"], True))


def test_group_quant_bf16_query():
    L = LAYERS["sliding"]
    args = make_inputs(workload(32, "mixed"), L["num_heads"], L["head_size"], L["block_size"], fp8_q=False)
    assert_bitwise(run_pair(args, L["window"], True))


def test_group_quant_3d_single_segment(monkeypatch):
    """3d kernel writing the output directly (NUM_SEGMENTS == 1): the epilogue runs in the 3d kernel."""
    orig = ua_mod.get_unified_attention_config

    def one_segment(op, params, backend=None):
        cfg = orig(op, params, backend=backend)
        if op == "kv_split":
            cfg["NUM_SEGMENTS"] = 1
        return cfg

    monkeypatch.setattr(ua_mod, "get_unified_attention_config", one_segment)
    L = LAYERS["full"]
    args = make_inputs(workload(16, "decode"), L["num_heads"], L["head_size"], L["block_size"])
    assert_bitwise(run_pair(args, L["window"], True))


def test_group_quant_determinism():
    L = LAYERS["full"]
    args = make_inputs(workload(32, "decode"), L["num_heads"], L["head_size"], L["block_size"])
    first = run_pair(args, L["window"], True)
    for _ in range(19):
        r = run_pair(args, L["window"], True)
        assert torch.equal(r[2].view(torch.uint8), first[2].view(torch.uint8))
        assert torch.equal(r[3].view(torch.int32), first[3].view(torch.int32))
