# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL absorbed sparse MLA 8-wave ping-pong host wrapper. bf16, H=16, D=512."""

from __future__ import annotations

import math

import flydsl.expr as fx
import torch

from aiter.ops.flydsl.kernels.sparse_mla_pp import (
    BLOCK_K,
    TOKENS_PER_WG,
    compile_sparse_mla_pp,
)
from aiter.ops.flydsl.kernels.sparse_mla_qblock import HEAD_DIM, NUM_HEADS
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg
from aiter.ops.flydsl.sparse_mla_qblock_kernels import (
    _index_span_supported,
    _require_contract,
    _require_gfx942,
    _uniform_topk,
)

__all__ = ["sparse_mla_pp_fwd_flydsl"]


def sparse_mla_pp_fwd_flydsl(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor,
    kv_indptr: torch.Tensor | None = None,
    *,
    attn_sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    uniform_topk: int | None = None,
    verify_csr: bool = True,
    mfma_transpose: bool = True,
    q_lds: int = 6,
):
    """Same contract as sparse_mla_one_query_fwd_flydsl; topk must be a multiple of 32."""
    _require_gfx942()
    n_tok, _n_kv = _require_contract(q, kv, out)
    flat, topk = _uniform_topk(
        kv_indices, n_tok, kv_indptr, uniform_topk, verify_csr=verify_csr
    )
    if topk % (2 * BLOCK_K):
        raise ValueError("FlyDSL sparse MLA ping-pong requires topk divisible by 32")
    if not _index_span_supported(n_tok, topk):
        raise ValueError("FlyDSL sparse MLA index byte span must fit uint32")
    scale = 1.0 / math.sqrt(HEAD_DIM) if softmax_scale is None else float(softmax_scale)
    if not (0 < scale < float("inf")):
        raise ValueError("FlyDSL sparse MLA softmax_scale must be finite and positive")
    has_sink = attn_sink is not None
    kernel = compile_sparse_mla_pp(
        has_sink, mfma_transpose=bool(mfma_transpose), q_lds=int(q_lds)
    )
    n_tiles = -(-n_tok // TOKENS_PER_WG)
    idx = flat.to(dtype=torch.int32).contiguous()
    if has_sink:
        attn_sink = attn_sink.reshape(-1).to(torch.float32).contiguous()
        if attn_sink.numel() != NUM_HEADS:
            raise ValueError(f"attn_sink must have {NUM_HEADS} elements")
    else:
        attn_sink = torch.empty(1, dtype=torch.float32, device=q.device)
    stream = torch.cuda.current_stream(q.device)
    _run_compiled(
        kernel,
        ptr_arg(q, fx.BFloat16),
        ptr_arg(kv, fx.BFloat16),
        ptr_arg(idx, fx.Int32),
        ptr_arg(out, fx.BFloat16),
        int(n_tok),
        int(topk),
        float(scale),
        ptr_arg(attn_sink, fx.Float32),
        int(n_tiles),
        fx.Stream(stream),
    )
    return out
