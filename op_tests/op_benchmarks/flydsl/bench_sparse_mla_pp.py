# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Latency of the gfx942 FlyDSL sparse MLA prefill kernels on a production-like KV pool.

Kernels (out[t] = softmax(scale * q[t] . kv[idx[t]]) @ kv[idx[t]], bf16, H=16, D=512):
  fly_one_q  one-query kernel, 2 waves / token   (sparse_mla_one_query_fwd_flydsl)
  fly_pp     8-wave ping-pong, 4 tokens / WG     (sparse_mla_pp_fwd_flydsl)
  fly_bq2/4  Q-block kernels; only when G tokens share one list (G % BQ == 0)
  triton     the shipping pa_decode_sparse path (--triton)

Index pattern, default ``--pattern paged`` (GLM-5.3-Flash serving log, SILOTIGER-1215): the
KV pool is every block of the server's cache, 3640 blocks x 1152 rows = 4,193,280 rows
(~4 GiB). One 131,072-token request owns ~114 of those blocks at random positions; the chunk
is its last T positions, and each query selects topk distinct rows of its own causal prefix,
mapped through the request's block table. The launch touches ~128 MiB of distinct rows.
``--pattern uniform`` draws every list from the whole pool instead (no-reuse stress case).
``--groups G`` makes G consecutive tokens share one list (the Q-block reuse case).

Before the table, the compiled resources (VGPR / LDS / occupancy) of fly_pp and of the
Q-block kernels the run uses are read from the ISA of a fresh compile (``--no-meta`` skips).
Every row is checked against an fp32 torch oracle on the first/last ``--check-tokens`` tokens;
the L2 / Infinity Cache is flushed before every timed call unless ``--no-flush``.

Run (MI300X):
    python op_tests/op_benchmarks/flydsl/bench_sparse_mla_pp.py
    python op_tests/op_benchmarks/flydsl/bench_sparse_mla_pp.py --gpu 3 -t 1024 16384
    python op_tests/op_benchmarks/flydsl/bench_sparse_mla_pp.py --reqs 2          # 2 requests
    python op_tests/op_benchmarks/flydsl/bench_sparse_mla_pp.py --seq-len 262144  # max_model_len
    python op_tests/op_benchmarks/flydsl/bench_sparse_mla_pp.py --pattern uniform --pool 4193280
    python op_tests/op_benchmarks/flydsl/bench_sparse_mla_pp.py --groups 1 2 4 --sink --triton
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

H, D = 16, 512
BLOCK_K = 16
MAX_POOL = (1 << 31) // D - 1  # rows < 2^31 / D: the kernels' int32 element offsets
SIMDS_PER_CU, VGPR_GRANULE, LDS_PER_CU, WAVE_SLOTS = 4, 8, 64 * 1024, 8
ATOL = RTOL = 1e-2


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gpu", type=int, default=0, help="HIP_VISIBLE_DEVICES for the run")
    p.add_argument("-t", "--tokens", type=int, nargs="+", default=[16384])
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--groups", type=int, nargs="+", default=[1], choices=[1, 2, 4],
                   help="tokens per shared list")  # fmt: skip
    p.add_argument("--pattern", choices=["paged", "uniform"], default="paged")
    p.add_argument("--pool", type=int, default=None,
                   help=f"KV rows (default: paged {MAX_POOL - MAX_POOL % 1152}, uniform 1 GiB)")  # fmt: skip
    p.add_argument("--pool-gib", type=float, help="KV pool in GiB; overrides --pool")
    p.add_argument("--page", type=int, default=1152, help="paged: rows per KV block")
    p.add_argument(
        "--seq-len", type=int, default=131072, help="paged: prompt length per request"
    )
    p.add_argument(
        "--reqs", type=int, default=1, help="paged: requests mixed in the chunk"
    )
    p.add_argument(
        "--lists",
        type=int,
        default=0,
        help="uniform: cap on distinct lists (0 = one per group)",
    )
    p.add_argument("--sink", action="store_true", help="add a per-head attn_sink")
    p.add_argument(
        "--triton", action="store_true", help="add the shipping Triton kernel"
    )
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--flush", action=argparse.BooleanOptionalAction, default=True,
                   help="cold L2/MALL before every timed call")  # fmt: skip
    p.add_argument(
        "--check-tokens", type=int, default=128, help="0 = skip the oracle check"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--meta", action=argparse.BooleanOptionalAction, default=True,
                   help="print the compiled resources first")  # fmt: skip
    p.add_argument(
        "--mix", action="store_true", help="meta: add the static instruction mix"
    )
    p.add_argument(
        "--stats-only", nargs="*", help=argparse.SUPPRESS
    )  # internal: meta subprocess
    return p.parse_args(argv)


# ---- compiled resources (run in a subprocess with a fresh, dumping FlyDSL compile) ------------

FIELDS = [
    (".vgpr_count", "VGPRs"), (".agpr_count", "AGPRs"), (".vgpr_spill_count", "VGPR spills"),
    (".sgpr_count", "SGPRs"), (".sgpr_spill_count", "SGPR spills"),
    (".private_segment_fixed_size", "Scratch bytes"), (".group_segment_fixed_size", "LDS bytes"),
]  # fmt: skip
MIX = [
    ("v_mfma", r"^\s*v_mfma"), ("buffer_load", r"^\s*buffer_load"), ("ds_read", r"^\s*ds_read"),
    ("ds_write", r"^\s*ds_write"), ("s_waitcnt", r"^\s*s_waitcnt"), ("s_barrier", r"^\s*s_barrier"),
    ("v_exp_f32", r"^\s*v_exp_f32"), ("v_readlane", r"^\s*v_readlane"),
    ("total instrs", r"^\s+[a-z_][a-z0-9_]*(\s|$)"),
]  # fmt: skip
# label, waves per workgroup
META_KERNELS = {
    "fly_pp": ("ping-pong (fly_pp)", 8),
    "fly_bq2": ("BLOCK_Q=2", 2),
    "fly_bq4": ("BLOCK_Q=4", 4),
}


def print_meta(args):
    """Re-run this script with --stats-only under a dumping, cache-free FlyDSL compile."""
    names = ["fly_pp"] + [
        n
        for bq, n in ((2, "fly_bq2"), (4, "fly_bq4"))
        if any(g % bq == 0 for g in args.groups)
    ]
    with tempfile.TemporaryDirectory(prefix="spmla_meta_") as tmp:
        env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=f"{tmp}/isa",
                   FLYDSL_RUNTIME_CACHE_DIR=f"{tmp}/cache", HIP_VISIBLE_DEVICES=str(args.gpu))  # fmt: skip
        cmd = [sys.executable, __file__, "--stats-only", *names] + (
            ["--mix"] if args.mix else []
        )
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        out = "\n".join(
            ln for ln in r.stdout.splitlines() if not ln.startswith("[flydsl.compile]")
        )
        print(out)
        if r.returncode:
            print(r.stderr[-3000:])


def stats_only(names, mix):
    import triton  # noqa: F401  # isort: skip
    import flydsl
    import torch

    print(
        f"flydsl: {getattr(flydsl, '__version__', '?')} ({Path(flydsl.__file__).parent})"
    )
    torch.set_default_device("cuda")
    dump = os.environ["FLYDSL_DUMP_DIR"]
    topk = 64
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(8, H, D, generator=gen).to(torch.bfloat16)
    kv = torch.randn(4096, D, generator=gen).to(torch.bfloat16)
    idx = (
        torch.randperm(4096, device="cuda", generator=gen)[:topk]
        .to(torch.int32)
        .repeat(8)
        .contiguous()
    )
    indptr = torch.arange(0, 9 * topk, topk, dtype=torch.int32)
    for name in names:
        before = set(glob.glob(f"{dump}/**/*final_isa*.s", recursive=True))
        CANDIDATES[name](q, kv, idx, indptr, None, topk, D**-0.5)()
        torch.cuda.synchronize()
        new = sorted(
            set(glob.glob(f"{dump}/**/*final_isa*.s", recursive=True)) - before
        )
        new = [f for f in new if "sparse_mla" in f] or new
        label, wpw = META_KERNELS[name]
        if not new:
            print(f"\n{label}: no ISA dump found")
            continue
        text = Path(new[-1]).read_text()
        stats = {}
        for key, _ in FIELDS:
            m = re.search(
                rf"^\s*(?:-\s*)?{re.escape(key)}:\s*(\d+)\s*$", text, re.MULTILINE
            )
            if m:
                stats[key] = int(m.group(1))
        print(f"\n{label}  ({wpw} waves / workgroup)   {Path(new[-1]).parent.name}")
        for key, lab in FIELDS:
            if key in stats:
                print(f"  {lab:<22} {stats[key]}")
        vgprs = stats.get(".vgpr_count", 0) + stats.get(".agpr_count", 0)
        by_vgpr = 512 // max(VGPR_GRANULE, -(-vgprs // VGPR_GRANULE) * VGPR_GRANULE)
        lds = stats.get(".group_segment_fixed_size", 0)
        by_lds = (LDS_PER_CU // lds if lds else 99) * wpw / SIMDS_PER_CU
        waves = min(by_vgpr, by_lds, WAVE_SLOTS)
        print(
            f"  {'Waves / SIMD':<22} {waves:g}   (VGPR cap {by_vgpr}, LDS cap {by_lds:g}, hw cap {WAVE_SLOTS})"
        )
        print(f"  {'Workgroups / CU':<22} {int(waves * SIMDS_PER_CU // wpw)}")
        if mix:
            body = "\n".join(
                ln
                for ln in text.splitlines()
                if ln.startswith(("\t", " ")) and not ln.strip().startswith((".", ";"))
            )
            print("  static instruction mix:")
            for n, rx in MIX:
                print(f"    {n:<22} {len(re.findall(rx, body, re.MULTILINE))}")


# ---- kernels under test: each factory returns fn() -> out, or None if it does not apply -----


def cand_one_q(q, kv, idx, indptr, sink, topk, scale):
    from aiter.ops.flydsl.sparse_mla_qblock_kernels import (
        sparse_mla_one_query_fwd_flydsl,
    )

    out = q.new_empty(q.shape)
    return lambda: sparse_mla_one_query_fwd_flydsl(
        q,
        kv,
        idx,
        out,
        attn_sink=sink,
        softmax_scale=scale,
        uniform_topk=topk,
        verify_csr=False,
    )


def cand_pp(q, kv, idx, indptr, sink, topk, scale):
    from aiter.ops.flydsl.sparse_mla_pp_kernels import sparse_mla_pp_fwd_flydsl

    if topk % (2 * BLOCK_K):
        return None  # two KV tiles per loop trip
    out = q.new_empty(q.shape)
    return lambda: sparse_mla_pp_fwd_flydsl(
        q,
        kv,
        idx,
        out,
        attn_sink=sink,
        softmax_scale=scale,
        uniform_topk=topk,
        verify_csr=False,
    )


def cand_qblock(block_q):
    def make(q, kv, idx, indptr, sink, topk, scale):
        from aiter.ops.flydsl.sparse_mla_qblock_kernels import (
            pack_qblock_union,
            sparse_mla_qblock_fwd_flydsl,
        )

        packed = pack_qblock_union(
            idx, q.shape[0], topk, block_q, BLOCK_K
        )  # outside the timed region
        if not packed[2]:
            return None
        out = q.new_empty(q.shape)
        return lambda: sparse_mla_qblock_fwd_flydsl(
            q, kv, idx, out, block_q=block_q, packed=packed, attn_sink=sink,
            softmax_scale=scale, uniform_topk=topk, verify_csr=False,
        )  # fmt: skip

    return make


def cand_triton(q, kv, idx, indptr, sink, topk, scale):
    import torch

    from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

    if sink is None:  # pa_decode_sparse always takes a sink; -inf adds exp(-inf) = 0
        sink = torch.full((H,), float("-inf"), dtype=torch.float32)
    out = q.new_empty(q.shape)
    return lambda: pa_decode_sparse(
        q, kv, idx, indptr, sink, scale, has_invalid=False, out=out
    )


CANDIDATES = {
    "fly_one_q": cand_one_q,
    "fly_pp": cand_pp,
    "fly_bq2": cand_qblock(2),
    "fly_bq4": cand_qblock(4),
}


# ---- index patterns -------------------------------------------------------------------------


def make_group_indices(torch, n_tok, group, pool, topk, n_lists, gen):
    """Uniform: token t reads list (t // group) % n_lists; each list = topk distinct sorted rows."""
    n_grp = -(-n_tok // group)
    n_lists = min(n_lists or n_grp, n_grp)
    draw = torch.randint(
        0, pool - topk + 1, (n_lists, topk), device="cuda", generator=gen
    )
    lists = (draw.sort(dim=1).values + torch.arange(topk, device="cuda")).to(
        torch.int32
    )
    per_tok = lists[torch.arange(n_grp, device="cuda") % n_lists].repeat_interleave(
        group, dim=0
    )[:n_tok]
    return per_tok.reshape(-1).contiguous(), n_lists


def make_paged_indices(torch, n_tok, group, pool, topk, seq_len, page, n_reqs, gen):
    """One chunked-prefill step over a paged KV cache (vLLM serving GLM-5.3-Flash).

    The pool is ``pool // page`` blocks; each request owns ceil(seq_len / page) of them at
    random positions (a fragmented free list). The chunk's queries split evenly over the
    requests, each contributing the LAST n_tok / n_reqs positions of its prompt (the largest
    prefixes). A query at position p selects topk distinct rows of its own causal prefix
    [0, p] (sorted draws + arange), mapped through its request's block table. With
    group > 1, G consecutive queries share the first one's list.
    """
    n_blocks = pool // page
    blocks_per_req = -(-seq_len // page)
    if n_reqs * blocks_per_req > n_blocks:
        raise ValueError(
            f"{n_reqs} requests x {blocks_per_req} blocks exceed the {n_blocks}-block pool"
        )
    if n_tok % n_reqs:
        raise ValueError(f"T={n_tok} must split evenly over {n_reqs} requests")
    per_req = n_tok // n_reqs
    if seq_len - per_req + 1 < topk:
        raise ValueError(
            f"prefix {seq_len - per_req + 1} < topk {topk}: those rows would be ragged"
        )
    dev = "cuda"
    table = torch.randperm(n_blocks, device=dev, generator=gen)[
        : n_reqs * blocks_per_req
    ].view(n_reqs, -1)
    first = torch.arange(n_tok, device=dev)[::group]
    prefix = (seq_len - per_req + first % per_req + 1).unsqueeze(1)
    draw = (
        torch.rand(first.numel(), topk, device=dev, generator=gen) * (prefix - topk + 1)
    ).long()
    logical = draw.sort(dim=1).values + torch.arange(topk, device=dev)
    phys = table[first // per_req].gather(1, logical // page) * page + logical % page
    per_tok = phys.repeat_interleave(group, dim=0)[:n_tok].to(torch.int32)
    return per_tok.reshape(-1).contiguous(), first.numel()


# ---- oracle and timing ----------------------------------------------------------------------


def reference(torch, q, kv, idx, topk, scale, sink, chunk=64):
    """fp32 torch oracle, chunked over tokens."""
    n_tok = q.shape[0]
    rows = idx.view(n_tok, topk).long()
    out = torch.empty(n_tok, H, D, dtype=torch.float32, device=q.device)
    for t0 in range(0, n_tok, chunk):
        t1 = min(t0 + chunk, n_tok)
        g = kv[rows[t0:t1]].float()
        s = torch.einsum("thd,tkd->thk", q[t0:t1].float(), g) * scale
        if sink is not None:
            sk = sink.float().view(1, H, 1).expand(t1 - t0, H, 1)
            p = torch.softmax(torch.cat([s, sk], -1), -1)[..., :topk]
        else:
            p = torch.softmax(s, -1)
        out[t0:t1] = torch.einsum("thk,tkd->thd", p, g)
    return out


def check(torch, fn, q, kv, idx, topk, sink, scale, n_check):
    out = fn()
    n_tok = q.shape[0]
    err, ok = 0.0, True
    for t0 in sorted({0, max(0, n_tok - n_check)}):
        t1 = min(t0 + n_check, n_tok)
        ref = reference(
            torch, q[t0:t1], kv, idx[t0 * topk : t1 * topk], topk, scale, sink
        )
        diff = (out[t0:t1].float() - ref).abs()
        ok &= bool((diff <= ATOL + RTOL * ref.abs()).all())
        err = max(err, diff.max().item())
    return err, ok


def time_fn(torch, fn, iters, warmup, flush):
    buf = torch.empty(512 << 20, dtype=torch.uint8, device="cuda") if flush else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if buf is not None:
            buf.fill_(i & 0xFF)  # evicts L2 and the 256 MiB Infinity Cache
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(st, en))
    return ts[len(ts) // 2], ts[0]


# ---- main -----------------------------------------------------------------------------------


def run_bench(args):
    import triton  # noqa: F401  # isort: skip  # this HIP image aborts if torch loads triton later
    import flydsl
    import torch

    if args.topk % BLOCK_K:
        sys.exit(f"topk must be a multiple of {BLOCK_K}")
    if args.pool_gib is not None:
        args.pool = int(args.pool_gib * (1 << 30)) // (D * 2)
    elif args.pool is None:
        args.pool = MAX_POOL if args.pattern == "paged" else (1 << 30) // (D * 2)
    if args.pattern == "paged":
        args.pool -= args.pool % args.page  # whole blocks: 3640 x 1152 by default
    if not args.topk <= args.pool <= MAX_POOL:
        sys.exit(
            f"pool={args.pool} rows must be in [topk, {MAX_POOL}]: int32 element offsets"
        )

    torch.set_default_device("cuda")
    props = torch.cuda.get_device_properties(0)
    print("=== latency: one list per group of G tokens ===")
    print(
        f"device: {props.name} ({props.gcnArchName}), CUs={props.multi_processor_count}"
    )
    print(
        f"flydsl: {getattr(flydsl, '__version__', '?')} ({Path(flydsl.__file__).parent})"
    )
    print(f"H={H} D={D} topk={args.topk} sink={args.sink} flush={args.flush}")
    if args.pattern == "paged":
        print(f"pattern=paged: {args.pool:,} rows = {args.pool // args.page} blocks x {args.page}, "
              f"{args.reqs} request(s) x {args.seq_len:,}-token prompt, chunk = last T/{args.reqs} positions")  # fmt: skip
    else:
        print("pattern=uniform: every list drawn from the whole pool")
    print()
    hdr = (f"{'T':>6} {'G':>2} {'pool GiB':>8} {'lists':>6} {'kernel':<10} {'median us':>10} {'min us':>9} "
           f"{'TFLOP/s':>8} {'KV MB':>9} {'TB/s':>6} {'max|err|':>9} {'acc':>4}")  # fmt: skip
    print(hdr)
    print("-" * len(hdr))

    gen = torch.Generator(device="cuda")
    scale = D**-0.5
    pool_gib = args.pool * D * 2 / (1 << 30)
    failed = False
    for n_tok in args.tokens:
        gen.manual_seed(args.seed)
        q = torch.randn(n_tok, H, D, generator=gen).to(torch.bfloat16)
        kv = torch.randn(args.pool, D, generator=gen).to(torch.bfloat16)
        sink = torch.randn(H, generator=gen) if args.sink else None
        flop = 4.0 * n_tok * H * D * args.topk
        io_b = 2 * n_tok * H * D * 2
        for g in args.groups:
            if args.pattern == "paged":
                idx, n_lists = make_paged_indices(
                    torch,
                    n_tok,
                    g,
                    args.pool,
                    args.topk,
                    args.seq_len,
                    args.page,
                    args.reqs,
                    gen,
                )
            else:
                idx, n_lists = make_group_indices(
                    torch, n_tok, g, args.pool, args.topk, args.lists, gen
                )
            uniq = torch.unique(idx)
            pages = (
                torch.unique(uniq // args.page).numel()
                if args.pattern == "paged"
                else None
            )
            print(f"# T={n_tok} G={g}: {uniq.numel():,} distinct rows = {uniq.numel() * D * 2 / (1 << 20):.1f} MiB"
                  + (f" in {pages} pages" if pages is not None else ""))  # fmt: skip
            del uniq
            indptr = torch.arange(
                0, (n_tok + 1) * args.topk, args.topk, dtype=torch.int32
            )
            runs = [(1, "triton", cand_triton)] if args.triton else []
            runs += [(1, "fly_one_q", cand_one_q), (1, "fly_pp", cand_pp)]
            runs += [
                (bq, f"fly_bq{bq}", CANDIDATES[f"fly_bq{bq}"])
                for bq in (2, 4)
                if g % bq == 0
            ]
            for bq, name, make in runs:
                fn = make(q, kv, idx, indptr, sink, args.topk, scale)
                if fn is None:
                    print(
                        f"{n_tok:>6} {g:>2} {pool_gib:>8.2f} {n_lists:>6} {name:<10}   (not applicable)"
                    )
                    continue
                if args.check_tokens:
                    err, ok = check(
                        torch, fn, q, kv, idx, args.topk, sink, scale, args.check_tokens
                    )
                    acc = "PASS" if ok else "FAIL"
                    failed |= not ok
                else:
                    err, acc = float("nan"), "skip"
                med, mn = time_fn(torch, fn, args.iters, args.warmup, args.flush)
                kv_b = (
                    -(-n_tok // bq) * args.topk * D * 2
                )  # rows streamed: one list per BQ tile
                print(
                    f"{n_tok:>6} {g:>2} {pool_gib:>8.2f} {n_lists:>6} {name:<10} {med:>10.1f} {mn:>9.1f} "
                    f"{flop / med / 1e6:>8.1f} {kv_b / 1e6:>9.1f} "
                    f"{(kv_b + io_b) / med / 1e6:>6.2f} {err:>9.2e} {acc:>4}"
                )
            print()
            del idx, indptr
            torch.cuda.empty_cache()
    if failed:
        print(f"accuracy FAIL: some kernel exceeded atol={ATOL} + rtol={RTOL}*|ref|")
        sys.exit(1)


def main():
    args = parse_args()
    os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")
    if args.stats_only is not None:  # meta subprocess: env already set by print_meta
        stats_only(args.stats_only, args.mix)
        return
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)

    def git(*a):
        r = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), *a],
            capture_output=True,
            text=True,
            check=False,
        )
        return r.stdout.strip() or "?"

    print(
        f"using GPU {args.gpu}   branch {git('rev-parse', '--abbrev-ref', 'HEAD')} @ {git('rev-parse', '--short', 'HEAD')}"
    )
    print()
    if args.meta:
        print("=== compiled-kernel resources (VGPR / LDS / occupancy) ===")
        print_meta(args)
        print()
    run_bench(args)


if __name__ == "__main__":
    main()
