# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the HSTU forward at the SILOTIGER-1337 shapes: ping-pong vs shipping vs CK.

Three providers, all run on the same tensors and the same seeded jagged lengths:

  ``pingpong``  the FlyDSL ping-pong forward, by default at the geometry its
                dispatcher picks for the mask (hstu: 512,32; windowed: 256,32;
                8 waves, 2 waves/EU). --pp-tile pins one BLOCK_M,BLOCK_N for all.
  ``flydsl``    the shipping FlyDSL forward, at the tile its tuned CSV selects.
  ``ck``        the CK-tile forward binary (tile_example_hstu_attention_fwd).

Protocol (SILOTIGER-1337): production ``hstu`` mask -- causal with the target
diagonal, max_attn_len=0, contextual_seq_len=0 -- plus a ``windowed`` mask that
is the same but with max_attn_len=W (CK: -local_len=W), W=512 unless --window
lists other sizes. Sparsity 0.95, bf16, N=16384,
d_qk=d_v=64, targets 10 for Training and 300 for Inference. One pass is 3
warmup launches then 11 timed launches. The cell reports the median over
``--reps`` passes (default 3), with the passes interleaved across providers so
card drift lands on all of them alike. Run on one idle GPU via
HIP_VISIBLE_DEVICES.

Timing statistics: a FlyDSL pass reports the median of its 11 launches. CK
prints one mean per process over its 11 timed launches (the local patch honours
HSTU_WARMUP / HSTU_ITERS), so its pass figure is that mean. The per-cell figure
is the median of pass figures for every provider.

Lengths come from one generator and the CK command is built from the same
tensor. CK's -seqlens is the uih length while aiter's lengths already include
the targets, so CK receives lengths - targets with -context_len=0.

Usage (from the repo root, so the ``op_tests`` import resolves):

    # Every cell, all three providers
    python -m op_tests.op_benchmarks.flydsl.bench_hstu_fwd

    # Windowed mask (max_attn_len=512) only
    python -m op_tests.op_benchmarks.flydsl.bench_hstu_fwd --mask windowed

    # Windowed mask, sweeping the window size
    python -m op_tests.op_benchmarks.flydsl.bench_hstu_fwd --mask windowed \
        --window 32 64 128 256 512 1024 2048 4096 --no-ck

    # Inference cells only, FlyDSL arms only
    python -m op_tests.op_benchmarks.flydsl.bench_hstu_fwd --regime Inference --provider pingpong flydsl

    # Written to CSV
    python -m op_tests.op_benchmarks.flydsl.bench_hstu_fwd -o fwd.csv

CK needs the forward binary built with the HSTU_WARMUP/HSTU_ITERS patch:

    cd 3rdparty/composable_kernel && mkdir -p build && cd build
    cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
          -G Ninja -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" ..
    ninja -j 50 tile_example_hstu_attention_fwd

The binary location can be overridden with CK_HSTU_FWD_BIN.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import aiter
from aiter.ops.flydsl.hstu_attention import flydsl_hstu_attention_fwd, fwd_pp_tiles
from aiter.ops.flydsl.kernels.hstu.hstu_attention_fwd_pp import (
    build_hstu_attention_fwd_pp,
)

REPO = Path(__file__).resolve().parents[3]
CK_BIN = Path(
    os.environ.get(
        "CK_HSTU_FWD_BIN",
        REPO / "3rdparty/composable_kernel/build/bin/tile_example_hstu_attention_fwd",
    )
)
CK_SRC = REPO / (
    "3rdparty/composable_kernel/example/ck_tile/18_hstu_attention/"
    "example_hstu_attention_fwd.cpp"
)

# Fixed across every cell.
SEQ_LEN = 16384
HEAD_DIM = 64
DTYPE = torch.bfloat16
# Length-distribution knob, NOT model sparsity: `length = uniform[1, N] * 0.95`.
SPARSITY = 0.95
SEED = 1001
ALPHA = 1.0 / HEAD_DIM * 10000
WARMUP = 3
ITERS = 11

# mask name -> default max_attn_len (0 = no window). --window overrides the
# windowed mask's size, and may list several.
MASKS = {"hstu": 0, "windowed": 512}

# (regime, batch, heads, target_size). Regime names are the ticket's; both are
# forward measurements here. Every cell runs once per selected mask.
CELLS = [
    ("Training", 1024, 4, 10),
    ("Training", 1024, 8, 10),
    ("Inference", 120, 4, 300),
    ("Inference", 120, 8, 300),
]

PROVIDERS = ("pingpong", "flydsl", "ck")
LABELS = {"pingpong": "FlyDSL ping-pong", "flydsl": "FlyDSL shipping", "ck": "CK"}
BASELINE = "flydsl"
CK_TIMING = re.compile(r"execution time .*? is ([0-9.]+) milli-seconds")

def make_inputs(b, h, target_size):
    """Jagged q/k/v, offsets, targets and lengths. One seed fixes all of them."""
    g = torch.Generator(device="cuda").manual_seed(SEED)
    lengths = torch.randint(1, SEQ_LEN + 1, (b,), device="cuda", generator=g)
    lengths = (lengths.float() * SPARSITY).clamp(min=1.0).to(torch.int64)
    offsets = torch.zeros(b + 1, dtype=torch.int64, device="cuda")
    offsets[1:] = torch.cumsum(lengths, dim=0)
    total = int(offsets[-1].item())
    x = torch.empty((total, h, 3 * HEAD_DIM), dtype=DTYPE, device="cuda").uniform_(
        -0.01, 0.01, generator=g
    )
    q, k, v = (t.contiguous() for t in torch.split(x, HEAD_DIM, dim=-1))
    num_targets = torch.minimum(
        torch.full((b,), target_size, device="cuda"), lengths
    ).to(torch.int32)
    return q, k, v, offsets, num_targets, lengths


def time_pass(fn) -> np.ndarray:
    """Per-launch latencies in ms for one pass of WARMUP + ITERS launches.

    Each launch has its own event pair and the pass syncs once at the end, so
    launches stay pipelined. The warmup absorbs JIT compile, which is never timed.
    """
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])


def run_ck(b, h, lengths, num_targets, max_attn_len) -> float:
    """One CK process = one pass. Returns its reported mean over the timed launches."""
    uih = (lengths - num_targets.to(lengths.dtype)).tolist()
    cmd = [
        str(CK_BIN), "-prec=bf16", f"-b={b}", f"-nhead={h}",
        "-hdim_qk=64", "-hdim_v=64", "-jagged=1",
        "-v=0", "-perf=1", "-causal=1",
        f"-local_len={max_attn_len}", "-context_len=0", "-minfull_len=0",
        f"-alpha={ALPHA}",
        "-seqlens=" + ",".join(map(str, uih)),
        "-targets=" + ",".join(map(str, num_targets.tolist())),
    ]  # fmt: skip
    env = dict(os.environ, HSTU_WARMUP=str(WARMUP), HSTU_ITERS=str(ITERS))
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600, env=env, check=False
    )
    match = CK_TIMING.search(proc.stdout)
    if not match:
        raise RuntimeError(
            f"CK produced no timing (rc={proc.returncode})\n"
            f"stdout: {proc.stdout.strip()[-500:]}\n"
            f"stderr: {proc.stderr.strip()[-500:]}"
        )
    return float(match.group(1))


def bench_cell(mask, max_attn_len, regime, b, h, target_size, providers, reps, pp_tile):
    """Time the selected providers on one cell. Returns one row per provider."""
    q, k, v, offsets, num_targets, lengths = make_inputs(b, h, target_size)
    total = int(offsets[-1].item())
    out = torch.empty_like(v)
    stream = torch.cuda.current_stream(q.device)
    fwd_args = (SEQ_LEN, ALPHA, q, k, v, offsets, True, num_targets, max_attn_len, 0)

    launch = {}
    if "flydsl" in providers:
        launch["flydsl"] = lambda: flydsl_hstu_attention_fwd(*fwd_args)
    if "pingpong" in providers:
        tiles = fwd_pp_tiles(max_attn_len)
        if pp_tile is not None:
            tiles["block_m"], tiles["block_n"] = pp_tile
        pp = build_hstu_attention_fwd_pp(
            num_heads=h, head_dim=HEAD_DIM, hidden_dim=HEAD_DIM, causal=True,
            max_attn_len=max_attn_len, contextual_seq_len=0, has_targets=True,
            alpha=ALPHA, dtype_str="bf16", **tiles,
        )  # fmt: skip
        launch["pingpong"] = lambda: pp(
            SEQ_LEN, b, q, k, v, offsets, num_targets, out, stream
        )  # fmt: skip

    # Each pass is (median, min) over its launches. CK's pass is one mean.
    passes = {p: [] for p in providers}
    for rep in range(reps):
        order = providers if rep % 2 == 0 else providers[::-1]
        for prov in order:
            if prov == "ck":
                value = run_ck(b, h, lengths, num_targets, max_attn_len)
                passes[prov].append((value, value))
            else:
                times = time_pass(launch[prov])
                passes[prov].append((float(np.median(times)), float(times.min())))

    medians = {p: float(np.median([m for m, _ in passes[p]])) for p in providers}
    mins = {p: float(min(x for _, x in passes[p])) for p in providers}
    base = medians.get(BASELINE, float("nan"))
    ck = medians.get("ck", float("nan"))

    rows = []
    for prov in providers:
        rows.append({
            "mask": mask, "regime": regime, "B": b, "H": h, "targets": target_size,
            "provider": LABELS[prov], "L": total,
            "median_ms": medians[prov], "min_ms": mins[prov],
            "vs shipping": base / medians[prov],
            "vs CK": ck / medians[prov],
        })  # fmt: skip

    del q, k, v, offsets, num_targets, lengths, out, launch, fwd_args
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def format_table(df):
    """One row per shape, one ms column per provider, like the SILOTIGER-1337 tables.

    Ratios are time ratios (a / b): below 1.00 means the first provider is faster.
    The fastest provider in each row is bold.
    """
    short = {"pingpong": "ping-pong", "flydsl": "shipping", "ck": "CK"}
    present = [p for p in PROVIDERS if LABELS[p] in set(df["provider"])]
    ratios = [("pingpong", "flydsl")] if {"pingpong", "flydsl"} <= set(present) else []
    cols = (
        ["Shape"]
        + [f"{short[p]} (ms)" for p in present]
        + [f"{short[a]} / {short[b]}" for a, b in ratios]
    )
    keys = ["mask", "regime", "B", "H", "targets"]
    out = []
    for (mask, regime, b, h, _t), g in df.groupby(keys, sort=False):
        ms = {p: float(g[g["provider"] == LABELS[p]]["median_ms"].iloc[0]) for p in present}
        best = min(ms.values())
        kind = "train" if regime == "Training" else "infer"
        shape = f"{kind} B={b} H={h}" + (f" {mask}" if mask != "hstu" else "")
        row = [shape]
        for p in present:
            cell = f"{ms[p]:.1f}"
            row.append(f"**{cell}**" if ms[p] == best else cell)
        row += [f"{ms[a] / ms[bb]:.2f}" for a, bb in ratios]
        out.append(row)
    return pd.DataFrame(out, columns=cols).to_markdown(index=False, disable_numparse=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="bench_hstu_fwd",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark the HSTU forward at the SILOTIGER-1337 shapes "
        "(N=16384, d=64, bf16, hstu mask): FlyDSL ping-pong vs FlyDSL shipping vs CK.",
    )
    parser.add_argument(
        "--provider", nargs="*", default=list(PROVIDERS), choices=list(PROVIDERS),
        help="""Providers to time.
        e.g.: --provider pingpong flydsl""",
    )  # fmt: skip
    parser.add_argument(
        "--mask", nargs="*", default=list(MASKS), choices=list(MASKS),
        help="""Masks to run: hstu (max_attn_len=0) and/or windowed (max_attn_len=512,
        or the sizes given by --window).
        e.g.: --mask windowed""",
    )  # fmt: skip
    parser.add_argument(
        "--window", type=int, nargs="+", default=[MASKS["windowed"]],
        help="""max_attn_len values for the windowed mask; each one is a full set
        of cells. Default 512.
        e.g.: --window 32 64 128 256 512 1024 2048 4096""",
    )  # fmt: skip
    parser.add_argument(
        "--no-ck", action="store_true",
        help="""Skip the CK provider (same as leaving ck out of --provider).""",
    )  # fmt: skip
    parser.add_argument(
        "--regime", nargs="*", default=["Training", "Inference"],
        choices=["Training", "Inference"],
        help="""Regimes to run. These name B=1024 (targets 10) and B=120 (targets 300).
        e.g.: --regime Inference""",
    )  # fmt: skip
    parser.add_argument(
        "-H", "--heads", type=int, nargs="*", default=[4, 8],
        help="""Head counts to keep.
        e.g.: -H 4""",
    )  # fmt: skip
    parser.add_argument(
        "--reps", type=int, default=1,
        help="""Passes of (3 warmup + 11 timed) per provider. The cell reports the
        median over passes. Default 1.""",
    )  # fmt: skip
    parser.add_argument(
        "--pp-tile", default=None,
        help="""Ping-pong BLOCK_M,BLOCK_N for every mask. Default: the dispatcher's
        geometry per mask -- 512,32 for hstu, 256,32 for windowed (a taller tile
        sweeps more out-of-window KV under a window).""",
    )  # fmt: skip
    parser.add_argument(
        "-o", type=str, default=None,
        help="""Also write the full results to this CSV (incl. L and min_ms).""",
    )  # fmt: skip
    return parser.parse_args()


def main():
    args = parse_args()
    providers = [
        p for p in PROVIDERS if p in args.provider and not (args.no_ck and p == "ck")
    ]

    if "ck" in providers:
        if not CK_BIN.exists():
            aiter.logger.warning("CK forward binary missing at %s; dropping ck", CK_BIN)
            providers.remove("ck")
        elif "HSTU_WARMUP" not in CK_SRC.read_text():
            aiter.logger.warning(
                "CK source lacks the HSTU_WARMUP/HSTU_ITERS patch; CK will run its "
                "hardcoded iteration count with no warmup"
            )
    if not providers:
        return

    # The shipping arm must not pick up the ping-pong path from the environment.
    os.environ.pop("AITER_HSTU_FWD_PINGPONG", None)
    pp_tile = tuple(int(x) for x in args.pp_tile.split(",")) if args.pp_tile else None

    # (label, max_attn_len) per mask instance; a swept window gets one per size.
    masks = []
    if "hstu" in args.mask:
        masks.append(("hstu", MASKS["hstu"]))
    if "windowed" in args.mask:
        sweep = args.window != [MASKS["windowed"]]
        masks += [(f"windowed W={w}" if sweep else "windowed", w) for w in args.window]
    cells = [
        (m, w, *c)
        for m, w in masks
        for c in CELLS
        if c[0] in args.regime and c[2] in args.heads
    ]
    rows = []
    for index, (mask, max_attn_len, regime, b, h, target_size) in enumerate(cells, 1):
        tag = f"{mask} {regime} B={b} H={h} targets={target_size}"
        # Printed before the cell runs so a long compile is not silent.
        print(f"[{index}/{len(cells)}] {tag} ... ", end="", flush=True)
        cell_rows = bench_cell(
            mask, max_attn_len, regime, b, h, target_size, providers, args.reps,
            pp_tile,
        )
        rows += cell_rows
        print(
            "  ".join(f"{r['provider']} {r['median_ms']:.1f}" for r in cell_rows),
            flush=True,
        )

    df = pd.DataFrame(rows)
    aiter.logger.info("hstu fwd, all cells (markdown):\n%s", format_table(df))
    if args.o:
        df.to_csv(args.o, index=False)
        aiter.logger.info("results saved to %s", args.o)


if __name__ == "__main__":
    main()
