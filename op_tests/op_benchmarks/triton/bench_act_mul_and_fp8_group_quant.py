# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the fused activation, multiply and FP8 group quantization kernel.

Times with HIP graph replay (``triton.testing.do_bench_cudagraph``), so host launch
cost is not counted. Consecutive launches read different input buffers (a pool larger
than the last-level cache), so the input comes from HBM.
"""

import argparse
import itertools
import sys

import torch
import triton

from aiter.ops.triton.activation import act_mul_and_fp8_group_quant
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_ROWS = (1, 128, 2048, 8192, 32768)
_WIDTHS = (2176, 4352, 8704, 17408)
_POOL_BYTES = 1 << 30


def _input_pool(rows: int, width: int, dtype: torch.dtype) -> list[torch.Tensor]:
    one = rows * 2 * width * torch.tensor([], dtype=dtype).element_size()
    count = max(2, min(64, _POOL_BYTES // max(one, 1)))
    return [
        torch.randn((rows, 2 * width), dtype=dtype, device="cuda") for _ in range(count)
    ]


def _benchmark(rows: int, width: int, metric: str, args):
    dtype_quant = get_fp8_e4m3_dtype()
    pool = _input_pool(rows, width, _DTYPES[args.dtype])
    inputs = itertools.cycle(pool)

    def fn():
        act_mul_and_fp8_group_quant(
            next(inputs), args.activation, args.group_size, dtype_quant
        )

    fn()  # compile outside the graph
    ms = triton.testing.do_bench_cudagraph(fn, rep=args.rep)
    if metric == "time":
        return ms * 1000
    if metric == "bandwidth":
        # Read gate and up; write one FP8 byte per output and one fp32 scale per group.
        scales = rows * triton.cdiv(width, args.group_size)
        x_bytes = pool[0].element_size()
        logical_bytes = rows * width * (2 * x_bytes + 1) + 4 * scales
        return logical_bytes / (ms * 1e-3) * 1e-9
    raise ValueError(f"unknown metric: {metric}")


def run_benchmark(args):
    rows = (args.rows,) if args.rows is not None else _ROWS
    widths = (args.width,) if args.width is not None else _WIDTHS
    unit = "us" if args.metric == "time" else "GB/s"

    # perf_report and benchmark_utils import matplotlib, which the image does not ship.
    lines = [f"rows,width,{args.metric}_{unit}"]
    for n in widths:
        for m in rows:
            value = _benchmark(m, n, args.metric, args)
            lines.append(f"{m},{n},{value:.3f}")
            print(lines[-1], flush=True)
    if args.output:
        name = f"bench_act_mul_and_fp8_group_quant_{args.dtype}_{args.metric}.csv"
        with open(name, "w") as f:
            f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Act-Mul and FP8 Group Quant",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument(
        "--width", type=int, default=None, help="N // 2, the output width"
    )
    parser.add_argument(
        "--activation", choices=("silu", "gelu", "gelu_tanh"), default="silu"
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    parser.add_argument("--metric", choices=("time", "bandwidth"), default="time")
    parser.add_argument("--rep", type=int, default=100, help="ms of graph replay")
    parser.add_argument("-o", "--output", action="store_true")
    return parser.parse_args()


def main():
    run_benchmark(parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
