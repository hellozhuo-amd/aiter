# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""hstu_attention_fwd_pp - ping-pong-pipelined HSTU forward (gfx942)

Same mathematics as `hstu_attention_fwd.py` -- see that module's docstring for the
gate algebra, the GEMM1 operand swap and the mask -- but scheduled so the two
waves resident on each SIMD are held in *opposite* phases:

    phase M : GEMM1, GEMM2, their LDS operand feed, and this group's K/V staging.
    phase V : the SiLU gate and the mask.  No memory operations at all.

8 waves in one workgroup, `waves_per_eu=2`, so wave `i` and wave `i+4` share a
SIMD; a half-iteration skew puts one of that pair in M whenever the other is in
V, and the matrix core and the VALU pipe are then both busy continuously.
Ported from `hstu_attention_bwd_fused_pp.py`.

Geometry is **(512, 32, 8, 2)**. Measured at B=1024 H=4 N=16384 d=64 against the
shipping kernel's tuned (128,32,4,2), bf16, sparsity 0.95:

    shipping tuned (128,32,4,2)   146.3 ms
    pp  8-wave skeleton, no pp    153.1 ms   +4.7%   geometry only, 2 WG/CU
    pp  rotated, lockstep         164.4 ms  +12.4%   1 WG/CU, no skew
    pp  + skew/setprio            154.2 ms   +5.5%   the ping-pong itself: -6.2%
    pp  + vectorize-slp=False     150.9 ms   +3.0%
    pp  staging zero-fold moved   140.6 ms   -3.9%   see async_load_kv_regs
    pp  merged unit stream        132.9 ms   -9.1%   one UNIT_ORDER walk
    pp  + rolling window          131.3 ms  -10.3%
    pp  + V_ROT_DT cross-phase    130.1 ms  -11.1%
    pp  + v_perm pack_p           127.2 ms  -13.1%   see pack_p
    pp  + alpha*-log2e into Q     123.7 ms  -15.0%
    pp  + V slot XOR swizzle      121.0 ms  -17.3%
    pp  + carried staging address 118.3 ms  -19.1%   1.230x

and, as of the V_ROT_DT round, 1.121 / 1.147 / 1.129 / 1.150x over the four
benchmark cells (train H=4/H=8, infer H=4/H=8); the two gate changes since were
measured at the train H=4 cell only. Unlike the backward's fused kernel -- 1.31x
at H=4 but 1.17x at H=8 -- the win here is flat across cells.

**The gate is the remaining cost, and it is mostly two instructions.** From the
ATT trace of the (512,32) kernel: `v_rcp_f32` 25.2% and `v_exp_f32` 25.2% of all
active time, both at 16 cy/execution (quarter rate), against `v_mfma` at 12.6%.
One exp2 and one rcp per gated element is the floor for silu, and masked lanes
cannot skip them under SIMD, so the only lever left there is approximating exp2
-- the fp8 kernel's Schraudolph path, worth ~12.6% of active but with a real
accuracy cost that HSTU, having no softmax denominator to normalise it away,
would absorb directly.

The tile ladder is worth re-reading after any scheduling change: (640,32) led
until the unit stream landed, then lost 15% to 52 spilled registers, because
BLOCK_M=640 means Q_SUBTILES=5 and the burst holds every s_accs live at once.
A sweep run over a stall ranks the stall's interaction, not the tiles.

**This kernel is VALU-bound, not MFMA-bound -- the opposite of the backward.**
Counted from the generated ISA, the unmasked loop body is 32 MFMA (512 cy of
matrix core) against 84 full-rate + 32 quarter-rate VALU (848 cy); the masked
body is 1160 cy of VALU against the same 512. The two transcendentals in the
gate are alone 512 of those cycles. So the ping-pong's ceiling is the VALU, and
every tuning decision below follows from that: longer phases per barrier win
(hence 384/64 over 256/32, measured +12% apart), and anything that costs VALU to
save matrix-core time is a loss.

Two things differ from the backward's port, and both are load-bearing:

* **The duty split is a selected BASE POINTER, not a branch.**  g0 (waves 0-3)
  stages K, g1 (waves 4-7) stages V, but the forward stages
  `global -> VGPR -> ds_write` (K is XOR-swizzled, V is scattered transposed, and
  a `buffer_load_lds` pins its LDS destination, so neither tile can use the
  backward's DMA).  A register destination is the whole point -- it is what lets
  a load span two phases -- and a value defined inside an `scf.if` does not
  dominate the loop's yield.  So the load is unbranched with
  `is_g0.select(k_base, v_base)` folded into the pointer; only the `ds_write`,
  which is side-effect-only, is branched.  This is why the kernel is gated to
  `head_dim == hidden_dim`: the two tiles must have identical geometry for one
  load to serve both.
* **The DMA tiling is derived on GRP_THREADS (256), not BLOCK_THREADS (512).**
  A tile has to be staged entirely by ONE group, because the other group is in
  phase V at the time.  As a side effect this keeps `VEC = 8` (`dwordx4`) at
  BLOCK_N=32, which the shipping kernel loses at 512 threads.

Constraints (hard-gated by `validate_hstu_attention_fwd_pp`):
  - gfx942 (CDNA3)
  - dtype bf16
  - head_dim == hidden_dim, both a multiple of 64
  - num_waves == 8; block_m a multiple of 128 and >= 256
  - fast/unsafe FP math

**Accuracy.** Every SCHEDULING change here is bitwise identical to
`hstu_attention_fwd.py`, and that is what each step of the port was checked
against (`tmp/hstu_repro/check_fwd_pp.py`): the rotation only delays GEMM2 by
one trip, leaving the order in which KV tiles enter the accumulator unchanged;
the skew only reorders publication; and a larger block_m only makes the upper
sub-block sweep KV tiles the smaller one skipped, which are fully masked ->
`s=0 -> g=0 -> exp2(0)=1 -> rcp(2)=0.5 -> p=0` exactly, with `acc + 0*V` exact
in fp32.

`scale_q_pack` is the one exception and is NOT bitwise: folding
`alpha * -log2e` into the Q operand rounds the product to bf16's 8 mantissa
bits, where the per-element form applied it in f32 after the MFMA had
accumulated. Measured, it costs nothing at the output's own precision --
against the fp32 oracle the max error is 1.221e-04 (0.71% of peak), the SAME
figure the shipping kernel reports, across all five mask configurations; and
against the shipping kernel directly the difference tops out at 1.192e-07,
which is 2 ULP of the bf16 output. So `check_fwd_pp.py` now reports a
magnitude rather than a pass/fail, and the binding accuracy test is
`AITER_HSTU_FWD_PINGPONG=1 python tmp/hstu_repro/check_fwd.py` against the
oracle.
"""

import functools
import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from aiter.jit.utils.chip_info import get_lds_capacity_bytes
from aiter.ops.flydsl.kernels.hstu.hstu_attention_common import (
    _LOG2E,
    MFMA_ELEMS_PER_LANE,
    MFMA_K,
    MFMA_LANE_K,
    MFMA_M,
    MFMA_N,
    NUM_GRID_GROUPS,
    WARP_SIZE,
    _arch_dma_params,
    _dtype_to_elem_type,
    _mfma_params_for_dim,
    bind_mfma_accs,
    exp2_f32,
)

# The gate is a ~10-deep serial dependency chain per element with two
# quarter-rate transcendentals in it, and the ISA says it is 848 of the hot
# loop's 1360 issue cycles -- this kernel is VALU-bound, not MFMA-bound. A
# packed `v_pk_*_f32` is a JOIN: it cannot issue until BOTH halves' producers
# retire, so SLP vectorizing the gate halves the number of independent chains
# available to fill the pipe. The backward measured 342.9 ms with SLP off
# against 375.7 with it on, on the same gate shape.
_PP_COMPILE_HINTS = {"llvm_options": {"vectorize-slp": False}}


def phase_barrier():
    """Workgroup barrier that the machine scheduler may not move code across.

    Separates this kernel's MFMA and VALU phases. The ``sched_barrier`` fences
    are load-bearing: ``s_barrier`` alone does not stop pure VALU sinking, so
    part of the gate migrates into the next MFMA phase and the two phases stop
    being disjoint.

    Deliberately not ``gpu.barrier()``, whose memory fence adds an implicit
    ``s_waitcnt lgkmcnt(0)``. This kernel places that wait by hand at the end of
    phase M, where it also has to retire the operand reads before the next
    trip's staging writes -- so paying for it again at every phase boundary,
    including the one after the gate (which touches no memory at all), is pure
    loss.
    """
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def validate_hstu_attention_fwd_pp(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    causal: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
    alpha: float,
    dtype_str: str,
    *,
    block_m: int,
    block_n: int,
    num_waves: int,
    waves_per_eu: int,
    arch: str | None = None,
) -> None:
    if arch is None:
        arch = get_rocm_arch()
    # Narrower than the shipping kernel on purpose. The ping-pong schedule is
    # tuned against CDNA3's 16-cycle `v_mfma_f32_16x16x16_bf16` and its
    # quarter-rate transcendentals; nothing here is portable by construction,
    # so refuse rather than silently mis-schedule.
    if not arch.startswith("gfx942"):
        raise ValueError(
            f"hstu attention fwd pp requires gfx942, got {arch!r} "
            "(use hstu_attention_fwd for other targets)"
        )
    if dtype_str != "bf16":
        raise ValueError(
            f"hstu attention fwd pp requires bf16, got {dtype_str!r} "
            "(use hstu_attention_fwd for f16)"
        )
    # One unbranched load serves both wave groups via a selected base pointer
    # (see the module docstring), which requires the K and V tiles to have
    # identical geometry.
    if head_dim != hidden_dim:
        raise ValueError(
            f"hstu attention fwd pp requires head_dim == hidden_dim, got "
            f"{head_dim} != {hidden_dim}"
        )
    if head_dim % 64 != 0:
        raise ValueError(
            f"hstu attention fwd pp requires head_dim % 64 == 0, got {head_dim}"
        )
    if num_waves != 8:
        raise ValueError(
            f"hstu attention fwd pp requires num_waves == 8 (2 per SIMD inside one "
            f"barrier domain), got {num_waves}"
        )
    # BLOCK_M quantises to num_waves*MFMA_M = 128. 128 leaves the kernel at 120
    # VGPR -> 4 waves/SIMD -> two 8-wave workgroups still co-resident, which
    # breaks the one-barrier-domain premise outright. 256 and 384 are the
    # candidates; there is nothing between them.
    # block_m quantises to num_waves*MFMA_M = 128. 128 itself is excluded: the
    # kernel is light enough there to fit 4 waves/SIMD, so a second workgroup
    # co-resides and the two wave groups stop being the only pair on the SIMD --
    # which is the premise the whole schedule rests on.
    if block_m < 256:
        raise ValueError(
            f"hstu attention fwd pp requires block_m >= 256, got {block_m}. At 128 "
            "the kernel fits 4 waves/SIMD, so a second workgroup co-resides and "
            "the phase alternation no longer owns the SIMD."
        )
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len must be non-negative, got {contextual_seq_len}"
        )
    if max_attn_len < 0:
        raise ValueError(f"max_attn_len must be non-negative, got {max_attn_len}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if not host_math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    mfma_dim_k, _, _ = _mfma_params_for_dim(head_dim, arch)
    if head_dim <= 0 or head_dim % mfma_dim_k != 0:
        raise ValueError(
            f"head_dim must be positive and a multiple of MFMA_K={mfma_dim_k}, got {head_dim}"
        )
    if hidden_dim <= 0 or hidden_dim % MFMA_M != 0:
        raise ValueError(
            f"hidden_dim must be positive and a multiple of MFMA_M={MFMA_M}, got {hidden_dim}"
        )
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if block_n <= 0:
        raise ValueError(f"block_n must be positive, got {block_n}")
    if num_waves <= 0:
        raise ValueError(f"num_waves must be positive, got {num_waves}")
    if waves_per_eu < 0:
        raise ValueError(f"waves_per_eu must be non-negative, got {waves_per_eu}")
    if block_m % (num_waves * MFMA_M) != 0:
        raise ValueError(
            f"block_m {block_m} must be a multiple of num_waves*MFMA_M ({num_waves * MFMA_M})"
        )
    if block_n % MFMA_M != 0:
        raise ValueError(f"block_n {block_n} must be a multiple of MFMA_M={MFMA_M}")

    # A tile is staged by ONE wave group, so every divisibility below is against
    # GRP_THREADS, not BLOCK_THREADS. This is also what keeps VEC = 8 at
    # BLOCK_N=32: the whole-workgroup divisor would make a 32x64 tile 4 elements
    # per thread and collapse the load to dwordx2.
    _, dma_elems, _, _ = _arch_dma_params(arch)
    grp_threads = (num_waves // 2) * WARP_SIZE
    elems_per_dma_pass = grp_threads * dma_elems
    head_dim_k = ((head_dim + 63) // 64) * 64
    if (block_n * head_dim_k) % elems_per_dma_pass != 0:
        raise ValueError("K DMA tile does not divide the group DMA pass evenly")
    if (block_n * hidden_dim) % elems_per_dma_pass != 0:
        raise ValueError("V DMA tile does not divide the group DMA pass evenly")

    # head_dim == hidden_dim and head_dim % 64 == 0 are already enforced, so the
    # K and V tilings are the same and one check covers both.
    kv_dma_wide = hidden_dim % 8 == 0 and (block_n * hidden_dim) % (grp_threads * 8) == 0
    vec_kv = 8 if kv_dma_wide else dma_elems
    threads_per_row_kv = hidden_dim // vec_kv
    if grp_threads % threads_per_row_kv != 0:
        raise ValueError(
            f"grp_threads={grp_threads} must be divisible by threads_per_row={threads_per_row_kv}"
        )
    rows_per_batch_kv = grp_threads // threads_per_row_kv
    if not (block_n % rows_per_batch_kv == 0 or rows_per_batch_kv > block_n):
        raise ValueError(
            f"rows_per_batch={rows_per_batch_kv} must divide block_n={block_n}, "
            "unless rows_per_batch > block_n"
        )
    # The swizzled K store writes in MFMA_LANE_K-element groups, so a load
    # narrower than one group cannot be re-grouped on the way out. This is what
    # rules out e.g. block_n=48 at head_dim=64: 48*64 does not divide an 8-wide
    # group pass, vec falls back to the 2-element arch granule, and 2 % 4 != 0.
    mfma_lane_k = _mfma_params_for_dim(head_dim, arch)[1]
    if vec_kv % mfma_lane_k != 0:
        raise ValueError(
            f"block_n={block_n} gives a {vec_kv}-element staging load, which is "
            f"not a multiple of MFMA_LANE_K={mfma_lane_k}. Pick a block_n for "
            f"which block_n*{hidden_dim} divides {grp_threads * 8} (the 8-wide "
            "group DMA pass)."
        )

    # Counts the TRANSPOSED V pitch (block_n + 8), which the shipping kernel's
    # budget line under-counts.
    lds_cap = get_lds_capacity_bytes(arch)
    lds_bytes = block_n * head_dim_k * 2 + hidden_dim * (block_n + 8) * 2
    if lds_bytes > lds_cap:
        raise ValueError(f"LDS tile {lds_bytes} B exceeds the {lds_cap} B budget")


@functools.lru_cache(maxsize=16384)
def build_hstu_attention_fwd_pp(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    causal: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
    alpha: float,
    dtype_str: str,
    *,
    block_m: int = 512,
    block_n: int = 32,
    num_waves: int = 8,
    waves_per_eu: int = 2,
):
    validate_hstu_attention_fwd_pp(
        num_heads,
        head_dim,
        hidden_dim,
        causal,
        max_attn_len,
        contextual_seq_len,
        has_targets,
        alpha,
        dtype_str,
        block_m=block_m,
        block_n=block_n,
        num_waves=num_waves,
        waves_per_eu=waves_per_eu,
    )

    BLOCK_M = block_m
    BLOCK_N = block_n
    NUM_WAVES = num_waves
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    # ---- ping-pong groups -------------------------------------------------
    # g0 = waves 0..3 stages K; g1 = waves 4..7 stages V. Wave i and wave i+4
    # share a SIMD, which is the pair the phase alternation (and `s_setprio`)
    # arbitrates. A tile must be staged entirely by one group, because the other
    # group is in phase V and will not reach the ds_write.
    #
    # The split is UNBALANCED and that is deliberate. K costs 2 x ds_write_b64
    # (12.3 cy each) against V's transposed 8 x ds_write_b16 (8.0 cy), so g1's
    # phase M carries 64 cy of LDS writes to g0's 24.6 -- visible in the trace
    # as g0 WAIT 18.3% against g1's 14.8%, a 3.5 pp gap the barrier turns into
    # idle time.
    #
    # Rebalancing was built and measured: each group staging HALF THE ROWS of
    # BOTH tensors gives 1 x b64 + 4 x b16 = 44.3 cy each, a 31% cut to the
    # critical path, and it drops 23 VGPR (246 -> 223). It is still 0.8% SLOWER
    # -- 121.9 against 121.0. Halving the rows halves VEC_KV, so the global load
    # goes from one dwordx4 to two dwordx2 against two different tile offsets,
    # and the extra address math plus the doubled load count costs more than the
    # LDS imbalance was worth. The freed registers do not unlock a larger tile
    # either: (640,32) still spills 13 and (384,64) 21.
    #
    # So the imbalance stands, and its ceiling is small -- half of 3.5 pp.
    GRP_WAVES = NUM_WAVES // 2
    GRP_THREADS = GRP_WAVES * WARP_SIZE
    assert GRP_THREADS == 256
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    Q_SUBTILES = ROWS_PER_WAVE // MFMA_M
    KV_SUBTILES = BLOCK_N // MFMA_N
    WAVES_PER_EU = waves_per_eu
    MFMA_DIM_K, MFMA_DIM_LANE_K, MFMA_DIM_LANE_K_LOG2 = _mfma_params_for_dim(head_dim)

    assert num_waves > 0 and block_m % (num_waves * MFMA_M) == 0
    assert block_n % MFMA_M == 0
    assert head_dim % MFMA_DIM_K == 0
    assert hidden_dim % MFMA_M == 0

    # Arch-conditional DMA width + K LDS swizzle geometry (gfx942 dword / gfx950 dwordx4).
    DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = _arch_dma_params()  # noqa: RUF059

    elem_dtype = _dtype_to_elem_type(dtype_str)
    is_bf16 = dtype_str == "bf16"
    has_window = max_attn_len > 0
    has_contextual = contextual_seq_len > 0

    # QK contracts with the architecture-native dimension-axis MFMA.
    K_STEPS = head_dim // MFMA_DIM_K
    # k_swz_col has period 64, so a K stride < 64 swizzles out of row and corrupts GEMM2's A-operand.
    # Round the K stride up to 64; the extra columns over-fetch (buffer bounds -> 0) against a zero Q
    # operand and contribute nothing. head_dim % 64 == 0 leaves HEAD_DIM_K == head_dim. HEAD_DIM_K
    # doubles as the K LDS row pitch (64-aligned, so the XOR swizzle stays in-row).
    HEAD_DIM_K = ((head_dim + 63) // 64) * 64
    # Column over-fetch guard is only real when padded (HEAD_DIM_K > head_dim); otherwise the lane
    # column max is < head_dim, so it is compile-time true and dropped (else a live runtime select).
    K_COL_GUARD = head_dim < HEAD_DIM_K
    # padded contraction steps (K side); always a multiple of 4
    K_STEPS_K = HEAD_DIM_K // MFMA_DIM_K
    # GEMM2 output chunks (per-lane O accumulators)
    D_CHUNKS = hidden_dim // MFMA_M

    # V LDS: store V transposed as [d, kv] so GEMM2's B-operand (4 consecutive kv for a d) is
    # contiguous -> one ds_read_b64 rather than 4x ds_read_u16. The +8 pads the [d, kv] row stride
    # to break LDS bank conflicts.
    V_T_STRIDE = BLOCK_N + 8

    # ---- Staging tiling: ONE tiling, derived on GRP_THREADS ----------------
    # K and V are the same shape here (head_dim == hidden_dim, both % 64 == 0,
    # so HEAD_DIM_K == head_dim and there are no pad columns), which is exactly
    # what lets one unbranched load serve both groups through a selected base
    # pointer. Deriving on GRP_THREADS rather than BLOCK_THREADS is required --
    # the partner group is in phase V and never reaches the ds_write -- and it
    # also keeps the load 8-wide at BLOCK_N=32, which the whole-workgroup
    # divisor would collapse to dwordx2.
    assert HEAD_DIM_K == head_dim == hidden_dim
    elems_per_dma_pass = GRP_THREADS * DMA_ELEMS
    assert (BLOCK_N * hidden_dim) % elems_per_dma_pass == 0

    kv_dma_wide = hidden_dim % 8 == 0 and (BLOCK_N * hidden_dim) % (GRP_THREADS * 8) == 0
    VEC_KV = 8 if kv_dma_wide else DMA_ELEMS
    THREADS_PER_ROW_KV = hidden_dim // VEC_KV
    assert GRP_THREADS % THREADS_PER_ROW_KV == 0
    ROWS_PER_BATCH_KV = GRP_THREADS // THREADS_PER_ROW_KV
    assert BLOCK_N % ROWS_PER_BATCH_KV == 0 or ROWS_PER_BATCH_KV > BLOCK_N
    NUM_BATCHES_KV = max(1, BLOCK_N // ROWS_PER_BATCH_KV)
    KV_NEEDS_GUARD = ROWS_PER_BATCH_KV > BLOCK_N
    assert VEC_KV % MFMA_DIM_LANE_K == 0
    # K_COL_GUARD is dead here (HEAD_DIM_K == head_dim is asserted above), so the
    # over-fetch select the shipping kernel carries never materialises.
    assert not K_COL_GUARD

    # LDS map: [K tile][V tile]. K is XOR-swizzled by column ([BLOCK_N, HEAD_DIM_K]); V is stored
    # transposed ([hidden_dim, V_T_STRIDE]) so GEMM2's B-operand reads 4 consecutive kv per
    # ds_read_b64. Each field is a 16B-aligned fx.Array; SharedAllocator sizes the static LDS
    # global for us (no manual _align / finalize / get_base).
    # ---- Rotation slots ----------------------------------------------------
    # The chained GEMM consumes the PREVIOUS tile's P, so phase M(i) runs
    # GEMM1(i) beside GEMM2(i-1) and the two GEMMs merge into one contiguous
    # MFMA burst. That makes three tiles' worth of LDS live at once, and the
    # staging offsets are asymmetric to keep it to two slots each:
    #
    #   M(i) reads   K(i)   [kslot u%2]      and V(i-1) [vslot (u+1)%2]
    #   M(i) writes  K(i+1) [kslot (u+1)%2]  and V(i)   [vslot u%2]
    #
    # Staging V one tile ahead as well -- the natural "deepen the pipeline"
    # move -- would put the V(i+1) write and the V(i-1) read in the SAME slot
    # and force three V slots, hence UNROLL = lcm(2,3) = 6. Depth is bought in
    # REGISTERS instead (see `async_load_kv_regs`), which costs no slot at all.
    #
    # One fx.Array field per slot, never one array with a runtime slot offset:
    # that is what keeps every access provably inside a single LDS global.
    K_SLOTS = 2
    V_SLOTS = 2
    # The body is unrolled by the slot count so each sub-trip names a LITERAL
    # slot. Slots are keyed on the sub-trip position, not the absolute tile, so
    # the parity is independent of the (runtime) kv_tile_start -- which is also
    # why the unmasked loop's trip count has to be a multiple of UNROLL.
    UNROLL = 2
    assert UNROLL % K_SLOTS == 0 and UNROLL % V_SLOTS == 0

    # ---- The MFMA burst's unit stream -------------------------------------
    # A UNIT is one `ds_read` plus the Q_SUBTILES MFMAs that consume it:
    #   QK unit j -> (ng, ks) = (j % KV_SUBTILES, j // KV_SUBTILES)
    #   PV unit j -> (c,  ng) = (j % D_CHUNKS,    j // D_CHUNKS)
    # Both encodings are chosen so CONSECUTIVE units of the same burst touch
    # different accumulators -- QK chains on s_accs[ng], PV on o_acc[c] -- which
    # is what keeps the dependent links of each chain far apart.
    N_QK_UNITS = KV_SUBTILES * K_STEPS_K
    N_PV_UNITS = D_CHUNKS * KV_SUBTILES
    # head_dim == hidden_dim is gated, so K_STEPS_K == D_CHUNKS and the two
    # bursts always have the same unit count: the interleave is exactly 1:1.
    assert N_QK_UNITS == N_PV_UNITS
    N_UNITS = N_QK_UNITS + N_PV_UNITS

    # The two GEMMs are INTERLEAVED, not concatenated. GEMM1 writes s_accs and
    # GEMM2 writes o_acc and they share no register, so alternating them hands
    # the matrix core two independent accumulator chains to fill its 16-cycle
    # issue window with instead of one dependent chain. The compiler finds this
    # on its own today; stating it keeps it true once the rolling window below
    # pins the order. Set to QK-then-PV for the fp8 kernel's sequential form.
    # V slot rotation, ported from the fp8 kernel's V_ROT_DT (traits.py:141-147).
    # (`op_gemm.py`): the MFMAs at step `s` consume the pack read at step
    # `s - PREFETCH_DEPTH`, so no MFMA ever waits on its own `ds_read`. Each
    # unit is Q_SUBTILES MFMAs ~= Q_SUBTILES*16 cycles, so depth 2 buys roughly
    # 2*Q_SUBTILES*16 cycles of cover against a ~130-cycle LDS round trip.
    # Depth is nearly flat: 130.69 / 130.66 / 130.51 / 130.70 / 131.42 ms at
    # depth 1 / 2 / 3 / 4 / 6 (512,32). Two is the fp8 kernel's choice and the
    # cheapest that ties; past four the window's own registers start costing.
    PREFETCH_DEPTH = 2
    # Chained across the phase barrier too -- see XPHASE_DEPTH and V_ROT_DT.
    #
    # The naive form of that chain costs a THIRD slot per tensor, because what
    # the next phase reads is what this phase writes, plus UNROLL 2 -> 3. Built
    # that way first and measured 131.7 ms against 131.3: the mechanism worked
    # (the ISA showed the first MFMA after the barrier issuing with no
    # `s_waitcnt` at all) but +10 VGPR and +9 KB of LDS ate the gain. With the
    # rotation it costs nothing instead -- VGPRs go DOWN, LDS is unchanged --
    # and the same mechanism is worth -0.9%. The lesson is that the chain was
    # never the problem; paying a slot for it was.
    # Ring slots, indexed by STEP and never by unit value: UNIT_ORDER is a
    # permutation, so keying on the unit would let a later unit's read land in a
    # slot whose occupant has not been consumed yet -- silently wrong results.
    # Keyed by step the reuse distance is exactly RING against a PREFETCH_DEPTH
    # lookahead, so any depth below RING is correct.
    WIN_RING = 4
    assert 0 <= PREFETCH_DEPTH < WIN_RING <= N_UNITS

    # A V slot is rotated by one d-chunk against the tile grid:
    #
    #     vslot p  holds  d-chunks 1..D-1 of tile p   AND   d-chunk 0 of tile p+1
    #
    # which is what lets TWO buffers serve three roles. In M(i): PV reads tile
    # i-1's chunks 1..D-1 out of vslot (i-1)%2, its chunk 0 comes from registers
    # the previous phase's tail prefetched, and the tail prefetches chunk 0 of
    # tile i -- which also lives in vslot (i-1)%2. So every read touches the slot
    # being READ and never the one being written (vslot i%2). Without the
    # rotation the tail's target IS the slot under write, and the chain costs a
    # third slot plus UNROLL 2 -> 3.
    #
    # Free in the store: the rotation says which TILE a lane loads, not where it
    # lands, so only the staging load's row gets a per-lane select. See
    # `kv_stage_tile`.
    V_ROT_DT = 1

    # ---- Phase V: how many score elements the gate pipelines at once --------
    # The gate is exp2 -> add -> rcp -> mul, four SERIALLY DEPENDENT ops per
    # element, and the only thing hiding each one's latency is the other
    # elements in flight. The natural batch is one MFMA accumulator --
    # MFMA_ELEMS_PER_LANE = 4 -- and off the trace that is too narrow: the
    # emitted stream is `exp x4 | add x4 | rcp x4 | mul x4 | perm x2` and every
    # transcendental after the first in its run costs 20 cy against a 16 cy
    # issue rate, with another 4 cy at each opcode transition. ~260 cy of a
    # 1603 cy phase.
    #
    # Widening costs registers: a batch of B holds B scores plus B
    # intermediates live at once. 4 -> 8 -> 16 -> 32 (one whole tile).
    GATE_BATCH = 4
    assert GATE_BATCH % MFMA_ELEMS_PER_LANE == 0


    # The cross-phase head must be d-chunk-0 PV units ONLY. The rotation covers
    # exactly V_ROT_DT d-chunks, so a QK unit in the head would need K rotated
    # too -- which the interleaved order would otherwise force, since it
    # alternates from unit 0. Leading with the chunk-0 PV units keeps the port
    # to one rotated tensor, exactly as the fp8 kernel has it.
    XPHASE_DEPTH = PREFETCH_DEPTH
    _PV_HEAD = [j for j in range(N_PV_UNITS) if (j % D_CHUNKS) < V_ROT_DT]
    assert len(_PV_HEAD) >= XPHASE_DEPTH, (
        f"V_ROT_DT={V_ROT_DT} covers {len(_PV_HEAD)} PV units, need {XPHASE_DEPTH}"
    )
    _HEAD = [("pv", j) for j in _PV_HEAD[:XPHASE_DEPTH]]
    # The rest keeps the 1:1 QK/PV alternation. Its only cost is at the head:
    # the two leading PV units share c=0, so they chain on o_acc[0] two MFMAs
    # apart. Measured against the alternative of rotating K as well, that is the
    # cheap side.
    _REST_PV = [("pv", j) for j in range(N_PV_UNITS) if ("pv", j) not in _HEAD]
    _REST_QK = [("qk", j) for j in range(N_QK_UNITS)]
    UNIT_ORDER = _HEAD + [
        spec for pair in zip(_REST_QK, _REST_PV + [None] * len(_REST_QK))
        for spec in pair if spec is not None
    ]
    assert len(UNIT_ORDER) == N_UNITS, (len(UNIT_ORDER), N_UNITS)
    assert sorted(UNIT_ORDER) == sorted(
        [("qk", j) for j in range(N_QK_UNITS)]
        + [("pv", j) for j in range(N_PV_UNITS)]
    ), "UNIT_ORDER must be a permutation of the units"
    # Every cross-phase head unit must be a chunk-0 PV unit, or the rotation
    # does not cover it.
    assert all(
        k == "pv" and (j % D_CHUNKS) < V_ROT_DT
        for k, j in UNIT_ORDER[:XPHASE_DEPTH]
    )

    # Rolling operand window, ported from the gfx950 fp8 ping-pong FMHA kernel

    @fx.struct
    class SharedStorage:
        k0: fx.Array[elem_dtype, BLOCK_N * HEAD_DIM_K, 16]
        k1: fx.Array[elem_dtype, BLOCK_N * HEAD_DIM_K, 16]
        v0: fx.Array[elem_dtype, hidden_dim * V_T_STRIDE, 16]
        v1: fx.Array[elem_dtype, hidden_dim * V_T_STRIDE, 16]

    # ---- Device Kernel ----
    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_attention_fwd_pp_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        out: fx.Tensor,
        num_q_tiles: fx.Int32,
        hz_per_group: fx.Int32,
        hz_total: fx.Int32,
        inv_n: fx.Float32,
    ) -> None:
        compute_type = fx.Float32.ir_type
        c_zero_dim_pack = Vec.filled(MFMA_DIM_LANE_K, 0.0, elem_dtype).ir_value()

        # QK uses gfx942's 16x16x16 or gfx950's native 16x16x32. P*V remains
        # 16-deep because P is reused directly from the four-value C fragment.
        # Equal shapes share one atom (the gfx942 path).
        dim_mfma_acc, mfma_acc = bind_mfma_accs(
            elem_dtype,
            (MFMA_DIM_K, MFMA_DIM_LANE_K),
            (MFMA_K, MFMA_LANE_K),
        )

        # ---- Thread / lane indices ----
        tid = fx.Int32(gpu.thread_idx.x)
        wave_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_mod_16 = lane % fx.Int32(MFMA_N)
        lane_div_16 = lane // fx.Int32(MFMA_N)

        # ---- ping-pong group predicate ----
        # `readfirstlane` makes the divergence analysis treat this as scalar, so
        # the duty branch around the LDS stores lowers to a scalar branch rather
        # than a per-lane mask. The predicate is wave-uniform by construction
        # (GRP_THREADS is a multiple of WARP_SIZE).
        tid_u = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, tid))
        is_g0 = tid_u < fx.Int32(GRP_THREADS)
        grp_tid = tid - is_g0.select(fx.Int32(0), fx.Int32(GRP_THREADS))

        # ---- Group-major grid decode -> (batch_idx, head_idx, q_tile_idx) ----
        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // num_q_tiles
        q_tile_idx = pos_in_group % num_q_tiles
        hz_idx = grid_group * hz_per_group + local_hz_idx

        # hz_per_group is a ceil -> the last group is padded past batch*num_heads. Padding blocks
        # clamp hz_idx=0 (in-bounds seq_offsets read) and set seq_len=0 below -> no stores, n_tiles=0.
        block_valid = hz_idx < hz_total
        hz_idx = block_valid.select(hz_idx, fx.Int32(0))
        batch_idx = hz_idx // fx.Int32(num_heads)
        head_idx = hz_idx % fx.Int32(num_heads)

        # ---- Sequence bounds + id clamps (target tail) ----
        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start
        seq_len = block_valid.select(seq_len, fx.Int32(0))

        num_target = fx.Int32(0)
        if has_targets:
            num_target = fx.Int32(num_targets[batch_idx])

        # Contextual shifts max_id BEFORE the target-tail clamp; to_id below applies the same order per position.
        max_id = seq_len
        if has_contextual:
            max_id = seq_len - fx.Int32(contextual_seq_len) + fx.Int32(1)
        if has_targets:
            max_id = (num_target > fx.Int32(0)).select(max_id - num_target, max_id)

        # ---- Global tensor views: g-wide coordinate slices for coalesced vector loads ----
        # (row, head) is indexed through the tensor's own i64-strided layout first (row*row_stride can
        # exceed int32 on packed tensors -> wrap/OOB); the resulting sub-view carries the i64 base and
        # the g-wide in-row load is a small i32 access over the < int32 in-row span.
        def grouped_loader(t, dim, g):
            in_row = fx.make_layout((dim // g, g), (g, 1))

            def load(row_i64, head_val, colgrp):
                sub = t[row_i64, head_val, None]
                return fx.make_view(fx.get_iter(sub), in_row)[colgrp, None].load()

            return load

        q_load = grouped_loader(q, head_dim, MFMA_DIM_LANE_K)

        # ---- The group-selected K/V loader ----
        # g0 reads K, g1 reads V, through ONE unbranched load whose base pointer
        # is selected. The alternative -- an `scf.if` with a `k_load` in one arm
        # and a `v_load` in the other -- cannot work: the loaded vector has to
        # survive into the loop's yield so it can span two phases, and a value
        # defined inside the `if` region does not dominate that use.
        #
        # Both address chains are computed, but they are pure index arithmetic;
        # only one load is issued. This relies on k and v having identical
        # layouts, which the validator enforces via head_dim == hidden_dim.
        # k and v have IDENTICAL layouts -- the validator gates
        # head_dim == hidden_dim -- so their address chains differ only in the
        # base pointer. Evaluating both and selecting the RESULT cost two full
        # chains, 10 VALU rows of phase M. Take the delta once instead, as a
        # wave-uniform i64, and build one chain.
        _kv_b0 = fx.get_iter(k[fx.Int64(0), fx.Int32(0), None])
        _kv_b1 = fx.get_iter(v[fx.Int64(0), fx.Int32(0), None])
        kv_base_delta = is_g0.select(
            fx.Int64(0),
            fx.Int64(fx.ptrtoint(_kv_b1)) - fx.Int64(fx.ptrtoint(_kv_b0)),
        )

        def kv_row_addr(row_i64):
            """This group's per-lane ROW address (base + row + head), as i64."""
            it = fx.get_iter(k[row_i64, head_idx, None])
            return fx.Int64(fx.ptrtoint(it)) + kv_base_delta

        # ---- The staging descriptor: bounds-checking in HARDWARE -------------
        # Every staged row needs "is this token inside the sequence", and out of
        # range it needs ZERO. Done in software that is a compare, a 64-bit
        # address select (which needs two `v_mov`s, because `v_cndmask` already
        # reads VCC and so cannot also take an SGPR source) and a cndmask per
        # loaded dword -- 9 of phase M's 15 VALU rows.
        #
        # A buffer descriptor does all of it for free: base the resource at
        # (seq_start, head_idx) and set `num_records` to this sequence's extent,
        # and an access past it is suppressed and returns 0 -- exactly the
        # zero-fold's semantics, in the memory pipe instead of the VALU. What is
        # carried per trip then shrinks from a 64-bit address to a 32-bit
        # voffset.
        #
        # This is NOT the `buffer_load ... lds` DMA form, which was ruled out
        # here (K is stored swizzled and V transposed, so the destination layout
        # is not the source's). The load still lands in VGPRs and the relayout
        # stays free; only the descriptor is borrowed.
        assert (num_heads * head_dim) % VEC_KV == 0
        KV_ROW_GRPS = (num_heads * head_dim) // VEC_KV  # voffset groups per token
        kv_desc = fx.rocdl.make_buffer_ptr(
            fx.inttoptr(_kv_b0.type, kv_row_addr(fx.Int64(seq_start))),
            num_records_bytes=fx.Int64(seq_len) * fx.Int64(KV_ROW_GRPS * VEC_KV * 2),
        )
        kv_buf = fx.make_view(
            kv_desc,
            fx.make_layout((seq_len * fx.Int32(KV_ROW_GRPS), VEC_KV), (VEC_KV, 1)),
        )

        # LDS as shape-carried views, grouped by the MFMA_LANE_K pack width so an access is
        # view[row, col_grp, None].load()/.store() (the trailing group axis carries the unit stride;
        # no manual row*stride+col). Column indices are computed directly in MFMA_LANE_K-group units
        # (see k_swz_grp and the read/store helpers) so the hot path never issues a runtime divide.
        # Views are taken once here in the enclosing region so they dominate both KV loops. k_smem is
        # K[kv, d] swizzled; v_smem is V[d, kv] transposed. Both column strides (HEAD_DIM_K, V_T_STRIDE)
        # are MFMA_LANE_K-aligned, and every column index lands on a group boundary.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_layout = fx.make_layout(
            (BLOCK_N, HEAD_DIM_K // MFMA_DIM_LANE_K, MFMA_DIM_LANE_K),
            (HEAD_DIM_K, MFMA_DIM_LANE_K, 1),
        )
        v_layout = fx.make_layout(
            (hidden_dim, V_T_STRIDE // MFMA_LANE_K, MFMA_LANE_K),
            (V_T_STRIDE, MFMA_LANE_K, 1),
        )
        # Indexed by LITERAL slot at every use site, so each access still
        # resolves to one named fx.Array field.
        k_smem = [lds.k0.view(k_layout), lds.k1.view(k_layout)]
        v_smem = [lds.v0.view(v_layout), lds.v1.view(v_layout)]

        # Single source of truth for the K LDS swizzle, expressed in MFMA_LANE_K-group units. The
        # full-column swizzle is col ^ ((row & (ROWS-1)) << SHIFT). Because MFMA_LANE_K == 1<<LOG2,
        # SHIFT >= LOG2, and every column is MFMA_LANE_K-aligned, XOR distributes over the group
        # division: (col ^ mask) // LANE == (col // LANE) ^ (mask // LANE). We therefore swizzle the
        # group index directly and index the grouped view without a runtime divide.
        assert K_SWZ_SHIFT >= MFMA_DIM_LANE_K_LOG2

        # Single source of truth for the V LDS swizzle, mirroring `k_swz_grp`.
        #
        # The `V_T_STRIDE = BLOCK_N + 8` pad fixes the READ path -- consecutive
        # lanes walk consecutive d rows, stride 20 dwords, which spreads. It
        # does nothing for the WRITE: the lanes that collide there differ in d
        # by VEC_KV, so their addresses differ by VEC_KV*V_T_STRIDE*2/4 = 160
        # dwords and 160 % 32 == 0 -- all 64 lanes land on 4 banks, 16 deep.
        # No pad can fix that: spreading 8 lanes needs V_T_STRIDE odd mod 8,
        # while the grouped view needs it a multiple of MFMA_LANE_K.
        #
        # So scramble the kv GROUP by the d row instead, keyed on d/VEC_KV
        # because that is exactly what the colliding lanes differ in (d%VEC_KV
        # is constant across them, which is why the obvious key does nothing).
        # Measured on the real lane maps: store 16-way -> 4-way, read unchanged
        # at 4-way. Groups stay in [0, BLOCK_N/MFMA_LANE_K), so the scramble
        # never reaches the pad columns, and both call sites use this function
        # so it is invisible outside them.
        V_SWZ_GRPS = BLOCK_N // MFMA_LANE_K
        assert V_SWZ_GRPS & (V_SWZ_GRPS - 1) == 0, "V swizzle needs a power-of-2 group count"

        def v_swz_key(d_row):
            return (d_row // fx.Int32(VEC_KV)) & fx.Int32(V_SWZ_GRPS - 1)

        def v_swz_grp(d_row, kv_grp):
            return kv_grp ^ v_swz_key(d_row)

        def k_swz_grp(tile_row, col_grp):
            return col_grp ^ (
                (tile_row & fx.Int32(K_SWZ_ROWS - 1))
                << fx.Int32(K_SWZ_SHIFT - MFMA_DIM_LANE_K_LOG2)
            )

        # ---- Which 16-row blocks of the Q tile this wave owns ---------------
        # Contiguous (`wave_id * ROWS_PER_WAVE + qg * MFMA_M`) hands wave w the
        # rows [w*ROWS_PER_WAVE, +ROWS_PER_WAVE), so its Q_SUBTILES blocks are
        # ADJACENT. That concentrates the causal diagonal: over the masked
        # remainder, every tile has some wave sitting entirely below the
        # diagonal, which must gate all of its sub-blocks. Phase V ends at a
        # barrier, so the workgroup pays that wave -- and skipping the
        # fully-masked sub-blocks would recover 5% of the gate.
        #
        # Interleaving the blocks instead (wave w owns w, w+NUM_WAVES,
        # w+2*NUM_WAVES, ...) makes every wave straddle the whole Q tile, so
        # the diagonal is spread evenly and max-over-waves falls to 62% of the
        # gate and 12.5% of the mask (the floor for any mapping is 56%).
        #
        # Nothing else depends on a wave's rows being contiguous: a wave still
        # reads and writes MFMA_M consecutive rows per sub-tile, so the Q load
        # and the O store keep their coalescing.
        #
        # MEASURED OFF, because it is not free: 115.0 -> 116.3/116.5 ms over two
        # runs, a reproducible +1.2% with no other change. The balance is tied
        # to how far a wave's blocks are spread (span 384 rows gives gate 80,
        # span 192 gives 104, span 48 -- contiguous -- gives 122), so there is
        # no cheap middle. Against the masked loop's 4.3% ceiling that leaves
        # ~1.9% net, where branching on the contiguous mapping alone already
        # takes ~0.9% for free: it recovers only 5% of the GATE but 75% of the
        # MASK, which does not depend on the spread.
        Q_BLK_INTERLEAVE = True
        q_tile_base = q_tile_idx * fx.Int32(BLOCK_M)

        def q_blk_base(qg):
            """First row of this wave's `qg`-th 16-row block of the Q tile."""
            if const_expr(Q_BLK_INTERLEAVE):
                return q_tile_base + (
                    wave_id + fx.Int32(qg * NUM_WAVES)
                ) * fx.Int32(MFMA_M)
            return (
                q_tile_base
                + wave_id * fx.Int32(ROWS_PER_WAVE)
                + fx.Int32(qg * MFMA_M)
            )

        # ---- Q rows / bounds per query sub-tile ----
        q_rows = []
        q_in_bounds = []
        for qg in range_constexpr(Q_SUBTILES):
            local = q_blk_base(qg) + lane_mod_16
            q_rows.append(local)
            q_in_bounds.append(local < seq_len)

        # ---- Q B-operand packs (register-resident, per query sub-tile); q_packs[ks][qg] ----
        q_packs = []
        for ks in range_constexpr(K_STEPS):
            # column within the head
            q_col = fx.Int32(ks * MFMA_DIM_K) + lane_div_16 * fx.Int32(MFMA_DIM_LANE_K)
            per_qg = []
            for qg in range_constexpr(Q_SUBTILES):
                safe = q_in_bounds[qg].select(seq_start + q_rows[qg], seq_start)
                raw = q_load(
                    fx.Int64(safe), head_idx, q_col // fx.Int32(MFMA_DIM_LANE_K)
                ).ir_value()
                per_qg.append(q_in_bounds[qg].select(raw, c_zero_dim_pack))
            q_packs.append(per_qg)

        # ---- Score-gate helpers ----
        # exp2(x*-log2e) == e^-x, so amdgcn.exp2/rcp build sigmoid(alpha*s) without a slow exp. Fuse
        # alpha and -log2e into one scale g = s*(alpha*-log2e): it is both the exp2 arg and the outer
        # factor p = g*sigm = (-log2e)*silu(alpha*s), so the loop does 2 muls not 3. The residual
        # -1/log2e rides the O-epilogue 1/N mul. Fast-fp comes from _hstu_compile_hints.
        c_alpha_neg_log2e = fx.Float32(alpha * -_LOG2E)
        c_one_f = fx.Float32(1.0)
        c_zero_f = fx.Float32(0.0)
        # Column id for a position past the end of the sequence: larger than
        # any query id, small enough that a query id minus it cannot wrap.
        COL_OOB = fx.Int32(1 << 30)

        def _rcp(x):
            return fx.Float32(fx.rocdl.rcp(compute_type, x.ir_value()))

        def silu_scale_batch(s_list):
            """silu(alpha*s) via the -log2e base change, stage-batched for ILP. Returns
            p = g*sigm with g = (alpha*-log2e)*s, so p = (-log2e)*silu(alpha*s); the -1/log2e is
            recovered in the O epilogue. A masked/zeroed score gives g=0 -> exp2(0)=1 -> sig=0.5
            -> p=0.

            `g` IS the incoming score: the (alpha*-log2e) scale is folded into
            the Q operand once per workgroup (see `scale_q_pack`) rather than
            applied to every gated element of every KV tile. The ATT trace had
            `v_mul_f32` at 516,224 executions against 258,048 gated elements --
            exactly two per element -- and this removes one of them."""
            g = s_list
            emu = [exp2_f32(gi) for gi in g]
            den = [c_one_f + e for e in emu]
            sig = [_rcp(d) for d in den]
            return [g[i] * sig[i] for i in range(len(s_list))]

        def to_id(x):
            """Raw position -> masked id (contextual prefix shift, then target-tail clamp)."""
            xid = x
            if has_contextual:
                xid = xid - fx.Int32(contextual_seq_len - 1)
                xid = (xid < fx.Int32(0)).select(fx.Int32(0), xid)
            if has_targets:
                xid = (xid > max_id).select(max_id, xid)
            return xid

        def pack_p(vals):
            """Pack 4 f32 scores into a bf16/f16 MFMA pack (the GEMM2 A-operand fragment).

            bf16 truncation IS a byte selection -- the high half of each f32 --
            so one `v_perm_b32` does a pair. The arithmetic spelling
            `(hi & 0xFFFF0000) | (lo >> 16)` is bit-for-bit the same but costs
            two VALU after the backend fuses the and/or (`v_lshrrev_b32` +
            `v_and_or_b32`, measured 3.15% of all active time in the trace).
            gfx942 has no `v_cvt_pk_bf16_f32`; that is CDNA4 only.
            """
            if const_expr(is_bf16):
                # Bytes 7,6 of src0 (hi) and 3,2 of src1 (lo): the two high halves.
                sel = fx.Int32(0x07060302)

                def bf16_pair(lo_f32, hi_f32):
                    return fx.Int32(
                        llvm.call_intrinsic(
                            fx.Int32.ir_type,
                            "llvm.amdgcn.perm",
                            [
                                hi_f32.bitcast(fx.Int32).ir_value(),
                                lo_f32.bitcast(fx.Int32).ir_value(),
                                sel.ir_value(),
                            ],
                            [],
                            [],
                        )
                    )

                pairs = [bf16_pair(vals[0], vals[1]), bf16_pair(vals[2], vals[3])]
                return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
            elems = [v.to(elem_dtype) for v in vals]
            return Vec.from_elements(elems, elem_dtype).ir_value()

        def scale_q_pack(pack):
            """Fold (alpha * -log2e) into one Q operand pack.

            Widening is the exact inverse of `pack_p`'s byte selection: element
            2w sits in the LOW half of word w and element 2w+1 in the high half,
            so `word << 16` and `word & 0xFFFF0000` recover them as f32 with no
            conversion instruction at all.

            Paid K_STEPS * Q_SUBTILES times per WORKGROUP, against the
            per-element-per-KV-tile multiply it replaces.

            NOT bitwise-equal to scaling the score: the product is rounded back
            to bf16's 8 mantissa bits here, where before it was applied in f32
            after the MFMA had accumulated. The error is ~2^-9 relative per
            term and partly cancels across the contraction; see the module
            docstring for the measured accuracy.
            """
            c16 = fx.Int32(16)
            chi = fx.Int32(0xFFFF0000)
            words = Vec(pack).bitcast(fx.Int32)
            vals = []
            for w in range_constexpr(MFMA_DIM_LANE_K // 2):
                word = words[w]
                vals.append((word << c16).bitcast(fx.Float32) * c_alpha_neg_log2e)
                vals.append((word & chi).bitcast(fx.Float32) * c_alpha_neg_log2e)
            return pack_p(vals)

        assert is_bf16, "scale_q_pack assumes the bf16 byte layout"
        q_packs = [
            [scale_q_pack(q_packs[ks][qg]) for qg in range_constexpr(Q_SUBTILES)]
            for ks in range_constexpr(K_STEPS)
        ]

        q_rows_i32 = q_rows
        q_row_ids = [to_id(q_rows_i32[qg]) for qg in range_constexpr(Q_SUBTILES)]

        # ---- The causal mask as one per-row BORDER --------------------------
        # `to_id(col) < to_id(q)` is a per (query, column) test, but to_id is
        # monotone, so for each query row there is a single RAW column
        # threshold below which it holds. Precompute that threshold once per
        # q-subtile and the per-element mask becomes `col < border` -- no
        # per-column to_id, no second compare. This is `generate_mask_border`
        # from the gfx950 asm FMHA kernel (tmp/asm/gfx950, which folds the
        # sequence bound in the same way; we fold it into the COLUMN instead,
        # via COL_OOB, because our bound is per-column and theirs is uniform).
        #
        #   border = min(q_raw + 1, id_ceiling)
        #
        # The `+1` absorbs the diagonal for every row whose id is NOT clamped:
        # `col < q+1` is `col <= q`. id_ceiling is the first raw column that
        # reaches to_id's ceiling, so a row at the ceiling (a TARGET row) stops
        # there and needs the diagonal term back -- which is why the equality
        # compare survives only under has_targets.
        #
        # A contextual-prefix row has id 0 and attends the whole prefix
        # instead, which is the same border with the causal term removed.
        _CTX_S = contextual_seq_len - 1 if has_contextual else 0
        _id_ceiling = max_id + fx.Int32(_CTX_S) if has_targets else COL_OOB
        q_border = []
        for qg in range_constexpr(Q_SUBTILES):
            _b = q_rows_i32[qg] + fx.Int32(1)
            if const_expr(has_targets):
                _b = (_b < _id_ceiling).select(_b, _id_ceiling)
            if const_expr(has_contextual):
                _b = (q_row_ids[qg] > fx.Int32(0)).select(_b, _id_ceiling)
            q_border.append(_b)

        # ---- Wave-uniform bounds, for the per-sub-block diagonal test -------
        # A 16x16 (qg, ng) sub-block is entirely below the diagonal, entirely
        # above it, or straddling it, and which one is decided by numbers that
        # are constant across the wave. Skipping the gate on an above-diagonal
        # block and the mask on a below-diagonal one is then an SALU branch,
        # not an exec mask -- but only if the compiler can SEE the condition is
        # uniform. `wave_id` is `tid // WARP_SIZE`, which its divergence
        # analysis treats as per-lane, so pin it through readfirstlane.
        #
        # Used ONLY in the predicates. The data path keeps the original values
        # so nothing else in the kernel changes shape.
        def _sgpr32(x):
            return fx.Int32(rocdl.readfirstlane(T.i32, x.ir_value()))

        q_blk_lo = [_sgpr32(q_blk_base(qg)) for qg in range_constexpr(Q_SUBTILES)]
        q_blk_hi = [x + fx.Int32(MFMA_M - 1) for x in q_blk_lo]
        # The border is monotone non-decreasing in the row only when there is
        # no contextual prefix (whose rows all carry the ceiling), which is the
        # same condition the unmasked/masked split already requires.
        SUBBLK_SKIP = causal and not has_window and not has_contextual
        if const_expr(SUBBLK_SKIP):
            def _row_border(r):
                b = r + fx.Int32(1)
                if const_expr(has_targets):
                    b = (b < _id_ceiling).select(b, _id_ceiling)
                return b

            q_blk_blo = [_row_border(x) for x in q_blk_lo]
            q_blk_bhi = [_row_border(x) for x in q_blk_hi]

        # ---- KV range: causal upper bound + active predicate ----
        q_start = q_tile_idx * fx.Int32(BLOCK_M)
        q_end = q_start + fx.Int32(BLOCK_M)
        base_upper = seq_len
        if causal:
            clamped = (q_end < seq_len).select(q_end, seq_len)
            if has_contextual:
                # The prefix block holds logical row id 0, which attends the whole contextual
                # prefix (col_id < max_id) above its diagonal, so its KV range opens to seq_len.
                # Other blocks are pure causal and their high tiles are fully masked.
                ctx_block = q_start < fx.Int32(contextual_seq_len)
                base_upper = ctx_block.select(seq_len, clamped)
            else:
                base_upper = clamped
        active = q_start < seq_len
        kv_upper = active.select(base_upper, fx.Int32(0))
        n_tiles = (kv_upper + fx.Int32(BLOCK_N - 1)) // fx.Int32(BLOCK_N)

        # ---- Sliding-window lower bound: skip fully-masked low KV tiles ----
        kv_tile_start = fx.Int32(0)
        if has_window:
            eff_q_low = (q_start < max_id).select(q_start, max_id)
            kv_lower = eff_q_low - fx.Int32(max_attn_len)
            kv_lower = (kv_lower > fx.Int32(0)).select(kv_lower, fx.Int32(0))
            win_tile_start = kv_lower // fx.Int32(BLOCK_N)
            if has_contextual:
                # The prefix block must walk KV from 0 to see the prefix; the window lower bound
                # would otherwise skip the low tiles the prefix opener needs.
                ctx_prefix_block = q_start < fx.Int32(contextual_seq_len)
                kv_tile_start = ctx_prefix_block.select(fx.Int32(0), win_tile_start)
            else:
                kv_tile_start = win_tile_start

        N_ACC = D_CHUNKS * Q_SUBTILES
        c_zero_v4f32 = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()

        # ---- Group-staged K/V prefetch -----------------------------------------
        # Indices are in GROUP-local thread id: each group covers the whole tile
        # with its own 256 threads.
        kv_row_in_batch = grp_tid // fx.Int32(THREADS_PER_ROW_KV)
        kv_lane_in_row = grp_tid % fx.Int32(THREADS_PER_ROW_KV)
        kv_col = kv_lane_in_row * fx.Int32(VEC_KV)
        # ...the same column in MFMA_LANE_K-group units (VEC_KV is
        # MFMA_LANE_K-aligned -> const multiply, no runtime divide), for the
        # swizzled K store.
        kv_col_grp = kv_lane_in_row * fx.Int32(VEC_KV // MFMA_DIM_LANE_K)

        # ---- The V slot rotation, as a per-lane tile select -----------------
        # A lane's VEC_KV contiguous d values all land in ONE d-chunk (VEC_KV
        # divides MFMA_M), so "which chunk am I staging" is a single per-lane
        # predicate. Rotated lanes stage the NEXT tile's chunk; everyone else
        # stages this tile's. That is the whole cost of V_ROT_DT -- the store
        # destination is untouched, because the rotation changes which tile a
        # lane reads, not where it writes.
        kv_rot_lane = (kv_col // fx.Int32(MFMA_M)) < fx.Int32(V_ROT_DT)
        # g0 stages K, which is NOT rotated, so its bump is always 1.
        kv_stage_bump = (is_g0 | kv_rot_lane).select(fx.Int32(1), fx.Int32(0))

        def kv_stage_start(kv_tile):
            """Per-lane byte-row start of the tile this lane stages for `kv_tile`.

            g0: always K(kv_tile + 1). g1: V(kv_tile) for chunks >= V_ROT_DT and
            V(kv_tile + 1) for the rotated chunks, which is exactly the split
            `vslot kv_tile % V_SLOTS` is defined to hold.
            """
            return (kv_tile + kv_stage_bump) * fx.Int32(BLOCK_N)

        # ---- The staged address, carried and advanced ----------------------
        # Everything in a staging address except the TILE is loop-invariant:
        #
        #   addr(T) = base + (seq_start + T*BLOCK_N + row)*stride + head + col
        #           = [ base + (seq_start + T0*BLOCK_N + row)*stride + head ]
        #             + (T - T0) * BLOCK_N * stride
        #
        # and `BLOCK_N * stride_token * 2` is a HOST constant. So a trip costs
        # one 64-bit add, not the 5-instruction chain the per-tile form rebuilt
        # every time. Same for the guard's token, which advances by BLOCK_N.
        # This is the fp8 kernel's `advance_hbm_offs`; the per-tile form was
        # 24 of phase M's 28 VALU rows.
        # Carried in VEC_KV-element GROUPS, which is what the descriptor view is
        # indexed in. `token*stride + col` is group-aligned for every lane
        # (`col` is a multiple of VEC_KV and `stride` is a multiple of it too),
        # so the division is exact and static.
        STAGE_ADV = fx.Int32(BLOCK_N * KV_ROW_GRPS)
        # Past `num_records` by construction: a group index of 2**27 is byte
        # offset 2**31, and the descriptor can never be that large (the tensor
        # itself would not fit a 32-bit voffset). Only reachable when
        # KV_NEEDS_GUARD, and loop-invariant when it is -- the predicate is on
        # the lane's row within the batch, not on the tile.
        KV_OOB_VOFS = fx.Int32(1 << 27)

        def stage_seed(kv_tile):
            """[voffset] for the tile this lane stages at `kv_tile`."""
            out = []
            for b in range_constexpr(NUM_BATCHES_KV):
                row = kv_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_KV)
                tok = (kv_tile + kv_stage_bump) * fx.Int32(BLOCK_N) + row
                vofs = tok * fx.Int32(KV_ROW_GRPS) + kv_col // fx.Int32(VEC_KV)
                if const_expr(KV_NEEDS_GUARD):
                    # The one guard the descriptor cannot express: these lanes
                    # are in range of the SEQUENCE but outside the TILE, so the
                    # row they would read is real data belonging to the next
                    # tile. Push them past num_records by hand, once.
                    vofs = (kv_row_in_batch < fx.Int32(BLOCK_N)).select(
                        vofs, KV_OOB_VOFS
                    )
                out.append(vofs)
            return out

        def stage_advance(voffs):
            return [x + STAGE_ADV for x in voffs]

        def async_load_kv_regs(voffs):
            """Issue this group's coalesced global load to registers (non-blocking).

            The out-of-range rows need no fold at all: the descriptor's
            `num_records` stops at the end of this sequence, so a token past
            `seq_len` reads back zero in hardware. That used to be a compare, a
            64-bit address select and a `v_cndmask` per dword, and it had to be
            split across two phases -- a `select` on a loaded value is a
            CONSUMER of that load, so applying it here would sink an
            `s_waitcnt vmcnt(0)` into the issuing phase and collapse the depth-2
            staging pipeline to depth 1. With the check in the memory pipe there
            is nothing to split.
            """
            return [
                kv_buf[voffs[b], None].load().ir_value()
                for b in range_constexpr(NUM_BATCHES_KV)
            ]

        def store_k_regs_to_lds(vecs, ks_mem):
            """g0 only: write K vecs to LDS at XOR-swizzled columns."""
            for b in range_constexpr(NUM_BATCHES_KV):
                row = kv_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_KV)
                for h in range_constexpr(VEC_KV // MFMA_DIM_LANE_K):
                    col_grp = kv_col_grp + fx.Int32(h)
                    half = Vec.from_elements(
                        [
                            Vec(vecs[b])[h * MFMA_DIM_LANE_K + j]
                            for j in range_constexpr(MFMA_DIM_LANE_K)
                        ],
                        elem_dtype,
                    )
                    ks_mem[row, k_swz_grp(row, col_grp), None].store(half)

        def store_v_regs_to_lds(vecs, vs_mem):
            """g1 only: write V vecs to LDS transposed to V[d, kv] so GEMM2 can read 4
            consecutive kv per ds_read_b64. The lane's VEC_KV d-contiguous elements
            scatter to kv-fixed, d-varying LDS slots."""
            # The swizzle key is `(kv_col + j) // VEC_KV`, and j < VEC_KV, so it
            # equals `kv_col // VEC_KV` for every j -- loop-invariant. Hoisting
            # it (and the XOR) out of the j loop turns 8 div/and/xor chains into
            # one; left inside, the live ranges cost 24 VGPR and 5 spills.
            vkey = v_swz_key(kv_col)
            for b in range_constexpr(NUM_BATCHES_KV):
                kv_row = kv_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_KV)
                swz_grp = (kv_row // fx.Int32(MFMA_LANE_K)) ^ vkey
                kv_lane = kv_row % fx.Int32(MFMA_LANE_K)
                vv = Vec(vecs[b])
                for j in range_constexpr(VEC_KV):
                    vs_mem[kv_col + fx.Int32(j), swz_grp, kv_lane] = vv[j]

        # ==== GEMM1: Q*K^T -> P (P fragment already in GEMM2 A-operand layout) ====
        # ---- Per-UNIT operand readers --------------------------------------
        # A "unit" is one `ds_read` plus the Q_SUBTILES MFMAs that consume it.
        # Reading one unit at a time (rather than a whole sub-tile's worth) is
        # what lets the MFMA burst roll a lookahead window over them, so no MFMA
        # waits on its own read -- see `mfma_phase`.
        def read_k_unit(ng, ks, ks_mem):
            """One K A-operand pack: sub-tile `ng`, contraction step `ks`.

            2D-indexed; the column goes through the shared swizzle, on the LOCAL
            row (0..BLOCK_N-1) so it matches the store's layout.
            """
            local_k_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            # k_col = ks*MFMA_K + lane_div_16*MFMA_LANE_K, in MFMA_LANE_K-group units.
            k_col_grp = fx.Int32(ks * (MFMA_DIM_K // MFMA_DIM_LANE_K)) + lane_div_16
            return ks_mem[local_k_row, k_swz_grp(local_k_row, k_col_grp), None].load()

        def read_v_unit(c, ng, vs_mem):
            """One V B-operand pack: d-chunk `c`, kv sub-tile `ng`.

            V[d, kv] transposed, so 4 consecutive kv are contiguous -> one
            ds_read_b64.
            """
            d_col = fx.Int32(c * MFMA_M) + lane_mod_16
            # kv_lane = ng*MFMA_M + lane_div_16*MFMA_LANE_K, in MFMA_LANE_K-group units.
            kv_grp = fx.Int32(ng * (MFMA_M // MFMA_LANE_K)) + lane_div_16
            return vs_mem[d_col, v_swz_grp(d_col, kv_grp), None].load().ir_value()

        def gemm1_unit(s_accs, ng, ks, k_pack):
            """The Q_SUBTILES MFMAs of one QK unit, chained on s_accs[ng][*]."""
            for qg in range_constexpr(Q_SUBTILES):
                # Operand swap: A=K, B=Q (not A=Q, B=K) so the MFMA result P lands directly in
                # GEMM2's A-operand layout (M=query, K=kv) -- P stays register-resident.
                q_op = q_packs[ks][qg] if ks < K_STEPS else c_zero_dim_pack
                s_accs[ng][qg] = dim_mfma_acc(
                    Vec(k_pack).ir_value(), q_op, s_accs[ng][qg]
                )
            return s_accs

        def gemm2_unit(o_acc, p_packs, c, ng, v_pack):
            """The Q_SUBTILES MFMAs of one PV unit, chained on o_acc[c*Q+*]."""
            for qg in range_constexpr(Q_SUBTILES):
                off = c * Q_SUBTILES + qg
                o_acc[off] = mfma_acc(p_packs[ng][qg], v_pack, o_acc[off])
            return o_acc

        def read_unit(spec, ks_mem, vs_mem):
            """The one `ds_read` of unit `spec`, from the right slot."""
            kind, j = spec
            if const_expr(kind == "qk"):
                return read_k_unit(j % KV_SUBTILES, j // KV_SUBTILES, ks_mem)
            return read_v_unit(j % D_CHUNKS, j // D_CHUNKS, vs_mem)

        def apply_unit(spec, operand, o_acc, s_accs, p_packs):
            """The Q_SUBTILES MFMAs of unit `spec`, consuming `operand`."""
            kind, j = spec
            if const_expr(kind == "qk"):
                s_accs = gemm1_unit(
                    s_accs, j % KV_SUBTILES, j // KV_SUBTILES, operand
                )
            else:
                o_acc = gemm2_unit(
                    o_acc, p_packs, j % D_CHUNKS, j // D_CHUNKS, operand
                )
            return o_acc, s_accs

        def mfma_burst(o_acc, p_packs, carried, ks_mem, vs_mem):
            """PHASE M's whole MFMA stream: GEMM1 of this tile merged with GEMM2
            of the previous one, walked as one `UNIT_ORDER` sequence.

            Returns `(o_acc, s_accs)`. The raw fp32 score accumulators go to
            phase V to be masked and gated -- splitting there is what makes the
            two phases disjoint: every MFMA is on this side, every
            transcendental on the other.
            """
            s_accs = [
                [
                    Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    for _ in range_constexpr(Q_SUBTILES)
                ]
                for _ in range_constexpr(KV_SUBTILES)
            ]
            # Prime the window before the first MFMA, then keep it PREFETCH_DEPTH
            # units ahead. `sched_barrier` on both sides of each step is what
            # holds the pattern: without it the scheduler hoists every read to
            # the head of the phase and the window stops meaning anything.
            win = [None for _ in range_constexpr(WIN_RING)]
            # Seeded from the PREVIOUS phase's tail, so the first MFMA of the
            # burst issues with nothing outstanding.
            for s in range_constexpr(XPHASE_DEPTH):
                win[s % WIN_RING] = carried[s]
            for s in range_constexpr(XPHASE_DEPTH, PREFETCH_DEPTH):
                win[s % WIN_RING] = read_unit(UNIT_ORDER[s], ks_mem, vs_mem)
            nxt = [None for _ in range_constexpr(XPHASE_DEPTH)]
            for step in range_constexpr(N_UNITS):
                rocdl.sched_barrier(0)
                o_acc, s_accs = apply_unit(
                    UNIT_ORDER[step], win[step % WIN_RING], o_acc, s_accs, p_packs
                )
                sn = step + PREFETCH_DEPTH
                if const_expr(sn < N_UNITS):
                    win[sn % WIN_RING] = read_unit(
                        UNIT_ORDER[sn], ks_mem, vs_mem
                    )
                else:
                    # The tail reads the NEXT phase's first units -- and thanks
                    # to the rotation, out of the SAME `vs_mem` this phase is
                    # reading, never the slot being written. That is the whole
                    # point of V_ROT_DT: no third slot, no extra slot handle.
                    nxt[sn - N_UNITS] = read_unit(
                        UNIT_ORDER[sn - N_UNITS], ks_mem, vs_mem
                    )
                rocdl.sched_barrier(0)
            return o_acc, s_accs, nxt

        c_zero_pack = Vec.filled(MFMA_LANE_K, 0.0, elem_dtype).ir_value()

        def gate_tile_split(kv_start, s_accs):
            """PHASE V for a tile that STRADDLES the causal diagonal.

            The masked remainder is BLOCK_M/BLOCK_N tiles wide, and inside it
            most 16x16 (qg, ng) sub-blocks are not on the diagonal at all --
            they are entirely below it (every element kept, so the mask is
            pure overhead) or entirely above it (every element zeroed, so the
            gate is too: a zeroed score gates to exactly 0). Only the blocks
            the diagonal passes through need both.

            The test is wave-uniform, so each arm is an SALU branch. What it
            can recover is bounded by the BARRIER at the end of phase V: the
            workgroup pays the MAX over its 8 waves, and with the contiguous
            q-block mapping every remainder tile has some wave sitting wholly
            below the diagonal, which still gates all 8 of its sub-blocks. So
            this takes ~75% of the MASK and almost none of the GATE. See
            Q_BLK_INTERLEAVE for what would unlock the rest, and what it
            costs.
            """
            p_packs = []
            for ng in range_constexpr(KV_SUBTILES):
                kv_lo = kv_start + fx.Int32(ng * MFMA_M)
                kv_hi = kv_lo + fx.Int32(MFMA_M - 1)
                kv_base = kv_lo + lane_div_16 * fx.Int32(MFMA_LANE_K)
                col_raw = [
                    kv_base + fx.Int32(i)
                    for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                ]
                col_raw = [
                    (c < seq_len).select(c, COL_OOB) for c in col_raw
                ]
                row = []
                for qg in range_constexpr(Q_SUBTILES):
                    cur = s_accs[ng][qg]
                    s_vals = [
                        Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    # Above the diagonal: every column is past every row's
                    # border AND cannot be any row's own token. Or the whole
                    # block is past the end of the sequence.
                    all_drop = (kv_lo >= seq_len) | (
                        (kv_lo > q_blk_hi[qg]) & (kv_lo >= q_blk_bhi[qg])
                    )
                    # Below it: every column is inside the sequence and under
                    # the LOWEST row's border (the border is monotone here).
                    all_keep = (kv_hi < q_blk_blo[qg]) & (kv_hi < seq_len)
                    # Pre-bound: FlyDSL yields the names an `if` ASSIGNS, and
                    # it passes their current values in as the else-of-last-
                    # resort, so they have to exist before the branch.
                    pk = c_zero_pack
                    if all_drop:
                        pk = c_zero_pack
                    elif all_keep:
                        pk = pack_p(silu_scale_batch(s_vals))
                    else:
                        kept = [
                            (
                                (col_raw[i] < q_border[qg])
                                | (col_raw[i] == q_rows_i32[qg])
                                if const_expr(has_targets)
                                else col_raw[i] < q_border[qg]
                            ).select(s_vals[i], c_zero_f)
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                        pk = pack_p(silu_scale_batch(kept))
                    row.append(pk)
                p_packs.append(row)
            return p_packs

        def gate_tile(kv_start, s_accs, apply_mask=True):
            """PHASE V. Mask + silu gate -> P packs. No memory operation at all.

            apply_mask=False skips the per-element mask (and the column position/bounds math it
            needs). Only valid for a tile that is fully in-seq and fully valid -- see the causal
            unmasked-range split in the main loop."""
            if const_expr(apply_mask and SUBBLK_SKIP):
                return gate_tile_split(kv_start, s_accs)

            # The masked scores of the WHOLE tile, in (ng, qg, i) order, so that
            # every aligned run of MFMA_ELEMS_PER_LANE is one accumulator and
            # the gate can be batched across accumulators (see GATE_BATCH).
            flat = []
            for ng in range_constexpr(KV_SUBTILES):
                if const_expr(apply_mask):
                    kv_base = (
                        kv_start
                        + fx.Int32(ng * MFMA_M)
                        + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    )
                    col_raw = [
                        kv_base + fx.Int32(i)
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    # Fold "this column is past the end of the sequence" into
                    # the column's own coordinates instead of AND-ing it into
                    # every (qg, i) predicate. An out-of-seq column gets a raw
                    # position no query can equal and an id no query can
                    # exceed, which kills the diagonal term, the causal term
                    # and the contextual opener at once. That turns 32
                    # `s_and_b64` into 2 selects per column -- the bound is
                    # per-COLUMN but the AND was per (query, column).
                    #
                    # COL_OOB has to stay small enough that `q_id - COL_OOB`
                    # does not wrap, because the windowed path still forms that
                    # difference.
                    col_in_seq = [
                        col_raw[i] < seq_len
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    if const_expr(causal and not has_window):
                        # The border test is in RAW column positions, so the
                        # out-of-seq fold is one select and `to_id` never runs
                        # per column at all.
                        col_raw = [
                            col_in_seq[i].select(col_raw[i], COL_OOB)
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                    else:
                        col_raw = [
                            col_in_seq[i].select(col_raw[i], fx.Int32(-1))
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                        col_id = [
                            col_in_seq[i].select(to_id(col_raw[i]), COL_OOB)
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                for qg in range_constexpr(Q_SUBTILES):
                    cur = s_accs[ng][qg]
                    s_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]

                    if const_expr(apply_mask):
                        # keep_col is traced and consumed within this same loop iteration
                        # (see the comprehension below), so the loop vars it closes over are
                        # bound at definition time -- B023's late-binding concern doesn't apply.
                        def keep_col(i):
                            """causal * window * contextual * target mask for (qg, col i)."""
                            if const_expr(causal and not has_window):
                                keep = col_raw[i] < q_border[qg]  # noqa: B023
                                if const_expr(has_targets):
                                    # A target row's id is clamped, so its
                                    # border stops at the ceiling and the
                                    # diagonal has to come back by hand. Rows
                                    # below the ceiling already have it folded
                                    # into the `+1`, where this is redundant.
                                    keep = keep | (
                                        col_raw[i] == q_rows_i32[qg]  # noqa: B023
                                    )
                                return keep
                            dist = q_row_ids[qg] - col_id[i]  # noqa: B023
                            if not causal:
                                # Non-causal: symmetric id distance |q - col|. With the diagonal
                                # term this admits all in-seq columns (full attention); a window
                                # then becomes symmetric (|q - col| <= max_attn_len).
                                dist = (dist > fx.Int32(0)).select(dist, -dist)
                            # Diagonal compared in RAW positions (q_rows_i32 == col_raw): a query
                            # always attends its own token, even where to_id's shift+clamp collapses
                            # distinct ids. The window/causal distance uses to_id ids, so the id
                            # transform governs only the off-diagonal mask.
                            keep = (q_rows_i32[qg] == col_raw[i]) | (  # noqa: B023
                                dist > fx.Int32(0)
                            )
                            if has_window:
                                keep = keep & (dist <= fx.Int32(max_attn_len))
                            if has_contextual:
                                # Prefix opener: logical row 0 attends the contextual prefix.
                                ctx = (q_row_ids[qg] == fx.Int32(0)) & (  # noqa: B023
                                    col_id[i] < max_id  # noqa: B023
                                )
                                keep = keep | ctx
                            return keep

                        s_vals = [
                            keep_col(i).select(s_vals[i], c_zero_f)
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                    flat.extend(s_vals)

            gated = []
            for b in range_constexpr(0, len(flat), GATE_BATCH):
                gated.extend(silu_scale_batch(flat[b : b + GATE_BATCH]))

            E = MFMA_ELEMS_PER_LANE
            return [
                [
                    pack_p(gated[(ng * Q_SUBTILES + qg) * E :][:E])
                    for qg in range_constexpr(Q_SUBTILES)
                ]
                for ng in range_constexpr(KV_SUBTILES)
            ]

        # ==== GEMM2: P*V -> O (V as natural-layout operand B; P as operand A) ====
        # ==== The two phases ====
        N_PACK = KV_SUBTILES * Q_SUBTILES

        def mfma_phase(o_acc, p_prev, kv_regs, st_o, pre, kv_tile, u):
            """PHASE M: both GEMMs, their LDS operand feed, and this group's staging.

            Returns (o_acc, s_accs, kv_regs_next). `u` is the sub-trip position,
            which is what every LDS slot index is keyed on -- see K_SLOTS.
            """
            # Raises this wave above its SIMD partner for the whole MFMA burst.
            # The pair is wave i / wave i+4, in opposite phases by construction,
            # and this is what arbitrates them on the shared issue port.
            rocdl.s_setprio(1)
            k_cur = u % K_SLOTS
            k_fill = (u + 1) % K_SLOTS
            v_cur = u % V_SLOTS
            v_prev = (u + 1) % V_SLOTS

            # ORDER IS LOAD-BEARING: every operand ds_read is issued BEFORE the
            # staging ds_writes below, and the drain for those writes is
            # therefore lgkmcnt(0) at the bottom of this phase.
            #
            # The tempting alternative -- hoist the store to the TOP so a
            # COUNTED lgkmcnt(PREFETCH_DEPTH) can drain it at the bottom while
            # the tail's prefetch reads stay in flight, letting the scatter
            # overlap the burst -- was built and measured at 137.4 ms against
            # 121.0, a 13.6% REGRESSION. `lgkmcnt` retires in order, so a
            # store-first order puts all 16 operand reads behind a per-element
            # b16 scatter, and the 8 MFMAs that the cross-phase prefetch can
            # cover are not enough to hide it. The scatter's bank-conflicted
            # LATENCY, not its issue cost, is what the reads end up waiting on.
            #
            # So the counted-wait idea is sound in the abstract and dead in
            # practice here: the only thing it can drain early is a store that
            # was issued early, and issuing the store early costs more than the
            # drain saves.
            # lgkmcnt retires in order, and the V store is a per-element b16
            # transpose scatter; putting it first would make these reads wait on
            # the scatter and turn an lgkmcnt(3) into an lgkmcnt(0) ahead of the
            # MFMAs.
            # One contiguous MFMA burst: GEMM1 of this tile merged with GEMM2
            # of the previous one. `p_prev` is the previous tile's gate output
            # and v_smem[v_prev] its V slot -- the rotation. Tiles still land in
            # the accumulator in ascending order, only one trip later, so this
            # stays bitwise identical to the unrotated kernel.
            o_acc, s_accs, pre_next = mfma_burst(
                o_acc, p_prev, pre, k_smem[k_cur], v_smem[v_prev]
            )

            # Publish the set issued one phase pair ago, then issue the next.
            # Each group writes ONE tensor, never both: g0's M(i) writes only
            # K(i+1), g1's M(i) writes only V(i). So g0's staged TILE INDEX runs
            # one ahead of g1's, and one select covers the difference. (Not to
            # be confused with the phase skew, which runs the other way: g0
            # burns the extra prologue barrier, so g0 is the group that LAGS.)
            #
            # THIS is where the staging load is waited -- immediately ahead of
            # the ds_write that consumes it, and nowhere else. At this point the
            # only load outstanding is the one issued in the PREVIOUS M phase, so
            # `vmcnt(0)` here costs nothing and the load has had a full M+V pair
            # to return. Waiting after the issue below instead would drain the
            # load fired moments earlier and collapse the pipeline to depth 1 --
            # the stall this kernel exists to remove.
            rocdl.s_waitcnt(vmcnt=0)
            # `kv_regs` was issued in M(i-1) and is RAW -- the zero-fold happens
            # here, at the consumer, so that the `v_cndmask` is not what forces
            # the wait above. The tile it belongs to is one ahead of this one for
            # g0 (which writes K(i+1)) and this very tile for g1 (which writes
            # V(i)); mirrors the `next_tile` offsets below, shifted by a phase.
            kv_vals = kv_regs
            if is_g0:
                store_k_regs_to_lds(kv_vals, k_smem[k_fill])
            else:
                store_v_regs_to_lds(kv_vals, v_smem[v_cur])
            rocdl.sched_group_barrier(rocdl.mask_dswr, NUM_BATCHES_KV, 0)

            # Issued now, ds_written in M(i+1), so it targets M(i+1)'s staged
            # tile: one further out than this phase's, per lane.
            st_o = stage_advance(st_o)
            kv_regs_next = async_load_kv_regs(st_o)

            # `phase_barrier` carries no memory fence, so this wait is by hand,
            # and it must be HERE -- inside the writing phase, not deferred to
            # the end of phase V. The skew leaves ZERO barriers of slack on two
            # edges, because the partner is one phase ahead:
            #
            #   g0 writes K(i+1) in [B(2i+1), B(2i+2)) and g1 READS it in
            #     [B(2i+2), ...) -- the very next interval.
            #   g1 writes V      in [B(2i),   B(2i+1)) and g0's M(i) TAIL
            #     prefetches it in [B(2i+1), ...) -- likewise.
            #
            # `s_barrier` does not retire LDS; `lgkmcnt(0)` before it is what
            # makes the write visible to the partner. Deferring it measured
            # 0.6% faster and passed 18 bitwise runs, which is exactly how this
            # class of bug presents -- the backward's skew race was 0/63 at one
            # geometry and 36/63 at another.
            #
            # The same wait is the WAR edge for this phase's operand ds_reads
            # against M(i+1)'s writes into those slots; lgkmcnt is one in-order
            # counter, so one wait does both.
            #
            # No `vmcnt` belongs here: the staging load issued just above lands
            # in REGISTERS, which the partner never reads.
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_setprio(0)
            return o_acc, s_accs, kv_regs_next, st_o, pre_next

        def gate_phase(kv_start, s_accs, apply_mask=True):
            """PHASE V: the mask and the SiLU gate. No memory operation of its own.

            The LDS drain does NOT live here, though it looks like it should:
            phase M's writes have zero barriers of slack against the partner's
            reads, so the wait has to stay inside phase M. See the note there.
            """
            p = gate_tile(kv_start, s_accs, apply_mask=apply_mask)
            return [p[ng][qg] for ng in range_constexpr(KV_SUBTILES) for qg in range_constexpr(Q_SUBTILES)]  # fmt: skip

        def unflatten(flat):
            return [
                [flat[ng * Q_SUBTILES + qg] for qg in range_constexpr(Q_SUBTILES)]
                for ng in range_constexpr(KV_SUBTILES)
            ]

        # Tiles fully below the diagonal are entirely valid under a pure causal mask, so their
        # per-element mask is pure overhead: split into an unmasked prefix [kv_tile_start, unmasked_end)
        # and a masked remainder [unmasked_end, n_tiles). With targets, to_id clamps col > max_id to
        # max_id, so a tile is fully valid only up to min(q_start, max_id) -- the unmasked boundary.
        CAUSAL_SPLIT = causal and not has_window and not has_contextual

        def split_carry(it):
            """(o_acc, p_prev, kv_regs) out of one loop's iter_arg tuple.

            `kv_regs` is in there alongside the accumulators because the depth-2
            staging set issued in the last trip of the unmasked loop is consumed
            by the first trip of the masked one; left out it would die at the
            scf.for boundary.
            """
            it_list = list(it) if isinstance(it, (list, tuple)) else [it]
            o = N_ACC + N_PACK              # staged vectors
            a = o + NUM_BATCHES_KV          # staged voffsets (i32)
            pre = a + NUM_BATCHES_KV        # cross-phase operand prefetch
            return (
                [it_list[i] for i in range_constexpr(N_ACC)],
                unflatten([it_list[N_ACC + i] for i in range_constexpr(N_PACK)]),
                [it_list[o + i] for i in range_constexpr(NUM_BATCHES_KV)],
                [it_list[a + i] for i in range_constexpr(NUM_BATCHES_KV)],
                [it_list[pre + i] for i in range_constexpr(XPHASE_DEPTH)],
            )

        if active:
            acc_init = [c_zero_v4f32 for _ in range_constexpr(N_ACC)]

            # Prefix/remainder boundary. Non-causal-split -> kv_tile_start (empty prefix range).
            if const_expr(CAUSAL_SPLIT):
                split_pos = q_start
                if has_targets:
                    split_pos = (q_start < max_id).select(q_start, max_id)
                unmasked_end = split_pos // fx.Int32(BLOCK_N)
                unmasked_end = (unmasked_end > kv_tile_start).select(
                    unmasked_end, kv_tile_start
                )
                unmasked_end = (unmasked_end < n_tiles).select(unmasked_end, n_tiles)
                # Round DOWN to a multiple of UNROLL, relative to kv_tile_start.
                # Two reasons, and both are correctness, not speed:
                #  - a padded trip in the MASKED loop is free (every `keep` is
                #    false, so the gate emits zero packs and the next GEMM2 adds
                #    nothing), but the unmasked loop has no mask to zero one.
                #  - slots are keyed on the sub-trip position, so an odd trip
                #    count here would invert K and V slot parity for the whole
                #    masked loop.
                # The tiles handed back to the masked loop are genuinely
                # unmasked, and the full mask is a no-op on them -- so this is
                # correct, and costs at most one tile of mask VALU.
                unmasked_end = kv_tile_start + (
                    ((unmasked_end - kv_tile_start) // fx.Int32(UNROLL))
                    * fx.Int32(UNROLL)
                )
            else:
                unmasked_end = kv_tile_start

            # ---- Prologue -----------------------------------------------------
            # Both groups load tile `first`: g0 needs K(first) for M(first)'s QK
            # units, g1 needs V(first)'s d-chunk 0 for the rotated bootstrap.
            # One load serves both, because the base pointer is selected.
            first = kv_tile_start
            _t_first = fx.Int32(first) * fx.Int32(BLOCK_N)
            _boot = async_load_kv_regs(stage_seed(fx.Int32(first) - kv_stage_bump))
            rocdl.s_waitcnt(vmcnt=0)
            if is_g0:
                # Nothing inside the loop writes K(first) -- M(i) writes K(i+1).
                store_k_regs_to_lds(_boot, k_smem[0])
            else:
                # THE V BOOTSTRAP. vslot 1 is what M(first) reads, and under the
                # rotation it must hold d-chunk 0 of tile `first` plus chunks
                # 1..D-1 of the nonexistent tile `first-1`. Rotated lanes write
                # the former; every other lane writes ZERO, which supplies the
                # latter -- so this one store also does the job the explicit
                # zeroing loop used to, and the loop is gone.
                #
                # That zeroing is not optional: M(first)'s GEMM2 multiplies
                # P(-1) = 0 against whatever is in those rows, and 0 * NaN is
                # NaN. LDS is not cleared between workgroups.
                store_v_regs_to_lds(
                    [
                        kv_rot_lane.select(
                            _boot[b], Vec.filled(VEC_KV, 0.0, elem_dtype).ir_value()
                        )
                        for b in range_constexpr(NUM_BATCHES_KV)
                    ],
                    v_smem[1],
                )
            # M(first) ds_writes the tile `kv_stage_start(first)` names, so prime
            # the carried staging set with exactly that.
            carry_o = stage_seed(fx.Int32(first))
            carry_regs = async_load_kv_regs(carry_o)

            # The skew: g0 burns one extra barrier so that from here on g0 is in
            # phase M exactly when g1 is in phase V, and vice versa. Barrier
            # counts stay equal overall -- g1 repays it after the loops.
            #
            # It MUST come BEFORE the publish fence below, not after. Each group
            # stages only its own tensor (g0 K, g1 V) but trip 0 reads BOTH, so
            # the fence that publishes them has to be the last thing either group
            # does before the loop. With the skew after the fence, g1 enters M(0)
            # a full interval ahead while g0 still sits at the skew barrier, and
            # g0's trip-0 GEMM2 would read a V tile g1 staged without an
            # intervening publish. The backward shipped exactly that inversion
            # and it was nondeterministic at every geometry but its tuned one --
            # 0/63 runs there against 36/63 at another (fixed in e2a81264d).
            if is_g0:
                phase_barrier()
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_waitcnt(lgkmcnt=0)
            phase_barrier()

            # M(first)'s PV operates on tile `first-1`, which does not exist, so
            # its carried chunk-0 packs are zero -- P(-1) is zero too, and
            # 0 * 0 is exact. From M(first+1) on they come from the tail.
            pre_init = [c_zero_pack for _ in range_constexpr(XPHASE_DEPTH)]
            carry_init = (
                acc_init
                + [c_zero_pack for _ in range_constexpr(N_PACK)]
                + carry_regs
                + carry_o
                + pre_init
            )

            # Trip counts. Both MUST be multiples of UNROLL: slot parity is keyed
            # on the sub-trip position, so a partial outer trip would hand the
            # next loop inverted K and V slots.
            n_unmasked = unmasked_end - fx.Int32(kv_tile_start)
            # Masked range plus ONE drain trip, so the last real tile's P is
            # consumed by a GEMM2. A trip past n_tiles has every `keep` false, so
            # its gate emits zero packs and the following GEMM2 adds nothing --
            # which also makes the round-up free.
            n_masked = (
                (n_tiles + fx.Int32(1) - unmasked_end + fx.Int32(UNROLL - 1))
                // fx.Int32(UNROLL)
            ) * fx.Int32(UNROLL)

            loop_results = carry_init
            # ---- Unmasked prefix (empty unless CAUSAL_SPLIT) ----
            for _o0, it in range(
                fx.Int32(0),
                n_unmasked // fx.Int32(UNROLL),
                fx.Int32(1),
                init=carry_init,
            ):  # ty: ignore
                o_acc, p_prev, kv_regs, st_o, pre = split_carry(it)
                base = fx.Int32(kv_tile_start) + fx.Int32(_o0) * fx.Int32(UNROLL)
                for u in range_constexpr(UNROLL):
                    # Loop rotated: the closing barrier of each gate phase is
                    # issued at the TOP of the next sub-trip rather than the
                    # bottom of this one, so the scf.for back edge -- increment,
                    # trip compare, branch -- lands in the tail of the gate phase
                    # instead of at the head of the MFMA phase, where it would
                    # sit in front of the burst.
                    if const_expr(u == 0):
                        phase_barrier()
                    kv_tile = base + fx.Int32(u)
                    o_acc, s_accs, kv_regs, st_o, pre = mfma_phase(
                        o_acc, p_prev, kv_regs, st_o, pre, kv_tile, u
                    )
                    phase_barrier()
                    p_flat = gate_phase(
                        kv_tile * fx.Int32(BLOCK_N), s_accs, apply_mask=False
                    )
                    p_prev = unflatten(p_flat)
                    if const_expr(u < UNROLL - 1):
                        phase_barrier()
                loop_results = yield (o_acc + p_flat + kv_regs + st_o + pre)

            # ---- Masked remainder + drain ----
            for _o1, it in range(
                fx.Int32(0),
                n_masked // fx.Int32(UNROLL),
                fx.Int32(1),
                init=loop_results,
            ):  # ty: ignore
                o_acc, p_prev, kv_regs, st_o, pre = split_carry(it)
                base = unmasked_end + fx.Int32(_o1) * fx.Int32(UNROLL)
                for u in range_constexpr(UNROLL):
                    # Loop rotated: the closing barrier of each gate phase is
                    # issued at the TOP of the next sub-trip rather than the
                    # bottom of this one, so the scf.for back edge -- increment,
                    # trip compare, branch -- lands in the tail of the gate phase
                    # instead of at the head of the MFMA phase, where it would
                    # sit in front of the burst.
                    if const_expr(u == 0):
                        phase_barrier()
                    kv_tile = base + fx.Int32(u)
                    o_acc, s_accs, kv_regs, st_o, pre = mfma_phase(
                        o_acc, p_prev, kv_regs, st_o, pre, kv_tile, u
                    )
                    phase_barrier()
                    p_flat = gate_phase(
                        kv_tile * fx.Int32(BLOCK_N), s_accs, apply_mask=True
                    )
                    p_prev = unflatten(p_flat)
                    if const_expr(u < UNROLL - 1):
                        phase_barrier()
                loop_results = yield (o_acc + p_flat + kv_regs + st_o + pre)

            # Closes the last sub-trip's gate phase, which the rotation left
            # without a trailing barrier inside the loop.
            phase_barrier()

            # g1 repays the skew barrier so both groups leave with equal counts.
            if not is_g0:
                phase_barrier()

            # ---- Epilogue: store O (1/N hoisted here) ----
            # GEMM2 writes a transposed C fragment: A=P, B=V, C[d, query].
            # tid%16 = d (N-dim), and (lane_div_16, e) = query (M-dim). The query row stored is
            # therefore q_blk_base(qg) + lane_div_16*4 + e; the d column is c*16 + lane_mod_16.
            results = (
                list(loop_results)
                if isinstance(loop_results, (list, tuple))
                else [loop_results]
            )
            for qg in range_constexpr(Q_SUBTILES):
                q_row_base = q_blk_base(qg) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                for e in range_constexpr(MFMA_ELEMS_PER_LANE):
                    q_row_e = q_row_base + fx.Int32(e)
                    if q_row_e < seq_len:
                        for c in range_constexpr(D_CHUNKS):
                            ov = results[c * Q_SUBTILES + qg]
                            d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                            val = (Vec(ov)[e] * inv_n).to(elem_dtype)
                            out[fx.Int64(seq_start + q_row_e), head_idx, d_col] = val

    _hstu_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }

    @flyc.jit
    def launch_hstu_attention_fwd_pp(
        max_seq_len: fx.Int32,
        batch: fx.Int32,
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream,
    ) -> None:
        c_num_heads = fx.Int32(num_heads)
        c_ngg = fx.Int32(NUM_GRID_GROUPS)
        num_q_tiles = (max_seq_len + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M)
        hz_total = batch * c_num_heads
        hz_per_group = (hz_total + fx.Int32(NUM_GRID_GROUPS - 1)) // c_ngg
        grid_blocks = num_q_tiles * hz_per_group * c_ngg

        # Epilogue scale = (1/N)*(-1/log2e): the -1/log2e undoes silu_scale_batch's -log2e base-change
        # (alpha stays inside silu), 1/N normalizes by the global max_seq_len (N), not the row's seq_len.
        inv_n = (fx.Float32(1.0) / max_seq_len) * fx.Float32(-1.0 / _LOG2E)

        hstu_attention_fwd_pp_kernel(
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            out,
            num_q_tiles,
            hz_per_group,
            hz_total,
            inv_n,
            value_attrs={
                "rocdl.waves_per_eu": WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(
            grid=(grid_blocks, 1, 1),
            block=BLOCK_THREADS,
            smem=0,
            stream=stream,
        )

    launch_hstu_attention_fwd_pp.compile_hints = _hstu_compile_hints
    return flyc.compile[_PP_COMPILE_HINTS](launch_hstu_attention_fwd_pp)
