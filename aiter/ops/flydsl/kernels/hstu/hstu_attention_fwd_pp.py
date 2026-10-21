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

Geometry is **(640, 32, 8, 2)**. Measured at B=1024 H=4 N=16384 d=64 against the
shipping kernel's tuned (128,32,4,2), bf16, sparsity 0.95:

    shipping tuned (128,32,4,2)   146.3 ms
    pp  8-wave skeleton, no pp    153.1 ms   +4.7%   geometry only, 2 WG/CU
    pp  rotated, lockstep         164.4 ms  +12.4%   1 WG/CU, no skew
    pp  + skew/setprio            154.2 ms   +5.5%   the ping-pong itself: -6.2%
    pp  + vectorize-slp=False     150.9 ms   +3.0%
    pp  staging zero-fold moved   140.6 ms   -3.9%   see async_load_kv_regs
    pp  (640,32) tuned            134.2 ms   -8.2%   1.089x

and 1.088 / 1.113 / 1.096 / 1.103x over the four benchmark cells (train H=4/H=8,
infer H=4/H=8). Unlike the backward's fused kernel -- 1.31x at H=4 but 1.17x at
H=8 -- the win here is flat across cells.

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

Everything this kernel produces is **bitwise identical** to
`hstu_attention_fwd.py` at every geometry, which is what the port was checked
against at each step (`tmp/hstu_repro/check_fwd_pp.py`). The rotation only
delays GEMM2 by one trip, leaving the order in which KV tiles enter the
accumulator unchanged; the skew only reorders publication; and a larger block_m
only makes the upper sub-block sweep KV tiles the smaller one skipped, which are
fully masked -> `s=0 -> g=0 -> exp2(0)=1 -> rcp(2)=0.5 -> p=0` exactly, and
`acc + 0*V` is exact in fp32.
"""

import functools
import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
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
    block_m: int = 640,
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
        kv_in_row = fx.make_layout((hidden_dim // VEC_KV, VEC_KV), (VEC_KV, 1))

        def kv_load(row_i64, head_val, colgrp):
            it_k = fx.get_iter(k[row_i64, head_val, None])
            it_v = fx.get_iter(v[row_i64, head_val, None])
            base = fx.inttoptr(
                it_k.type,
                is_g0.select(
                    fx.Int64(fx.ptrtoint(it_k)), fx.Int64(fx.ptrtoint(it_v))
                ),
            )
            return fx.make_view(base, kv_in_row)[colgrp, None].load()

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

        def k_swz_grp(tile_row, col_grp):
            return col_grp ^ (
                (tile_row & fx.Int32(K_SWZ_ROWS - 1))
                << fx.Int32(K_SWZ_SHIFT - MFMA_DIM_LANE_K_LOG2)
            )

        q_wave_base = q_tile_idx * fx.Int32(BLOCK_M) + wave_id * fx.Int32(ROWS_PER_WAVE)

        # ---- Q rows / bounds per query sub-tile ----
        q_rows = []
        q_in_bounds = []
        for qg in range_constexpr(Q_SUBTILES):
            local = q_wave_base + fx.Int32(qg * MFMA_M) + lane_mod_16
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

        def _rcp(x):
            return fx.Float32(fx.rocdl.rcp(compute_type, x.ir_value()))

        def silu_scale_batch(s_list):
            """silu(alpha*s) via the -log2e base change, stage-batched for ILP. Returns
            p = g*sigm with g = (alpha*-log2e)*s, so p = (-log2e)*silu(alpha*s); the -1/log2e is
            recovered in the O epilogue. A masked/zeroed score gives g=0 -> exp2(0)=1 -> sig=0.5
            -> p=0."""
            g = [s * c_alpha_neg_log2e for s in s_list]
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
            """Pack 4 f32 scores into a bf16/f16 MFMA pack (the GEMM2 A-operand fragment)."""
            if const_expr(is_bf16):
                c16 = fx.Int32(16)
                cmask = fx.Int32(0xFFFF0000)

                # bf16 = the high 16 bits of each f32 (round-toward-zero truncation, no round-to-nearest).
                def bf16_pair(lo_f32, hi_f32):
                    lo_i32 = lo_f32.bitcast(fx.Int32)
                    hi_i32 = hi_f32.bitcast(fx.Int32)
                    return (hi_i32 & cmask) | lo_i32.shrui(c16)

                pairs = [bf16_pair(vals[0], vals[1]), bf16_pair(vals[2], vals[3])]
                return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
            elems = [v.to(elem_dtype) for v in vals]
            return Vec.from_elements(elems, elem_dtype).ir_value()

        q_rows_i32 = q_rows
        q_row_ids = [to_id(q_rows_i32[qg]) for qg in range_constexpr(Q_SUBTILES)]

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

        def async_load_kv_regs(kv_start):
            """Issue this group's coalesced global load to registers (non-blocking).

            g0 gets K[kv_start], g1 gets V[kv_start]; see `kv_load`. Returns an
            ordinary SSA vector, which is the point -- it can be carried across a
            phase barrier and the loop backedge.

            Returns the RAW loaded vectors. The out-of-range zero-fold is NOT
            applied here -- `kv_zero_fold` does it at the consumer, one phase
            pair later. That split is the whole reason the staging pipeline has
            any depth: a `select` on the loaded value is a CONSUMER of the load,
            so folding here forces the compiler to sink an `s_waitcnt vmcnt(0)`
            into the phase that issued it, and the latency this kernel carries
            registers across a barrier to hide gets paid immediately instead.
            Measured in the ISA before the split:

                global_load_dwordx4 v[122:125], v[122:123], off
                s_waitcnt vmcnt(0)                     <- drains what was just issued
                v_cndmask_b32_e32   v125, 0, v125, vcc

            The address guard below is fine to keep here: it feeds the ADDRESS,
            not the result, so it does not wait on anything.

            There is no `full_tile` variant. The shipping kernel drops the
            `tok < seq_len` guard on tiles it knows are fully in-seq, which is
            sound only because it loads and computes the SAME tile. Under the
            ping-pong rotation the load runs one or two tiles ahead of the
            compute, so the flag no longer tracks the loop and the unmasked loop
            would prefetch a masked tile with its bounds guard elided. The guard
            costs one compare+select per batch; the elision costs an
            out-of-range global read.
            """
            vecs = []
            for b in range_constexpr(NUM_BATCHES_KV):
                row = kv_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_KV)
                tok = kv_start + row
                guard = _kv_row_guard(row, tok)
                safe = guard.select(seq_start + tok, seq_start)
                vecs.append(
                    kv_load(
                        fx.Int64(safe), head_idx, kv_col // fx.Int32(VEC_KV)
                    ).ir_value()
                )
            return vecs

        def _kv_row_guard(row, tok):
            """In-range predicate for one staging row."""
            guard = tok < seq_len
            if const_expr(KV_NEEDS_GUARD):
                guard = guard & (row < fx.Int32(BLOCK_N))
            return guard

        def kv_zero_fold(vecs, kv_start):
            """Zero the out-of-range rows of a staged set, at its CONSUMER.

            The predicate is RECOMPUTED here from `kv_start` rather than carried
            alongside the vectors. Carrying it would mean threading an i1 through
            the loop's iter_args, and recomputing costs one compare per batch
            against a scarce register budget (250 of 256 VGPRs at the tuned
            geometry). The caller is responsible for passing the kv_start the
            vectors were LOADED with, not the tile it is currently computing --
            see `mfma_phase`, where those differ by one or two tiles.
            """
            folded = []
            for b in range_constexpr(NUM_BATCHES_KV):
                row = kv_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_KV)
                guard = _kv_row_guard(row, kv_start + row)
                folded.append(
                    guard.select(
                        vecs[b], Vec.filled(VEC_KV, 0.0, elem_dtype).ir_value()
                    )
                )
            return folded

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
            for b in range_constexpr(NUM_BATCHES_KV):
                kv_row = kv_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_KV)
                kv_grp = kv_row // fx.Int32(MFMA_LANE_K)
                kv_lane = kv_row % fx.Int32(MFMA_LANE_K)
                vv = Vec(vecs[b])
                for j in range_constexpr(VEC_KV):
                    vs_mem[kv_col + fx.Int32(j), kv_grp, kv_lane] = vv[j]

        # ==== GEMM1: Q*K^T -> P (P fragment already in GEMM2 A-operand layout) ====
        def read_k_packs(ng, ks_mem):
            """LDS-read K A-operand packs for sub-tile ng (2D-indexed; col via the shared swizzle).
            The swizzle uses the LOCAL row (0..BLOCK_N-1) so it matches the store's layout.
            """
            local_k_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(K_STEPS_K):
                # k_col = ks*MFMA_K + lane_div_16*MFMA_LANE_K, in MFMA_LANE_K-group units.
                k_col_grp = fx.Int32(ks * (MFMA_DIM_K // MFMA_DIM_LANE_K)) + lane_div_16
                packs.append(
                    ks_mem[local_k_row, k_swz_grp(local_k_row, k_col_grp), None].load()
                )
            return packs

        def gemm1_tile(k_packs_by_ng):
            """PHASE M. Q*K^T MFMA only -- no mask, no gate.

            Returns the raw fp32 score accumulators s_accs[ng][qg], which phase V
            then masks and gates. Splitting here is what makes the two phases
            disjoint: every MFMA is on this side, every transcendental on the
            other. k_packs_by_ng comes from `read_k_packs`.
            """
            s_accs = [
                [None for _ in range_constexpr(Q_SUBTILES)]
                for _ in range_constexpr(KV_SUBTILES)
            ]
            for ng in range_constexpr(KV_SUBTILES):
                k_packs = [
                    Vec(k_packs_by_ng[ng][ks]) for ks in range_constexpr(K_STEPS_K)
                ]
                for qg in range_constexpr(Q_SUBTILES):
                    cur = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    # Operand swap: A=K, B=Q (not A=Q, B=K) so the MFMA result P lands directly in
                    # GEMM2's A-operand layout (M=query, K=kv) -- P stays register-resident.
                    for ks in range_constexpr(K_STEPS_K):
                        q_op = q_packs[ks][qg] if ks < K_STEPS else c_zero_dim_pack
                        cur = dim_mfma_acc(k_packs[ks].ir_value(), q_op, cur)
                    s_accs[ng][qg] = cur
            return s_accs

        def gate_tile(kv_start, s_accs, apply_mask=True):
            """PHASE V. Mask + silu gate -> P packs. No memory operation at all.

            apply_mask=False skips the per-element mask (and the column position/bounds math it
            needs). Only valid for a tile that is fully in-seq and fully valid -- see the causal
            unmasked-range split in the main loop."""
            p_packs = [
                [None for _ in range_constexpr(Q_SUBTILES)]
                for _ in range_constexpr(KV_SUBTILES)
            ]
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
                    col_in_seq = [
                        col_raw[i] < seq_len
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    col_id = [
                        to_id(col_raw[i]) for i in range_constexpr(MFMA_ELEMS_PER_LANE)
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
                            keep = keep & col_in_seq[i]  # noqa: B023
                            return keep

                        s_vals = [
                            keep_col(i).select(s_vals[i], c_zero_f)
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                    p_packs[ng][qg] = pack_p(silu_scale_batch(s_vals))
            return p_packs

        # ==== GEMM2: P*V -> O (V as natural-layout operand B; P as operand A) ====
        def accum_o_tile(o_acc, p_packs, vs_mem):
            """O[m,d] += P[m,n]*V[n,d]. A = P (GEMM1 frag, M=query K=kv), B = V[kv,d] natural.

            `p_packs` is the PREVIOUS tile's gate output and `vs_mem` the previous
            tile's V slot -- see the rotation note at K_SLOTS. The tiles still
            land in the accumulator in ascending order, only one trip later, so
            this is bitwise identical to the unrotated kernel.
            """

            def read_v_pack(c, ng):
                # V[d, kv] transposed: 4 consecutive kv contiguous -> one ds_read_b64 = B pack.
                d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                # kv_lane = ng*MFMA_M + lane_div_16*MFMA_LANE_K, in MFMA_LANE_K-group units.
                kv_grp = fx.Int32(ng * (MFMA_M // MFMA_LANE_K)) + lane_div_16
                return vs_mem[d_col, kv_grp, None].load().ir_value()

            for c in range_constexpr(D_CHUNKS):
                v_packs = [read_v_pack(c, ng) for ng in range_constexpr(KV_SUBTILES)]
                for qg in range_constexpr(Q_SUBTILES):
                    acc_off = c * Q_SUBTILES + qg
                    cur = o_acc[acc_off]
                    for ng in range_constexpr(KV_SUBTILES):
                        cur = mfma_acc(p_packs[ng][qg], v_packs[ng], cur)
                    o_acc[acc_off] = cur
            return o_acc

        # ==== The two phases ====
        N_PACK = KV_SUBTILES * Q_SUBTILES
        c_zero_pack = Vec.filled(MFMA_LANE_K, 0.0, elem_dtype).ir_value()

        def mfma_phase(o_acc, p_prev, kv_regs, kv_tile, u):
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

            # Every operand ds_read is issued BEFORE the staging ds_writes below.
            # lgkmcnt retires in order, and the V store is a per-element b16
            # transpose scatter; putting it first would make these reads wait on
            # the scatter and turn an lgkmcnt(3) into an lgkmcnt(0) ahead of the
            # MFMAs.
            k_packs = [
                read_k_packs(ng, k_smem[k_cur]) for ng in range_constexpr(KV_SUBTILES)
            ]

            # One contiguous MFMA burst: GEMM1 of this tile beside GEMM2 of the
            # previous one. That merge is the whole point of the rotation.
            s_accs = gemm1_tile(k_packs)
            o_acc = accum_o_tile(o_acc, p_prev, v_smem[v_prev])

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
            staged_start = (
                kv_tile + is_g0.select(fx.Int32(1), fx.Int32(0))
            ) * fx.Int32(BLOCK_N)
            kv_vals = kv_zero_fold(kv_regs, staged_start)
            if is_g0:
                store_k_regs_to_lds(kv_vals, k_smem[k_fill])
            else:
                store_v_regs_to_lds(kv_vals, v_smem[v_cur])
            rocdl.sched_group_barrier(rocdl.mask_dswr, NUM_BATCHES_KV, 0)

            next_tile = kv_tile + is_g0.select(fx.Int32(2), fx.Int32(1))
            kv_regs_next = async_load_kv_regs(next_tile * fx.Int32(BLOCK_N))

            # `phase_barrier` carries no memory fence, so this wait is placed by
            # hand. No `vmcnt` belongs here: the load issued just above lands in
            # REGISTERS, which the partner group never reads, so nothing about it
            # has to be visible at the barrier. Only the ds_write does.
            #   lgkmcnt(0) : publishes this phase's staging ds_write AND retires
            #                its operand ds_reads, which is the WAR edge against
            #                the next trip's writes into the same slots.
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_setprio(0)
            return o_acc, s_accs, kv_regs_next

        def gate_phase(kv_start, s_accs, apply_mask=True):
            """PHASE V: the mask and the SiLU gate. No memory operation at all."""
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
            return (
                [it_list[i] for i in range_constexpr(N_ACC)],
                unflatten([it_list[N_ACC + i] for i in range_constexpr(N_PACK)]),
                [it_list[N_ACC + N_PACK + i] for i in range_constexpr(NUM_BATCHES_KV)],
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
            # Trip 0's GEMM2 consumes P(-1) = 0 against V(-1), which does not
            # exist. A zero A operand does NOT make that a no-op: 0 * NaN is NaN
            # and would poison the accumulators for the whole sweep. LDS is not
            # zeroed between workgroups, so zero the slot it will read.
            # Sub-trip u=0 reads v_smem[(0+1) % V_SLOTS] == v_smem[1] == lds.v1.
            V_ELEMS = hidden_dim * V_T_STRIDE
            c_zero_elem = fx.Float32(0.0).to(elem_dtype)
            for _z in range_constexpr(
                (V_ELEMS + BLOCK_THREADS - 1) // BLOCK_THREADS
            ):
                _zi = tid + fx.Int32(_z * BLOCK_THREADS)
                _zi = (_zi < fx.Int32(V_ELEMS)).select(_zi, fx.Int32(0))
                lds.v1[_zi] = c_zero_elem

            # Nothing inside the loop writes K(first) -- M(i) writes K(i+1) -- so
            # the prologue stages it, and primes each group's carried register
            # set: g0 carries K(first+1), g1 carries V(first).
            first = kv_tile_start
            _t_first = fx.Int32(first) * fx.Int32(BLOCK_N)
            k0_vecs = async_load_kv_regs(_t_first)
            rocdl.s_waitcnt(vmcnt=0)
            if is_g0:
                store_k_regs_to_lds(kv_zero_fold(k0_vecs, _t_first), k_smem[0])
            prime_tile = first + is_g0.select(fx.Int32(1), fx.Int32(0))
            carry_regs = async_load_kv_regs(prime_tile * fx.Int32(BLOCK_N))

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

            carry_init = (
                acc_init + [c_zero_pack for _ in range_constexpr(N_PACK)] + carry_regs
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
                o_acc, p_prev, kv_regs = split_carry(it)
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
                    o_acc, s_accs, kv_regs = mfma_phase(
                        o_acc, p_prev, kv_regs, kv_tile, u
                    )
                    phase_barrier()
                    p_flat = gate_phase(
                        kv_tile * fx.Int32(BLOCK_N), s_accs, apply_mask=False
                    )
                    p_prev = unflatten(p_flat)
                    if const_expr(u < UNROLL - 1):
                        phase_barrier()
                loop_results = yield (o_acc + p_flat + kv_regs)

            # ---- Masked remainder + drain ----
            for _o1, it in range(
                fx.Int32(0),
                n_masked // fx.Int32(UNROLL),
                fx.Int32(1),
                init=loop_results,
            ):  # ty: ignore
                o_acc, p_prev, kv_regs = split_carry(it)
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
                    o_acc, s_accs, kv_regs = mfma_phase(
                        o_acc, p_prev, kv_regs, kv_tile, u
                    )
                    phase_barrier()
                    p_flat = gate_phase(
                        kv_tile * fx.Int32(BLOCK_N), s_accs, apply_mask=True
                    )
                    p_prev = unflatten(p_flat)
                    if const_expr(u < UNROLL - 1):
                        phase_barrier()
                loop_results = yield (o_acc + p_flat + kv_regs)

            # Closes the last sub-trip's gate phase, which the rotation left
            # without a trailing barrier inside the loop.
            phase_barrier()

            # g1 repays the skew barrier so both groups leave with equal counts.
            if not is_g0:
                phase_barrier()

            # ---- Epilogue: store O (1/N hoisted here) ----
            # GEMM2 writes a transposed C fragment: A=P, B=V, C[d, query].
            # tid%16 = d (N-dim), and (lane_div_16, e) = query (M-dim). The query row stored is
            # therefore q_wave_base + qg*16 + lane_div_16*4 + e; the d column is c*16 + lane_mod_16.
            results = (
                list(loop_results)
                if isinstance(loop_results, (list, tuple))
                else [loop_results]
            )
            for qg in range_constexpr(Q_SUBTILES):
                q_row_base = (
                    q_wave_base
                    + fx.Int32(qg * MFMA_M)
                    + lane_div_16 * fx.Int32(MFMA_LANE_K)
                )
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
