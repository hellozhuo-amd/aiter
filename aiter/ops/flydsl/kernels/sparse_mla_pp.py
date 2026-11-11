# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL absorbed sparse MLA, one query per token, 8-wave ping-pong.

Same math as the BLOCK_Q=1 kernel of ``sparse_mla_qblock.py``, and the same summation
order, so the output is bitwise identical to it. Host wrapper:
``aiter/ops/flydsl/sparse_mla_pp_kernels.py``. Different mapping:

* WG = 8 waves = 4 tokens. Wave w serves token 4 * block + w // 2 and D-half w & 1,
  for both QK and PV. Waves 0-3 are group g0, waves 4-7 group g1.
* Wave (token, h) gathers D-half h of all 16 rows of a KV tile, row per lane: lane
  (r, g) loads 16 B of row idx[r] at d = 256 h + 32 p + 8 g, p = 0..7. Those
  registers are the QK A operands as loaded.
* The PV operands (V^T) are the same registers transposed. Default
  (``mfma_transpose``): 16 identity MFMAs per tile, C = K_block @ I^T, exact for
  bf16. 6 of the 8 Q fragment pairs are parked in LDS to make room for the V^T
  registers, which then stay live across the V phase. Alternative: write the tile
  into a private 8 KiB LDS region and read it back transposed (8 x 8 KiB = the
  whole 64 KiB).
* Chained schedule, two phases per tile:
    M(t): PV(t-1) from V^T / P^T registers, QK(t) from the gather registers,
          [the V^T(t) transposes], and the partial S(t) written into the PEER
          wave's region. MFMA only, apart from the parked-Q reads.
    V(t): peer partial read, [tile write + V^T reads], gather(t + 1), softmax(t)
          (gather(t + 2) with ``prefetch_depth=2``).
  With ``skew`` g0 runs one phase ahead of g1, so the two groups of each SIMD
  alternate matrix and memory/LDS/VALU work (ping-pong). Without it (``skew=False``)
  the kernel is the lockstep control.

Nothing is shared across groups; the phase barriers only schedule.
"""

from __future__ import annotations

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.sparse_mla_qblock import (
    HEAD_DIM,
    NEG_INF,
    NUM_HEADS,
    RESCALE_TAU,
    VEC,
    WAVE,
    _colmajor,
    _elem,
    _exp2,
    _ext,
    _halves,
    _head_max,
    _mfma_ssa,
    _py,
    _rescale_all,
    _rowmajor,
    _to_bf16,
    _zero4,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

NWAVE = 8
TOKENS_PER_WG = 4
BLOCK_THREADS = NWAVE * WAVE
BLOCK_K = 16
HALF = HEAD_DIM // 2


def _phase_barrier():
    # Not gpu.barrier(): its fence adds lgkmcnt(0); waits are placed by hand.
    fx.rocdl.sched_barrier(0)
    fx.rocdl.s_barrier()
    fx.rocdl.sched_barrier(0)


def _bp_mfma_schedule(pv_blocks, qk_pairs, halves):
    """interleave_bp: static issue order of M's ds_bpermute and PV / QK MFMAs.

    Returns ("bp", k) (k-th bpermute: pair k // 4, dword k % 4), ("pv", g, h) and
    ("qk", pair, h) entries: one bpermute before each MFMA, plus whatever a QK still needs
    of its own pair. Plain Python (outside the kernel: the AST rewriter would turn its
    while loops into scf.while).
    """
    acts, bi, nbp = [], 0, 4 * qk_pairs
    mm = [("pv",) + b for b in pv_blocks]
    mm += [("qk", pr, h) for pr in range(qk_pairs) for h in range(halves)]
    for it in mm:
        if bi < nbp:
            acts.append(("bp", bi))
            bi += 1
        need = 0 if it[0] == "pv" else 4 * it[1] + 4
        while bi < need:
            acts.append(("bp", bi))
            bi += 1
        acts.append(it)
    while bi < nbp:
        acts.append(("bp", bi))
        bi += 1
    return acts


def _with_q_reads(acts, q_reg, q_lds, first):
    """Insert ("q", j) (read parked Q pair q_reg + j from LDS) after MFMA first + j of
    acts (1-based), one per MFMA, and in any case before the first QK of its pair."""
    out, n_mm, j = [], 0, 0
    for it in acts:
        while j < q_lds and it[0] == "qk" and it[1] >= q_reg + j:
            out.append(("q", j))
            j += 1
        out.append(it)
        if it[0] in ("pv", "qk"):
            n_mm += 1
            if j < q_lds and n_mm >= first + j:
                out.append(("q", j))
                j += 1
    return out + [("q", k) for k in range(j, q_lds)]


@lru_cache(maxsize=8)
def compile_sparse_mla_pp(
    has_sink=False,
    skew=True,
    setprio=True,
    mfma_transpose=True,
    q_lds=6,
    gather_split=4,
    slp=False,
    coalesce=True,
    interleave_bp=True,
    prefetch_depth=1,
    q_read_at=2,
):
    """Launcher of the ping-pong kernel; grid = ceil(T / TOKENS_PER_WG) workgroups.

    ``interleave_bp`` (default; coalesce + mfma_transpose): M issues its parked-Q reads,
    then one sched_barrier-pinned stream that alternates the 32 ds_bpermute of K with the
    16 PV + 16 QK MFMAs (bpermute -> MFMA -> bpermute -> ...; a QK waits only for its own
    pair's 4 bpermutes), then the transposes; off, all bpermutes go out before the MFMAs.

    ``skew=False`` is the lockstep 8-wave control (same schedule, no phase offset).
    ``mfma_transpose`` builds the PV operands with identity MFMAs in the M phase
    instead of an LDS tile write + transposed read in the V phase. Paged T=16384:
    0.96x the LDS path (8.03 vs 9.12 M cycles, at 1.29 vs 1.40 GHz). ``q_lds`` Q pairs
    are parked in LDS; with ``mfma_transpose`` fewer than 6 spill VGPRs.
    """
    HAS_SINK = bool(has_sink)
    SKEW = bool(skew)
    SETPRIO = bool(setprio)
    MFMA_T = bool(mfma_transpose)
    T_BATCH = 2  # MFMA_T: QK pairs transposed per batch (2 MFMAs each)
    # the last Q_LDS of the 8 Q pairs live in LDS, read back at the start of every M phase:
    # frees 4 VGPR per pair (MFMA_T keeps V^T live across the V phase and needs them)
    Q_LDS = int(q_lds)
    # The 8 gather loads of a V phase in G_SPLIT chunks spread through the softmax.
    # One burst per wave makes the 4 waves of a group queue for the CU's one vector
    # memory pipe while their VALU waits behind the stalled load.
    G_SPLIT = int(gather_split)
    # LLVM SLP vectorization pairs fp32 ops into v_pk_*_f32; a packed op waits for BOTH
    # halves' producers (as in the HSTU kernels, where turning it off was a win)
    SLP = bool(slp)
    # Gather in a coalescable lane order (the 4 lanes that read one row's 64 B are adjacent),
    # then ds_bpermute the tile into the MFMA layout. In the MFMA order the 4 lanes of a row
    # are 16 apart, so the address unit cannot merge them: ~58 L1 tag lookups per load
    # instead of ~16, and tag-RAM conflicts stall the L1 (TA busy 95%).
    COALESCE = bool(coalesce)
    INTER_BP = COALESCE and MFMA_T and bool(interleave_bp)
    # interleave_bp: the parked-Q reads go one per MFMA, from MFMA Q_READ_AT of the M stream
    # on (-1: all before it). Up front, M slot 0's 4 x 6 reads (right after the barrier)
    # beat the other group's single partial-S read at the start of V to the LDS: ~250 cy.
    Q_READ_AT = int(q_read_at) if INTER_BP else -1
    # Tiles between a gather and its M phase: 1 (default) = gather(t + 1) in V(t) into the
    # buffer M(t) just drained; 2 = gather(t + 2) in V(t), one register buffer per slot
    # (+32 VGPR). The L1's outstanding misses are full at either depth (the extra requests
    # only queue), yet depth 1 is 0.98x on paged T=16384. Index loads run PD + 2 tiles ahead.
    PD = int(prefetch_depth)
    if PD not in (1, 2):
        raise ValueError(f"prefetch_depth must be 1 or 2, got {PD}")
    IDX_AHEAD = PD + 2
    # the last trip (tiles n - 2, n - 1): V(n - 2 + s) gathers n - 2 + s + PD iff that exists
    LAST_PREFETCH = tuple(s + PD < 2 for s in range(2))
    if G_SPLIT not in (1, 2, 4):
        raise ValueError(f"gather_split must be 1, 2 or 4, got {G_SPLIT}")
    # only non-default values are named, so the default build is plain "sparse_mla_pp"
    # (the WaveScope profile's kernelRegex ^sparse_mla_pp$)
    module_name = (
        "sparse_mla_pp"
        + ("_sink" if HAS_SINK else "")
        + ("" if SKEW else "_noskew")
        + ("" if SETPRIO else "_noprio")
        + ("" if MFMA_T else "_nomt")
        + (f"_ql{Q_LDS}" if Q_LDS != 6 else "")
        + (f"_gs{G_SPLIT}" if G_SPLIT != 4 else "")
        + ("_slp" if SLP else "")
        + ("" if COALESCE else "_nocoal")
        + ("" if INTER_BP or not (COALESCE and MFMA_T) else "_nointer")
        + (
            ""
            if not INTER_BP or Q_READ_AT == 2
            else ("_qfront" if Q_READ_AT < 0 else f"_qra{Q_READ_AT}")
        )
        + (f"_pd{PD}" if PD != 1 else "")
    )
    BIG = 1 << 24
    CHUNKS = HEAD_DIM // VEC
    # elements of one wave's private LDS region: the KV tile, or only the partial S
    REGION = WAVE * VEC if MFMA_T else BLOCK_K * HALF

    def _lds_load(view, n):
        r = fx.make_rmem_tensor(fx.make_layout(n, 1), fx.BFloat16)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy(16 * n), fx.BFloat16), view, r)
        return r.load()

    def _lds_store(vec, view, n):
        r = fx.make_rmem_tensor(fx.make_layout(n, 1), fx.BFloat16)
        r.store(vec)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy(16 * n), fx.BFloat16), r, view)

    def _swz(layout):
        # the 16-B chunk (offset bits 3..5) XORed with row & 7 (row pitch 256: bits 8..10)
        return fx.make_composed_layout(fx.static(fx.SwizzleType.get(3, 3, 5)), layout)

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def sparse_mla_pp_kernel(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        idx_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        n_tok: fx.Int32,
        n_unique: fx.Int32,
        scale: fx.Float32,
        sink_ptr: fx.Pointer,
    ):
        # ---- static tilings (one D-half per wave)
        n_d_tiles = _ext(fx.zipped_divide(fx.make_layout(HALF, 1), (16,)), 1)
        qk_tiles = fx.zipped_divide(fx.make_layout(n_d_tiles, 1), (2,))
        pv_tiles = fx.zipped_divide(fx.make_layout(n_d_tiles, 1), (4,))
        QK_PAIRS = _ext(qk_tiles, 1)
        PV_GROUPS = _ext(pv_tiles, 1)
        HALVES = _ext(qk_tiles, 0)
        H4 = _ext(pv_tiles, 0)
        halves = fx.zipped_divide(fx.make_layout(VEC, 1), (H4,))
        acc_l = _colmajor(H4, PV_GROUPS)
        N_ACC = _py(fx.size(acc_l))

        tid = fx.Int32(gpu.thread_id("x"))
        tc = fx.idx2crd(tid, _colmajor(WAVE, NWAVE))
        lane = fx.get_(tc, 0).unpack()
        warp = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.get_(tc, 1).unpack()))
        # wave -> (D-half, token slot)
        wc = fx.idx2crd(warp, _colmajor(2, TOKENS_PER_WG))
        half = fx.get_(wc, 0).unpack()
        slot = fx.get_(wc, 1).unpack()
        lanes = _colmajor(16, 4)
        lc = fx.idx2crd(lane, lanes)
        lane_row = fx.get_(lc, 0).unpack()
        lane_g = fx.get_(lc, 1).unpack()
        head_xor = [_py(fx.crd2idx((0, b), lanes)) for b in (1, 2)]
        # the lane's gather row and 16-B chunk: MFMA order (row = lane % 16, chunk = lane // 16)
        # or, COALESCE, load order (row = lane // 4, chunk = lane % 4)
        if const_expr(COALESCE):
            lc_ld = fx.idx2crd(lane, _colmajor(4, 16))
            ld_chunk = fx.get_(lc_ld, 0).unpack()
            ld_row = fx.get_(lc_ld, 1).unpack()
            # MFMA lane (r, g) takes load lane 4 r + g; ds_bpermute addresses are bytes
            perm_addr = fx.get_scalar(
                fx.crd2idx((lane_g, lane_row), _colmajor(4, 16))
            ) * fx.Int32(4)
        else:
            ld_row, ld_chunk = lane_row, lane_g
        block = fx.Int32(gpu.block_id("x"))
        t_idx = fx.get_scalar(fx.crd2idx((slot, block), _colmajor(TOKENS_PER_WG, BIG)))
        n_tok_i = fx.Int32(n_tok)
        live = t_idx < n_tok_i
        q_row = live.select(t_idx, fx.Int32(0))
        is_g0 = warp < fx.Int32(NWAVE // 2)

        # ---- global views (elements)
        q_g = fx.make_view(
            fx.rocdl.make_buffer_ptr(q_ptr),
            _rowmajor(BIG, NUM_HEADS, 2, QK_PAIRS, 4, VEC),
        )
        out_g = fx.make_view(
            fx.rocdl.make_buffer_ptr(out_ptr),
            _rowmajor(BIG, NUM_HEADS, 2, PV_GROUPS, 4, 4, H4),
        )
        # MFMA_T: PV block (pair p, half hf) row i is d = 32 p + 8 (i // 4) + 4 hf + i % 4,
        # so a lane's two blocks of pair p hold 8 adjacent d: out[t, head, half, p, k-group, 8]
        out_t = fx.make_view(
            fx.rocdl.make_buffer_ptr(out_ptr),
            _rowmajor(BIG, NUM_HEADS, 2, QK_PAIRS, 4, VEC),
        )
        n_u = fx.Int32(n_unique)
        idx_tiles = fx.zipped_divide(_rowmajor(BIG, n_u), (1, BLOCK_K))
        n_kt = fx.get_scalar(fx.size(fx.get_(fx.get_shape(idx_tiles), [1, 1])))
        idx_lane = fx.make_view(
            fx.rocdl.make_buffer_ptr(idx_ptr), _rowmajor(BLOCK_K, 1)
        )[ld_row, None]
        idx_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        kv_g = fx.make_view(
            fx.rocdl.make_buffer_ptr(kv_ptr), _rowmajor(BIG, CHUNKS, VEC)
        )
        # lane (r, g), pair p: 16-B chunk (half, p, g) of row idx[r]
        g_chunks = _rowmajor(2, QK_PAIRS, 4)

        # ---- LDS: 8 private regions [16 rows][256 d], swizzled
        @fx.struct
        class SharedStorage:
            kv: fx.Array[fx.BFloat16, NWAVE * REGION, 16]
            q: fx.Array[fx.BFloat16, max(1, NWAVE * Q_LDS * WAVE * VEC), 16]

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        regions = _colmajor(REGION, NWAVE)
        own_base = fx.get_scalar(fx.crd2idx((0, warp), regions))
        peer_base = fx.get_scalar(fx.crd2idx((0, warp ^ fx.Int32(1)), regions))
        own_ptr = fx.add_offset(storage.kv.ptr, own_base)
        peer_ptr = fx.add_offset(storage.kv.ptr, peer_base)
        tile_l = fx.make_layout((BLOCK_K, HALF // VEC, VEC), (HALF, VEC, 1))
        own_tile = own_ptr.view(_swz(tile_l))
        rows = fx.make_layout(BLOCK_K, HALF)
        vt_tok = fx.flat_divide(rows, (4,))
        vt_rows = _swz(
            fx.make_layout((fx.get_shape(vt_tok), 16, 4), (fx.get_stride(vt_tok), 4, 1))
        )
        # partial S [(lane % 32, lane // 32), 8] over rows 0-1 of a region
        part_l = fx.make_ordered_layout(((32, 2), VEC), ((1, 2), 0))
        part_peer = peer_ptr.view(part_l)
        part_own = own_ptr.view(part_l)

        qk_scale = fx.Float32(scale) * fx.Float32(LOG2E)
        neg_inf = fx.Float32(arith.constant(NEG_INF, type=T.f32))
        zero_f = fx.Float32(arith.constant(0.0, type=T.f32))
        one_f = fx.Float32(arith.constant(1.0, type=T.f32))
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))

        # identity B operand: lane (r, g) holds I[r][4 g + e], e = 0..3
        eye = fx.Vector.from_elements(
            [
                (lane_row == lane_g * fx.Int32(4) + fx.Int32(e)).select(one_f, zero_f)
                for e in range_constexpr(4)
            ],
            fx.Float32,
        ).to(fx.BFloat16)
        # PV block of the MFMA_T path: (half within the 8 loaded bf16, pair) -> acc index
        blk_l = _colmajor(HALVES, QK_PAIRS)

        # Q pairs [wave, pair, lane, 8] parked in LDS (each wave reads back only its own)
        q_park = storage.q.view(_rowmajor(NWAVE, max(1, Q_LDS), WAVE, VEC))
        Q_REG = QK_PAIRS - Q_LDS
        q_frags = []
        for pr in range_constexpr(QK_PAIRS):
            raw = q_g[q_row, lane_row, half, pr, lane_g, None].load()
            if const_expr(pr < Q_REG):
                q_frags.append(_halves(raw, halves))
            else:
                q_park[warp, pr - Q_REG, lane, None].store(fx.Vector(raw))

        def _load_idx(kt):
            kt = (kt < n_kt).select(kt, fx.Int32(0))
            off = fx.get_scalar(fx.crd2idx(((0, 0), (q_row, kt)), idx_tiles))
            atom = fx.atom_set_value(idx_atom, "soffset", off)
            r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            fx.copy(atom, idx_lane, r)
            return fx.Int32(fx.Vector(r.load())[0])

        def _chunk(pr):
            return fx.get_scalar(fx.crd2idx((half, pr, ld_chunk), g_chunks))

        def _to_mfma(frags):
            # load order -> MFMA order, one ds_bpermute per dword (in place: the value is
            # read at issue)
            out = []
            for v in frags:
                d = fx.Vector(v).bitcast(fx.Int32)
                moved = [
                    fx.Int32(fx.rocdl.ds_bpermute(T.i32, perm_addr, fx.Int32(d[i])))
                    for i in range_constexpr(4)
                ]
                out.append(
                    fx.Vector.from_elements(moved, fx.Int32).bitcast(fx.BFloat16)
                )
            return out

        def _gather(idx_v):
            return [
                kv_g[idx_v, _chunk(pr), None].load() for pr in range_constexpr(QK_PAIRS)
            ]

        def _store_tile(frags):
            lane_chunks = _rowmajor(QK_PAIRS, 4)
            for pr in range_constexpr(QK_PAIRS):
                c = fx.get_scalar(fx.crd2idx((pr, lane_g), lane_chunks))
                _lds_store(fx.Vector(frags[pr]), own_tile[lane_row, c, None], VEC)

        def _read_vt():
            # V^T A operands of tiles (h, g): row r is d = 64 g + 4 r + h
            ops = []
            for g in range_constexpr(PV_GROUPS):
                vt_op = fx.add_offset(own_ptr, g * 64).view(vt_rows)
                rows_g = [
                    fx.Vector(_lds_load(vt_op[(e, lane_g), lane_row, None], 4))
                    for e in range_constexpr(4)
                ]
                ops += [
                    fx.Vector.from_elements(
                        [rows_g[e][h] for e in range_constexpr(4)], fx.BFloat16
                    )
                    for h in range_constexpr(H4)
                ]
            return ops

        def _pv(acc, vt, p_b):
            acc = list(acc)
            for g in range_constexpr(PV_GROUPS):
                for h in range_constexpr(H4):
                    t = _py(fx.crd2idx((h, g), acc_l))
                    acc[t] = _mfma_ssa(mma, vt[t], p_b, acc[t])
            return acc

        def _qk(frags, q_all):
            s_accs = [_zero4() for _ in range_constexpr(HALVES)]
            for pr in range_constexpr(QK_PAIRS):
                k = _halves(frags[pr], halves)
                for h in range_constexpr(HALVES):
                    s_accs[h] = _mfma_ssa(mma, k[h], q_all[pr][h], s_accs[h])
            s_acc = s_accs[0]
            for a in s_accs[1:]:
                s_acc = fx.Vector.from_elements(
                    [_elem(s_acc, i) + _elem(a, i) for i in range_constexpr(4)],
                    fx.Float32,
                )
            return s_acc

        def _transpose(frags):
            # C = K_block @ I^T: lane (r, g) gets K[token 4 g + c][d(r)] for c = 0..3, the
            # PV A layout. x * 1.0 in fp32 is exact, so the bf16 repack is too.
            # Batches of T_BATCH pairs, each packed before the next is issued: left to the
            # scheduler, all 16 fp32 results (64 VGPR) are live at once and the kernel spills.
            ops = [None] * N_ACC
            for b0 in range_constexpr(0, QK_PAIRS, T_BATCH):
                cs = []
                for pr in range_constexpr(b0, b0 + T_BATCH):
                    k = _halves(frags[pr], halves)
                    for hf in range_constexpr(HALVES):
                        cs.append((pr, hf, _mfma_ssa(mma, k[hf], eye, _zero4())))
                for pr, hf, c in cs:
                    # the high bf16 of each fp32 is the transposed value
                    hi = fx.Vector(c).bitcast(fx.BFloat16)
                    ops[_py(fx.crd2idx((hf, pr), blk_l))] = fx.Vector.from_elements(
                        [hi[2 * i + 1] for i in range_constexpr(4)], fx.BFloat16
                    )
                fx.rocdl.sched_barrier(0)
            return ops

        def _m_phase_bp(acc, vt, p_b, frags):
            # interleave_bp: bpermute / MFMA alternating (the parked-Q reads spread through
            # it from MFMA Q_READ_AT on, or all up front), then transposes
            q_all = q_frags + [None] * Q_LDS
            if const_expr(Q_READ_AT < 0):
                for j in range_constexpr(Q_LDS):
                    q_all[Q_REG + j] = _halves(q_park[warp, j, lane, None].load(), halves)
            fx.rocdl.sched_barrier(0)
            if const_expr(SETPRIO):
                fx.rocdl.s_setprio(1)
            dw = [fx.Vector(v).bitcast(fx.Int32) for v in frags]
            moved = [[None] * 4 for _ in range_constexpr(QK_PAIRS)]
            acc = list(acc)
            s_accs = [_zero4() for _ in range_constexpr(HALVES)]
            kops = [None] * QK_PAIRS
            sched = _bp_mfma_schedule(
                [(g, h) for g in range(PV_GROUPS) for h in range(H4)], QK_PAIRS, HALVES
            )
            if const_expr(Q_READ_AT >= 0):
                sched = _with_q_reads(sched, Q_REG, Q_LDS, Q_READ_AT)
            for n in range_constexpr(len(sched)):
                act = sched[n]
                if const_expr(act[0] == "q"):
                    j = act[1]
                    q_all[Q_REG + j] = _halves(q_park[warp, j, lane, None].load(), halves)
                elif const_expr(act[0] == "bp"):
                    pr, i = act[1] // 4, act[1] % 4
                    moved[pr][i] = fx.Int32(
                        fx.rocdl.ds_bpermute(T.i32, perm_addr, fx.Int32(dw[pr][i]))
                    )
                elif const_expr(act[0] == "pv"):
                    t = _py(fx.crd2idx((act[2], act[1]), acc_l))
                    acc[t] = _mfma_ssa(mma, vt[t], p_b, acc[t])
                else:
                    pr, h = act[1], act[2]
                    if const_expr(kops[pr] is None):
                        kops[pr] = _halves(
                            fx.Vector.from_elements(moved[pr], fx.Int32).bitcast(
                                fx.BFloat16
                            ),
                            halves,
                        )
                    s_accs[h] = _mfma_ssa(mma, kops[pr][h], q_all[pr][h], s_accs[h])
                fx.rocdl.sched_barrier(0)
            s_acc = s_accs[0]
            for a in s_accs[1:]:
                s_acc = fx.Vector.from_elements(
                    [_elem(s_acc, i) + _elem(a, i) for i in range_constexpr(4)],
                    fx.Float32,
                )
            frags = [
                fx.Vector.from_elements(moved[pr], fx.Int32).bitcast(fx.BFloat16)
                for pr in range_constexpr(QK_PAIRS)
            ]
            vt = _transpose(frags)
            if const_expr(SETPRIO):
                fx.rocdl.s_setprio(0)
            part_peer[lane, None].store(fx.Vector(s_acc).bitcast(fx.BFloat16))
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _phase_barrier()
            return acc, s_acc, vt

        def _m_phase(acc, vt, p_b, frags):
            if const_expr(INTER_BP):
                return _m_phase_bp(acc, vt, p_b, frags)
            if const_expr(COALESCE):
                # issued before the PV burst, which covers their latency
                frags = _to_mfma(frags)
            # parked Q pairs: read first, so the PV burst covers the LDS latency
            q_all = q_frags + [
                _halves(q_park[warp, j, lane, None].load(), halves)
                for j in range_constexpr(Q_LDS)
            ]
            if const_expr(SETPRIO):
                fx.rocdl.s_setprio(1)
            acc = _pv(acc, vt, p_b)
            if const_expr(MFMA_T):
                # PV retires the old V^T operands before the transposes build new ones;
                # interleaved, both sets (2 x 32 VGPR) are live and the kernel spills
                fx.rocdl.sched_barrier(0)
            s_acc = _qk(frags, q_all)
            if const_expr(MFMA_T):
                vt = _transpose(frags)
            if const_expr(SETPRIO):
                fx.rocdl.s_setprio(0)
            part_peer[lane, None].store(fx.Vector(s_acc).bitcast(fx.BFloat16))
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _phase_barrier()
            return acc, s_acc, vt

        def _at(hooks, point):
            # a gather chunk pinned at this point of the softmax (G_SPLIT > 1)
            if const_expr(point in hooks):
                fx.rocdl.sched_barrier(0)
                hooks[point]()
                fx.rocdl.sched_barrier(0)

        def _softmax(s_acc, m_i, l_i, acc, hooks):
            scores = [live.select(_elem(s_acc, i), neg_inf) for i in range_constexpr(4)]
            m_w = scores[0]
            for i in range_constexpr(1, 4):
                m_w = (m_w > scores[i]).select(m_w, scores[i])
            _at(hooks, 1)
            moved = fx.Int64(
                fx.rocdl.ballot(
                    fx.Int64.ir_type, (m_w - m_i) * qk_scale > fx.Float32(RESCALE_TAU)
                )
            )
            m_n = m_i
            alpha = one_f
            if moved != fx.Int64(0):
                m_h = _head_max(m_w, head_xor)
                m_n = (m_h > m_i).select(m_h, m_i)
                alpha = (m_n == neg_inf).select(zero_f, _exp2((m_i - m_n) * qk_scale))
                acc = _rescale_all(acc, alpha)
            _at(hooks, 2)
            empty = m_n == neg_inf
            p_t = []
            for i in range_constexpr(4):
                p = empty.select(zero_f, _exp2((scores[i] - m_n) * qk_scale))
                p_t.append(live.select(p, zero_f))
            l_i = l_i * alpha + ((p_t[0] + p_t[1]) + (p_t[2] + p_t[3]))
            _at(hooks, 3)
            p_b = fx.Vector.from_elements([_to_bf16(p) for p in p_t], fx.BFloat16)
            return m_n, l_i, acc, p_b

        def _v_phase(t, s_acc, m_i, l_i, acc, vt, frags, idx_s, prefetch):
            peer = fx.Vector(part_own[lane, None].load()).bitcast(fx.Float32)
            if const_expr(not MFMA_T):
                _store_tile(frags)
            # prefetch is static: a runtime skip would merge a gather-free path into the
            # loop's waitcnt state and drain the next tile's gather one phase early.
            new_frags = frags
            hooks = {}
            if const_expr(prefetch and G_SPLIT == 1):
                new_frags = _gather(idx_s)
                idx_s = _load_idx(t + fx.Int32(IDX_AHEAD))
            elif const_expr(prefetch):
                # chunk c of the 8 loads at softmax point POINTS[c]; the index load
                # (tile t + IDX_AHEAD) stays last so the M-phase vmcnt counts keep their meaning
                per = QK_PAIRS // G_SPLIT
                points = (0, 3) if G_SPLIT == 2 else (0, 1, 2, 3)
                got = {}
                idx_cur = idx_s

                def _chunk_loads(c):
                    def go():
                        for pr in range_constexpr(c * per, (c + 1) * per):
                            got[pr] = kv_g[idx_cur, _chunk(pr), None].load()
                        if const_expr(c == G_SPLIT - 1):
                            got["idx"] = _load_idx(t + fx.Int32(IDX_AHEAD))

                    return go

                hooks = {pt: _chunk_loads(c) for c, pt in enumerate(points)}
            _at(hooks, 0)
            s_acc = fx.Vector.from_elements(
                [_elem(s_acc, i) + peer[i] for i in range_constexpr(4)], fx.Float32
            )
            m_i, l_i, acc, p_b = _softmax(s_acc, m_i, l_i, acc, hooks)
            if const_expr(prefetch and G_SPLIT > 1):
                new_frags = [got[pr] for pr in range_constexpr(QK_PAIRS)]
                idx_s = got["idx"]
            if const_expr(not MFMA_T):
                vt = _read_vt()
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _phase_barrier()
            return m_i, l_i, acc, p_b, vt, new_frags, idx_s

        # ---- prologue: gathers 0 and 1 in flight, index tiles 2 and 3 queued
        m_i = neg_inf
        l_i = zero_f
        acc = [_zero4() for _ in range_constexpr(N_ACC)]
        # PV(-1) adds 0 x 0: zero operands keep uninitialised registers (NaN) out of acc
        vt = [
            fx.Vector.filled(4, 0.0, fx.Float32).to(fx.BFloat16)
            for _ in range_constexpr(N_ACC)
        ]
        p_b = fx.Vector.filled(4, 0.0, fx.Float32).to(fx.BFloat16)
        # idxs[j] holds the index of the next tile = j (mod 2) to gather: PD + 0 / PD + 1
        if const_expr(PD == 2):
            # Same issue order as the loop (gather t, index t + 2, gather t + 1, index t + 3):
            # the loop's waits are merged with the prologue's, and vmcnt counts in order.
            i0 = _load_idx(fx.Int32(0))
            i1 = _load_idx(fx.Int32(1))
            bufs0 = _gather(i0)
            idx0 = _load_idx(fx.Int32(2))
            bufs1 = _gather(i1)
            idx1 = _load_idx(fx.Int32(3))
            bufs = [bufs0, bufs1]
        else:
            i0 = _load_idx(fx.Int32(0))
            idx1 = _load_idx(fx.Int32(1))
            bufs = [_gather(i0)]
            idx0 = _load_idx(fx.Int32(2))
        # const_expr is resolved at trace time, is_g0 at run time: the two cannot be one `if`
        if const_expr(SKEW):  # noqa: SIM102
            if is_g0:
                _phase_barrier()

        NB = QK_PAIRS
        NBUF = PD

        NS = 5

        def _unflat(state):
            m_i, l_i, p_b, idx0, idx1 = state[:NS]
            rest = state[NS:]
            acc = [fx.Vector(a) for a in rest[:N_ACC]]
            vt = [fx.Vector(a) for a in rest[N_ACC : 2 * N_ACC]]
            bufs = [
                [fx.Vector(a) for a in rest[2 * N_ACC + b * NB : 2 * N_ACC + (b + 1) * NB]]
                for b in range_constexpr(NBUF)
            ]
            return [
                fx.Float32(m_i), fx.Float32(l_i), fx.Vector(p_b), [fx.Int32(idx0), fx.Int32(idx1)],
                acc, vt, bufs,
            ]  # fmt: skip

        def _trip(t0, state, prefetch):
            # tiles t0 (slot 0) and t0 + 1 (slot 1). PD = 2: each slot keeps its own buffer;
            # PD = 1: V(t) gathers t + 1 into the buffer M(t) drained. Either way the
            # registers return to place after the trip: the loop carries no copies of
            # in-flight loads. prefetch[s]: whether V(t0 + s) gathers.
            m_i, l_i, p_b, idxs, acc, vt, bufs = _unflat(state)
            for s in range_constexpr(2):
                b, j = s % NBUF, (s + PD) % 2
                acc, s_acc, vt = _m_phase(acc, vt, p_b, bufs[b])
                m_i, l_i, acc, p_b, vt, bufs[b], idxs[j] = _v_phase(
                    t0 + fx.Int32(s), s_acc, m_i, l_i, acc, vt, bufs[b], idxs[j], prefetch[s]
                )
            return [m_i, l_i, p_b] + idxs + acc + vt + [v for bb in bufs for v in bb]

        loop_init = [m_i, l_i, p_b, idx0, idx1] + acc + vt + [v for bb in bufs for v in bb]
        last = n_kt - fx.Int32(2)
        for kt, state in range(fx.Int32(0), last, fx.Int32(2), init=loop_init):
            results = yield _trip(fx.Int32(kt), state, (True, True))
        m_i, l_i, p_b, _idxs, acc, vt, _bufs = _unflat(
            _trip(last, list(results), LAST_PREFETCH)
        )
        # the last tile's PV (the M phase the rotation left over)
        acc = _pv(acc, vt, p_b)
        if const_expr(SKEW):  # noqa: SIM102
            if not is_g0:
                _phase_barrier()

        for sh in head_xor:
            l_i = l_i + l_i.shuffle_xor(sh, WAVE)
        tiny = fx.Float32(1.0e-30)
        head = lane_row
        denom = l_i
        alpha_q = fx.Float32(1.0)
        if const_expr(HAS_SINK):
            sink = fx.make_view(
                fx.rocdl.make_buffer_ptr(sink_ptr), fx.make_layout(NUM_HEADS, 1)
            )[head]
            # no `contract`: with SLP off the backend would fuse m_i * scale - m_final into
            # one FMA (one rounding), unlike the one-query kernel; keep the two roundings
            with arith.fastmath(arith.FastMathFlags.none):
                m_scaled = m_i * fx.Float32(scale)
                m_final = (sink > m_scaled).select(sink, m_scaled)
                alpha_q = _exp2((m_scaled - m_final) * fx.Float32(LOG2E))
                sink_term = _exp2((sink - m_final) * fx.Float32(LOG2E))
                denom = denom * alpha_q + sink_term
        denom = (denom > tiny).select(denom, tiny)
        inv = alpha_q / denom
        if const_expr(MFMA_T):
            if live:
                for pr in range_constexpr(QK_PAIRS):
                    vals = [
                        _to_bf16(_elem(acc[_py(fx.crd2idx((hf, pr), blk_l))], i) * inv)
                        for hf in range_constexpr(HALVES)
                        for i in range_constexpr(4)
                    ]
                    out_t[t_idx, head, half, pr, lane_g, None].store(
                        fx.Vector.from_elements(vals, fx.BFloat16)
                    )
        elif live:
            for g in range_constexpr(PV_GROUPS):
                for i in range_constexpr(4):
                    vals = [
                        _to_bf16(_elem(acc[_py(fx.crd2idx((h, g), acc_l))], i) * inv)
                        for h in range_constexpr(H4)
                    ]
                    out_g[t_idx, head, half, g, lane_g, i, None].store(
                        fx.Vector.from_elements(vals, fx.BFloat16)
                    )

    @flyc.jit
    def launch_sparse_mla_pp(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        idx_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        n_tok: fx.Int32,
        n_unique: fx.Int32,
        scale: fx.Float32,
        sink_ptr: fx.Pointer,
        n_tiles: fx.Int32,
        stream: fx.Stream,
    ):
        sparse_mla_pp_kernel(
            q_ptr, kv_ptr, idx_ptr, out_ptr, n_tok, n_unique, scale, sink_ptr
        ).launch(
            grid=(fx.Int64(n_tiles), 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_sparse_mla_pp.compile_hints = {
        "waves_per_eu": 1,
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
            "vectorize-slp": SLP,
        },
    }
    return launch_sparse_mla_pp
