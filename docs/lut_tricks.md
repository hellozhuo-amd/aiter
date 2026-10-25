# Replacing the HSTU forward gate's transcendentals with an LDS lookup table

**Result: 113.95 ms → 104.61 ms, 1.277x → 1.391x over the shipping tuned
kernel, with accuracy unchanged.** Committed as `f9bca5799` on
`zhuo/hstu-fwd`. This is the largest single win of the optimisation effort,
and it came from a pipe I had twice measured as unusable — both measurements
were artifacts. The debugging of those artifacts is the most transferable part
of this document; see §6.

Target: `aiter/ops/flydsl/kernels/hstu/hstu_attention_fwd_pp.py`, gfx942 /
CDNA3 / MI300X.

---

## 1. What the gate was, and what it is now

The HSTU gate computes, per score element,

```
p = g / (1 + 2^g)          g = alpha * (-log2 e) * s,  folded into Q once per workgroup
```

which is `(-log2 e) * silu(alpha * s)`; the residual `-1/log2 e` rides the
O-epilogue's existing `1/N` multiply.

**Before** — four instructions plus half a pack, per element:

| instruction | n | cy | what |
|---|---|---|---|
| `v_exp_f32` | 32 | 597.0 | `e = 2^g` |
| `v_add_f32` | 32 | 180.9 | `d = 1 + e` |
| `v_rcp_f32` | 32 | 587.1 | `sigma = 1/d` |
| `v_mul_f32` | 32 | 175.1 | `p = g * sigma` |
| `v_perm_b32` | 16 | 64.0 | truncate + pack 2x f32 -> 2x bf16 |
| `s_barrier` | 2 | 160.1 | phase boundary |
| **total** | **146** | **1764** | body 1604 cy + 160 cy barrier |

**After** — the exp/add/rcp chain becomes one indexed read:

| instruction | n | what |
|---|---|---|
| `v_fmamk_f32` | 32 | `t = g * (1/h) + bias` — g to fractional node index |
| `v_med3_f32` | 32 | clamp `t` to `[0, N-1]` — **this is the saturating branch** |
| `v_cvt_i32_f32` | 32 | integer node index |
| `v_lshlrev_b32` | 32 | byte address `= i * 4` |
| `ds_read_b32` | 32 | **sigma, from the table** |
| `s_waitcnt` | 32 | |
| `v_mul_f32` | 32 | `p = g * sigma` — unchanged |
| `v_perm_b32` | 16 | pack — unchanged |
| `s_barrier` | 2 | |
| **total** | **242** | |

Instruction count goes **up** (146 → 242). What falls is the **issue floor**,
because every quarter-rate op is gone:

```
before   64 x 16 (transcendental) + 80 x 4 (full-rate)  = 1344 cy
after     0 x 16                  + 176 x 4             =  704 cy
```

**Phase V now contains zero transcendentals.**

Measured: 244 VGPR, 0 spills, 26976 B LDS, 1 WG/CU. Tile ladder re-swept and
unchanged — (512,32) 1.391x, (384,32) 1.363x, (384,64) 1.337x.

---

## 2. Why the accuracy target is 2^-8, not 2^-24

This is the hinge of the whole idea and it is the same argument the Schraudolph
analysis rests on.

`pack_p` truncates `p` to **bf16** to feed GEMM2's A-operand. bf16 has 8 bits
of precision (1 implicit + 7 stored mantissa bits), so it already injects up to
`2^-7 = 0.78%` relative error on every element, whatever the gate does. The
accuracy target for any replacement is therefore `2^-8` (round-to-nearest), not
f32's `2^-24` — a factor of ~65,000 more headroom than a naive reading gives.

Measured consequence: **max error against the fp32 oracle is 1.221e-04 on every
mask configuration, bit-identical to the exact gate.** The table's error lands
under a floor that was already there.

Also note `p = g * sigma` means the relative error of `p` *equals* the relative
error of `sigma`, since `g` is exact. That makes the table's error analysis a
pure statement about sigma.

---

## 3. Table sizing, derived

### 3.1 The step `h`

Nearest-node lookup: the worst case is landing half a step from a node.

```
abs err in sigma  =  |sigma'| * h/2,        sigma'(g) = -ln2 * sigma * (1-sigma)

rel err of p == rel err of sigma
              =  |sigma'| * (h/2) / sigma
              =  ln2 * (1 - sigma) * h / 2
```

Worst as `sigma -> 0` (large positive g), where `(1-sigma) -> 1`:

```
rel_max = ln2 * h / 2 = 0.3466 * h  =  2^-8   ->   h = 0.01127
```

A bisection over 4M samples gives **h = 0.01125** — the closed form agrees to
0.2%, which is the check that the derivation is right rather than a curve fit.

Note what this says: the binding constraint is the **large-positive-g tail**,
not the interesting middle. At `g ~ 0` where sigma actually curves,
`(1-sigma) = 0.5` and half the resolution would do. All that precision is spent
where sigma is nearly flat but relatively small — exactly where `p = g*sigma`
is tiny. See §8.1.

### 3.2 The range

```
g < -8:   1 - sigma = 2^g < 2^-8   ->  sigma rounds to EXACTLY 1.0 in bf16
g > +16:  p = g * sigma ~ 2.4e-4   ->  under bf16's noise floor vs an O(1) accumulator
```

Span **24**.

### 3.2.1 Why the cut is safe when sigma's relative error past it is unbounded

§3.1 sizes `h` by the **relative** error of sigma, and notes that the binding
constraint is the large-positive-g tail. §3.2 then cuts the range at `g = 16`,
abandoning that metric exactly where it was tightest. The two do not obviously
compose, and §3.2's two lines are loose in three ways:

1. The `g > +16` line describes `p` decaying to 2.4e-4 and being **dropped**.
   The kernel does not drop it -- it clamps the index, so `p` grows *linearly*
   past the cut. The error is `0` at `g = 16`, not 2.4e-4.
2. "under bf16's noise floor" invites reading `2^-7` as an absolute
   flush-to-zero threshold. It is a *relative* resolution, and it applies to
   the final `O`, not to `p` or to sigma.
3. The `g < -8` line needs `g < -9`. See the end of this section.

This section is the actual argument.

**What the clamp does.** It clamps the **index**, not `g`:

```python
t = gv * c_lut_scale + c_lut_big
t = fmed3(t, c_lut_lo, c_lut_hi)        # index -> [0, _LUT_N-1]
...
return [g[i] * sig[i] for i in ...]      # g is NOT clamped
```

So past the cut, sigma freezes at the endpoint while `g` keeps growing:

```
sigma_c(g) = 2^-16          (constant, the g=16 node)
sigma_t(g) = 2^-g
p_c(g)     = g * 2^-16      LINEAR IN g -- it does not decay
p_t(g)     = g * 2^-g       -> 0
```

§3.2's "`p = g*sigma ~ 2.4e-4`" is the value **at the cut point**, and it
describes a truncation of `p` to zero that the kernel does not perform. The
error is **zero at `g = 16`** (it is a table node, exact) and grows from there.

**Relative error of sigma past the cut is unbounded.**

```
rel_err(sigma) = sigma_c/sigma_t - 1 = (2^g - 2^16)/(1 + 2^16) ~ 2^(g-16) - 1
```

| g | rel err of sigma (== rel err of p, §3.1) |
|---|---|
| 16 | 0 |
| 17 | 1.0 -> **100%** |
| 20 | 15 |
| 34.5 (§6.4, scale 10) | 3.7e5 |
| 3455 (§6.4, scale 100) | `sigma_t` underflows fp32; undefined |

It doubles per unit of `g`. There is no `g` past the cut at which the relative
error is acceptable by §3.1's own standard.

**Absolute error of sigma past the cut is bounded, at `2^-16`.** Both the true
and the computed value live in `(0, 2^-16]`, so their difference cannot exceed
the interval:

```
|sigma_c - sigma_t| = 2^-16 - 2^-g  <  2^-16 = 1.526e-5
```

The relative error explodes only because the **denominator** collapses. A
relative error of 1e6 on a quantity of size 1e-300 is not an error anyone can
observe. Relative error is the scale-free metric, which is what makes it the
right one *inside* the table -- where sigma spans four orders of magnitude and
we do not know in advance which part of the range an element will land in. Past
the cut we do know: sigma is tiny. At that point absolute error is the honest
metric, and it is the one that is bounded.

**Which metric is right depends on where the error comes to rest.** The general
rule, and the thing §3.1 and §3.2 are each applying correctly to different
regimes:

> An error introduced at one point in a computation must be compared against
> the local grid at the point where it is finally **stored**, not at the point
> where it was made. Relative error is the right currency while a value is
> being **multiplied**; absolute error is the right currency once it is
> **added** into something of known magnitude.

Sigma's entire fate is: multiplied by `g`, packed to bf16, multiplied by `V`,
added into an fp32 accumulator, scaled by `1/N`, stored as bf16. Trace it:

```
err_abs(sigma)  <  2^-16                                 (bounded, above)
err_abs(p)      =  g * err_abs(sigma)  <  g * 2^-16      (g is exact)
err_abs(o)      =  (1/N) * |SUM_tail err_abs(p_j) * v_j|
```

Sigma itself is **never stored**. It exists in fp32 registers for two
instructions. No bf16 grid is ever applied to it, so asking what bf16 can
resolve near sigma's magnitude is asking about a quantization that does not
happen.

**ulp(1.0), and why not ulp(0).** `ulp` is the spacing of the float grid *at a
magnitude* -- a property of a location on the number line, not of a number. Over
the normal range, binary floating point is **scale-invariant**: the grid at
magnitude `m` has spacing `m * 2^-7` for bf16 (7 stored mantissa bits). So
`ulp(1.0) = 2^-7 = 7.8125e-3` is just a convenient name for the *relative*
resolution `2^-7`, and the criterion it expresses --

```
an error is invisible if  err / |value stored|  <  2^-8   (half-ulp, RNE)
                                                   2^-7   (full ulp, truncation)
```

-- holds at any magnitude, not only at 1.0. The value stored here is `O`. Its
raw peak is ~2e-5 (`check_fwd.py` scales by `N` to bring that to ~0.02); the
criterion is unchanged either way, because both sides scale together.

`ulp(0)` is where scale-invariance ends, which is why it is the wrong question.
For bf16:

| | value |
|---|---|
| ulp(1.0) = 2^-7 | 7.8125e-3 |
| min normal = 2^-126 | 1.176e-38 |
| min subnormal = 2^-133 (`ulp(0)`) | 9.18e-41 |

Measured against `ulp(0)`, essentially no error in any computation is ever
invisible -- which is the tell that the metric is misapplied, not that the
kernel is broken. Note also what this table says about the intermediate
`p`: bf16 is **fp32's exponent field** with the mantissa truncated, so
`p = 2^-12 = 2.4414e-4` is not flushed to zero, it is *exactly* representable
(mantissa all zeros). Low precision and small dynamic range are independent
properties, and bf16 only has the first. The tail error survives the pack
intact; what makes it invisible happens later.

**The end-to-end budget, scale-free.** Express the tail error in the currency
of the rounding the kernel *already* commits. `pack_p` truncates f32 -> bf16
(a `v_perm_b32` byte select, §1), so every `p` already carries a one-signed
error of up to `2^-7 * |p|`. Compare one tail column against one peak column:

```
err_tail(g)            =  g * 2^-16
err_pack(p_peak)       =  2^-7 * |p_peak|

ratio = (g / |p_peak|) * 2^-9
```

And `|p_peak|` is set by the *same* spread of `g`: for `g << 0`, `sigma -> 1`
and `p -> g`, so `|p_peak| ~ |g_min|`. For a roughly symmetric score
distribution `g_max ~ |g_min|` and the ratio collapses to a constant:

```
ratio ~ 2^-9 = 1.95e-3     INDEPENDENT of the input scale
```

A tail column's error is ~0.2% of the quantization error the kernel commits on
a single peak column. This is the real reason the cut is safe, and it is why it
stays safe as the scores are scaled -- the tail magnitude and the peak both
track `max|g|`, so the ratio does not move. Worked at §6.4's extremes:

| §6.4 scale | max\|g\| | err_tail (max) | err_pack(p_peak) | ratio |
|---|---|---|---|---|
| 1 | 0.3 | tail is EMPTY (0.3 < 16) | 2.3e-3 | -- |
| 10 | 34.5 | 5.3e-4 | 0.270 | 2.0e-3 |
| 100 | 3455.4 | 5.27e-2 | 27.0 | 2.0e-3 |

The absolute numbers move by 100x; the ratio does not move at all.

**What §6.4 shows empirically.** That table scales q/k over 100x, which takes
`max|g|` from 0.3 to 3455 -- i.e. from a tail that is **empty** to one that is
heavily populated. If the clamped tail were a meaningful error source, the
`rel err` column would trend. It does not:

```
scale   1      3      10     30     100
max|g|  0.3    3.1    34.5   310.6  3455.4
rel err 0.641% 0.529% 0.824% 0.962% 0.694%
```

Non-monotonic, inside a 2x band, with the largest value at scale 30 rather
than 100. The tail accounts for at most a few hundredths of a percentage point
of a ~0.7% budget that is dominated by bf16 rounding throughout the kernel. The
independent check is that the LUT and the exact `exp2`/`rcp` gate report the
**same** 1.221e-04 against the fp32 oracle (§1, §9) -- the exact gate has no
clamp at all, so any tail contribution would separate them.

**What would break it.** The per-column bound is solid; the aggregation over
columns is where the assumptions live.

1. **Coherent accumulation.** `err_tail` is one-signed in `p` (always positive:
   `p_c > p_t` everywhere past the cut). It multiplies `v_j`, which is
   mixed-sign and roughly zero-mean, so the sum over `n_tail` columns grows
   like `sqrt(n_tail)`, not `n_tail`. A workload with correlated `V` along the
   tail columns would convert that `sqrt` back into a linear term -- a 1024x
   amplification at `N = 1024` in the limit.
2. **An asymmetric `g` distribution.** The `ratio ~ 2^-9` result assumes
   `g_max ~ |g_min|`. A distribution with a long positive-`g` (strongly
   negative score) tail but no corresponding negative-`g` mass has a small
   `|p_peak|` and a large `g_max`, and the ratio grows as `g_max/|p_peak|`.
3. **A smaller `N`.** Everything above survives the `1/N` because the signal is
   divided by `N` too; that is not a safety margin, just a cancellation.

All three are the same gap §8.2 names: the per-wave spread of `g` on a
production workload is unmeasured, and the inputs here are synthetic. The
instrumentation that closes this closes that.

**The left cut is off by one binade.** Unrelated to the above, but it is the
other half of §3.2's range. `g < -8` gives `1 - sigma = 2^g < 2^-8`, which puts
sigma in `[1 - 2^-8, 1)`. Near 1 *from below* the binade is `[0.5, 1)`, whose
spacing is `2^-8` -- so `1 - 2^-8` is the next representable value down, not a
tie. Round-to-nearest only snaps to exactly 1.0 once `1 - sigma < 2^-9`:

```
g < -9:   1 - sigma < 2^-9   ->  sigma rounds to EXACTLY 1.0 in bf16
```

By that criterion the cut belongs at `-9` and the span is 25, which costs 89
more entries (2135 -> 2224, +356 B).

But that criterion is stricter than the table needs, and the cut at `-8` is
right for a different reason -- the same absolute-vs-relative argument as the
right cut, run in the other direction. Clamping `g < -8` to the left node gives
`sigma_c = sigma(-8)` against a true value approaching 1:

```
rel_err <= 1 - sigma(-8) = 2^-8/(1 + 2^-8) = 3.891e-3 ~ 2^-8
```

which is **exactly the `2^-8` budget §3.1 sized `h` to hit**. The left clamp
adds no error beyond what the table already commits at any interior node, so
the cut is free. "Rounds to exactly 1.0" is a stronger claim that happens to be
false at `-8`; "stays inside the node-error budget" is the true one, and it is
the claim that justifies the span. No change to the shipped table.

### 3.3 Entries and bytes

```
entries = ceil(24 / 0.01125) + 1 = 2135
bytes   = 2135 x 4 B (one f32)   = 8540 B
```

LDS goes 18432 -> 26976 B; occupancy is unaffected (1 WG/CU, VGPR-limited at 2
waves/SIMD either way).

The index bias carries a **`+0.5`** so that `v_cvt_i32_f32`'s truncation
behaves as round-to-nearest, which is what the `h/2` in §3.1 assumes:

```python
_LUT_SCALE = 1.0 / _LUT_H
_LUT_BIAS  = -_LUT_GLO / _LUT_H + 0.5
```

### 3.4 Alternatives considered

Interpolation collapses the size by two orders of magnitude (error goes `O(h)`
-> `O(h^2)`) but costs two more VALU ops per element:

| scheme | 2^-8 | 2^-7 | VALU/elem | LDS budget to break even |
|---|---|---|---|---|
| direct, nearest node | 2135 entries, 8540 B | 1070, 4280 B | 4 | **20 cy** |
| value + slope (1 fma) | 195, 1560 B | 140, 1120 B | 6 | 12 cy |
| lerp between nodes | 96, 384 B | 68, 272 B | 6 | 12 cy |

**Direct wins** — capacity was never the constraint, and fewer VALU ops means
a larger tolerance for the LDS read. This is counterintuitive: the *bigger*
table is the better one.

**Measured, not assumed** (secant-slope variant: entry = `(sigma_i,
sigma_{i+1} - sigma_i)`, h = 0.2548, 96 entries x 8 B = 768 B, LDS
26976 -> 19200 B):

| | phase V | VALU | floor | ms | oracle |
|---|---|---|---|---|---|
| direct, nearest node | 242 instr | 176 | 704 cy | **104.43** | 1.221e-04 |
| interpolated (lerp) | 306 instr | 240 | 960 cy | 126.45 (**+21%**) | 1.221e-04 |

The interpolated form adds exactly the two ops predicted — `v_fract_f32` for
the fraction and `v_fmac_f32` for the interpolation, 32 of each — and the
+256 cy floor difference accounts for essentially all of the +21%. The
24x smaller table buys nothing, because **conflicts were already near-free**
(§6.4), so there was nothing for the extra accuracy-per-byte to pay for.

Accuracy is identical either way (1.221e-04 against the oracle, 1.192e-07
against the shipping kernel) — both are pinned by the bf16 truncation floor,
which is the point of §2. The interpolated table is *more* accurate in
principle and that accuracy is invisible.

**The general lesson:** when a lookup's cost is dominated by ALU, not by the
memory it touches, spend table bytes to buy instructions — never the reverse.
Interpolation is the right trade only when the table is capacity-bound.

Unused refinement: `ds_read_u16_d16_hi` exists on gfx942 and loads 16 bits
straight into bits [31:16] of the destination. Store sigma as **bf16** and read
it that way into a register whose low half is zero, and the result *is* the f32
— no conversion instruction, and the table halves to 4270 B. Not needed here
(LDS is not tight) but worth knowing.

---

## 4. The clamp is the saturating branch, not a safety check

`v_med3_f32` clamping the index to node 0 gives `sigma ~ 1`, and the surviving
multiply then produces `p = g * 1 = g` — the correct answer for a
strongly-passing score **at any magnitude**:

| g | true p | sigma-table (clamp -> sigma, then x g) | |
|---|---|---|---|
| -2 | -1.60 | -1.60 | ok |
| -8 | -7.97 | -7.97 | ok |
| -20 | -20.00 | -19.92 | ok |
| -100 | -100.00 | -99.61 | ok |
| -3455 | -3455.00 | -3441.56 | ok |

This is structurally the same as the Schraudolph clamp rule: **clamp the index,
never the `g` in the outer multiply.**

### 4.1 Why a direct `p`-table does NOT work

Tempting idea: tabulate `p(g)` directly and drop the multiply. It fails, and
the failure is instructive.

**sigma is bounded** — it lives in `(0, 1]` and flattens to 1. **p is
unbounded** — `p -> g` as `g -> -infinity`. When the index clamps:

| g | true p | sigma-table | **p-table** |
|---|---|---|---|
| -20 | -20.00 | -19.92 ok | **-7.97 BAD** |
| -100 | -100.00 | -99.61 ok | **-7.97 BAD** |
| -3455 | -3455.00 | -3441.56 ok | **-7.97 BAD** |

A p-table clamps to a **constant** and is wrong by `|g|`. You cannot fix it by
extending the range, because `p` has no bound — you would need entries down to
whatever the largest `|g|` is, which is data-dependent (measured 3455 at 100x
the benchmark's input scale).

Patching it costs a compare plus a select — **two instructions to save the one
multiply**:

| | index | LDS | interp | fix-up | total |
|---|---|---|---|---|---|
| sigma-table | 4 | 1 | 0 | 1 multiply | **5 VALU + 1 LDS** |
| p-table | 4 | 1 | 0 | 1 cmp + 1 select | **6 VALU + 1 LDS** |

**The multiply that survives is load-bearing. It is what makes the clamp
correct.**

### 4.2 Masked elements stay exactly zero

A masked score gives `g = 0`, hence `p = 0 * sigma = 0` **exactly**, whatever
sigma is. The kernel's documented correctness property (`acc + 0*V` is exact in
f32) survives untouched — this is independent of the table's accuracy.

---

## 5. Building the table

Filled in LDS by the workgroup itself, using the exact `exp2`/`rcp`, before the
prologue's publish fence:

```python
def build_sigma_lut():
    for r in range_constexpr(_LUT_ROUNDS):          # 5 rounds over BLOCK_THREADS
        _idx = tid + fx.Int32(r * BLOCK_THREADS)
        _idx = (_idx < fx.Int32(_LUT_N)).select(_idx, fx.Int32(_LUT_N - 1))
        _gv = _idx.to(fx.Float32) * fx.Float32(_LUT_H) + fx.Float32(_LUT_GLO)
        lut_mem[_idx, fx.Int32(0)] = _rcp(c_one_f + exp2_f32(_gv))
```

Five rounds against the ~270 KV tiles that follow — amortises to nothing. No
host-side table, no extra kernel argument.

**Two FlyDSL gotchas, both cost time:**

1. **A store inside a dynamic `if` fails.** `if _idx < N: lut_mem[...] = v`
   raises `TypeError: Variable(s) ['lut_mem'] initialized as None before a
   dynamic if/else (scf.if)` — the rewriter collects `lut_mem` as an assigned
   name and tries to yield it as an scf.if result. Fix: **clamp instead of
   branch.** Surplus threads in the last round all write the last entry, and
   because the value derives from the *clamped* index they write identical
   bits, so the redundant stores are benign rather than a race.

2. The `lut` field can be declared with a different element type from the rest
   of `SharedStorage`: `fx.Array[fx.Float32, _LUT_N, 16]` alongside the bf16
   K/V arrays. Viewed as `fx.make_layout((_LUT_N, 1), (1, 1))` and read with
   `lut_mem[i, None].load()`.

---

## 6. LDS bank conflicts — I measured this wrong TWICE

**This section is the most transferable part of the document.**

### 6.1 The correct model

LDS on CDNA3: 32 banks x 4 B, `bank = (byte_addr / 4) mod 32`. Within a
half-wave of 32 lanes:

- **different banks** -> parallel, free
- **same address** -> **broadcast**, free
- **different addresses in the SAME bank** -> conflict, serialized

Not sharing an address only removes the *broadcast* benefit; it does not create
conflicts. With indices spread uniformly over a large table, `bank = index mod
32` is uniform, and 32 lanes into 32 bins is balls-in-bins: **expected max
occupancy 3-4, not 32.** Random access is mildly conflicted.

**Maximal conflict needs the opposite of randomness** — a *structured* stride
that is a multiple of 32 dwords (128 B), so every lane lands in the same bank
at a different address.

### 6.2 Artifact #1: 73 cy/read — a pathological stride

My first control derived the index from the score's mantissa bits and read
`k_smem[r, 0, None]`. But:

```
k_layout strides = (HEAD_DIM_K, MFMA_DIM_LANE_K, 1) = (64, 4, 1) in bf16 elements
row stride = 64 * 2 B = 128 B = 32 dwords
-> bank = (r * 32) mod 32 = 0   for EVERY row r
```

**All 32 lanes hit bank 0 at 32 different addresses** — a guaranteed 32-way
conflict. That measured a 2.7x kernel slowdown and looked fatal. It was
measuring an accidental worst-case stride that no real 4 B-entry table has.

### 6.3 Artifact #2: ~0 cy/read — clustered indices

Switching to a value-derived index measured the read as essentially free. But
this benchmark's `U(-0.01, 0.01)` inputs give `|g| <= 0.3`, which at
`h = 0.01125` is ~54 of 2135 entries — **about 2.5% of the table**, so 64 lanes
crowd into a handful of adjacent dwords and the access is near-broadcast by
construction.

### 6.4 The real number

Measured by **scaling q/k until the index actually spreads** (B=256, N=4096,
time relative to F=1 within the same kernel, so the conflict effect is
isolated):

| scale | max\|g\| | entries hit | vs F=1 | rel err | NaN |
|---|---|---|---|---|---|
| 1 | 0.3 | 1 | 1.00x | 0.641% | 0 |
| 3 | 3.1 | 534 | 1.11x | 0.529% | 0 |
| 10 | 34.5 | 2135 | **1.13x** | 0.824% | 0 |
| 30 | 310.6 | 2135 | 1.03x | 0.962% | 0 |
| 100 | 3455.4 | 2135 | 1.02x | 0.694% | 0 |

**Worst case 1.13x on the LDS step** — at the *middle* scale, where the index
spreads across the whole table. It *improves* again at larger scales because
the scores clamp to the table ends and broadcast. Zero NaN throughout, error
stays under 1%.

### 6.5 Methodology lessons

- **A subtraction model is unreliable here.** Estimating the LDS cost as
  `(total delta) - (added VALU x 4 cy)` produced a *negative* per-read cost.
  The "4 cy per full-rate op" calibration does not hold when the phase is
  bloated with added work. Use a **three-way A/B** with a no-read control that
  keeps the identical index math.
- **When building a "remove X" control, watch for DCE.** `pack_p` consumes all
  four `vals`, so a control that drops two of them lets dead-code elimination
  delete half the gate — that measured a fake -18.3%. **Add instructions
  instead of removing them** where possible.
- **Verify the control in the ISA.** Every control in this effort was checked
  by dumping `HSTU_ISA_DIR` and counting opcodes in the `s_setprio`-delimited
  phase regions. Two of them were doing something other than intended.

---

## 6a. Gate grouping: GATE_BATCH 4 -> 32, -1.5% (`a9daf70ce`)

With the LUT in place, the gate's grouping decides how far ahead the table
reads issue. The compiler emits one group at a time:

```
GATE_BATCH=4/8   index x8  | ds_read x8  | (s_waitcnt, mul) x8   -- first waitcnt covers 1 mul
GATE_BATCH=16    index x16 | ds_read x16 | ...                   -- covers 2
GATE_BATCH=32    index x32 | ds_read x32 | ...                   -- covers 18
```

Phase V 242 -> 225 instructions (17 fewer `s_waitcnt`), **free in registers**
(244 VGPR either way -- the address register is reused in place as the loaded
sigma, so a wider group holds no more live values).

| GATE_BATCH | run1 | run2 | run3 | run4 | mean |
|---|---|---|---|---|---|
| 4 | 105.11 | 105.45 | 104.81 | 105.20 | 105.14 |
| 16 | 106.14 | 106.41 | — | — | 106.28 |
| **32** | 103.66 | 103.74 | 103.41 | 103.52 | **103.58** |

**-1.5%, and NON-MONOTONIC** -- 16 is worse than both 4 and 32. Accuracy
unchanged.

**This corrects an earlier conclusion in this document's history.** I built an
explicit software-pipelined gate, measured it "within noise", and reverted it.
That was a MEASUREMENT-QUALITY problem, not a real null: the machine was
drifting 3-5% between runs and could not resolve a 1.5% effect. Always put a
stable in-process control (here, the shipping kernel timed in the same process)
in the comparison before concluding "no effect" -- a blocked A-then-B over a
drifting machine will manufacture nulls and false positives at this magnitude.

What does NOT change is the other half of that claim: `ds_read_b32` still
measures 10.31 cy against phase M's 4.56, and most of that gap is NOT
recoverable, because the wave stalls there while the partner wave in phase M
keeps issuing. **ATT's stall is per-wave, not per-SIMD.** The 1.5% here is the
part that was recoverable; the rest is not time.

---

## 6b. Magic-number addressing: 4 address ops -> 3 (`344694d31`), -3.3%

### 6b.1 The problem

The address was four instructions:

```
v_fmamk_f32     t, g, 1/h, bias        t = (g - g_lo)/h + 0.5
v_med3_f32      t, t, 0, N-1           clamp
v_cvt_i32_f32   i, t                   truncate -> node index
v_lshlrev_b32   a, 2, i                byte address = i*4
```

The obvious saving is to fold the `x4` into the fma -- scale by `4/h` instead
of `1/h` -- and drop the shift. **It does not work**, because truncation does
not commute with scaling:

```
trunc(4x) != 4 * trunc(x)          e.g. x = 1.3:  trunc(5.2) = 5,  4*trunc(1.3) = 4
```

`trunc(4t) = 4i + f` with `f` in {0,1,2,3}, i.e. a byte address *inside* entry
`i` rather than at its start. `ds_read_b32` needs 4-byte alignment, so you
would have to mask the low two bits back off -- one op to remove one op.

### 6b.2 The trick: let the exponent do the scaling

Recall the f32 layout (§3.1): `v = (1 + m/2^23) * 2^E`, where `m` is the
23-bit **stored** mantissa. Fix the exponent at `E = 21`:

```
v = (1 + m/2^23) * 2^21
  = 2^21 + m * 2^(21-23)
  = 2^21 + m/4

  =>   m = 4 * (v - 2^21)
```

**The mantissa field IS the offset-from-2^21, pre-multiplied by four.** And
four is exactly the scale a 4-byte table entry needs. So if the index arrives
biased by `2^21`, its byte address is sitting in the mantissa already --
no convert, no shift, just a mask.

Equivalently, in ULP terms: the spacing of representable floats in
`[2^21, 2^22)` is `2^21 / 2^23 = 0.25`. A float there is stored as *a count of
quarters above 2^21*, and `index * 4` is precisely "index, counted in
quarters".

### 6b.3 Why 2^21 specifically

The magic constant choice is not arbitrary -- it is picked so the ULP matches
the entry size:

| magic | ULP | mantissa holds | to get a BYTE address |
|---|---|---|---|
| `2^23` | 1 | `index` | needs `m << 2` -- an extra op |
| **`2^21`** | **0.25** | **`index * 4`** | **already there** |
| `2^19` | 0.0625 | `index * 16` | would need `m >> 2` |

`2^23` is the textbook magic number for float->int conversion, and here it
would be one op *worse*. The entry size dictates the constant.

### 6b.4 The bit picture

For `t` in `[2^21, 2^22)` the exponent field is fixed at `127 + 21 = 148`:

```
                 31 30      23 22                           0
 bits(t)        [ 0][ 1 0 0 1 0 1 0 0 ][       m (23 bits)      ]
                sign    exp = 148                 = index*4
                 0     0x4A000000                  <= 0x2158

 mask 0x7FFFFC   0  0 0 0 0 0 0 0 0     1 1 1 ... 1 1 0 0
                 \--- strips the exponent ---/     \-- clears the
                                                      low 2 bits --/
```

So `bits(t) & 0x7FFFFC` is the byte address, in **one `v_and_b32`**. The
exponent is constant and masked away; the low two bits are the sub-node
fraction and are cleared.

### 6b.5 Worked examples

Constants as they appear in the ISA: `big = 2^21 + 711.5 = 0x4A000B1E`,
clamp bounds `0x4A000000` and `0x4A002158`.

| g | t = g*88.889 + big | bits(t) | exp | mantissa m | m & ~3 | idx | g_i |
|---|---|---|---|---|---|---|---|
| -3455.0 | 2097152.00 | `0x4A000000` | 148 | 0 | 0 | 0 | -8.0000 |
| -8.0 | 2097152.50 | `0x4A000002` | 148 | 2 | 0 | 0 | -8.0000 |
| -1.2785 | 2097749.75 | `0x4A000957` | 148 | 2391 | 2388 | 597 | -1.2838 |
| 0.0 | 2097863.50 | `0x4A000B1E` | 148 | 2846 | 2844 | 711 | -0.0013 |
| 0.3 | 2097890.25 | `0x4A000B89` | 148 | 2953 | 2952 | 738 | 0.3025 |
| 16.0 | 2099285.75 | `0x4A002157` | 148 | 8535 | 8532 | 2133 | 15.9962 |

Read the `0.0` row: `g = 0` gives exactly the big constant back, mantissa
2846, byte address 2844, node 711 at `g_i = -0.0013`. The node is slightly off
zero and it does not matter -- `p = 0 * sigma = 0` exactly, whatever sigma is.

Read the `-3455` row: `t` underflows the clamp, pinned to `0x4A000000`,
mantissa 0, node 0, `sigma ~ 1`, and the surviving multiply gives `p = g`.

### 6b.6 What the compiler actually did

The source writes the ELEMENT index and lets the view scale it:

```python
idx = (t.bitcast(fx.Int32) >> fx.Int32(2)) & fx.Int32(0x1FFFFF)
return Vec(lut_mem[idx, None].load())[0]
```

That is `>>2`, `&`, then the view's `x4`. The backend folded all three into a
single `v_and_b32` with mask `0x7FFFFC`, because
`((b >> 2) & 0x1FFFFF) << 2 == b & 0x7FFFFC`. So two instructions
(`v_cvt_i32_f32` and `v_lshlrev_b32`) are replaced by one -- the address math
goes **4 ops -> 3**, a net saving of one per element. The ISA confirms it:
phase V 225 -> 193 is exactly -32 over 32 elements.

Worth checking the ISA rather than assuming here: written naively the source
looks like 3 ops (`>>`, `&`, and the view's `x4`) replacing 2, i.e. a loss.
The fold is what makes it a win.

```
phase V      225 -> 193 instructions
VALU floor   176x4 = 704  ->  144x4 = 576 cy
measured     104.41 -> 101.07 ms      (-3.3%, 4 vs 3 interleaved runs)
```

### 6b.7 The cost: the bias is quantised too

This is a real tradeoff, not a free win. The big constant must itself be
representable in `[2^21, 2^22)`, i.e. **a multiple of the 0.25 ULP**. The
ideal bias is `-g_lo/h + 0.5 = 711.6111`, which is not. It snaps to a quarter,
so the rounding point jitters by up to `+-0.125` node instead of sitting at
exactly `0.5` -- the worst-case node distance widens from `0.5h` to `0.625h`.

Measured over 3M samples:

| bias | max rel err of sigma |
|---|---|
| ideal 711.611 (old form, exact) | 3.906e-3 |
| **711.5** (chosen) | **3.998e-3** |
| 711.75 | 5.939e-3 |
| 711.25 | 5.756e-3 |
| bf16 truncation floor (§2) | 7.812e-3 |

2% worse at the best quarter, and all three quarters stay under the floor --
so end-to-end accuracy does not move: **1.221e-04** against the fp32 oracle on
every mask configuration. Pick the quarter by measurement; `711.25` and
`711.75` are needlessly 50% worse for no gain.

### 6b.8 Preconditions, if this is reused elsewhere

* The index range must fit the mantissa: `index * 4 < 2^23`, i.e. fewer than
  2^21 entries. Ours uses 8536 of 8388608.
* The clamp must be applied **in the biased domain** (bounds `2^21` and
  `2^21 + N-1`), so the exponent is guaranteed to be 148 before the mask. An
  unclamped value outside `[2^21, 2^22)` has a different exponent and the mask
  returns garbage -- the clamp is load-bearing twice over, here and in §4.
* The entry size fixes the magic constant (§6b.3). A 2-byte table wants
  `2^22`, an 8-byte table `2^20`.
* Negative inputs never reach the mask: the clamp pins them to `2^21` first.

---

## 7. What was ruled out along the way

| idea | verdict | why |
|---|---|---|
| **MFMA for the pack** | no | No MFMA has a packed-bf16 destination on gfx942 or gfx950 (`v_mfma_bf16_*` does not exist). And MFMA's output rate per lane is **4 cy/result — identical to `v_perm_b32`**. Its 16x advantage is arithmetic *density* (4096 MACs/instr), useless for data movement with 1 MAC per output. |
| **MFMA eating f32 P** (skip the pack) | no | `v_mfma_f32_16x16x4_f32` *does* exist (my earlier "NO" was a test-harness bug — a `:` inside `a[0:3]` truncated the string). But K drops 16 -> 4, so GEMM2 goes 32 -> 128 MFMAs, +1536 cy against saving 64 cy. V would also have to be f32. |
| **f32-as-two-bf16 + zero mask** | no | Numerically correct — the high half of an f32 *is* bf16(p), and zeroing matching V entries kills the garbage term. But K doubles, so GEMM2 goes 32 -> 64 MFMAs (+554 cy) to save 64 cy, and V needs interleaved zeros (2x LDS, 2x ds_write). **On gfx950 this changes**: `v_mfma_f32_16x16x32_bf16` exists there, so the instruction count would be unchanged — worth re-examining if ported. |
| **Sparse MFMA** (`v_smfmac_f32_16x16x32_bf16`) | no | Exists on gfx942, but its A-operand is the *compressed* non-zeros — contiguous packed bf16, exactly what `pack_p` produces. It **requires** the pack rather than replacing it. |
| **32x32x8 instead of 16x16x16** | no | Half the instructions, each 2x as long — **matrix time identical at 1024 cy**. Only the ~84 cy of per-instruction excess is on the table, in a phase that is not the critical path. And the 32x32 accumulator layout breaks the deliberate operand swap that keeps P register-resident between the GEMMs. |
| **Removing one of the two barriers** | no, **14% slower** | The barrier is the ping-pong's arbitration mechanism, not overhead. Without it the wave groups drift out of opposite phases and contend for the same unit. (It is also a correctness fence for the LDS publish, and barrier counts must stay equal because of the prologue skew.) |
| **Batch inversion for `rcp`** | no | `1 rcp + 3(n-1) muls` asymptotes at **12 cy/element** vs 16 — only 4 cy of headroom, because full-rate:transcendental is just 4:1 here. Normalising by the per-lane max costs 31 `v_max_f32` = 124 cy, more than the entire saving, **and converts overflow into underflow** (needs `D < 15.3` where unnormalised needs only `sum(max(g,0)) < 128`). Measured 8-12% slower at n=4/8/16/32 in all three variants; the real killer is the serial prefix-product chain replacing 32 independently-pipelined reciprocals. |
| **gfx942 packed FP32** (`v_pk_add/mul_f32`) | no, **16% slower** | Works — 64 of each, phase V 146 -> 114 instructions — and still loses. A packed op is a **join**: it cannot issue until BOTH halves' producers retire, halving the independent chain count. This is already documented beside `_PP_COMPILE_HINTS`, which disables SLP for the same reason. |
| **SALU** | n/a | Wave-uniform only; the gate is per-lane f32. Note the *masked* path already uses `s_and_b64` / `s_or_b64` / `s_cselect_b64` for predicate combination — the compiler moves boolean work to the scalar pipe automatically wherever the type permits. |

---

## 8. Open questions and risks

### 8.1 The table is sized conservatively

§3.1 uses **relative error of sigma**, which is dominated by the
large-positive-g tail where `p` is tiny and contributes almost nothing. If the
metric is switched to **absolute error normalised to peak `|p|`** — which is
what `check_fwd.py` actually asserts — the same table collapses from **2135
entries to 17** (lerp) or a few hundred (direct). The 2135 figure is the
defensible one to ship without knowing the score distribution; a much smaller
table is probably fine and would conflict *less*.

### 8.2 Bank conflicts depend on data the benchmark does not have

The 1.13x worst case in §6.4 is measured by artificially scaling synthetic
inputs. Real HSTU score distributions are **unmeasured**. The same gap blocks
the Schraudolph clamp analysis (§7.1 of `hstu_fwd_schraudolph_exp2.md`). The
instrumentation that would close both: record the per-wave spread of `g` on a
production workload.

### 8.3 Relationship to Schraudolph

The reverted Schraudolph commit (`18e42128d`, patch kept at
`tmp/schraudolph_exp2.patch`) measured **1.320x**. The LUT beats it at
**1.391x**, and for the same underlying reason — the bf16 truncation floor
means the accuracy target is `2^-8`, not `2^-24`. The two are **mutually
exclusive** (both replace `exp2`); the LUT also removes the `rcp` and the
`1 + e` add, which Schraudolph leaves in place. That is the whole difference:

```
Schraudolph   removes 1 transcendental per element     ->  1.320x
sigma LUT     removes 2 transcendentals + 1 add        ->  1.391x
```

### 8.4 Not re-examined after this change

The masked-remainder loop was measured as memory/barrier-bound *before* the
LUT, and both its phases could be gutted with zero gain. The LUT substantially
shortens masked phase V too, so that balance may have moved. Worth re-running
the phase-V/phase-M stub controls.

---

## 9. Reproducing

```bash
# resources: VGPR / spills / LDS / "Workgroups per CU" must stay 1
python tmp/hstu_repro/stats_fwd_pp.py -H 4 -s 16384 --head-dim 64 \
    --target-size 10 --block-m 512 --block-n 32

# the binding accuracy test -- must stay 1.221e-04
AITER_HSTU_FWD_PINGPONG=1 python tmp/hstu_repro/check_fwd.py \
    -b 31 32 33 -H 4 -s 1024 --head-dim 64 --mask hstu --target-size 10

# magnitude vs the shipping kernel across all six mask configurations
python tmp/hstu_repro/check_fwd_pp.py -b 31 32 33 -H 4 -s 1024 \
    --head-dim 64 --repeats 6 --block-m 512 --block-n 32

# perf, ABBA against the shipping tuned kernel
python tmp/hstu_repro/bench_fwd_pp.py --reps 3 -b 1024 -H 4 -s 16384 \
    --head-dim 64 --target-size 10 --tiles 512,32

# ISA: phase regions are delimited by `s_setprio 1` ... `s_setprio 0`
HSTU_ISA_DIR=/tmp/isa FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python tmp/hstu_repro/stats_fwd_pp.py --block-m 512 --block-n 32
```

Tunables live at build scope in `build_hstu_attention_fwd_pp` (FlyDSL's disk
cache keys on the kernel function's source text and closure values, **not** on
module globals — a tunable as a module global silently re-measures the old
binary):

```python
USE_SIGMA_LUT = True
_LUT_GLO, _LUT_GHI = -8.0, 16.0
_LUT_H = 0.01125
_LUT_N = ceil((_LUT_GHI - _LUT_GLO) / _LUT_H) + 1     # 2135
_LUT_SCALE = 1.0 / _LUT_H
_LUT_BIAS  = -_LUT_GLO / _LUT_H + 0.5                 # +0.5 -> trunc == round-to-nearest
```

Setting `USE_SIGMA_LUT = False` restores the exact `exp2`/`rcp` gate.
