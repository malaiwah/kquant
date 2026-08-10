# Kimi-K3-QSRT technical brief

Status: TP-independent canonical storage and TP12 runtime qualification,
2026-08-07.

`QSRT` means **Quantile-Stratified Rate-shifted Trellis codec**. QSRT is an
expert-static, fixed-payload mixed-rate trellis codec for gated
mixture-of-experts weights. The Kimi-K3 construction combines three
independently useful ideas:

1. an L16 stratified-quantile graph whose transitions reconstruct finite E4M3
   values (`SQG-E4M3`);
2. equal-byte K2/K4 exchanges around K3 (`R0`, `R1`, and `R2`), selected
   separately for fused `w1`/`w3` and for `w2`; and
3. `X4T`, an exact high-quality endpoint that preserves the official MXFP4
   nibble plane and losslessly compresses its UE8M0 scale plane.

The intended artifact name is `Kimi-K3-QSRT`. The first usable checkpoint will
contain QSRT experts selected from a fresh all-expert candidate pool and X4T
experts chosen by an exact-byte global allocator. There is no raw-MXFP4 keep
tier in the storage contract.

## Frozen scope

The initial production experiment is intentionally narrow:

```text
canonical storage              TP-independent 32-channel balanced atoms
qualified runtime              TP12 first; TP is not serialized in the codec
trellis window                 L16
reconstruction family         SQG normal -> finite E4M3, RNE
lossy rate candidates          R0, R1, R2
gate/up decision               one shared r13
down-projection decision       independent r2
high-quality endpoint          exact X4T
source weights                 official Kimi-K3 MXFP4 checkpoint
calibration teacher            resident interim EXL3 checkpoint
encoder objective              expert-stratified dense-H BlockLDLQ + routed replay
```

The current performance gate is TP12 because that is the available Kimi-K3
deployment. TP4, TP8, TP16, TP24, and TP32 are storage-valid direct-load views
of the same artifact and require kernel qualification, not re-encoding.
Alternate companders, wider rate ladders, learned per-layer tables, and
entropy-coded hot streams remain outside the supported surface.

## Encoder ownership

QSRT's offline implementation is owned by kquant. The mixed-rate dense-H
BlockLDLQ backend is `kquant/exl3_encoder_backend.py`; SQG label generation,
packed traceback, tail-biting Viterbi, and its CUDA sources live under
`kquant/sqg_e4m3.py`, `kquant/sqg_quantizer.py`, and `kquant/csrc`.

The ExLlamaV3 checkout is an unmodified upstream dependency. It supplies only
the established EXL packing, Hadamard, and tensor utilities used by the
encoder. No QSRT format, rate-selection, SQG, LDLQ, or CUDA change may be
carried as a local ExLlamaV3 patch. The exact upstream-derived source retained
in kquant is covered by `THIRD_PARTY_NOTICES.md`.

## QSRT-E4M3 reconstruction

QSRT's reconstruction mechanism is the **Stratified Quantile Graph (SQG)**.
SQG assigns the $2^L$ directed edges of an $L$-bit de Bruijn trellis
bijectively to equal-probability microcells of a reference distribution. At
rate $K$, each state retains $L-K$ history bits and has $2^K$ outgoing
branches. A history-dependent branch permutation selects one branch from each
of $2^K$ coarse quantile strata, while a bijective state permutation selects
the within-stratum phase. Consequently, every state exposes exactly one
reconstruction candidate from every stratum, and every global probability
rank occurs exactly once across the directed edge set.

The graph and scalar reconstruction law are separate design objects. For a
reference distribution with quantile function $F^{-1}$, microcell $r$ spans

$$
I_r = \left[\frac{r}{2^L},\frac{r+1}{2^L}\right),
$$

and its canonical representative is the conditional mean

$$
c_r = \mathbb{E}\!\left[X\mid F(X)\in I_r\right].
$$

This value is MSE-optimal within that microcell. It is then projected with
round-to-nearest-even to finite E4M3. Numerically identical E4M3 labels may
remain on different directed edges and lead to different successors; scalar
label collisions therefore do not collapse the richer trellis geometry.

The public runtime profile is **QSRT-E4M3**. Its definition is the composition
of an SQG rank map and a finite reconstruction staircase:

$$
\text{codeword}
\xrightarrow{G_K}
r
\xrightarrow{Y_{12}}
\text{finite E4M3}.
$$

The graph $G_K$ and scalar law $Y_{12}$ are independent mathematical objects.
In particular, the carry-mixed graph does not approximate an inverse CDF or a
Chebyshev polynomial.

### Carry-mixed SQG rank map

For $L=16$, rate $K\in\{2,3,4\}$, and $w=16-K$, split a codeword $t$ into
history and physical branch:

$$
h=t\mathbin{\gg}K,
\qquad
b=t\mathbin{\&}(2^K-1).
$$

With $M_w=2^w-1$, define

$$
\begin{aligned}
x_0 &= h\oplus(h\gg11),\\
x_1 &= x_0\oplus((x_0\ll11)\mathbin{\&}M_w),\\
p &= (\mathtt{0x3FA7D929}\,x_1+\mathtt{0xC928FD8E})\bmod2^{32},\\
\phi &= p\mathbin{\&}M_w,\\
s_K &= p\gg(32-K),\\
j &= \operatorname{rev}_K(b)\oplus s_K,\\
G_K(h,b)=r &= (j\ll w)\mathbin{|}\phi.
\end{aligned}
$$

Here $\operatorname{rev}_K$ reverses the $K$ branch bits. Both xorshifts are
triangular bijections on $w$ bits, and `0x3FA7D929` is odd, so multiplication
by it is invertible modulo $2^w$. Therefore $h\mapsto\phi$ is a permutation.
For fixed $h$, $b\mapsto j$ is also a permutation. It follows that

$$
(h,b)\longleftrightarrow(j,\phi)
$$

is a bijection over all $2^{16}$ directed edges. Every state has exactly one
outgoing branch in each of its $2^K$ strata, and every global rank occurs once.

The rank bijection does not by itself specify sequence behavior. If $P(h)=\phi$
is the phase permutation, $\pi_h(b)=j$ is the branch permutation, and $T$ is
the de Bruijn successor, then logical stratum $j$ induces the continuation map

$$
F_j(\phi)=P\!\left(
T\!\left(P^{-1}(\phi),\pi_{P^{-1}(\phi)}^{-1}(j)\right)
\right).
$$

The family $\{F_j\}$ is the branch-conditioned phase-transition geometry used
by Viterbi. It is part of $G_K$, not part of the scalar reconstruction law.

### Chebyshev-derived finite staircase

The exact normal staircase is defined on every global rank $r$ by

$$
u_r=\frac{r+\tfrac12}{65536},
\qquad
z_r=\Phi^{-1}(u_r),
\qquad
Y(r)=\operatorname{RNE}_{\mathrm{E4M3FN}}(1.5z_r).
$$

An E4M3-aware piecewise Chebyshev construction is a compact synthesis of this
discrete map. For the target byte $Y(r)$, its polynomial output is constrained
to lie inside the real interval that rounds to $Y(r)$. Exhaustive evaluation
over all 65,536 ranks proves byte identity. Chebyshev therefore derives the
rank-to-byte staircase; it does not participate in $G_K$.

### Twelve-bit execution staircase

QSRT-E4M3 compresses the exact staircase from 65,536 rank labels to 4,096
bytes. For $q\in\{0,\ldots,4095\}$, define

$$
Y_{12}(q)=
\operatorname{mode}\{Y(16q),Y(16q+1),\ldots,Y(16q+15)\},
$$

with the lower unsigned E4M3 byte selected on a modal tie. Runtime
reconstruction is

$$
\widehat Y_K(h,b)=Y_{12}\!\left(G_K(h,b)\gg4\right).
$$

Thus the 12-bit table is a piecewise-constant approximation to the discrete
Chebyshev-derived E4M3 staircase. It is not a Chebyshev evaluator and does not
change the reference distribution. The approximation chain is

$$
\text{normal equal-probability rank}
\longrightarrow
\text{Chebyshev-derived finite-E4M3 label}
\longrightarrow
\text{modal 16-rank execution label}.
$$

The authoritative construction is implemented independently in
`kquant/sqg_e4m3.py` and B12X. kquant generates the 4,096-byte staircase and
complete K2/K3/K4 direct encoder labels and passes them through
`kquant/sqg_quantizer.py`; B12X evaluates the same immutable construction at
runtime. Frozen SHA-256 checks over the T12 table and all three 65,536-byte
direct tables make cross-repository drift fail in unit tests. A payload
encoded under a different graph cannot be relabelled in place because
$\{F_j\}$ and the selected Viterbi paths differ.

## Fixed-payload rate shifting

The common 3,072-neuron intermediate axis is divided into 24 records of 128
neurons.  Each record retains the existing 16x16 coding tiles.  A single
function-preserving permutation is applied as

```text
W1' = P W1
W3' = P W3
W2' = W2 P^T.
```

Because SiTU is coordinatewise, this changes neither the expert function nor
the coordinate presented at the nonlinear boundary.  It makes importance
regions contiguous and makes the record rate derivable from a small mode ID,
without a per-channel rate map or runtime shuffle.

For a matrix family, mode `Rr` assigns

```text
first r records       K2
middle 24 - 2r        K3
last r records        K4.
```

Thus every mode averages exactly three path bits per weight:

```text
R0 =  0 K2 + 24 K3 + 0 K4
R1 =  1 K2 + 22 K3 + 1 K4
R2 =  2 K2 + 20 K3 + 2 K4
```

QSRT redistributes rate over paired 128-channel records without changing
payload size. A `P24` container assigns K2 to a low-priority donor record and
K4 to a high-priority recipient, while a `P33` container assigns K3 to both.
Each consumes six trellis bits per coefficient pair and occupies the same
physical size. Pair placement is rotated over a global 96-slot atom axis by
layer and expert so P24 work remains balanced at every supported TP view.

The mode is `(r13, r2)`. `w1` and `w3` share `r13` for fused execution;
`w2` selects `r2` independently. The common physical neuron permutation does
not require the three matrices to share a rate schedule.

### Allocation coordinate invariant

The neuron permutation and the rate allocator act at different granularities
and must be composed in a fixed order. First freeze one bijection $P$ over the
3,072 intermediate coordinates and apply it identically to `w1`, `w3`, and
`w2` as above. Then define every record or tile rate decision in that encoder
coordinate system. The same $P$ must be used to construct K2/K3/K4 candidate
errors, select the rate map, rerun BlockLDLQ with the selected map, and score
the reconstructed expert after undoing $P$.

The present conditioning search is a restricted subset of those bijections:
it moves indivisible contiguous four-channel groups and forms each 16-channel
neuron band from exactly four such groups. Each neuron band intersects all 224
orthogonal 16-channel bands, producing 224 distinct 16x16 coefficient tiles;
the funding decision belongs to one such intersection, not to the neuron band
alone. This restriction preserves the encoder's score and packing unit. It
must not be described as a tile permutation: one group move changes every
orthogonal tile incident on those four channels. The rate allocator sees only
the completed post-permutation tile grid and cannot move, split, or reinterpret
a group.

Conditioning policies span exact sensitivity order, within-record balancing,
source-shape clustering, and joint sensitivity/rate-response clustering. A
rate-response feature for one four-channel group is computed only after a
preliminary K2/K3/K4 encode and contains its regularized K3 error and K2/K3
and K3/K4 error ratios across every one of the 224 incident orthogonal bands,
separately for upstream and down. The current search preserves each
128-channel importance population and clusters only the 32 four-channel groups
inside that record. It does not move a favorable local group into a different
donor or recipient population. This is a fixed fit-only proposal rule, not a
fitted continuous hyperparameter.

A tile allocator is not allowed to change $P$. Searching a conditioning
permutation is an outer discrete experiment: each proposed $P$ receives a
complete, independently encoded and validated rate-allocation search. Within
a fixed 128-channel record, a conditioning policy may reorder its eight
16-channel stripes without changing record membership. Moving a channel
between records changes both the candidate basis and its donor/recipient
population and therefore requires rebuilding all rate-error surfaces.

Formally, let $A_{13}$ be the shared gate/up tile-rate map and $A_2$ the down
tile-rate map. The experiment is bilevel:

$$
(A_{13}^*(P),A_2^*(P))
=\arg\min_{A_{13},A_2}D_{\mathrm{fit}}(P,A_{13},A_2),
$$

followed by comparison of

$$
D_{\mathrm{confirm}}
\left(P,A_{13}^*(P),A_2^*(P)\right).
$$

The confirmation partition never selects a tile, a permutation, or a prefix.
When selecting the permutation policy itself, aggregate fit evidence across
training experts and report its result on held-out experts; do not choose the
policy from the same experts' confirmation scores. Every reported candidate
uses a complete BlockLDLQ re-encode and an independently fitted scale. Local
tile errors and single-toggle functional deltas are proposal statistics only,
because the channel permutation and BlockLDLQ feedback couple many tiles.

Gate and up share an intermediate-axis rate map. Down uses an independent map
over its matching intermediate axis because its orthogonal 16-channel tile
coordinate has different functional meaning. A tile-funding experiment may
therefore share a bitmap between `w1` and `w3`, but it must not silently reuse
that bitmap for `w2`. All compared candidates record the frozen permutation
identity, and the research encoder rejects implicit or mismatched permutation
bases.

### Constant-payload 3.083-bpw allocation

For 24 intermediate-axis records, an all-QSRT allocation with two more K4
records than K2 records satisfies

$$
N K2 + (22-2N)K3 + (N+2)K4 = 74
$$

trellis bits per 24 coefficients, or $74/24=3.083\overline{3}$ trellis bpw.
At record granularity this is one P44 pair, $N$ P24 pairs, and $11-N$ P33
pairs. At tile granularity the same identity is enforced independently for
every 16-channel strip after the neuron permutation is frozen. The shared
gate/up map and independent down map must each sum to 74 bits in every strip.

Two fixed-stride selector grammars were evaluated as research controls. The
paired grammar stores one 32-bit word per strip containing eleven P33/P24 bits
for gate/up and eleven for down. Across 1,792 strips this costs 7,168 bytes per
expert, or 0.001736 bpw over all three expert matrices. A top-two-K4 grammar
also stores one 32-bit word per strip: each of the gate/up and down K4-record
pairs is one of $\binom{24}{2}=276$ possibilities and therefore needs nine
bits. This likewise costs 7,168 bytes, or 0.001736 bpw. Its disposable
rank-local serving view may expand the canonical word into two 24-bit masks,
but those 14,336 prepared bytes are not checkpoint payload. Both grammars have
a trellis-plus-selector rate of 3.085069 bpw.
Scale and container metadata are accounted separately by exact serialized
bytes.

The current numerical gate uses `h2_reverse` as the frozen permutation. A
four-expert layer-24 screen compared two allocation-conditioned permutations
that moved only intact 16-channel bands within fixed 128-channel records. The
P24-pair objective and top-two-K4 objective improved their fit proxies, but
their isolated serial confirmation SSE regressed by 0.277% and 0.293%
respectively. A preceding 24-expert split likewise rejected rate-response
clustering by 0.205% on held-out experts. These policies remain research
controls; tile allocation does not authorize changing the production neuron
ordering.

A 24-expert panel spanning layers 1, 24, and 40 used fit documents for every
schedule decision and document-disjoint confirmation rows for the final
comparison. The isolated confirmation results were:

| 3.083-bpw schedule | Pooled SSE | Change from K3 |
| --- | ---: | ---: |
| K3 on all 24 records | 3.335706 | baseline |
| Fixed 22 K3 + 2 K4 records | 2.938512 | -11.907% |
| Tile-top-two K4 for gate/up only | 2.941872 | -11.807% |
| Tile-top-two K4 for down only | 2.944253 | -11.735% |
| Tile-top-two K4 for both axes | 2.950014 | -11.563% |

The fixed record schedule is therefore the qualified first 3.083-bpw profile.
It is slightly better than every tile-top-two selector while requiring no
selector payload, no tile-local rate bookkeeping, and no new kernel grammar.
An exploratory broad schedule search reached 2.932161 pooled SSE, only 0.216%
below the fixed schedule, but its candidate shortlist was formed from batched
fit proxies and is not production evidence. It does not justify the added
metadata or runtime surface.

All 24 fixed-schedule experts improved over K3. On a four-expert scale-control
subset, separately closing both sides left the high-rate schedules 17.79%
below scale-closed K3. The five-point path-aware schedule-specific scale search
regressed the high-rate pooled result by 0.113% and improved K3 by 0.174%, so
the profile retains the source-local scale fitted by the uniform-K3 procedure.
The serialized trellis rate is exactly $74/24=3.083\overline{3}$ bpw before
the existing scale and container metadata.

## TP-independent balanced-atom storage

Tensor parallelism is a view over the checkpoint, not part of the codec. The
canonical sharding unit is a balanced 32-channel atom. For logical mirrored
record pair $i\in\{0,\ldots,11\}$ and 16-channel stripe
$s\in\{0,\ldots,7\}$, define

$$
a=8i+s.
$$

The encoder serializes logical donor/recipient pair $i$ as physical records
$2i$ and $2i+1$. Atom $a$ owns stripe $s$ from both of those physical
records. In mode `Rr`, its rate pair is

$$
(K_\mathrm{low},K_\mathrm{high})=
\begin{cases}
(2,4),&i<r,\\
(3,3),&i\ge r.
\end{cases}
$$

Both cases contain exactly six trellis bits per coefficient pair. For one
matrix, one atom therefore occupies exactly

$$
32\cdot3584\cdot\frac{3}{8}=43{,}008\ \text{bytes}.
$$

The atom bundle stores the fixed trellis fragments for `w1`, `w3`, and `w2`
plus their three 32-value FP16 intermediate-side scale fragments:

$$
B_\mathrm{atom}
=3(43{,}008+32\cdot2)
=129{,}216\ \text{bytes}.
$$

There are 96 atoms per compressed expert, so atomization preserves the exact
payload:

$$
96B_\mathrm{atom}=12{,}404{,}736\ \text{bytes per expert}.
$$

It adds no rate padding and cannot separate coupled coordinates: both sides
of a P24/P33 pair, all three expert matrices, and all three local scale
fragments have one atom owner.

### Physical atom order

Let

$$
\rho_{\ell,e}=(5e+\ell)\bmod12.
$$

The physical slot of logical atom $a$ is

$$
p=(a+8\rho_{\ell,e})\bmod96.
$$

The rotation is defined over the model-global atom axis and contains no TP
rank. It rotates complete record pairs, leaves the stripe index unchanged,
and is bijective for every layer/expert. Both on-disk revisions are
atom-major:

```text
atoms-v1: [96 physical atom slots, compressed experts, 129216 bytes]
atoms-v2: [96 physical atom slots, fixed padded row stride]
```

The layer file is a standards-valid safetensors container. Its tensors are the
atom slab, a 4 KiB expert-format section, and a 24 KiB shared-scale section.
The safetensors header itself occupies a fixed 4 KiB. Each physical atom row
is padded once to a 4 KiB boundary, at most 4,095 bytes per slot per layer
rather than per expert. These fixed offsets permit direct range loading of
physical atom rows without parsing or copying unrelated payload bytes.

Atoms-v1 uses a uniform 129,216-byte bundle for each three-bit QSRT expert.
Atoms-v2 retains the same 96-row container and atom ownership, but permits a
fixed profile to divide each row into compact groups with different bundle
widths. Group membership is a deterministic function of layer, expert, and
physical record pair; it is not serialized as a TP-specific map.

### Shard views

For any TP size $T$ dividing 96, rank $q$ owns

$$
A=96/T
$$

consecutive physical atom slots beginning at $qA$. Its local intermediate
width is $32A=3072/T$. Consequently one rank loads one aligned contiguous
extent per layer; no trellis bit is decoded, shifted, concatenated, or
repacked. All practical Kimi-K3 views through TP32 are exact direct views:

```text
TP = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32
```

TP48 and TP96 are also equal-width views. A shard count that does not divide
96 still receives one aligned, contiguous range of complete atoms. The
quotient/remainder partition covers every atom once and differs by at most one
atom, or 32 intermediate channels, between shards. A runtime requiring equal
local shapes pads only its disposable prepared cache; canonical bytes remain
unchanged. Thus arbitrary resharding never separates a P24/P33 atom or
requires trellis re-encoding.

At TP12, a rank owns eight atoms, exactly 256 intermediate channels. This is
a consequence of the canonical layout, not a serialized TP12 contract.

### Load preparation

The loader reads the small layer metadata and uses InstantTensor to transfer
only its atom-row range. It then performs one GPU preparation pass that removes
slot padding and transposes atom-major
storage into the fused-MoE operand layout. It also derives P24/P33 work queues
from `(layer, expert, physical_slot, r13, r2)`. Rate metadata is per
expert/atom during preparation; the fused coefficient loop has no TP-dependent
addressing and no coefficient-level rate branch. Rank-local prepared buffers
are disposable caches and are never checkpoint files.

The canonical implementation and byte-accounting reference are in
`kquant/qsrt_storage.py`.

### Fixed high-rate `atoms_v2` profile

The 3.083-bpw all-QSRT profile has no P24 pairs or tile selector. In logical
importance order, records 0 through 21 are K3 and records 22 and 23 are K4.
The checkpoint then applies one expert-static permutation of complete
128-channel records, shared by `w1` rows, `w3` rows, and `w2` columns. This is
the exact symmetry

$$
W_1'=PW_1,\qquad W_3'=PW_3,\qquad W_2'=W_2P^\mathsf{T}.
$$

No channel, 16-channel tile, or transformed block is split by this placement.
Because the encoder's intermediate transform is block-128, moving complete
records commutes with that transform. The rate schedule, reconstructed expert
function, and source-local scales are therefore unchanged by physical
balancing.

The two K4 records are placed in distinct physical record pairs. The pair
assignment is rotated by

$$
\rho_{\ell,e}=(5e+\ell)\bmod12,
$$

which balances K4 work across serving shards without serializing a TP count.
The profile uses the atoms-v2 revision of the canonical atom container. Every
physical record pair contributes eight consecutive atom rows. Within each
row, experts using P33 are stored first in ascending expert order with a
129,216-byte bundle; experts using P43 follow in ascending expert order with a
150,720-byte bundle. The common row stride is the 4-KiB-aligned maximum over
all 96 rows. The layer, expert, and physical-pair rotation determines group
membership exactly, so the file stores neither a TP count nor an expert mode
bitmap.

A rank owns complete contiguous atom rows; at TP12 it reads eight rows, or 256
intermediate channels. The canonical expert-layer container size is
1,051,056,799,744 bytes across 92 layers, including safetensors headers and row
padding. Load preparation removes the group layout into a disposable compact
P33/P43 operand pool. The canonical layout and exact accounting are in
`kquant/qsrt_atoms_v2.py`.

## Dense-H encoding and statistical selection

Cheap importance scores only propose the permutation and donor/recipient
records. They do not authorize a rate shift. Down-projection candidates are
evaluated through complete dense-$H$ BlockLDLQ re-encodes so cross-record
covariance feedback is retained. For `w1`/`w3`, the common input covariance is
retained while the selected output-row records receive their assigned rates.

The encoder then reconstructs the full expert and scores applied-gate-square
weighted routed output error on document-disjoint samples.  A nonzero mode is
accepted only when its paired document-bootstrap lower confidence bound clears
the frozen improvement margin over matched SQG `R0`; uncertain experts fall
back to `(R0,R0)`.

The initial search evaluates only the 3x3 Cartesian grid

```text
(r13, r2) in {0,1,2} x {0,1,2}.
```

This keeps the all-expert encode operationally viable while retaining the
independent `w2` decisions that earlier studies showed were important. Modes
are expert-static: serving reads a compact format code and never performs
runtime rate selection.

## X4T exact endpoint

Official MXFP4 uses four E2M1 bits per weight plus one UE8M0 scale byte per 32
weights, or 4.25 bpw. X4T changes no represented value:

- every E2M1 nibble is preserved exactly, including both zero codes;
- each scale row chooses the adjacent UE8M0 pair that covers the most entries;
- selector bits and out-of-pair exceptions reproduce the complete official
  scale plane; and
- load preparation partitions the decoded exact matrix on 32-channel storage
  groups, with the same equal or quotient/remainder shard rule as QSRT.

The selector stream is directly indexable and needs no tile offset table,
prefix sum, or exception search.  X4T is therefore the high-quality
endpoint; uniform K4 remains lossy and is not treated as a substitute for the
official weights. The all-expert X4T index stores each expert's exact tensor
payload contribution rather than relying on a nominal bpw estimate.

The canonical X4T layer is a TP-independent safetensors file. It contains:

```text
expert_ids: int32[E]
w1/w3.packed: uint8[E, output_rows, input_columns / 2]
w2.packed: uint8[E, 32-channel input groups, output_rows, 16]
matrix.scale_fixed: uint8[E, fixed_stream_bytes]
matrix.scale_exceptions: uint8[concatenated exception bytes]
matrix.scale_exception_offsets: int64[E + 1]
```

The safetensors JSON directory is padded to 4 KiB, which makes storage exactly
additive: each layer has 4,128 fixed bytes, and each expert contributes its
three matrix tensor payloads plus 28 bytes for its ID and exception-offset
entries. Artifact SHA-256 closure receipts authenticate the complete files;
there is no private record header, directory, CRC, or padding convention.

The stored matrix is the source of truth; rank-local W4A16 tensors are a
load-time cache. `w1`/`w3` partition on 32-row groups and `w2` on 32-column
groups. Equal divisors produce identical local shapes; other shard counts use
the same bounded uneven partition and optional cache padding as the QSRT atom
reader. The checkpoint never stores a rank count or rank-local X4T copy.

The scalar scale codec remains `kquant/mxfp4_scale_codec.py`. The existing
full-matrix `kquant/x4t.py` layer container is the canonical exact endpoint.

### X4T runtime refinement

The compressed X4T scale planes can remain persistent in device memory rather
than being expanded for every expert at model initialization.  Immediately
before the ordinary W4A16 call, one graph-safe launch expands only the routed
experts into a caller-owned packed-scale scratch buffer.  That scratch is
reused across layers on the same stream; there is no per-call allocation, CPU
parsing, prefix scan, exception search, or disk access.

The TP12 implementation exactly reproduces the active packed W4A16 scale
bytes, folds the fused-`w1`/`w3` row rotation and BF16 E8M0 clamp into the same
launch, and survives scratch poisoning followed by CUDA graph replay.  On an
RTX PRO 6000 Blackwell Max-Q, a balanced 1,000-replay synthetic Kimi-K3 M=1
study measured the following complete routed-MoE costs:

| Active X4T experts | Dense W4A16 | X4T + W4A16 | Added latency |
| ---: | ---: | ---: | ---: |
| 1 | 22.08 us | 24.16 us | 2.08 us |
| 2 | 22.11 us | 26.11 us | 4.00 us |
| 4 | 22.11 us | 26.21 us | 4.10 us |
| 8 | 24.16 us | 28.26 us | 4.10 us |
| 16 | 30.30 us | 32.35 us | 2.05 us |

M=2 and M=4 sweeps also closed exact scale-byte reconstruction; their added
latency ranged from 1.25 to 8.19 us depending on routed density.  Output
differences once four or more experts contribute match the dense kernel's own
repeatability envelope and come from nondeterministic atomic accumulation,
not scale decode.  The benchmark is
`b12x/benchmarks/benchmark_x4t_w4a16_moe_tp12.py`.

The runtime result clears the latency plausibility gate.  The remaining X4T
work for this checkpoint is to build its own all-expert exact-byte index and
rerun the routed benchmark with checkpoint-derived selections.

## Global allocation

Rate shifting and high-tier selection solve different problems.

1. For each expert, the candidate pool freezes the statistically selected
   `(r13,r2)` at the same three-bit trellis payload.
2. X4T then competes against that selected lossy candidate.  Promoting an
   expert removes its measured routed damage and incurs that expert's exact X4T
   safetensors payload bytes rather than a fixed nominal four-bit cost.

The comparison byte cap inherited from the validated 3p09 allocation is:

```text
target container bytes = 1,058,586,247,168
```

The final cap must be restated in canonical atom-container bytes before the
next allocation; TP-rank padding is not a valid budget component. The global
allocator minimizes

```text
sum_e D_e(choice_e) + lambda * sum_e bytes_e(choice_e)
```

and sweeps `lambda` to meet the checkpoint budget. Its endpoint alphabet is
`qsrt_all` and `x4t_all`; exact trellis and X4T safetensors bytes are
charged by the same optimizer. Since X4T sizes vary by expert, candidate
generation and X4T cost indexing remain reusable when the target budget
changes.

## Evidence and current quality blocker

The initial production-path SQG study used 24 official-source experts across
layers 1, 24, and 40. At fixed K2/K3/K4 endpoints, SQG normal beat both
MUL1-E4M3 and FP16 MCG for all 24 experts and all 216 matrix/rate comparisons.
Those results established SQG as a serious candidate, but they are now
hypothesis-forming rather than a production quality gate: their `w2` metric
used a layer-global post-SiTU covariance whose coordinate indices are not
shared across independently permuted experts.

The matched R0/R1/R2 gate on the same panel found:

```text
SQG selected nonzero r13       6 / 24 experts
SQG selected nonzero r2        2 / 24 experts
SQG proposed nonzero r13       8 / 24 experts
SQG proposed nonzero r2        5 / 24 experts
aggregate SQG R0 vs MUL1 R0    2.2443% lower confirmation SSE
aggregate SQG selected vs
  MUL1 selected                2.1071% lower confirmation SSE
```

The small 21-document confirmation fold was never a final model-quality claim.
It did establish that SQG survives the real Hadamard, LDLQ, Viterbi,
official-weight, and routed-replay path, and that independent `w2` selection
can remain active. It did not establish that the captured Hessian geometry was
representative enough for a checkpoint. The resulting R44/X4T artifact failed
the expected quality trajectory, and its generation path has been stopped.

The replacement gate begins with a source-controlled one-million-token
training capture. `H13` remains layer-global because its latent input basis is
shared. `H2` is rebuilt from expert-stratified routed post-SiTU rows, shrunk
toward identity according to support, and falls back to identity for
unsupported experts. Mode-selection and final-validation corpora remain
document-disjoint. No old R44 candidate pool is eligible for the next
checkpoint merely because it is complete.

The mature B12X W4A16 kernel now has one QSRT serving reconstruction:
QSRT-E4M3. The slow exact profile-5 graph, R44, MUL1, and MCG codebook branches
have been removed from that kernel; the exact variants remain offline teachers
where comparisons require them. QSRT-E4M3 passes dense K2/K3/K4 reconstruction,
P24, P33, dynamic pair selection, and CUDA graph replay closure.

The current mature split-K W4A16 implementation measures about 56.90 us for
P33 and 65.09 us for P24 on the production-shaped benchmark and reproduces the
CPU decoder for P33, P24, and dynamic pair layouts. P24 remains above the
current latency target.

## Supported reconstruction path

QSRT exposes one serving profile:

| Profile | Role | Contract |
| --- | --- | --- |
| `qsrt_sqg_e4m3` | sole runtime encoding profile | `sqg_xor_cheb_t12`: the two-round XOR/odd-multiply bijective SQG graph plus the shared 12-bit approximation to the Chebyshev-derived finite-E4M3 staircase at K2/K3/K4 |

There is no runtime R44, MUL1, MCG, exact profile-5 graph, alternate K2
staircase, or per-expert codebook selector. Those names may appear in archived
measurements and offline research controls, but they are not valid payload or
kernel profile identities.

Rate shifting was checked on 384 unseen, support-stratified experts from layers
1, 24, and 40 using production Hadamard ordering, TF32 dense-H BlockLDLQ,
decoded-upstream conditional `H2`, the complete `R0/R1/R2` grid, and
document-disjoint confirmation and external validation. The native
`sqg_xor_cheb_t12` law retained 89 shifted experts, including 88 with `w2`
R1+ and 77 with `w2=R2`, at a pooled selected external SSE of 68.5222868.
This establishes the native four-stratum K2 mapping as the sole encoder
contract. No matrix- or rate-specific reconstruction-law selector is used.

### Offline trellis-encoder optimization

The SM120 offline tile encoder keeps the authoritative 128-symbol context on
both sides of each 256-value tile.  Its optimized implementation transposes
each state-indexed SQG byte table into predecessor-major groups so one thread
loads all K2 or K3 predecessor labels in one vector transaction and all K4
labels in two.  K2 traceback stores four two-bit decisions per byte, K3 uses a
768-thread forward pass while preserving the established 512-thread final
reduction tree, K4 uses a 512-thread/maximum-L1 configuration, and paired
half-precision comparisons update both paths together.

On 512 production-codebook tiles at C128, median kernel time changed as
follows on SM120:

| Rate | Previous encoder | Optimized encoder | Reduction |
| --- | ---: | ---: | ---: |
| K2 | 7.270 ms | 4.825 ms | 33.64% |
| K3 | 6.378 ms | 4.015 ms | 37.04% |
| K4 | 6.054 ms | 3.345 ms | 44.75% |

Safety was checked directly against the preceding CUDA extension in 63 cases
covering K2/K3/K4, C1/C32/C128, Gaussian/heavy/structured inputs, and the
production and control E4M3 tables.  Reconstructed values and trellis indices
were bit-identical in every case.  A complete 20-expert layer-24 endpoint
study also produced the identical serialized candidate-payload SHA-256 and
the same selected modes, while wall time fell from about 136 to 89 seconds.

A shorter C32 primer is not part of this optimization.  The initial
20-expert screening study put every C128 confirmation winner in C32's top
three, but that is not sufficient evidence for the all-expert pool or a future
arbitrary-record search.  The current production build remains C128 end to
end; C32 may be revisited only as a shortlist generator followed by exact C128
re-encoding after a substantially broader audit.

### All-expert mode selection

The sealed production candidate pool at
`/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-CANDIDATES-v1` contains all 92
MoE layers and 82,432 experts. A nonzero mode is retained only when its paired,
document-clustered confirmation lower bound clears the zero-improvement
margin.

| Selected `(r13,r2)` | Experts | Share |
| --- | ---: | ---: |
| `R0/R0` | 73,053 | 88.622% |
| `R0/R1` | 1,007 | 1.222% |
| `R0/R2` | 1,231 | 1.493% |
| `R1/R0` | 213 | 0.258% |
| `R1/R1` | 1,252 | 1.519% |
| `R1/R2` | 3,428 | 4.159% |
| `R2/R0` | 10 | 0.012% |
| `R2/R1` | 30 | 0.036% |
| `R2/R2` | 2,208 | 2.679% |

In aggregate, 9,379 experts (11.378%) select a confirmed nonzero shift. The
down projection selects R1+ in 9,156 experts (11.107%), the coupled gate/up
pair selects R1+ in 7,141 (8.663%), and R2 appears on at least one axis in
6,907 experts (8.379%). X4T endpoint allocation is a separate exact-byte
optimization over these sealed candidates.

### Extreme-rate K2 research register

Pure or nearly pure K2 operation makes individually small gains relevant. The
following mechanisms remain distinct research candidates; a negative result
for one parameterization does not remove the underlying symmetry or coding
degree of freedom.

| Mechanism | Mathematical role | Current evidence | Qualification |
| --- | --- | --- | --- |
| Coupled gate/up/down boundary Hadamard | Exact change of basis before the coordinatewise activation boundary | Fresh uniform-K2 SQG re-encodes improved routed error on 22/24 experts; pooled routed SSE fell 3.052% | Numerically promising; requires fused-transform latency and broader layer confirmation |
| Activation-metric W1/W3 pair code | Uses the local 2-by-2 SiTU metric so gate/up errors can cancel | 24/28 isolated pair-codebook wins; median functional metric improvement 4.90% | Codebook oracle; needs a joint vector trellis, decoded-payload scoring, and full-expert validation |
| W3/W2 sign gauge | Exact symmetry from the odd up activation | Zero payload and runtime cost after baking signs into both matrices | Retain for real SQG path search; symmetric scalar proxies cannot measure its trellis-path value |
| Positive W3/W2 scale gauge | Approximate symmetry while the up branch is linear | Across 28 routed experts, median route-weighted mass with $g'(u)\ge0.99$ is 99.9986%; the worst expert remains 97.7792%. A four-expert baked-gauge check changed full-precision expert SSE by only $1.39\times10^{-11}$, but naive RMS balancing worsened the 2-bit proxy by 0.283% median | Symmetry is valid; RMS balancing is a negative heuristic, not a rejection of activation-aware scale fitting |
| Co-routing-aware candidate phase | Chooses among near-equal expert errors to reduce top-16 cross terms | A 32-row layer-24 audit measured a positive cross term equal to 0.929% of diagonal mapped SSE; the linear metric matched exact post-projection SSE within 0.458% | Plausible sub-percent headroom; requires two or more retained trellis candidates per expert and document-disjoint selection |
| Aligned per-neuron shared bases | Stores a small number of layer bases and quantizes only expert residuals | In 32-expert coordinate sketches, rank-four excess capture over an energy-matched isotropic null was 1.18, 1.74, 0.09, and 0.39 percentage points at layers 1, 24, 64, and 92. Global expert coefficients were weaker | Track as a low-rate oracle, but the signal is not stable through depth; require full-coordinate/all-expert factorization and residual K2 encoding before implementation work |
| Reconstructed-activation W2 refit | Compensates upstream quantization before the final K2 encode | Dense refit won 20/28 experts with 1.55% median routed improvement | Upper bound only; the dense fitted matrix must be distilled into a cheap structured correction or used solely as the next W2 encoding target |

The remaining low-rate design space is retained explicitly even where no
production result exists yet:

| Mechanism | Pure K2 | Required decisive experiment |
| --- | --- | --- |
| One- or two-bit tile-local K2 codebook menu | Yes | Retain byte-distinct complementary K2 staircases or graphs, fit the mode on training documents, and verify mode stability and full-expert SSE on confirmation documents. One selector bit per 16-by-16 tile costs 0.00390625 bpw; two cost 0.0078125 bpw. |
| Joint gate/up vector trellis | Yes | Incorporate the measured 2-by-2 activation metric into full tail-biting assignment, preserve decoded scale closure, and score the complete reconstructed expert. |
| Successively refinable K2 base | Yes, as the base layer | Jointly train a two-plane base and K3/K4 refinement planes with the production transform, refitted scales, and functional objective; do not infer viability from native MXFP4 bit truncation. |
| Tile-local P33/P24 funding | No | Rerun the actual equal-byte pair allocator after every tile proposal. Keep its selector accounting and kernel grammar separate from pure-K2 quality claims. |
| Gate/up/down rate triples such as 234 permutations | No | Select complete equal-byte projection triplets through decoded whole-expert error; isolated matrix SSE cannot choose which projection receives K2 or K4. |
| Joint low/high K6 vector code | No | Treat as a six-bit pair-allocation oracle and require a real trellis realization plus P24/P33 allocator comparison before considering a runtime format. |

The co-routing objective for retained candidate mode $m_e$ is

$$
\min_{\{m_e\}}
\sum_n\left\|
\sum_{e\in\mathcal R_n}p_{n,e}J_n\epsilon_{n,e,m_e}
\right\|_2^2,
$$

where $J_n$ is the post-aggregate RMSNorm/output-projection Jacobian. Candidate
modes must first pass an expert-local unary-loss bound; otherwise cancellation
can hide an unacceptable individual regression. The repository analysis
solver implements this constrained objective, but the sealed candidate pool
contains only one payload per expert, so alternate paths must be generated in
a fresh research encode rather than inferred from aggregate SSE.

## Execution checklist

- [x] Implement and unit-test L16 SQG-normal E4M3 labels for K2/K3/K4.
- [x] Integrate SQG into dense-H rate-shifted encoding and stored-state decode.
- [x] Validate SQG endpoints and the separate `(r13,r2)` R0/R1/R2 gate.
- [x] Start the resumable all-82,432-expert SQG candidate pool on 12 GPUs.
- [x] Complete and seal the all-82,432-expert R44 SQG candidate pool.
- [x] Freeze and unit-test the exact X4T numerical representation.
- [x] Add exact X4T load-time reconstruction and TP12 W4A16 preparation.
- [x] Store exact X4T layers as TP-independent safetensors and implement the
      one-launch, graph-safe routed W4A16 scale predecoder.
- [x] Benchmark X4T inside the complete routed W4A16 path across M=1/2/4 and
      1/2/4/8/16 active-expert densities.
- [ ] Build and seal the all-expert X4T byte-cost index.
- [x] Close representative SQG K2/K3/K4, P24/P33, fused-SiTU, and graph-replay
      execution through the production B12X API.
- [x] Close native SQG W4A8 dense/routed execution and measure its full-path
      activation-quantization error and latency against matched W4A16.
- [x] Seal the 223-document, 128K-token document-disjoint validation capture.
- [x] Validate R44 and the shared SQG-Cheb normal staircase at K2/K3/K4 on the
      production path.
- [x] Confirm and freeze the `w2`-only bit-4 K2 graph on 384 unseen,
      support-stratified experts with both expert- and document-clustered
      positive confidence intervals for fixed `R0/R2`.
- [ ] Score selected candidates on the untouched validation capture and run
      the matched-R0 rate-shift policy audit.
- [ ] Freeze the global QSRT allocation at the target checkpoint budget.
- [ ] Materialize QSRT and X4T physical-slot extents into a fresh atom-major
      artifact.
- [ ] Close the materialized artifact's structural validation, malformed-input
      rejection, exact state decode, exact X4T source reconstruction, and exact
      byte accounting.
- [ ] Re-run the TP12 kernel/performance gate on checkpoint-derived routed
      mixtures; the synthetic P33/P24/sparse/mixed gate has passed.
- [ ] Package a fresh serve directory; never mutate the validated 3p09 model.
- [ ] Run streamed official-vs-packaged traces, live TP12 routing/logit checks,
      and the expanded end-to-end quality suite.

The later evaluation suite should incorporate the 32x2048 KLD reference
dataset and the Kimi-K3 evaluation tools identified for final end-to-end
testing.  The current calibration/confirmation corpus must also be expanded
substantially before production quality claims are made.
