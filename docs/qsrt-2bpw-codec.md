# QSRT two-bit expert-weight codec

## Definition

QSRT is the Quantile-Stratified Rate-shifted Trellis codec used to compress the
routed-expert matrices in Kimi-K3. The two-bit profile encodes every expert
weight with two path bits, plus a small set of block scales and one transform
identifier per expert.

Each Kimi-K3 routed expert contains three matrices:

$$
W_1,W_3\in\mathbb R^{3072\times3584},
\qquad
W_2\in\mathbb R^{3584\times3072}.
$$

For a residual row $z$, the expert computes

$$
g=zW_1^{\mathsf T},
\qquad
u=zW_3^{\mathsf T},
\qquad
h=\mathrm{SiTU}(g,u),
\qquad
y=hW_2^{\mathsf T}.
$$

The coordinatewise activation is

$$
\mathrm{SiTU}(g,u)
=\left[4\tanh(g/4)\,\sigma(g)\right]
 \odot
 \left[25\tanh(u/25)\right].
$$

It combines corresponding gate and up coordinates independently for each of
the 3,072 intermediate neurons.

The codec does not round these matrix coefficients independently. It encodes
each coefficient sequence as a path through a finite-state graph. Two new bits
are chosen at every step. The previous fourteen path bits determine the four
reconstruction values currently available and which four sets of values can
be reached next. An offline Viterbi search chooses the complete path that
minimizes an activation-weighted error objective.

The terms used elsewhere in the repository follow directly from this
construction:

- `K2` means that each trellis step adds two payload bits and therefore has
  four outgoing branches;
- `L16` means that the current edge is identified by a sixteen-bit window,
  consisting of fourteen retained history bits and two new branch bits; and
- a `K2 record` is a 128-neuron region whose coefficients all use this
  two-bit path law.

The uniform two-bit profile assigns K2 to all 24 records of every routed
expert. It does not use higher-rate records or higher-rate experts.

```text
weights per expert                33,030,144
trellis payload per weight        2.0 bits
trellis payload per expert        8,257,536 bytes
stored scales per expert             18,432 bytes
total payload per expert          8,275,968 bytes
total rate including scales        2.004464 bpw
routed experts                       82,432
```

The two-bit representation combines four mechanisms:

1. history-dependent four-way sequence quantization;
2. Gaussian-quantile reconstruction values rounded to finite E4M3;
3. an exact coupled Hadamard change of basis across all three matrices; and
4. covariance-aware error feedback evaluated on the reconstructed expert.

## History-dependent sequence quantization

### Four choices with fourteen bits of memory

At one encoding step, let $h$ be the fourteen retained path bits and let
$b\in\{0,1,2,3\}$ be the two new bits. The pair $(h,b)$ identifies one of
$2^{16}=65,536$ directed edges.

Every history exposes four reconstruction candidates. They are deliberately
drawn from four different quarters of a Gaussian probability distribution:

```text
branch choice 0    one value from probability interval [0, 1/4)
branch choice 1    one value from probability interval [1/4, 1/2)
branch choice 2    one value from probability interval [1/2, 3/4)
branch choice 3    one value from probability interval [3/4, 1)
```

The association between a physical two-bit branch and a probability quarter
changes with history. The exact value within each quarter also changes with
history. Every state therefore spans negative tail, negative center, positive
center, and positive tail values, but different states expose different
fine-grained choices.

Selecting a branch also changes the next fourteen-bit history. Two edges with
the same numerical value can remain useful alternatives because they lead to
different future states. The quantizer is consequently richer than its scalar
reconstruction alphabet.

This labelled finite-state construction is the Stratified Quantile Graph.

### Edge-to-quantile assignment

The graph assigns every directed edge a unique integer rank from 0 through
65,535. Those ranks are ordered from the most negative to the most positive
reconstruction region.

For a sixteen-bit edge word $t$, split the retained history and new branch:

$$
h=t\gg2,
\qquad
b=t\mathbin{\mathrm{AND}}3.
$$

With $M=2^{14}-1$, compute

$$
\begin{aligned}
x_0 &= h\oplus(h\gg11),\\
x_1 &= x_0\oplus((x_0\ll11)\mathbin{\mathrm{AND}}M),\\
p &= (\mathtt{0x3FA7D929}\,x_1+
      \mathtt{0xC928FD8E})\bmod2^{32},\\
\phi &= p\mathbin{\mathrm{AND}}M,\\
s &= p\gg30,\\
j &= \mathrm{rev}_2(b)\oplus s,\\
r &= (j\ll14)\mathbin{\mathrm{OR}}\phi.
\end{aligned}
$$

Here $j\in\{0,1,2,3\}$ selects the probability quarter and
$\phi\in\{0,\ldots,16383\}$ selects the position within that quarter.

Both exclusive-or shifts are invertible on fourteen-bit integers. The
multiplier is odd and is therefore invertible modulo $2^{14}$. It follows that
$h\mapsto\phi$ is a permutation. For fixed history, $b\mapsto j$ is also a
permutation.

The construction therefore guarantees:

- every one of the 65,536 ranks is assigned to exactly one directed edge;
- every state exposes one candidate from each probability quarter; and
- all four edges from a state share one fine position within their respective
  quarters.

The integer carries and invertible shifts also control how a chosen branch
changes the next fine position. This continuation structure is part of the
code. Matching the one-step value distribution without matching useful future
states produces materially worse quantization.

### Closed-path encoding

The encoder uses tail-biting Viterbi search: the final state must close onto
the initial state. This removes an arbitrary start-state penalty and makes the
stored bitstream a closed finite-state path. Each 256-coefficient coding tile
is evaluated with 128 coefficients of authoritative context on both sides so
the selected interior path is not optimized as an isolated scalar block.

## Gaussian reconstruction values in finite E4M3

The rank assignment defines probability regions, not floating-point values.
For rank $r$, the ideal value is based on the midpoint of its equal-probability
Gaussian interval:

$$
u_r=\frac{r+\tfrac12}{65536},
\qquad
z_r=\Phi^{-1}(u_r).
$$

The reconstruction value is scaled and rounded to the nearest finite E4M3
number:

$$
Y(r)=\mathrm{RNE}_{\mathrm{E4M3FN}}(1.5z_r).
$$

E4M3 is an eight-bit floating-point reconstruction alphabet with one sign bit,
four exponent bits, and three fraction bits. These eight-bit values are
reconstructed values, not payload: the stored path still costs two bits per
weight.

The normal quantile law gives every state broad sign and magnitude coverage.
Rounding creates repeated numerical labels, but repeated labels on different
edges retain different successors and therefore remain distinct coding
choices.

### Chebyshev-derived discrete law

The inverse Gaussian distribution is not evaluated during encoding or
decoding. An offline piecewise-Chebyshev construction establishes the desired
rank-to-E4M3 mapping. Its coefficients are fitted against the complete real
interval that rounds to each required E4M3 value. Exhaustive evaluation then
verifies the resulting byte for all 65,536 ranks.

Chebyshev approximation supplies the mathematical derivation of the discrete
staircase. It does not define the finite-state graph or its transitions.

### Shared 4,096-byte staircase

The exact 65,536-rank staircase is reduced to one globally shared 4,096-byte
table. For table index $q$,

$$
Y_{12}(q)=
\mathrm{mode}
\{Y(16q),Y(16q+1),\ldots,Y(16q+15)\},
$$

with the lower unsigned E4M3 byte selected on a tie. The reconstructed value
for an edge is

$$
\widehat Y(h,b)=Y_{12}\!\left(r(h,b)\gg4\right).
$$

The table index uses the upper twelve bits of the full sixteen-bit rank. This
is why the repository calls it the twelve-bit, or `T12`, staircase. It is a
single 4,096-byte codebook shared by the entire model, not twelve stored bits
per weight.

The approximation changes only the final reconstruction label. It retains the
complete edge permutation, four-way stratification, state transitions, and
Viterbi path search. Frozen hashes identify the shared table and the resulting
65,536-edge K2 label map.

## Exact coupled Hadamard conditioning

Two-bit quantization is highly sensitive to outliers and unequal coordinate
scales. Orthogonal Hadamard transforms spread concentrated energy across a
block, making the sequence presented to the quantizer more homogeneous.

The gate, up, and down matrices cannot be transformed independently: gate and
up meet at a nonlinear activation, and the resulting intermediate coordinates
are consumed by the down matrix. QSRT therefore uses a coupled change of basis
whose transforms are explicitly cancelled on the correct side of the
activation.

### Transformation

Interleave corresponding gate and up rows:

$$
Q_e=\mathrm{interleave}(W_{1,e},W_{3,e})
\in\mathbb R^{6144\times3584}.
$$

Define three orthogonal block transforms:

- $U_R$: normalized 512-coordinate Hadamard blocks on the residual input and
  expert-output axes, shared by every expert in a layer;
- $U_{A,e}$: normalized signed 128-coordinate Hadamard blocks across the
  interleaved gate/up preactivations of expert $e$; and
- $U_{B,e}$: normalized signed 128-coordinate Hadamard blocks across the
  post-activation intermediate coordinates of expert $e$.

The matrices given to the two-bit encoder are

$$
Q'_e=U_{A,e}^{\mathsf T}Q_eU_R,
\qquad
W'_{2,e}=U_R^{\mathsf T}W_{2,e}U_{B,e}.
$$

For a row vector,

$$
z'=zU_R,
\qquad
q'=z'Q_e'^{\mathsf T}=(zQ_e^{\mathsf T})U_{A,e}.
$$

The inverse of $U_{A,e}$ is applied before the preactivations are split into
gate and up vectors and passed through SiTU. The resulting hidden vector is
then transformed by $U_{B,e}$ before multiplication by the transformed down
matrix. Orthogonality gives

$$
y'_e=(hU_{B,e})W_{2,e}'^{\mathsf T}
     =(hW_{2,e}^{\mathsf T})U_R,
$$

followed by

$$
y_e=y'_eU_R^{\mathsf T}=hW_{2,e}^{\mathsf T}.
$$

The transformation is therefore exactly function-preserving before
quantization. Its only purpose is to present a better-conditioned coordinate
system to the lossy encoder.

### Expert-specific signed transforms

The residual-axis Hadamard has a fixed sign pattern shared by a layer. The two
intermediate-axis transforms use deterministic sign patterns selected
separately for each expert. One three-bit identifier generates the two required
sign vectors; the vectors themselves are not stored.

The format defines eight possible identifiers. The materialized model uses two
validated choices:

```text
transform identifier 0      60,277 experts
transform identifier 6      22,155 experts
stored metadata              3 bits per expert
```

For a given expert, training documents may propose the alternate transform.
A document-disjoint confirmation set can only accept or reject that proposal
against identifier zero. It cannot search for a different winner.

The selected transform affects every subsequent encoding operation: transformed
weights, scale fitting, trellis paths, activation covariances, and reconstructed
expert error are all recomputed for that candidate.

## Covariance-aware error feedback

The encoder minimizes error in the directions exercised by routed
activations, not unweighted coefficient error. For a linear matrix with input
rows $X$, the local quadratic metric is based on the dense covariance
$H=X^{\mathsf T}X$.

The encoder factors this dense matrix and quantizes coefficient blocks in an
order that feeds each block's reconstruction error into later blocks. This is
the BlockLDLQ procedure. It allows later coefficients to compensate for
earlier quantization error under the activation-weighted metric.

Every candidate receives its own fitted scales and a complete Viterbi and
BlockLDLQ encode. Tile-local scalar error is not used to select the final
payload.

### Gate and up covariance

All experts in a layer consume the same residual coordinate system. A common
gate/up input covariance can therefore be accumulated for the layer and
transformed into the residual Hadamard basis. The repository names this matrix
`H13` because it conditions the first and third expert projections, $W_1$ and
$W_3$.

### Down covariance conditioned on reconstructed gate and up

The input to $W_2$ is produced by the quantized gate and up matrices. It is
also expressed in an expert-specific Hadamard basis. Its covariance cannot be
pooled across experts or borrowed from the unquantized source matrices.

For each candidate transform, the encoder therefore:

1. transforms and two-bit encodes $W_1$ and $W_3$;
2. decodes their exact stored reconstruction;
3. replays naturally routed inputs through those reconstructed matrices;
4. cancels the preactivation transform and evaluates SiTU;
5. applies the candidate's post-activation transform;
6. accumulates the resulting expert-specific covariance for $W_2$; and
7. performs a complete two-bit BlockLDLQ encode of $W_2$.

The repository calls this covariance `H2`. With intermediate dimension
$d=3072$, its regularized estimate is

$$
\widehat H_{2,e}
=\alpha_e H^{\mathrm{sample}}_{2,e}
+(1-\alpha_e)
\frac{\mathrm{tr}(H^{\mathrm{sample}}_{2,e})}{d}I.
$$

The identity term stabilizes poorly sampled directions without importing a
coordinate system from another expert. Experts without sufficient routed
support use identity. A layer-wide post-activation covariance is never used.

## Whole-expert selection

The final candidate score reconstructs all three matrices, evaluates the
complete expert function on naturally routed rows, and weights output error by
the square of the router coefficient actually applied to that expert.

This objective captures interactions that independent matrix or weight errors
miss:

- gate and up errors interact through SiTU;
- their reconstructed activations determine the correct down covariance;
- the Hadamard candidate changes all three encoded matrices; and
- dense error feedback can move error between coefficient blocks.

Training and selection are separated by complete source document. A
source-controlled four-million-token natural-routing corpus supplies
covariances and transform proposals. Disjoint documents accept or reject each
proposal. A separate 128,000-token routed corpus measures transfer. Final model
KLD and task evaluations do not select codec parameters.

## Payload

The 3,072-neuron intermediate axis is divided into 24 records of 128 neurons.
Every record uses two path bits per coefficient. There is no rate bitmap and no
record- or tile-specific reconstruction table.

Canonical storage groups aligned fragments from $W_1$, $W_3$, and $W_2$ by
32 intermediate neurons. One such group is called an atom. Each expert
contributes 96 atoms, and each atom contains:

- the two-bit trellis payload for its gate fragment;
- the two-bit trellis payload for its up fragment;
- the two-bit trellis payload for its down fragment; and
- the corresponding half-precision scale fragments.

One atom occupies 86,208 bytes. Ninety-six atoms give the 8,275,968-byte
expert payload shown above. The 18,432 bytes above the exact two-bit weight
stream are the stored scales.

Atoms have complete ownership and the canonical file stores no tensor-parallel
degree. This permits the same encoded weights to be divided among different
numbers of devices without re-encoding. It does not alter the quantization
law.

Across 92 mixture-of-experts layers, the expert payload occupies
682,207,608,832 bytes including safetensors headers and aligned row padding.
The shared 4,096-byte reconstruction table and three-bit expert transform
identifiers are negligible at model scale.

## Measured codec contribution

On fresh two-bit SQG encodes, the fixed coupled Hadamard transformation reduced
pooled routed expert-output squared error by 3.052% and improved 22 of 24
experts.

Expert-specific signed-transform selection was fitted on the
four-million-token corpus and evaluated on the independent 128,000-token
corpus. It reduced pooled routed error by a further 1.308%, reduced the median
expert error by 0.447%, and improved 17 of 28 experts. A single transform for
all experts regressed, while choosing one transform per layer recovered only
0.030%.

The complete representation contains all 82,432 routed experts and closes the
payload for all 92 mixture-of-experts layers. A full-model A16 comparison over
32 windows and 65,504 scored token positions measured mean
reference-to-candidate Kullback-Leibler divergence of 0.0851995464.

## Representation boundary

The two-bit representation contains:

- one history-dependent four-way trellis construction;
- one shared 4,096-byte Gaussian-derived E4M3 table;
- one fixed residual-axis Hadamard basis per layer;
- one three-bit intermediate-axis Hadamard identifier per expert;
- candidate-specific scales and covariance-aware trellis paths; and
- one tensor-parallel-independent all-expert payload.

It does not contain mixed two-, three-, or four-bit records, a higher-rate
expert tier, tile-local codebooks, joint gate/up vector symbols, cross-expert
bases, or refinement bit planes.

## Authoritative implementation

The codec is defined by:

- `qsrt/sqg_e4m3.py`: edge ranks and the shared E4M3 table;
- `qsrt/sqg_quantizer.py` and `qsrt/csrc`: closed-path Viterbi encoding;
- `qsrt/qsrt_coupled.py`: exact coupled Hadamard coordinates;
- `qsrt/qsrt_coupled_plan.py`: expert transform selection;
- `qsrt/exl3_encoder_backend.py`: covariance-aware BlockLDLQ encoding; and
- `qsrt/qsrt_atoms_v2.py`: canonical payload and byte accounting.
