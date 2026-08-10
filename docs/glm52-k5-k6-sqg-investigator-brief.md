# Investigator brief: extending SQG beyond K4

Date: 2026-08-08

## Question

Can the SQG codec used by QSRT be generalized usefully to uniform K5 and K6,
when compared with freshly encoded EXL3/MCG K5 and K6 on the same GLM-5.2
weights?

The first literal extension of the frozen K2/K3/K4 SQG construction failed the
quality gate. Do not treat its partial measurements as a valid rate curve.

## Frozen experiment

- Source: `zai-org/GLM-5.2` revision
  `b4734de4facf877f85769a911abafc5283eab3d9`, official BF16 tensors.
- Panel: the same 48 error-blind, real-K4 experts used by the sealed uniform
  K2/K4 study: four experts in each of 12 layers across the model.
- Arms: fresh EXL3/MCG and fresh SQG at each rate. There is no materialized K5
  or K6 checkpoint arm.
- Matched controls: identical source tensors, transform seeds, identity H,
  `sigma_reg=0.025`, `tailbite_context=128`, FP16 reconstruction endpoint, and
  FP64 source-relative SSE accumulation.
- Partial run directory:
  `/data/kquant/pilots/glm52-uniform-k5-k6-codec-v1`.
  It contains a manifest and seven atomic expert receipts, but no completed
  report. The run was stopped deliberately.

## Observed failure

Across the seven completed experts, every one showed essentially the same
result:

| Rate | MCG relative SSE | SQG relative SSE | SQG / MCG |
|---|---:|---:|---:|
| K5 | 0.00119724776 | 0.00125915085 | 1.05170450 |
| K6 | 0.000320682791 | 0.000724382630 | 2.25887590 |

The K5 expert ratios ranged from 1.04925 to 1.05269. The K6 ratios ranged from
2.24480 to 2.26437. Gate, up, and down projections all agreed, so this is not a
single projection or scale-sharing outlier.

## Implementation bugs already ruled out

1. The K5/K6 CUDA Viterbi mechanics match upstream EXL3. With the MCG codebook,
   kquant's generalized kernel and upstream EXL3 produced exactly identical
   states, reconstructions, and SSE for K4, K5, and K6 on 64 random tiles.
2. SQG K5/K6 emitted tail-biting-consistent states. A 2,048-state check at each
   rate had zero closure failures.
3. CUDA SQG reconstruction matched direct lookup of the supplied E4M3 table
   exactly for every checked state.
4. Experimental native-word K5/K6 pack/unpack reconstructs the cyclic states,
   and the K2/K3/K4 form is byte-identical to the frozen QSRT packer.
5. The full repository suite passed: 381 tests.

These checks make a gross CUDA dispatch, traceback-buffer, LUT-indexing, or
payload-decode bug unlikely.

## T12 is not the main cause

On layer 3, expert 51, `gate_proj`, K6:

- T12 SQG relative SSE: `0.000724716417`
- exact Cheb-normal SQG relative SSE: `0.000718988207`
- MCG relative SSE: `0.000322282558`

Removing the 4,096-entry T12 approximation recovers less than one percent of
SQG error and leaves the roughly 2.2x gap intact.

## Confirmed cause: the E4M3 alphabet has saturated

The SQG T12 rank law has 4,096 table entries but only 151 distinct finite-E4M3
numerical levels. The exact 65,536-rank Cheb-normal law has only 155 distinct
finite-E4M3 levels after rounding.

EXL3/MCG is not an E4M3 codebook. Its multiply/permuted half construction has
approximately 10,746 distinct FP16 reconstruction values over the 65,536
trellis states. K5/K6 therefore give MCG meaningful additional scalar
resolution, while SQG is approaching the error floor of its much smaller
finite-E4M3 alphabet. The measured shape supports this explanation: SQG
improves from K5 to K6, but much less than MCG, and then clusters tightly near
`7.24e-4` relative SSE.

The requested oracle control is complete. On two million actual
base-regularized values from layer 3, expert 51, `gate_proj`, the best scalar
T12/E4M3 scale gave `0.000698197` relative SSE. The observed K6 trellis result
was `0.000724716`, only about four percent above that unattainable scalar
lower bound. The full MCG alphabet's scalar oracle was about `4.57e-8`.

Graph work cannot close the K6 gap while retaining the finite-E4M3 endpoint.
The endpoint alphabet, not the T12 approximation or a Viterbi bug, is the
dominant limitation.

## Quantile/phase allocation result

The tested alternative family was:

- K5: Q8/H4
- K6: Q8/H8

Use three physical predecessor bits for eight coarse quantile strata and the
remaining K-3 bits for phase alternatives within each stratum. Combine those
phase bits with the physical `(16-K)`-bit outgoing-edge coordinate into one
13-bit phase coordinate. The construction must be bijective over all 65,536
codewords and, for each actual EXL Viterbi predecessor menu, expose exactly
eight strata with H alternatives per stratum.

After adding an explicit single-matrix global-scale override and repeating the
full decode, the earlier enormous Q8/H8 reconstruction could not be
reproduced. Scales and reconstructed maxima were sane. Treat that observation
as a transient prototype failure, not a codec result.

On the real gate matrix, Q8/H did not win:

| Rate | Frozen-style graph | Q8/H graph | Q8/H penalty |
|---|---:|---:|---:|
| K5 | `0.00126197` | about `0.001313` | about 4.1% |
| K6 | `0.000724716` | about `0.0007286` | about 0.5% |

Scale searches did not change that conclusion. With the richer FP16 normal
law, the existing carry-mixed graph also beat both the high-bit native-strata
and Q8/H alternatives. More phase allocation is therefore not the current
priority.

## Graph-orientation finding

EXL's Viterbi menu for a fixed outgoing edge enumerates codewords

```text
(predecessor << (16 - K)) | outgoing_edge
```

so the high K bits vary within the menu. The frozen carry-mixed rank function
is expressed using a low-K-bit branch decomposition. When inspected along the
actual EXL menu axis, its average distinct coarse-stratum counts are:

| Rate | Nominal strata | Mean distinct strata in actual menu |
|---|---:|---:|
| K2 | 4 | 2.044 |
| K3 | 8 | 7.906 |
| K4 | 16 | 9.742 |
| K5 | 32 | 28.812 |
| K6 | 64 | 53.000 |

This helps explain why frozen K4 behaves more like an approximately
eight-quantile graph than a clean Q16 graph. However, a corrected high-bit
Q64 construction still measured about `0.000725` relative SSE on the real K6
gate matrix, so orientation alone does not explain the high-rate MCG gap. Do
not change the frozen K2/K3/K4 tables while investigating this.

## Richer SQG endpoint result

A 65,536-rank normal law rounded only to FP16 removes the alphabet floor. On
layer 3, expert 51, `gate_proj`, using the existing carry-mixed graph:

| Rate | MCG relative SSE | FP16 SQG relative SSE | SQG / MCG |
|---|---:|---:|---:|
| K5 | about `0.0012033` | `0.001093112` | `0.908426` |
| K6 | about `0.0003223` | `0.000281932` | `0.874798` |

Thus K5 and K6 mechanics are not broken. The literal E4M3 extension was the
broken scientific comparison: it asked a 151-value endpoint to exploit rates
at which MCG has more than ten thousand values.

### Seven-expert real-weight confirmation

The stopped E4M3 pilot's seven completed experts were replayed with the full
65,536-rank FP16 normal law on 2026-08-08. This is a direct measurement, not a
projection from the synthetic Gaussian control. It re-encoded both arms from
the official BF16 source with the same identity H, transform seeds,
regularization, tail-biting context, independent scale fits, payload decode,
and FP64 SSE accumulation used by the original pilot. All three projections
were included for every expert.

| Rate | Fresh MCG relative SSE | FP16 SQG relative SSE | SQG / MCG | Reduction | Expert wins |
|---|---:|---:|---:|---:|---:|
| K5 | `0.001197247761` | `0.001089862522` | `0.910306586` | `8.9693%` | `7/7` |
| K6 | `0.000320682791` | `0.000281199346` | `0.876876945` | `12.3123%` | `7/7` |

The per-expert SQG/MCG ranges were `0.908093`--`0.911356` at K5 and
`0.873443`--`0.879170` at K6. This confirms the investigator's projected
roughly 8%/12% advantages on the exact seven real experts from which the E4M3
plateau was measured.

### D3L serving-approximation confirmation

The same seven experts were freshly re-encoded with the 104-descriptor,
416-byte dyadic tail-adaptive linear approximation on 2026-08-08. The control
uses the exact primary carry-mixed graph from the preceding full-FP16 run;
MCG, full FP16 SQG, and D3L SQG each receive independent scale fitting and
independently decoded payload scoring.

| Rate | Fresh MCG relative SSE | Full FP16 SQG | D3L SQG | D3L / full FP16 | D3L reduction vs MCG | D3L wins |
|---|---:|---:|---:|---:|---:|---:|
| K5 | `0.001197247761` | `0.001089862522` | `0.001089916388` | `1.000049425` | `8.9648%` | `7/7` |
| K6 | `0.000320682791` | `0.000281199346` | `0.000281209732` | `1.000036935` | `12.3091%` | `7/7` |

D3L's pooled penalty versus the complete 128 KiB FP16 law is only `0.00494%`
at K5 and `0.00369%` at K6. Its per-expert D3L/full-FP16 ratio ranges were
`1.000004`--`1.000118` at K5 and `0.999902`--`1.000153` at K6. The small
sub-unity values are path/scale discretization effects, not evidence that the
approximation dominates its target law.

The fitted descriptor payload for the repository's scaled normal-law
convention has SHA-256
`17cf4ca9ef1e3a07c3354c12f7ac887b4e081b1668bea61eb37d8f2b410bb968`.
The generic fitter also reproduces the investigator's supplied unscaled
Gaussian D3L descriptor blob byte-for-byte. This establishes D3L's numerical
viability; fused-kernel latency remains unmeasured.

The 128 KiB state-indexed FP16 table used for this control is an oracle
implementation, not a proposed codec layout. A compact emulation computes the
carry-mixed rank procedurally and uses 256 piecewise-linear quantile segments,
with exact values only in the two tail pages. The straightforward tables total
2 KiB. On the real gate matrix its penalty versus the complete FP16 table was:

| Rate | Compact / full FP16 SQG |
|---|---:|
| K3 | `1.000211` |
| K5 | `1.000226` |
| K6 | `1.000228` |

These are numerical emulation results. The compact procedural CUDA decoder
still needs to be implemented and benchmarked before claiming a speed result.

## E4M3-aware path selection result

Do not search with the FP16 law and downcast the selected labels afterward.
At K3, one expert from each of 12 layers (36 matrices total) measured:

| Arm | Relative SSE |
|---|---:|
| FP16 SQG | `0.016966912` |
| FP16 path, then E4M3 labels | `0.017654529` |
| direct-E4M3-aware search, same scale | `0.016973188` |
| native T12/E4M3 search | `0.016978103` |

E4M3-aware search recovered 3.859% versus post-search downcasting. T12 was
only 0.029% worse than the directly rounded E4M3-aware control.

The effect is larger at K6. On layer 3, expert 51, all three projections:

| Arm | Relative SSE |
|---|---:|
| FP16 SQG | `0.000281728` |
| FP16 path, then E4M3 labels | `0.000981948` |
| direct-E4M3-aware search, same scale | `0.000719267` |
| native T12/E4M3 search | `0.000724675` |

Native T12 is 26.2% lower-error than FP16-search-then-E4M3. The richer SQG
gain survives only if the reconstruction endpoint remains richer than E4M3.

## Recommended investigation order

1. Implement procedural carry-mixed rank reconstruction plus the 2 KiB
   segmented FP16 normal law. Prove exact agreement with its expanded-table
   numerical definition and benchmark it against the 128 KiB oracle lookup.
2. Complete one real expert across gate, up, and down at K5/K6 against fresh
   MCG, then gate one expert per layer.
3. Only restart the frozen 48-expert panel if both rates retain the FP16 SQG
   advantage, absolute error decreases monotonically, and payload/proxy
   closure remains sane.
4. Keep K3 on native T12/E4M3. Its full-FP16 gain was only about 0.076% on the
   first complete expert and does not justify a richer runtime endpoint.

## Current code status

The working tree contains experimental K5/K6 support, a research-only FP16
table encoder, and generic fixed-path codebook replay. The production QSRT
mixed-rate container and K2/K3/K4 format remain unchanged. The incomplete
E4M3 pilot directory must not be presented as a completed report or resumed
after changing the graph/codebook identity; use a fresh destination for every
new construction.
