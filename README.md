# QSRT

QSRT is a fixed-payload weight codec for gated mixture-of-experts models. The
name means **Quantile-Stratified Rate-shifted Trellis**.

QSRT encodes each weight sequence as a path through a finite-state graph. The
stored bits select both the current reconstruction value and the state that
determines future choices. Each state exposes values from different probability
regions, giving a low-rate path broad sign and magnitude coverage. Reconstructed
weights use a finite eight-bit floating-point alphabet.

The Kimi-K3 implementation supports:

- a uniform two-bit routed-expert representation that occupies 2.004464 bits
  per weight including scales;
- a 3.083-bit all-QSRT representation with two four-bit records and twenty-two
  three-bit records per expert matrix;
- equal-size exchanges between two- and four-bit records around a three-bit
  baseline; and
- an exact high-quality endpoint that preserves the source model's four-bit
  microscaled (MXFP4) weights.

The two-bit representation uses an exact coupled Hadamard change of basis
across the gate, up, and down matrices. The encoder reconstructs the quantized
gate and up projections before deriving the down-projection covariance, then
selects complete expert candidates using naturally routed activation error.

This repository contains the offline encoder, calibration and covariance
tools, candidate selection, canonical checkpoint storage, and correctness
validation. Production model integration lives in vLLM, and GPU kernels live
in B12X.

## Setup

```bash
uv sync --dev
.venv/bin/pytest -q
```

## Documentation

- [Two-bit codec](docs/qsrt-2bpw-codec.md)
- [Complete technical specification](docs/qsrt-technical-brief.md)

Generated checkpoints, calibration captures, traces, and benchmark output are
not source artifacts and must not be committed.
