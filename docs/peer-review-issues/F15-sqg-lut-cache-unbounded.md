---
title: "install_sqg_quantizer: process-local LUT caches are never evicted"
labels: [enhancement]
---

`kquant/sqg_quantizer.py`. `install_sqg_quantizer` closes over two dictionaries:

```python
device_luts: dict[tuple[str, int, str], torch.Tensor] = {}
transposed_sqg_luts: dict[
    tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}
```

Neither is ever evicted, and both outlive every call because the closure is
installed onto the encoder module for the life of the process.

`transposed_sqg_luts` is keyed on `(device, bits, id(codebook)/data_ptr())` and
stores `(codebook, transposed_lut)`. Retaining the source `codebook` in
`cached[0]` is deliberate and correct — it is what keeps the identity guard
(`cached[0] is not codebook`) sound, since a bare `id()` key could otherwise
collide with a freed-and-reallocated tensor. The consequence is that every
distinct codebook that passes through stays alive, alongside its transpose.

This is a leak, not a correctness bug: the identity guard means no stale or
wrong LUT is ever returned. But an encode that pushes many distinct
`sqg_e4m3_lut` tensors through `quantize_tiles` — a rate-curve or
candidate-codebook sweep is exactly that shape of workload — grows both dicts
monotonically. Each retained pair pins a 65,536-byte source codebook plus its
transposed device copy, and nothing releases them until the process exits.

## Suggested fix

Either bound the caches with an LRU (a small `maxsize` is plenty; the working set
during any one layer is tiny), or key on a content digest of the codebook rather
than its identity. A digest key would let repeated codebooks collapse onto one
entry and would remove the reason to retain the source tensor at all, which
resolves the retention and the unbounded growth in the same change.
