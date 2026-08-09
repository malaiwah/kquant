---
title: "CUDA: launch batch is bounded by temp_costs, but the kernel indexes temp_edges, whose leading dimension is never validated"
labels: [bug]
---

Both exported entry points of the SQG CUDA extension size their launch batch
from one scratch tensor while the kernel slices a *different* one per block.

## The mismatch

`kquant/csrc/sqg_quantize.cu:225` (and `:308` for `quantize_tiles_procedural_cuda`):

```cpp
const int max_batch = min(
    static_cast<int>(temp_costs.size(0)), 3 * multiprocessors);
```

`launch_trellis` then issues `batch = min(max_batch, tiles - start)` blocks, and
each block takes its own slice of `temp_edges` indexed by `blockIdx.x`
(`kquant/csrc/qsrt_quantize_tiles_kernel.cuh:55`):

```cpp
const int tile_idx = blockIdx.x;
uint8_t* temp_edges = temp_edges_ptr + 256 * decisions_per_row * tile_idx;
```

So the tensor that must have at least `max_batch` leading rows is `temp_edges`.
The host validates its inner dimensions and nothing else:

```cpp
TORCH_CHECK(temp_edges.scalar_type() == at::kByte && temp_edges.dim() == 3,
            "temp_edges must be rank-3 uint8");
TORCH_CHECK(temp_edges.size(1) == 256 && temp_edges.size(2) == decision_entries,
            "temp_edges shape mismatch");
```

`temp_edges.size(0)` is never checked, against `max_batch` or against anything
else. Any block with `blockIdx.x >= temp_edges.size(0)` writes past the end of
the allocation — an out-of-bounds global write, in the traceback buffer the
Viterbi decode reads back at
`qsrt_quantize_tiles_kernel.cuh:425-435`.

The tensor that *is* bounding the batch, `temp_costs`, is not read or written by
the kernel at all for any supported `K` — see the companion issue about the dead
`temp_costs` buffer. The batch is bounded by a buffer the kernel never touches.

## Trigger

`temp_edges` with fewer leading rows than `min(temp_costs.size(0), 3 * SM)`, and
`tiles` greater than `temp_edges.size(0)`. For example, on an 80-SM device with
`temp_costs` shaped `[256, 2, edges]` and `temp_edges` shaped
`[64, 256, decisions]`, `max_batch` is 240 and blocks 64..239 each write
`256 * decisions_per_row` bytes past the end of `temp_edges`.

## Scope

No in-tree caller triggers this today. `kquant/sqg_quantizer.py:_sqg_temp_buffers`
allocates both buffers with the same `max_batch`, so the two always agree:

```python
costs = torch.zeros((max_batch, 2, edges), dtype=torch.float16, device=device)
traceback = torch.empty((max_batch, 256, decisions), dtype=torch.uint8, device=device)
```

But `quantize_tiles_sqg` and `quantize_tiles_procedural` are public pybind11
entry points (`kquant/csrc/sqg_quantize.cpp:24-33`) with no Python wrapper
enforcing the pairing, and every other precondition in these functions *is*
checked. This is the one hole in an otherwise careful validation block.

## Suggested fix

Bound the batch by the buffer that is actually indexed, and assert the pair
agrees, in both entry points:

```cpp
TORCH_CHECK(temp_edges.size(0) == temp_costs.size(0),
            "temp_edges and temp_costs must share a leading dimension");
const int max_batch = min(
    static_cast<int>(temp_edges.size(0)), 3 * multiprocessors);
```

Note `max_batch` can also reach 0 if a caller passes an empty scratch tensor,
which yields a zero-grid launch; a `TORCH_CHECK(max_batch > 0, …)` would turn
that into a named error too.
