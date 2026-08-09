---
title: "CUDA: the temp_costs global buffer is unreachable for every supported K, wasting ~28 MiB of VRAM per GPU"
labels: [enhancement]
---

The SQG kernel takes a `temp_costs` global buffer that it never reads or writes
at any rate the extension accepts.

## The dead branch

`temp_costs_ptr` occurs exactly twice in
`kquant/csrc/qsrt_quantize_tiles_kernel.cuh` — once as a parameter (line 31),
and once here (line 62):

```cpp
half* temp_costs = K >= 2 ? sh_temp_costs : temp_costs_ptr + 2 * edges * tile_idx;
```

The host constrains `K` to 2–4 in both entry points:

```cpp
TORCH_CHECK(K >= 2 && K <= 4, "SQG validation kernel supports K2--K4");
```

so the `temp_costs_ptr` branch is unreachable for every `K` that can reach the
kernel. Cost accumulation always uses the shared-memory `sh_temp_costs`. The
buffer is allocated, shape-validated, and passed across the ABI for nothing.

## The cost

`kquant/sqg_quantizer.py:_sqg_temp_buffers` allocates it with `torch.zeros`, so
it also pays a device memset:

```python
costs = torch.zeros((max_batch, 2, edges), dtype=torch.float16, device=device)
```

| K | `edges` = 65536 >> K | `costs` bytes at `max_batch` = 256 |
| --- | --- | --- |
| 2 | 16,384 | 16 MiB |
| 3 | 8,192 | 8 MiB |
| 4 | 4,096 | 4 MiB |

`_sqg_temp_buffers` is `@lru_cache(maxsize=None)` keyed on `(device, bits)`, so a
run touching all three rates holds roughly 28 MiB of dead VRAM per GPU for the
lifetime of the process — on the order of 336 MiB across a TP12 encode.

There is a second-order effect. The same function derives `max_batch` from a
free-memory estimate that counts only the traceback buffer:

```python
decision_bytes_per_tile = 256 * decisions
affordable = max(256, int(free_bytes * 0.5) // decision_bytes_per_tile)
max_batch = min(max(256, 3 * multiprocessors), affordable)
```

`costs` is absent from that budget, so the allocator's own sizing heuristic is
wrong by exactly the amount of the unused buffer.

## Suggested fix

Drop the parameter and the allocation, and delete the `K < 2` branch. If the
sub-K2 path is deliberately being held for future work, keep it but say so in a
comment — and in either case stop deriving the launch batch from it, which is a
separate defect tracked in the companion issue about the batch bound.
