---
title: "CUDA: output and scratch tensors are never checked for device or contiguity in either entry point"
labels: [bug]
---

`kquant/csrc/sqg_quantize.cu` validates its inputs carefully — dtype, rank, and
exact shapes on every tensor, plus `K` and `tailbite_context` ranges. Two checks
are missing, and they are the two that keep raw pointer arithmetic safe.

## What is actually checked

Across all 342 lines and both exported entry points, `is_contiguous()` is called
exactly once and `is_cuda()` twice:

```
175:    const at::cuda::OptionalCUDAGuard guard(input_tiles.device());
178:    TORCH_CHECK(input_tiles.is_cuda(), "input_tiles must be CUDA");
193:    TORCH_CHECK(codebook.is_cuda() && codebook.device() == input_tiles.device(), …);
197:    TORCH_CHECK(codebook.is_contiguous(), "codebook must be contiguous");
271:    const at::cuda::OptionalCUDAGuard guard(input_tiles.device());
274:    TORCH_CHECK(input_tiles.is_cuda(), "input_tiles must be CUDA");
```

`output_tiles`, `output_indices`, `temp_costs` and `temp_edges` receive shape and
dtype checks only. `input_tiles` itself is checked for device but never for
contiguity. All of them are then handed to the kernel as raw `data_ptr<T>()` and
indexed with linear offsets that assume row-major contiguous storage:

```cpp
const float*    input_tile     = input_tiles_ptr    + 256 * tile_idx;
float*          output_tile    = output_tiles_ptr   + 256 * tile_idx;
uint16_t*       output_indices = output_indices_ptr + 256 * tile_idx;
uint8_t*        temp_edges     = temp_edges_ptr + 256 * decisions_per_row * tile_idx;
```

## Two reachable consequences

**Wrong device, or a CPU tensor.** An `output_tiles` allocated on a second CUDA
device — or a plain CPU tensor of the right shape and dtype — passes every
existing check. `OptionalCUDAGuard` sets the current device from `input_tiles`,
so `output_tiles.data_ptr<float>()` yields an address the guarded device cannot
resolve and the kernel faults with an illegal memory access. For a project whose
entire deployment story is TP12 multi-GPU, handing in a tensor from the wrong
device is an ordinary operator slip that deserves a named error rather than a
CUDA abort with no attribution.

**Non-contiguous view.** `big[:, :256]` on a wider tensor satisfies
`dim() == 2 && size(1) == 256`, is CUDA, and is float. It passes. The kernel then
walks `input_tiles_ptr + 256 * tile_idx` across a buffer whose real row stride is
larger, reading the wrong elements for every tile after the first — silently, with
no error, producing a corrupt encode. Same hazard for a transposed or `expand`ed
view on any of the five tensors.

## Suggested fix

One block per entry point, before the pointers are taken:

```cpp
TORCH_CHECK(input_tiles.is_contiguous(), "input_tiles must be contiguous");
for (const auto& t : {output_tiles, output_indices, temp_costs, temp_edges}) {
    TORCH_CHECK(t.is_cuda() && t.device() == input_tiles.device(),
                "all tensors must share the input CUDA device");
    TORCH_CHECK(t.is_contiguous(), "all tensors must be contiguous");
}
```

These are cheap host-side checks on a call that already launches a multi-block
kernel, so the cost is irrelevant next to turning two silent-corruption and
hard-fault paths into named errors.
