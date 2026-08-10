"""Research-only bridge from EXL's production encoder to SQG E4M3 tables."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping

import torch
from torch.utils.cpp_extension import load

from kquant.sqg_e4m3 import sqg_e4m3_bytes


def _debug_check_trellis_closure(indices: torch.Tensor, bits: int) -> None:
    """Fail at the CUDA boundary if a tile contains an inconsistent state.

    This is intentionally opt-in because it adds a device synchronization to
    every tile-quantizer call.  It distinguishes an invalid state written by
    the traceback kernel from corruption introduced later while mixed-rate
    candidates are assembled and scored.
    """

    edges = indices.to(torch.int64) & ((1 << bits) - 1)
    expected = torch.zeros_like(edges)
    for lag in range(math.ceil(16 / bits)):
        expected |= torch.roll(edges, shifts=lag, dims=-1) << (lag * bits)
    expected = (expected & 0xFFFF).to(torch.int16)
    mismatch = indices != expected
    if bool(torch.any(mismatch)):
        count = int(torch.count_nonzero(mismatch))
        first = tuple(
            int(value)
            for value in torch.nonzero(mismatch, as_tuple=False)[0].cpu().tolist()
        )
        encoded = int(indices[first].to(torch.int32).cpu())
        reconstructed = int(expected[first].to(torch.int32).cpu())
        raise RuntimeError(
            "SQG tile kernel emitted a non-closing trellis state: "
            f"K{bits}, {count} states differ; first at {first}, "
            f"encoded={encoded}, reconstructed={reconstructed}"
        )


@lru_cache(maxsize=1)
def _extension():
    project = Path(__file__).resolve().parents[1]
    return load(
        name="kquant_sqg_quantize_ext_v25",
        sources=[
            str(project / "kquant/csrc/sqg_quantize.cpp"),
            str(project / "kquant/csrc/sqg_quantize.cu"),
        ],
        extra_include_paths=[str(project / "kquant/csrc")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-Xcudafe",
            "--diag_suppress=177",
            "-Xcudafe",
            "--diag_suppress=20012",
        ],
        verbose=False,
    )


@lru_cache(maxsize=None)
def _sqg_temp_buffers(device: torch.device, bits: int):
    """Allocate the packed traceback buffers consumed by kquant's kernel."""

    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    edges = 65536 >> bits
    decisions = edges // 4 if bits == 2 else edges // 2 if bits <= 4 else edges
    free_bytes, _ = torch.cuda.mem_get_info(device)
    decision_bytes_per_tile = 256 * decisions
    affordable = max(256, int(free_bytes * 0.5) // decision_bytes_per_tile)
    max_batch = min(max(256, 3 * multiprocessors), affordable)
    costs = torch.zeros(
        (max_batch, 2, edges), dtype=torch.float16, device=device
    )
    traceback = torch.empty(
        (max_batch, 256, decisions), dtype=torch.uint8, device=device
    )
    return costs, traceback


def install_sqg_quantizer(quantizer_module) -> None:
    """Teach a loaded EXL encoder module to consume ``sqg_e4m3_lut``.

    The patch is process-local. SQG and explicit MCG/MUL1 controls use
    kquant's CUDA extension with one Viterbi/tail-biting implementation. A
    ``None`` entry in a rate-specific mapping explicitly selects MCG,
    permitting controlled hybrid rate-curve studies. Calls without any
    kquant codebook argument retain the unmodified upstream EXL behavior.
    """

    if getattr(quantizer_module, "_kquant_sqg_installed", False):
        return
    original = quantizer_module.quantize_tiles

    device_luts: dict[tuple[str, int, str], torch.Tensor] = {}
    transposed_sqg_luts: dict[
        tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]
    ] = {}

    def quantize_tiles(tiles: torch.Tensor, quant_args: dict):
        codebook = quant_args.get("sqg_e4m3_lut")
        fp16_codebook = quant_args.get("sqg_fp16_lut")
        rate_codebooks = quant_args.get("sqg_e4m3_luts_by_bits")
        mode = quant_args.get("sqg_e4m3_mode")
        if (
            codebook is None
            and fp16_codebook is None
            and rate_codebooks is None
            and mode is None
        ):
            return original(tiles, quant_args)
        if fp16_codebook is not None and (
            codebook is not None or rate_codebooks is not None or mode is not None
        ):
            raise ValueError("an FP16 SQG table cannot be combined with another codebook")
        if len(quant_args["devices"]) != 1:
            raise ValueError("the SQG validation hook currently requires one CUDA device")
        tiles = tiles.contiguous()
        if tiles.dtype != torch.float32 or tiles.ndim != 2 or tiles.shape[1] != 256:
            raise ValueError("SQG tiles must be contiguous FP32 [N, 256]")
        bits = int(quant_args["K"])
        if fp16_codebook is not None:
            output = torch.empty_like(tiles)
            indices = torch.empty_like(tiles, dtype=torch.int16)
            costs, edges = _sqg_temp_buffers(tiles.device, bits)
            lut = fp16_codebook.to(
                device=tiles.device, dtype=torch.float16
            ).contiguous()
            if lut.ndim != 1 or lut.numel() != 65536 or not bool(
                torch.isfinite(lut).all()
            ):
                raise ValueError(
                    "an experimental FP16 SQG table must contain 65,536 finite values"
                )
            _extension().quantize_tiles_fp16(
                tiles,
                output,
                indices,
                costs,
                edges,
                lut,
                bits,
                int(quant_args.get("tailbite_context", 128)),
            )
            if os.environ.get("KQUANT_QSRT_DEBUG_TILE_CLOSURE") == "1":
                _debug_check_trellis_closure(indices, bits)
            return output, indices
        if rate_codebooks is not None:
            if codebook is not None or mode is not None:
                raise ValueError(
                    "rate-specific SQG LUTs cannot be combined with another SQG law"
                )
            if not isinstance(rate_codebooks, Mapping):
                raise TypeError("sqg_e4m3_luts_by_bits must be a mapping")
            if set(rate_codebooks) - {2, 3, 4}:
                raise ValueError("rate-specific SQG LUT keys must be K2, K3, or K4")
            try:
                codebook = rate_codebooks[bits]
            except KeyError as exc:
                raise ValueError(f"missing rate-specific SQG K{bits} LUT") from exc
            if codebook is None:
                output = torch.empty_like(tiles)
                indices = torch.empty_like(tiles, dtype=torch.int16)
                costs, edges = _sqg_temp_buffers(tiles.device, bits)
                _extension().quantize_tiles_procedural(
                    tiles,
                    output,
                    indices,
                    costs,
                    edges,
                    bits,
                    1,
                    int(quant_args.get("tailbite_context", 128)),
                )
                return output, indices
        elif codebook is None:
            if mode != "normal":
                raise ValueError("the supported R44 mode is 'normal'")
            key = (str(tiles.device), bits, mode)
            codebook = device_luts.get(key)
            if codebook is None:
                codebook = sqg_e4m3_bytes(bits, mode, device=tiles.device)
                device_luts[key] = codebook
        output = torch.empty_like(tiles)
        indices = torch.empty_like(tiles, dtype=torch.int16)
        costs, edges = _sqg_temp_buffers(tiles.device, bits)
        lut = codebook.to(device=tiles.device, dtype=torch.uint8).contiguous()
        if bits in (2, 3, 4):
            source_key = codebook.data_ptr() if codebook.is_cuda else id(codebook)
            cache_key = (str(tiles.device), bits, source_key)
            cached = transposed_sqg_luts.get(cache_key)
            if cached is None or cached[0] is not codebook:
                # [predecessor, out-edge-pair, pair-byte] ->
                # [out-edge-pair, predecessor, pair-byte].  Each CUDA thread
                # can then fetch all predecessor labels in one uint2 (K2),
                # one uint4 (K3), or two uint4s (K4), rather than issuing one
                # gather per predecessor.
                predecessors = 1 << bits
                out_edge_pairs = (65536 >> bits) // 2
                lut = (
                    lut.reshape(predecessors, out_edge_pairs, 2)
                    .permute(1, 0, 2)
                    .contiguous()
                    .reshape(-1)
                )
                transposed_sqg_luts[cache_key] = (codebook, lut)
            else:
                lut = cached[1]
        _extension().quantize_tiles_sqg(
            tiles,
            output,
            indices,
            costs,
            edges,
            lut,
            bits,
            int(quant_args.get("tailbite_context", 128)),
        )
        if os.environ.get("KQUANT_QSRT_DEBUG_TILE_CLOSURE") == "1":
            _debug_check_trellis_closure(indices, bits)
        return output, indices

    quantizer_module.quantize_tiles = quantize_tiles
    quantizer_module._kquant_sqg_installed = True
