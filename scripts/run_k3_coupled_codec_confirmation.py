#!/usr/bin/env python3
"""Confirm promoted coupled transforms with fresh uniform SQG/MCG encodes.

This is a bounded research comparison at uniform K2 by default.  It compares
the official decoded Kimi expert against independently encoded MCG and the
production SQG-T12 scalar law, with and without an explicit coupled two-sided
block-Hadamard reparameterization.  It writes no checkpoint payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from kquant import constants as C
from kquant.capture import index_cached_layer_samples
from kquant.coupled_expert_study import (
    CoupledTriplet,
    RoutedOutputMetric,
    encode_coupled_block_hadamard,
    execute_coupled_block_hadamard,
    expert_hidden,
)
from kquant.exl3_loader import load_qsrt_encoder
from kquant.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from kquant.io.stream import load_tensor
from kquant.qsrt import matrix_rate_axis
from kquant.qsrt_codec_pilot import CODEBOOK_MCG, encode_uniform_candidate
from kquant.source_weights import OfficialMXFP4Store
from kquant.sqg_quantizer import install_sqg_quantizer


KIND = "kquant_k3_coupled_uniform_codec_confirmation"
SCHEMA_VERSION = 1
DEFAULT_CACHE = Path(
    "/data/kquant/captures/k3-codec-diverse-validation-v3-128k-input-v1.kqsamples"
)


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected a nonempty list of unique integers")
    return result


def _parse_names(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected a nonempty list of unique names")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _output_metric(store: OfficialMXFP4Store, layer: int) -> RoutedOutputMetric:
    prefix = f"{C.LM_PREFIX}layers.{layer}.block_sparse_moe"
    return RoutedOutputMetric(
        load_tensor(store.cache, f"{prefix}.routed_expert_norm.weight").float(),
        load_tensor(store.cache, C.latent_up_proj_tensor(layer)).float(),
    )


def _rows(samples: Any, expert: int, maximum: int) -> dict[str, torch.Tensor]:
    locations = torch.nonzero(samples.input_experts == expert, as_tuple=False)
    if locations.numel() == 0:
        raise ValueError(f"expert {expert} has no validation rows")
    if locations.shape[0] > maximum:
        indices = torch.linspace(0, locations.shape[0] - 1, maximum).round().long()
        locations = locations.index_select(0, indices)
    rows, slots = locations[:, 0], locations[:, 1]
    return {
        "inputs": samples.input_values.index_select(0, rows).float(),
        "gates": samples.input_gates[rows, slots].float(),
        "aggregate": samples.routed_latent.index_select(0, rows).float(),
        "documents": torch.bitwise_right_shift(
            samples.input_observations.index_select(0, rows), 32
        ),
    }


def _external_transform(source: CoupledTriplet) -> CoupledTriplet:
    return encode_coupled_block_hadamard(source, block_size=512)


def _execute_arm(inputs: torch.Tensor, reconstruction: CoupledTriplet, arm: str) -> torch.Tensor:
    if arm == "baseline":
        return expert_hidden(inputs, reconstruction) @ reconstruction.down.T
    if arm != "coupled_hadamard":
        raise ValueError(f"unknown confirmation arm {arm!r}")
    return execute_coupled_block_hadamard(
        inputs, reconstruction, block_size=512
    )


def _encode_triplet(
    source: CoupledTriplet,
    *,
    layer: int,
    expert: int,
    arm: str,
    bits: int,
    codebook: str,
    device: torch.device,
    quantizer_module: Any,
    ldlq_tf32: bool,
) -> tuple[CoupledTriplet, list[dict[str, Any]]]:
    reconstructions = []
    evidence = []
    for matrix, weight in zip(C.EXPERT_MATRICES, source.tensors(), strict=True):
        seed = layer * 1_000_000 + C.EXPERT_MATRICES.index(matrix)
        result = encode_uniform_candidate(
            weight,
            bits=bits,
            codebook=codebook,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=seed,
            output_sign_seed=seed + 499_979,
            rate_axis=matrix_rate_axis(matrix),
            scale_scope_key=(KIND, layer, expert, arm, codebook, matrix, bits),
            g_scale_into_sv=matrix in ("w1", "w3"),
            ldlq_tf32=ldlq_tf32,
        )
        reconstructions.append(result["reconstruction"].float())
        evidence.append(result["payload"])
        torch.cuda.empty_cache()
    return CoupledTriplet(*reconstructions), evidence


def _score(
    source: CoupledTriplet,
    encoded_source: CoupledTriplet,
    reconstruction: CoupledTriplet,
    rows: dict[str, torch.Tensor],
    output_metric: RoutedOutputMetric,
    arm: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    source_output = expert_hidden(rows["inputs"], source) @ source.down.T
    output = _execute_arm(rows["inputs"], reconstruction, arm)
    error = output - source_output
    routed_error = rows["gates"][:, None] * error
    exact = output_metric.exact_delta(rows["aggregate"], routed_error)
    weight_sse = sum(
        float((candidate.double() - target.double()).square().sum())
        for candidate, target in zip(
            reconstruction.tensors(), encoded_source.tensors(), strict=True
        )
    )
    weight_energy = sum(
        float(target.double().square().sum()) for target in encoded_source.tensors()
    )
    payload_bits = sum(int(item["trellis_bytes"]) * 8 for item in evidence)
    scale_bits = sum(int(item["scale_bytes"]) * 8 for item in evidence)
    return {
        "weight_nmse": weight_sse / weight_energy,
        "expert_output_nmse": float(
            error.double().square().sum()
            / source_output.double().square().sum().clamp_min(1e-30)
        ),
        "post_projection_sse": float(exact.double().square().sum()),
        "rows": int(rows["inputs"].shape[0]),
        "documents": int(torch.unique(rows["documents"]).numel()),
        "trellis_bpw": payload_bits / source.numel,
        "scale_bpw": scale_bits / source.numel,
        "all_in_bpw": (payload_bits + scale_bits) / source.numel,
        "matrices": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", type=_parse_ints, required=True)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument(
        "--codebooks",
        type=_parse_names,
        default=(CODEBOOK_MCG, CODEBOOK_SQG_XOR_CHEB_T12),
    )
    parser.add_argument(
        "--arms", type=_parse_names, default=("baseline", "coupled_hadamard")
    )
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--maximum-rows", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exllamav3-root", type=Path, default=Path("/home/luke/projects/exllamav3"))
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument("--ldlq-tf32", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.layer not in C.MOE_LAYERS:
        parser.error("--layer must be a Kimi MoE layer 1..92")
    if any(not 0 <= expert < C.NUM_EXPERTS for expert in args.experts):
        parser.error(f"--experts must be in 0..{C.NUM_EXPERTS - 1}")
    if args.bits not in range(2, 7):
        parser.error("--bits must be K2..K6")
    if any(item not in (CODEBOOK_MCG, CODEBOOK_SQG_XOR_CHEB_T12) for item in args.codebooks):
        parser.error("--codebooks supports only mcg and sqg_xor_cheb_t12")
    if any(item not in ("baseline", "coupled_hadamard") for item in args.arms):
        parser.error("--arms supports baseline and coupled_hadamard")
    if args.maximum_rows <= 0:
        parser.error("--maximum-rows must be positive")
    return args


def main() -> None:
    args = parse_args()
    signature = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "source_model": C.MODEL_ID,
        "source_revision": args.official_revision,
        "layer": args.layer,
        "experts": list(args.experts),
        "bits": args.bits,
        "codebooks": list(args.codebooks),
        "arms": list(args.arms),
        "validation_cache": str(args.validation_cache.resolve()),
        "validation_manifest_sha256": _sha256(args.validation_cache / "manifest.json"),
        "maximum_rows": args.maximum_rows,
        "ldlq_tf32": args.ldlq_tf32,
        "no_qat": True,
        "writes_checkpoint_payloads": False,
    }
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        payload = json.loads(args.output.read_text())
        if payload.get("signature") != signature:
            raise ValueError("resume output signature mismatch")
    else:
        payload = {"signature": signature, "results": {}, "complete": False}
        _atomic_json(args.output, payload)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("codec confirmation requires an available CUDA device")
    store = OfficialMXFP4Store(revision=args.official_revision)
    samples = index_cached_layer_samples(args.validation_cache, [args.layer - 1]).pop(
        args.layer - 1
    )
    output_metric = _output_metric(store, args.layer)
    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    for expert in args.experts:
        key = str(expert)
        if key in payload["results"]:
            continue
        started = time.time()
        selected_rows = _rows(samples, expert, args.maximum_rows)
        with store.open_layer(args.layer, experts=(expert,)) as layer_store:
            source = CoupledTriplet(
                layer_store.load_matrix(args.layer, expert, "w1"),
                layer_store.load_matrix(args.layer, expert, "w3"),
                layer_store.load_matrix(args.layer, expert, "w2"),
            )
        arm_sources = {
            "baseline": source,
            "coupled_hadamard": _external_transform(source),
        }
        results: dict[str, Any] = {}
        for codebook in args.codebooks:
            results[codebook] = {}
            for arm in args.arms:
                reconstruction, evidence = _encode_triplet(
                    arm_sources[arm],
                    layer=args.layer,
                    expert=expert,
                    arm=arm,
                    bits=args.bits,
                    codebook=codebook,
                    device=device,
                    quantizer_module=quantizer_module,
                    ldlq_tf32=args.ldlq_tf32,
                )
                results[codebook][arm] = _score(
                    source,
                    arm_sources[arm],
                    reconstruction,
                    selected_rows,
                    output_metric,
                    arm,
                    evidence,
                )
                print(
                    f"layer {args.layer} expert {expert} {codebook} {arm}: "
                    f"post SSE {results[codebook][arm]['post_projection_sse']:.6g}",
                    flush=True,
                )
                del reconstruction
                torch.cuda.empty_cache()
        comparisons = {}
        for codebook in args.codebooks:
            baseline = results[codebook].get("baseline")
            transformed = results[codebook].get("coupled_hadamard")
            if baseline and transformed:
                comparisons[f"{codebook}:coupled_hadamard_vs_baseline"] = {
                    metric: 1.0 - transformed[metric] / baseline[metric]
                    for metric in ("weight_nmse", "expert_output_nmse", "post_projection_sse")
                }
        if CODEBOOK_MCG in results and CODEBOOK_SQG_XOR_CHEB_T12 in results:
            for arm in args.arms:
                comparisons[f"sqg_vs_mcg:{arm}"] = {
                    metric: 1.0
                    - results[CODEBOOK_SQG_XOR_CHEB_T12][arm][metric]
                    / results[CODEBOOK_MCG][arm][metric]
                    for metric in ("weight_nmse", "expert_output_nmse", "post_projection_sse")
                }
        payload["results"][key] = {
            "expert": expert,
            "candidates": results,
            "comparisons": comparisons,
            "seconds": time.time() - started,
        }
        _atomic_json(args.output, payload)
    payload["complete"] = True
    _atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output.resolve()), "complete": True}, indent=2))


if __name__ == "__main__":
    main()
