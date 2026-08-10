#!/usr/bin/env python3
"""Measure the authenticated Fruit expert curve at uniform K2/K3/K4 rates."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.qsrt_import_guard import (  # isort: skip
    QSRT_IMPORT_IDENTITY as _QSRT_IMPORT_IDENTITY,
)

from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.fruit_calibration import (
    FRUIT_CALIBRATION_AUTHORITIES,
    FruitCalibrationStore,
    fruit_calibration_authority,
)
from qsrt.fruit_rate_evidence import (
    FRUIT_RATE_SWEEP_SAMPLE_ASSIGNMENTS,
    FRUIT_UNIFORM_RATE_SWEEP_RATES,
    run_fruit_uniform_rate_sweep,
)
from qsrt.fruit_source import FruitCheckpointStore, FruitSafetensorsStore
from qsrt.sqg_quantizer import install_sqg_quantizer
from scripts.build_fruit_qsrt_model import (
    _validate_source_evidence,
    current_encoder_provenance,
)
from scripts.encode_fruit_qsrt import _assignment

_RATES = FRUIT_UNIFORM_RATE_SWEEP_RATES


def _parse_rates(value: str) -> tuple[int, ...]:
    try:
        rates = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "rates must be comma-separated integers"
        ) from exc
    if (
        not rates
        or len(set(rates)) != len(rates)
        or any(rate not in _RATES for rate in rates)
    ):
        raise argparse.ArgumentTypeError("rates must be a unique subset of 2,3,4")
    return rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--variant",
        choices=tuple(FRUIT_CALIBRATION_AUTHORITIES),
        default="annealed",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--assignment", action="append", type=_assignment)
    selection.add_argument("--sample", action="store_true")
    parser.add_argument("--rates", type=_parse_rates, default=_RATES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exllamav3-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authority = fruit_calibration_authority(args.variant)
    spec = authority.spec
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Fruit uniform-rate sweep requires a CUDA device")
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    assignments = (
        FRUIT_RATE_SWEEP_SAMPLE_ASSIGNMENTS
        if args.sample
        else tuple(sorted(set(args.assignment or ())))
    )
    if not assignments:
        raise ValueError("no Fruit assignments selected")

    if args.checkpoint.is_dir():
        if spec.safetensors_manifest_sha256 is None:
            raise ValueError("Fruit variant has no pinned Safetensors source")
        store = FruitSafetensorsStore(
            args.checkpoint,
            spec=spec,
            expected_manifest_sha256=spec.safetensors_manifest_sha256,
        )
    else:
        store = FruitCheckpointStore(
            args.checkpoint,
            spec=spec,
            expected_sha256=spec.checkpoint_sha256,
        )
    try:
        calibration_store = FruitCalibrationStore(
            args.calibration,
            authority=authority,
        )
        try:
            encoder = current_encoder_provenance(
                exllamav3_root=args.exllamav3_root,
                calibration=calibration_store,
                qsrt_identity=_QSRT_IMPORT_IDENTITY,
            )
            if (
                encoder["qsrt_revision"],
                encoder["qsrt_source_sha256"],
            ) != _QSRT_IMPORT_IDENTITY:
                raise ValueError(
                    "QSRT sources changed while importing the rate sweep"
                )
            quantizer_module = load_qsrt_encoder(args.exllamav3_root)
            install_sqg_quantizer(quantizer_module)

            def verify_encoder() -> None:
                if (
                    current_encoder_provenance(
                        exllamav3_root=args.exllamav3_root,
                        calibration=calibration_store,
                        qsrt_identity=_QSRT_IMPORT_IDENTITY,
                    )
                    != encoder
                ):
                    raise ValueError(
                        "Fruit encoder sources changed during the rate sweep"
                    )

            run_fruit_uniform_rate_sweep(
                args.output,
                store=store,
                calibration=calibration_store,
                encoder=encoder,
                source_evidence=_validate_source_evidence(store.evidence, spec=spec),
                assignments=assignments,
                rates=args.rates,
                device=device,
                quantizer_module=quantizer_module,
                verify_encoder=verify_encoder,
                resume=args.resume,
            )
        finally:
            calibration_store.close()
    finally:
        store.close()


if __name__ == "__main__":
    main()
