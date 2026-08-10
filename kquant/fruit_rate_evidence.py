"""Sealed uniform-rate evidence for authenticated Fruit QSRT encoders."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from kquant.fruit_calibration import FruitCalibrationStore
from kquant.fruit_qsrt import FruitMatrixStore, measure_fruit_uniform_rates

FRUIT_UNIFORM_RATE_SWEEP_SCHEMA = "kquant_fruit_uniform_rate_sweep_v1"
FRUIT_UNIFORM_RATE_SWEEP_RATES = (2, 3, 4)
FRUIT_RATE_SWEEP_SAMPLE_ASSIGNMENTS: tuple[tuple[int, int], ...] = (
    (3, 0),
    (3, 1),
    (3, 255),
    (4, 17),
    (4, 128),
    (5, 31),
    (5, 224),
    (6, 63),
    (7, 95),
    (8, 127),
    (9, 159),
    (10, 191),
    (11, 223),
    (12, 0),
    (12, 64),
    (12, 192),
    (12, 255),
    (13, 0),
    (13, 255),
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(_canonical_json(value), encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_fruit_uniform_rate_sweep(
    output: Path,
    *,
    store: FruitMatrixStore,
    calibration: FruitCalibrationStore,
    encoder: dict[str, object],
    source_evidence: dict[str, object],
    assignments: tuple[tuple[int, int], ...],
    rates: tuple[int, ...],
    device: torch.device,
    quantizer_module: Any,
    verify_encoder: Callable[[], None],
    resume: bool = False,
) -> dict[str, object]:
    """Measure and seal one fixed assignment set under one encoder identity."""

    if not assignments or len(set(assignments)) != len(assignments):
        raise ValueError("Fruit rate-sweep assignments must be nonempty and unique")
    if (
        not rates
        or len(set(rates)) != len(rates)
        or any(rate not in FRUIT_UNIFORM_RATE_SWEEP_RATES for rate in rates)
    ):
        raise ValueError("Fruit rate-sweep rates must be a unique subset of 2,3,4")
    if output.is_symlink():
        raise ValueError(f"Fruit rate-sweep output must not be symbolic: {output}")
    if output.exists() and (not output.is_file() or output.stat().st_nlink != 1):
        raise ValueError(f"Fruit rate-sweep output must be private: {output}")

    signature = {
        "schema": FRUIT_UNIFORM_RATE_SWEEP_SCHEMA,
        "source": source_evidence,
        "calibration": {
            "capture_id": calibration.capture_id,
            "fingerprint": calibration.fingerprint,
            "manifest_sha256": calibration.manifest_sha256,
        },
        "encoder": encoder,
        "rates": list(rates),
        "assignments": [list(value) for value in assignments],
    }
    if resume:
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            raise ValueError("resume output does not match this Fruit rate sweep")
    else:
        if output.exists():
            raise FileExistsError(output)
        payload = {
            "schema": FRUIT_UNIFORM_RATE_SWEEP_SCHEMA,
            "signature": signature,
            "metric_target": (
                "local routed expert output on sealed document-disjoint Fruit "
                "fit/confirmation/validation rows"
            ),
            "storage_scope": (
                "uniform path-rate endpoint with per-candidate FP16 scales before "
                "layer deduplication; not a serving package"
            ),
            "results": {},
            "complete": False,
        }
        _atomic_json(output, payload)

    results = payload.get("results")
    if not isinstance(results, dict):
        raise TypeError("Fruit rate-sweep results must be a JSON object")
    started = time.perf_counter()
    calibration_layer = None
    calibration_layer_id = None
    with torch.inference_mode():
        for index, (layer, expert) in enumerate(assignments, start=1):
            key = f"{layer}:{expert}"
            if key in results:
                print(
                    f"[{index}/{len(assignments)}] layer {layer} expert {expert}: resume"
                )
                continue
            if calibration_layer_id != layer:
                calibration_layer = calibration.load_layer(layer)
                calibration_layer_id = layer
            assert calibration_layer is not None
            result = measure_fruit_uniform_rates(
                store,
                calibration_layer.expert_rows(expert),
                layer=layer,
                expert=expert,
                rates=rates,
                device=device,
                quantizer_module=quantizer_module,
            )
            results[key] = result
            payload["elapsed_seconds"] = time.perf_counter() - started
            _atomic_json(output, payload)
            print(
                f"[{index}/{len(assignments)}] layer {layer} expert {expert}: "
                f"{result['status']}"
            )
            torch.cuda.empty_cache()
    verify_encoder()
    payload["elapsed_seconds"] = time.perf_counter() - started
    payload["complete"] = True
    _atomic_json(output, payload)
    return payload
