"""Authenticated document-disjoint activation calibration for Fruit QSRT."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

import torch
from safetensors import safe_open

from qsrt.fruit_source import (
    FRUIT_ANNEALED_SPEC,
    FRUIT_INSTRUCT_SPEC,
    FruitModelSpec,
    _copy_authenticated_file,
    fruit_model_spec,
)

FRUIT_CALIBRATION_SCHEMA = "qsrt_fruit_qsrt_calibration_v2"
FRUIT_CALIBRATION_VERSION = 2
FRUIT_CALIBRATION_LAYERS = (*FRUIT_ANNEALED_SPEC.layers, FRUIT_ANNEALED_SPEC.mtp_layer)
FRUIT_CALIBRATION_TOPK = 8
CalibrationSplit = Literal["fit", "confirmation", "validation"]
CALIBRATION_SPLITS: tuple[CalibrationSplit, ...] = (
    "fit",
    "confirmation",
    "validation",
)
_LAYER_TENSORS = (
    "document_ids",
    "global_h13",
    "inputs",
    "route_ids",
    "route_weights",
)
FRUIT_CALIBRATION_AXES = (
    "axis1_general",
    "axis2_legal",
    "axis3_code_agentic",
    "axis4_reasoning_termination",
)
FRUIT_CALIBRATION_DOCUMENTS_PER_AXIS = 64
FRUIT_CALIBRATION_FIT_PER_AXIS = 40
FRUIT_CALIBRATION_CONFIRMATION_PER_AXIS = 16
FRUIT_CALIBRATION_VALIDATION_PER_AXIS = 8
FRUIT_CALIBRATION_TOKEN_BOUNDS = {"minimum": 32, "maximum": 256}
FRUIT_CALIBRATION_PROTOCOL = {
    "normalization": "fruit_data_prep_calib_jsonl_v1",
    "partition": "prompt_family_and_truncated_token_sha256_v1",
    "documents_per_axis": FRUIT_CALIBRATION_DOCUMENTS_PER_AXIS,
    "fit_per_axis": FRUIT_CALIBRATION_FIT_PER_AXIS,
    "confirmation_per_axis": FRUIT_CALIBRATION_CONFIRMATION_PER_AXIS,
    "validation_per_axis": FRUIT_CALIBRATION_VALIDATION_PER_AXIS,
}
FRUIT_CALIBRATION_CORPUS = {
    "filename": "reap_recall_calib.jsonl",
    "sha256": "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4",
}
FRUIT_CALIBRATION_TOKENIZER_FILES = {
    "config.json": "912cb50f96a42a21501367bfe38aa29d531146d76aadc5fd62ac20a3008d7c34",
    "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    "tokenizer_config.json": (
        "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"
    ),
}
FRUIT_CALIBRATION_TRAINER_FILES = {
    "checkpoint_contract.py": (
        "c8267e5c2ec9e8ed5195829020c7ad598d8a354e02cd26495fe7e767b8892d44"
    ),
    "run2_corpus_contract.py": (
        "443645d22fff071cf6410cca23730e011e2af89591f1a8cbb4fd044ccaea908f"
    ),
    "train_fruit.py": (
        "520be10eeaaf4bc525ba1f5d0d91b860556d9995e782a4c6ac225244105ffcd8"
    ),
}
FRUIT_CALIBRATION_CONVENTIONS = {
    "serve_conv_v": 2,
    "trained_rope_theta": FRUIT_ANNEALED_SPEC.trained_rope_theta,
    "moe_impl": "grouped",
    "sdpa_backend": "math",
    "deterministic_algorithms": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _snapshot_calibration_source(
    source_root: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[tempfile.TemporaryDirectory[str], Path, bytes]:
    temporary_directory = tempfile.TemporaryDirectory(
        prefix="qsrt-fruit-calibration-"
    )
    snapshot_root = Path(temporary_directory.name)
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    try:
        source_root_fd = os.open(source_root, directory_flags)
    except OSError as exc:
        temporary_directory.cleanup()
        raise ValueError(
            f"cannot securely open Fruit calibration root: {source_root}"
        ) from exc
    try:
        manifest_bytes, _ = _copy_authenticated_file(
            source_root_fd=source_root_fd,
            filename="calibration-manifest.json",
            destination=snapshot_root / "calibration-manifest.json",
            expected_sha256=expected_manifest_sha256,
            capture_bytes=True,
        )
        assert manifest_bytes is not None
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Fruit calibration manifest is malformed") from exc
        layers = manifest.get("layers") if isinstance(manifest, dict) else None
        if not isinstance(layers, dict):
            raise TypeError("Fruit calibration layer ledger is unavailable")
        copied: set[str] = set()
        for entry in layers.values():
            if not isinstance(entry, dict):
                raise TypeError("Fruit calibration layer ledger entry is invalid")
            filename = entry.get("file")
            expected_sha256 = entry.get("sha256")
            expected_bytes = entry.get("bytes")
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or filename in copied
                or not _is_sha256(expected_sha256)
                or type(expected_bytes) is not int
                or expected_bytes <= 0
            ):
                raise ValueError("Fruit calibration layer ledger entry is invalid")
            _copy_authenticated_file(
                source_root_fd=source_root_fd,
                filename=filename,
                destination=snapshot_root / filename,
                expected_sha256=expected_sha256,
            )
            if (snapshot_root / filename).stat().st_size != expected_bytes:
                raise ValueError(
                    f"Fruit calibration layer byte length mismatch: {filename}"
                )
            copied.add(filename)
    except BaseException:
        temporary_directory.cleanup()
        raise
    finally:
        os.close(source_root_fd)
    return temporary_directory, snapshot_root, manifest_bytes


@dataclass(frozen=True)
class FruitCalibrationAuthority:
    """Pinned source, reference, and closure contract for one Fruit variant."""

    variant: str
    spec: FruitModelSpec
    source_index_sha256: str
    source_file_count: int
    reference_sha256: str
    capture_id: str
    fingerprint: str
    manifest_sha256: str
    max_abs_logprob: float
    rms_logprob: float
    mean_forward_kl: float
    max_forward_kl: float

    def __post_init__(self) -> None:
        if fruit_model_spec(self.variant) is not self.spec:
            raise ValueError("Fruit calibration authority variant/spec mismatch")
        if self.spec.safetensors_manifest_sha256 is None:
            raise ValueError("Fruit calibration authority requires a BF16 source")
        if not _is_sha256(self.source_index_sha256):
            raise ValueError("Fruit calibration source index SHA-256 is invalid")
        if type(self.source_file_count) is not int or self.source_file_count <= 0:
            raise ValueError("Fruit calibration source file count must be positive")
        for name in ("capture_id", "fingerprint", "manifest_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"Fruit calibration {name} is invalid")
        if not _is_sha256(self.reference_sha256):
            raise ValueError("Fruit calibration reference SHA-256 is invalid")
        for name, value in self.closure_limits.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Fruit calibration {name} limit must be positive")
        geometry = (
            self.spec.layers,
            self.spec.mtp_layer,
            self.spec.num_experts,
            self.spec.hidden_size,
            self.spec.intermediate_size,
            self.spec.trained_rope_theta,
        )
        canonical_geometry = (
            FRUIT_ANNEALED_SPEC.layers,
            FRUIT_ANNEALED_SPEC.mtp_layer,
            FRUIT_ANNEALED_SPEC.num_experts,
            FRUIT_ANNEALED_SPEC.hidden_size,
            FRUIT_ANNEALED_SPEC.intermediate_size,
            FRUIT_ANNEALED_SPEC.trained_rope_theta,
        )
        if geometry != canonical_geometry:
            raise ValueError("Fruit calibration variant has unsupported geometry")

    @property
    def source(self) -> dict[str, object]:
        assert self.spec.safetensors_manifest_sha256 is not None
        return {
            "kind": "authenticated_bf16_export",
            "manifest_sha256": self.spec.safetensors_manifest_sha256,
            "index_sha256": self.source_index_sha256,
            "file_count": self.source_file_count,
        }

    @property
    def closure_limits(self) -> dict[str, float]:
        return {
            "max_abs_logprob": self.max_abs_logprob,
            "rms_logprob": self.rms_logprob,
            "mean_forward_kl": self.mean_forward_kl,
            "max_forward_kl": self.max_forward_kl,
        }

    @property
    def conventions(self) -> dict[str, object]:
        return {
            **FRUIT_CALIBRATION_CONVENTIONS,
            "trained_rope_theta": self.spec.trained_rope_theta,
        }


FRUIT_ANNEALED_CALIBRATION_AUTHORITY = FruitCalibrationAuthority(
    variant="annealed",
    spec=FRUIT_ANNEALED_SPEC,
    source_index_sha256=(
        "86e6cc1d8548c7bdbbc117e93b85b8ae249f446de9b48d2195e51f358674ba56"
    ),
    source_file_count=23,
    reference_sha256=(
        "1e11e847d745e7015291543620caec65c99ce8d9bb8c72f4db54f8790bb20ddc"
    ),
    capture_id="cc686d28f505d62653763cdb746e830374207137d103d3c26ebbcce070059053",
    fingerprint="fed8cd68311c3347791f81b817c7cc3cce704056792e56881b97c4319e07dd2b",
    manifest_sha256=(
        "77fd947235b89e67549fa264a08dc1436d3913ba864d58da558d78e75d892cde"
    ),
    max_abs_logprob=0.3,
    rms_logprob=0.08,
    mean_forward_kl=2e-4,
    max_forward_kl=6e-4,
)

FRUIT_INSTRUCT_CALIBRATION_AUTHORITY = FruitCalibrationAuthority(
    variant="instruct",
    spec=FRUIT_INSTRUCT_SPEC,
    source_index_sha256=(
        "86e6cc1d8548c7bdbbc117e93b85b8ae249f446de9b48d2195e51f358674ba56"
    ),
    source_file_count=23,
    reference_sha256=(
        "e838645989a37e651e59f2388bb55d16f9b33b9a76b0352628abf2d4e667f414"
    ),
    capture_id="ddf1b740a6b0a12f0bc447a22467e64b19213ba793d2d4651e0bcb6c27e56d7b",
    fingerprint="4feb5b05c4e078f4eaaa02635b0e62daca4e260dfb3213877d495b670f0624dc",
    manifest_sha256=(
        "b9d38a5a373f79b87e0a79cfba9057283f6ce92ffa4a8f7372e135124609685b"
    ),
    max_abs_logprob=1.0,
    rms_logprob=0.2,
    mean_forward_kl=1e-3,
    max_forward_kl=5e-3,
)

FRUIT_CALIBRATION_AUTHORITIES: Mapping[str, FruitCalibrationAuthority] = (
    MappingProxyType(
        {
            "annealed": FRUIT_ANNEALED_CALIBRATION_AUTHORITY,
            "instruct": FRUIT_INSTRUCT_CALIBRATION_AUTHORITY,
        }
    )
)


def fruit_calibration_authority(variant: str) -> FruitCalibrationAuthority:
    """Resolve the pinned calibration contract for one Fruit variant."""

    try:
        return FRUIT_CALIBRATION_AUTHORITIES[variant]
    except KeyError as exc:
        supported = ", ".join(FRUIT_CALIBRATION_AUTHORITIES)
        raise ValueError(
            f"unsupported Fruit calibration variant {variant!r}; "
            f"expected one of {supported}"
        ) from exc


def _capture_fingerprint(manifest: dict[str, object]) -> str:
    identity = dict(manifest)
    for name in ("capture_id", "complete", "fingerprint", "layers"):
        identity.pop(name, None)
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def calibration_fingerprint(manifest: dict[str, object]) -> str:
    """Hash the complete root manifest except for its self-referential fingerprint."""

    payload = dict(manifest)
    payload.pop("fingerprint", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FruitCalibrationRows:
    """Routed rows for one expert and one whole-document split."""

    inputs: torch.Tensor
    gates: torch.Tensor
    document_ids: torch.Tensor
    requests: dict[int, str]

    def __post_init__(self) -> None:
        rows = int(self.inputs.shape[0]) if self.inputs.ndim == 2 else -1
        if (
            self.inputs.ndim != 2
            or self.inputs.shape[1] != FRUIT_ANNEALED_SPEC.hidden_size
        ):
            raise ValueError("Fruit calibration inputs have the wrong shape")
        if self.gates.ndim != 1 or self.document_ids.ndim != 1:
            raise ValueError("Fruit calibration route metadata must be one-dimensional")
        if self.gates.numel() != rows or self.document_ids.numel() != rows:
            raise ValueError(
                "Fruit calibration route metadata does not align with inputs"
            )
        if self.inputs.dtype != torch.bfloat16:
            raise TypeError("Fruit calibration inputs must use bfloat16")
        if self.gates.dtype != torch.float32:
            raise TypeError("Fruit calibration gates must use float32")
        if self.document_ids.dtype != torch.int64:
            raise TypeError("Fruit calibration document IDs must use int64")
        if self.inputs.device.type != "cpu" or self.gates.device.type != "cpu":
            raise ValueError("Fruit calibration rows must remain on CPU")
        if not self.inputs.is_contiguous() or not self.gates.is_contiguous():
            raise ValueError("Fruit calibration rows must be contiguous")
        if rows:
            if not bool(torch.all(torch.isfinite(self.inputs))):
                raise ValueError("Fruit calibration inputs contain non-finite values")
            if not bool(torch.all(torch.isfinite(self.gates))) or bool(
                torch.any(self.gates <= 0)
            ):
                raise ValueError("Fruit calibration gates must be finite and positive")
            if not set(map(int, self.document_ids.tolist())).issubset(self.requests):
                raise ValueError("Fruit calibration rows reference an unknown document")
        if not self.requests or any(
            type(step) is not int or not isinstance(document, str) or not document
            for step, document in self.requests.items()
        ):
            raise ValueError("Fruit calibration request map is invalid")

    @property
    def row_count(self) -> int:
        return int(self.inputs.shape[0])


@dataclass(frozen=True)
class FruitExpertCalibration:
    """Fit, confirmation, and untouched validation rows for one expert."""

    layer: int
    expert: int
    fingerprint: str
    global_h13: torch.Tensor
    fit: FruitCalibrationRows
    confirmation: FruitCalibrationRows
    validation: FruitCalibrationRows

    def __post_init__(self) -> None:
        if self.layer not in FRUIT_CALIBRATION_LAYERS:
            raise ValueError("Fruit calibration layer is unsupported")
        if not 0 <= self.expert < FRUIT_ANNEALED_SPEC.num_experts:
            raise ValueError("Fruit calibration expert is unsupported")
        if len(self.fingerprint) != 64:
            raise ValueError("Fruit calibration fingerprint is invalid")
        expected = (FRUIT_ANNEALED_SPEC.hidden_size,) * 2
        if (
            self.global_h13.dtype != torch.float32
            or tuple(self.global_h13.shape) != expected
        ):
            raise ValueError("Fruit calibration global H13 has the wrong contract")
        if self.global_h13.device.type != "cpu" or not self.global_h13.is_contiguous():
            raise ValueError(
                "Fruit calibration global H13 must be contiguous CPU storage"
            )


@dataclass(frozen=True)
class _Document:
    document_id: int
    axis: str
    source: str
    digest: str
    family_digest: str
    token_digest: str
    split: CalibrationSplit


class FruitCalibrationLayer:
    """One authenticated layer capture with expert-route selection."""

    def __init__(
        self,
        *,
        layer: int,
        fingerprint: str,
        documents: tuple[_Document, ...],
        inputs: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        document_ids: torch.Tensor,
        global_h13: torch.Tensor,
        routed_scale: float,
    ) -> None:
        rows = int(inputs.shape[0]) if inputs.ndim == 2 else -1
        expected_hidden = FRUIT_ANNEALED_SPEC.hidden_size
        expected_experts = FRUIT_ANNEALED_SPEC.num_experts
        if layer not in FRUIT_CALIBRATION_LAYERS:
            raise ValueError("Fruit calibration layer is unsupported")
        if inputs.dtype != torch.bfloat16 or tuple(inputs.shape) != (
            rows,
            expected_hidden,
        ):
            raise ValueError("Fruit calibration inputs violate the tensor contract")
        if route_ids.dtype != torch.int16 or tuple(route_ids.shape) != (
            rows,
            FRUIT_CALIBRATION_TOPK,
        ):
            raise ValueError("Fruit calibration route IDs violate the tensor contract")
        if (
            route_weights.dtype != torch.float32
            or route_weights.shape != route_ids.shape
        ):
            raise ValueError(
                "Fruit calibration route weights violate the tensor contract"
            )
        if document_ids.dtype != torch.int32 or tuple(document_ids.shape) != (rows,):
            raise ValueError(
                "Fruit calibration document IDs violate the tensor contract"
            )
        if global_h13.dtype != torch.float32 or tuple(global_h13.shape) != (
            expected_hidden,
            expected_hidden,
        ):
            raise ValueError(
                "Fruit calibration global H13 violates the tensor contract"
            )
        for tensor in (inputs, route_ids, route_weights, document_ids, global_h13):
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(
                    "Fruit calibration tensors must be contiguous CPU storage"
                )
        if rows <= 0:
            raise ValueError("Fruit calibration layer must contain routed rows")
        if not bool(torch.all(torch.isfinite(inputs))) or not bool(
            torch.all(torch.isfinite(route_weights))
        ):
            raise ValueError("Fruit calibration layer contains non-finite values")
        if bool(torch.any(route_weights <= 0)):
            raise ValueError("Fruit calibration route weights must be positive")
        if bool(torch.any(route_ids < 0)) or bool(
            torch.any(route_ids >= expected_experts)
        ):
            raise ValueError("Fruit calibration route IDs are out of range")
        ordered_routes = torch.sort(route_ids, dim=1).values
        if bool(torch.any(ordered_routes[:, 1:] == ordered_routes[:, :-1])):
            raise ValueError("Fruit calibration top-k routes are not unique")
        route_sums = route_weights.float().sum(dim=1)
        if not torch.allclose(
            route_sums,
            torch.full_like(route_sums, float(routed_scale)),
            rtol=2e-3,
            atol=2e-3,
        ):
            raise ValueError(
                "Fruit calibration route weights do not close routed_scale"
            )
        document_by_id = {document.document_id: document for document in documents}
        if set(map(int, torch.unique(document_ids).tolist())) != set(document_by_id):
            raise ValueError("Fruit calibration layer does not cover every document")
        if not bool(torch.all(torch.isfinite(global_h13))):
            raise ValueError("Fruit calibration global H13 contains non-finite values")
        if not torch.allclose(global_h13, global_h13.T, rtol=2e-5, atol=2e-6):
            raise ValueError("Fruit calibration global H13 is not symmetric")
        if bool(torch.any(torch.diagonal(global_h13) <= 0)):
            raise ValueError("Fruit calibration global H13 has a non-positive diagonal")

        self.layer = layer
        self.fingerprint = fingerprint
        self._documents = documents
        self._document_by_id = document_by_id
        self.inputs = inputs
        self.route_ids = route_ids
        self.route_weights = route_weights
        self.document_ids = document_ids
        self.global_h13 = global_h13

    def _rows(self, expert: int, split: CalibrationSplit) -> FruitCalibrationRows:
        matches = self.route_ids == int(expert)
        positions = torch.nonzero(matches, as_tuple=False)
        token_rows = (
            positions[:, 0] if positions.numel() else torch.empty(0, dtype=torch.long)
        )
        slots = (
            positions[:, 1] if positions.numel() else torch.empty(0, dtype=torch.long)
        )
        selected_documents = self.document_ids.index_select(0, token_rows).to(
            torch.int64
        )
        if selected_documents.numel():
            keep = torch.tensor(
                [
                    self._document_by_id[int(document)].split == split
                    for document in selected_documents.tolist()
                ],
                dtype=torch.bool,
            )
            token_rows = token_rows[keep]
            slots = slots[keep]
            selected_documents = selected_documents[keep]
        requests = {
            document.document_id: document.digest
            for document in self._documents
            if document.split == split
        }
        return FruitCalibrationRows(
            inputs=self.inputs.index_select(0, token_rows).contiguous(),
            gates=self.route_weights[token_rows, slots].float().contiguous(),
            document_ids=selected_documents.contiguous(),
            requests=requests,
        )

    def expert_rows(self, expert: int) -> FruitExpertCalibration:
        if type(expert) is not int or not 0 <= expert < FRUIT_ANNEALED_SPEC.num_experts:
            raise ValueError("Fruit calibration expert is out of range")
        return FruitExpertCalibration(
            layer=self.layer,
            expert=expert,
            fingerprint=self.fingerprint,
            global_h13=self.global_h13,
            fit=self._rows(expert, "fit"),
            confirmation=self._rows(expert, "confirmation"),
            validation=self._rows(expert, "validation"),
        )


class FruitCalibrationStore:
    """Fail-closed reader for a complete Fruit calibration capture."""

    def __init__(
        self,
        root: str | Path,
        *,
        authority: FruitCalibrationAuthority = FRUIT_ANNEALED_CALIBRATION_AUTHORITY,
    ) -> None:
        source_root = Path(root)
        (
            self._snapshot,
            self.root,
            manifest_bytes,
        ) = _snapshot_calibration_source(
            source_root,
            expected_manifest_sha256=authority.manifest_sha256,
        )
        actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if not isinstance(manifest, dict):
            raise TypeError("Fruit calibration manifest must be a JSON object")
        expected_root_keys = {
            "capture_id",
            "checkpoint_sha256",
            "complete",
            "conventions",
            "corpus",
            "documents",
            "fingerprint",
            "geometry",
            "kind",
            "layers",
            "protocol",
            "routed_scale",
            "schema",
            "source",
            "source_closure",
            "token_bounds",
            "tokenizer",
            "trainer",
            "version",
        }
        if set(manifest) != expected_root_keys:
            raise ValueError("Fruit calibration manifest field inventory mismatch")
        expected = {
            "schema": FRUIT_CALIBRATION_SCHEMA,
            "version": FRUIT_CALIBRATION_VERSION,
            "kind": "fruit_qsrt_activation_calibration",
            "complete": True,
            "checkpoint_sha256": authority.spec.checkpoint_sha256,
            "corpus": FRUIT_CALIBRATION_CORPUS,
            "protocol": FRUIT_CALIBRATION_PROTOCOL,
            "token_bounds": FRUIT_CALIBRATION_TOKEN_BOUNDS,
            "conventions": authority.conventions,
        }
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise ValueError(f"Fruit calibration manifest {name!r} mismatch")
        fingerprint = manifest.get("fingerprint")
        if (
            not _is_sha256(fingerprint)
            or fingerprint != calibration_fingerprint(manifest)
            or fingerprint != authority.fingerprint
        ):
            raise ValueError("Fruit calibration manifest fingerprint mismatch")
        capture_id = manifest.get("capture_id")
        if (
            not _is_sha256(capture_id)
            or capture_id != _capture_fingerprint(manifest)
            or capture_id != authority.capture_id
        ):
            raise ValueError("Fruit calibration capture ID is invalid")

        source = manifest.get("source")
        expected_source = authority.source
        if (
            not isinstance(source, dict)
            or set(source) != {*expected_source, "directory"}
            or not isinstance(source.get("directory"), str)
            or not source["directory"]
            or any(source.get(name) != value for name, value in expected_source.items())
        ):
            raise ValueError("Fruit calibration source identity mismatch")
        source_closure = manifest.get("source_closure")
        closure_fields = {
            "max_abs_logprob",
            "max_forward_kl",
            "mean_forward_kl",
            "mtp_finite",
            "mtp_positions",
            "positions",
            "reference_sha256",
            "rms_logprob",
            "top1_matches",
            "vocab_size",
        }
        if (
            not isinstance(source_closure, dict)
            or set(source_closure) != closure_fields
        ):
            raise ValueError("Fruit calibration source closure contract mismatch")
        metric_names = (
            "max_abs_logprob",
            "max_forward_kl",
            "mean_forward_kl",
            "rms_logprob",
        )
        if (
            source_closure.get("reference_sha256") != authority.reference_sha256
            or source_closure.get("positions") != 6
            or source_closure.get("vocab_size") != 154880
            or source_closure.get("top1_matches") != 6
            or source_closure.get("mtp_positions") != 6
            or source_closure.get("mtp_finite") is not True
            or any(
                isinstance(source_closure.get(name), bool)
                or not isinstance(source_closure.get(name), (int, float))
                or not math.isfinite(float(source_closure[name]))
                or float(source_closure[name]) < 0
                for name in metric_names
            )
            or any(
                float(source_closure[name]) > limit
                for name, limit in authority.closure_limits.items()
            )
        ):
            raise ValueError("Fruit calibration source closure failed")
        tokenizer = manifest.get("tokenizer")
        if (
            not isinstance(tokenizer, dict)
            or set(tokenizer) != {"directory", "files"}
            or not isinstance(tokenizer.get("directory"), str)
            or not tokenizer["directory"]
            or tokenizer.get("files") != FRUIT_CALIBRATION_TOKENIZER_FILES
        ):
            raise ValueError("Fruit calibration tokenizer identity mismatch")
        trainer = manifest.get("trainer")
        if trainer != {
            "filename": "train_fruit.py",
            "files": FRUIT_CALIBRATION_TRAINER_FILES,
        }:
            raise ValueError("Fruit calibration trainer identity mismatch")

        expected_geometry = {
            "hidden_size": authority.spec.hidden_size,
            "intermediate_size": authority.spec.intermediate_size,
            "experts": authority.spec.num_experts,
            "topk": FRUIT_CALIBRATION_TOPK,
            "layers": [*authority.spec.layers, authority.spec.mtp_layer],
        }
        if manifest.get("geometry") != expected_geometry:
            raise ValueError("Fruit calibration geometry mismatch")
        routed_scale = manifest.get("routed_scale")
        if (
            isinstance(routed_scale, bool)
            or not isinstance(routed_scale, (int, float))
            or not math.isfinite(float(routed_scale))
            or float(routed_scale) != 2.5
        ):
            raise ValueError("Fruit calibration routed_scale mismatch")

        raw_documents = manifest.get("documents")
        expected_document_count = (
            len(FRUIT_CALIBRATION_AXES) * FRUIT_CALIBRATION_DOCUMENTS_PER_AXIS
        )
        if (
            not isinstance(raw_documents, list)
            or len(raw_documents) != expected_document_count
        ):
            raise ValueError("Fruit calibration document ledger size mismatch")
        document_fields = {
            "axis",
            "family_sha256",
            "id",
            "line",
            "sha256",
            "source",
            "split",
            "token_ids_sha256",
            "tokens",
        }
        documents: list[_Document] = []
        for expected_id, value in enumerate(raw_documents):
            if (
                not isinstance(value, dict)
                or set(value) != document_fields
                or value.get("id") != expected_id
            ):
                raise ValueError("Fruit calibration document IDs are not canonical")
            axis = value.get("axis")
            source_name = value.get("source")
            digest = value.get("sha256")
            family_digest = value.get("family_sha256")
            token_digest = value.get("token_ids_sha256")
            split = value.get("split")
            line = value.get("line")
            tokens = value.get("tokens")
            if (
                axis not in FRUIT_CALIBRATION_AXES
                or not isinstance(source_name, str)
                or not source_name
                or not _is_sha256(digest)
                or not _is_sha256(family_digest)
                or not _is_sha256(token_digest)
                or split not in CALIBRATION_SPLITS
                or type(line) is not int
                or line <= 0
                or type(tokens) is not int
                or not (
                    FRUIT_CALIBRATION_TOKEN_BOUNDS["minimum"]
                    <= tokens
                    <= FRUIT_CALIBRATION_TOKEN_BOUNDS["maximum"]
                )
            ):
                raise ValueError("Fruit calibration document evidence is invalid")
            documents.append(
                _Document(
                    document_id=expected_id,
                    axis=axis,
                    source=source_name,
                    digest=digest,
                    family_digest=family_digest,
                    token_digest=token_digest,
                    split=split,
                )
            )
        for name, values in {
            "documents": [document.digest for document in documents],
            "prompt families": [document.family_digest for document in documents],
            "token sequences": [document.token_digest for document in documents],
        }.items():
            if len(set(values)) != len(values):
                raise ValueError(f"Fruit calibration {name} are not unique")
        expected_partition_counts = {
            (axis, split): count
            for axis in FRUIT_CALIBRATION_AXES
            for split, count in (
                ("fit", FRUIT_CALIBRATION_FIT_PER_AXIS),
                ("confirmation", FRUIT_CALIBRATION_CONFIRMATION_PER_AXIS),
                ("validation", FRUIT_CALIBRATION_VALIDATION_PER_AXIS),
            )
        }
        if Counter(
            (document.axis, document.split) for document in documents
        ) != Counter(expected_partition_counts):
            raise ValueError("Fruit calibration axis/split quotas mismatch")

        layers = manifest.get("layers")
        if not isinstance(layers, dict) or set(layers) != {
            str(layer) for layer in FRUIT_CALIBRATION_LAYERS
        }:
            raise ValueError("Fruit calibration layer ledger is incomplete")
        self.manifest = manifest
        self.manifest_sha256 = actual_manifest_sha256
        self.fingerprint = fingerprint
        self.capture_id = capture_id
        self.routed_scale = float(routed_scale)
        self.authority = authority
        self._documents = tuple(documents)
        self._layers = layers

    def load_layer(self, layer: int) -> FruitCalibrationLayer:
        if layer not in FRUIT_CALIBRATION_LAYERS:
            raise ValueError("Fruit calibration layer is unsupported")
        entry = self._layers[str(layer)]
        if not isinstance(entry, dict):
            raise TypeError("Fruit calibration layer ledger entry is invalid")
        filename = entry.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Fruit calibration layer filename is invalid")
        path = self.root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if entry.get("bytes") != path.stat().st_size:
            raise ValueError(f"Fruit calibration layer {layer} byte length mismatch")
        if entry.get("sha256") != _sha256(path):
            raise ValueError(f"Fruit calibration layer {layer} checksum mismatch")
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if set(handle.keys()) != set(_LAYER_TENSORS):
                raise ValueError(
                    f"Fruit calibration layer {layer} tensor inventory mismatch"
                )
            tensors = {name: handle.get_tensor(name) for name in _LAYER_TENSORS}
        expected_metadata = {
            "schema": FRUIT_CALIBRATION_SCHEMA,
            "version": str(FRUIT_CALIBRATION_VERSION),
            "capture_id": self.capture_id,
            "layer": str(layer),
        }
        if any(
            metadata.get(name) != value for name, value in expected_metadata.items()
        ):
            raise ValueError(f"Fruit calibration layer {layer} metadata mismatch")
        if entry.get("rows") != int(tensors["inputs"].shape[0]):
            raise ValueError(f"Fruit calibration layer {layer} row ledger mismatch")
        return FruitCalibrationLayer(
            layer=layer,
            fingerprint=self.fingerprint,
            documents=self._documents,
            inputs=tensors["inputs"],
            route_ids=tensors["route_ids"],
            route_weights=tensors["route_weights"],
            document_ids=tensors["document_ids"],
            global_h13=tensors["global_h13"],
            routed_scale=self.routed_scale,
        )

    def close(self) -> None:
        """Remove the process-private calibration snapshot."""

        self._snapshot.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = [
    "CALIBRATION_SPLITS",
    "FRUIT_ANNEALED_CALIBRATION_AUTHORITY",
    "FRUIT_CALIBRATION_AUTHORITIES",
    "FRUIT_CALIBRATION_LAYERS",
    "FRUIT_CALIBRATION_SCHEMA",
    "FRUIT_CALIBRATION_TOPK",
    "FRUIT_CALIBRATION_VERSION",
    "FRUIT_INSTRUCT_CALIBRATION_AUTHORITY",
    "FruitCalibrationAuthority",
    "FruitCalibrationLayer",
    "FruitCalibrationRows",
    "FruitCalibrationStore",
    "FruitExpertCalibration",
    "calibration_fingerprint",
    "fruit_calibration_authority",
]
