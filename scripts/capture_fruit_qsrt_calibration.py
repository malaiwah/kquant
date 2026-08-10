#!/usr/bin/env python3
"""Capture document-disjoint routed activations for production Fruit QSRT."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import stat
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.torch_version import TorchVersion

from kquant.candidate_hessian import weighted_covariance
from kquant.fruit_calibration import (
    FRUIT_CALIBRATION_AUTHORITIES,
    FRUIT_CALIBRATION_AXES,
    FRUIT_CALIBRATION_CORPUS,
    FRUIT_CALIBRATION_LAYERS,
    FRUIT_CALIBRATION_PROTOCOL,
    FRUIT_CALIBRATION_SCHEMA,
    FRUIT_CALIBRATION_TOKEN_BOUNDS,
    FRUIT_CALIBRATION_TOKENIZER_FILES,
    FRUIT_CALIBRATION_TOPK,
    FRUIT_CALIBRATION_TRAINER_FILES,
    FRUIT_CALIBRATION_VERSION,
    FruitCalibrationAuthority,
    calibration_fingerprint,
    fruit_calibration_authority,
)
from kquant.fruit_source import FruitModelSpec


_BF16_BASE_INDEX = "model.safetensors.index.json"
_MODEL_MARKER_KEYS = frozenset({"rope_theta_trained", "serve_conv_v"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_safetensors(
    path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]
) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file(tensors, temporary, metadata=metadata)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _copy_pinned_input(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> str:
    try:
        resolved = source.resolve(strict=True)
        source_fd = os.open(
            resolved,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as exc:
        raise ValueError(f"cannot securely open calibration input: {source}") from exc
    destination_fd: int | None = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"calibration input is not a regular file: {source}")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o400,
        )
        digest = hashlib.sha256()
        bytes_read = 0
        while chunk := os.read(source_fd, 8 << 20):
            digest.update(chunk)
            bytes_read += len(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(destination_fd, chunk[offset:])
                if written <= 0:
                    raise OSError("short write while snapshotting calibration input")
                offset += written
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        actual_sha256 = digest.hexdigest()
        if bytes_read != before.st_size or _stable_file_identity(
            before
        ) != _stable_file_identity(after):
            raise ValueError(f"calibration input changed while copying: {source}")
        if actual_sha256 != expected_sha256:
            raise ValueError(f"calibration input identity mismatch: {source}")
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)
    if _sha256(destination) != expected_sha256:
        raise ValueError(f"private calibration snapshot is invalid: {source}")
    return actual_sha256


@dataclass
class _AuthenticatedCalibrationInputs:
    temporary_directory: tempfile.TemporaryDirectory[str]
    corpus: Path
    tokenizer: Path
    trainer: Path
    reference: Path
    tokenizer_hashes: dict[str, str]
    trainer_hashes: dict[str, str]

    def close(self) -> None:
        self.temporary_directory.cleanup()


def _snapshot_calibration_inputs(
    args: argparse.Namespace,
    authority: FruitCalibrationAuthority,
) -> _AuthenticatedCalibrationInputs:
    temporary_directory = tempfile.TemporaryDirectory(
        prefix="kquant-fruit-capture-inputs-"
    )
    root = Path(temporary_directory.name)
    corpus_root = root / "corpus"
    tokenizer_root = root / "tokenizer"
    trainer_root = root / "trainer"
    reference_root = root / "reference"
    for directory in (corpus_root, tokenizer_root, trainer_root, reference_root):
        directory.mkdir(mode=0o700)
    corpus = corpus_root / FRUIT_CALIBRATION_CORPUS["filename"]
    trainer = trainer_root / args.trainer.name
    reference = reference_root / args.reference.name
    try:
        _copy_pinned_input(
            args.corpus,
            corpus,
            expected_sha256=FRUIT_CALIBRATION_CORPUS["sha256"],
        )
        tokenizer_hashes = {}
        for name, expected_sha256 in FRUIT_CALIBRATION_TOKENIZER_FILES.items():
            tokenizer_hashes[name] = _copy_pinned_input(
                args.tokenizer / name,
                tokenizer_root / name,
                expected_sha256=expected_sha256,
            )
        if args.trainer.name not in FRUIT_CALIBRATION_TRAINER_FILES:
            raise ValueError("Fruit calibration trainer filename is not pinned")
        trainer_hashes = {}
        for name, expected_sha256 in sorted(FRUIT_CALIBRATION_TRAINER_FILES.items()):
            source = (
                args.trainer
                if name == args.trainer.name
                else args.trainer.parent / name
            )
            trainer_hashes[name] = _copy_pinned_input(
                source,
                trainer_root / name,
                expected_sha256=expected_sha256,
            )
        _copy_pinned_input(
            args.reference,
            reference,
            expected_sha256=authority.reference_sha256,
        )
    except BaseException:
        temporary_directory.cleanup()
        raise
    return _AuthenticatedCalibrationInputs(
        temporary_directory=temporary_directory,
        corpus=corpus,
        tokenizer=tokenizer_root,
        trainer=trainer,
        reference=reference,
        tokenizer_hashes=tokenizer_hashes,
        trainer_hashes=trainer_hashes,
    )


@dataclass(frozen=True)
class _CorpusCandidate:
    axis: str
    source: str
    text: str
    line: int
    digest: str
    family_digest: str
    token_ids: tuple[int, ...]
    token_digest: str


@dataclass(frozen=True)
class _CorpusDocument(_CorpusCandidate):
    split: str


@dataclass
class _LayerChunks:
    inputs: list[torch.Tensor] = field(default_factory=list)
    route_ids: list[torch.Tensor] = field(default_factory=list)
    route_weights: list[torch.Tensor] = field(default_factory=list)
    document_ids: list[torch.Tensor] = field(default_factory=list)

    def append(
        self,
        inputs: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        document_id: int,
    ) -> None:
        rows = int(inputs.shape[0])
        self.inputs.append(inputs.to(device="cpu", dtype=torch.bfloat16).contiguous())
        self.route_ids.append(
            route_ids.to(device="cpu", dtype=torch.int16).contiguous()
        )
        self.route_weights.append(
            route_weights.to(device="cpu", dtype=torch.float32).contiguous()
        )
        self.document_ids.append(
            torch.full((rows,), int(document_id), dtype=torch.int32)
        )

    def finish(self) -> dict[str, torch.Tensor]:
        if not self.inputs:
            raise ValueError("Fruit calibration layer captured no inputs")
        return {
            "inputs": torch.cat(self.inputs).contiguous(),
            "route_ids": torch.cat(self.route_ids).contiguous(),
            "route_weights": torch.cat(self.route_weights).contiguous(),
            "document_ids": torch.cat(self.document_ids).contiguous(),
        }


def _normalized_corpus_text(raw: str, source: str) -> tuple[str, str]:
    """Mirror Fruit training normalization and identify a prompt family."""

    try:
        envelope = json.loads(raw)
        messages = envelope.get("messages")
        if not isinstance(messages, list) or not messages:
            return raw, _sha256_bytes(f"{source}\0{raw}".encode())
        contents = [message.get("content", "") for message in messages]
        if any(not isinstance(content, str) for content in contents):
            return raw, _sha256_bytes(f"{source}\0{raw}".encode())
        normalized = "\n".join(contents)
        prompt = [
            {
                "role": message.get("role", ""),
                "content": content,
            }
            for message, content in zip(messages, contents, strict=True)
            if message.get("role") != "assistant"
        ]
        if not prompt:
            prompt = [
                {
                    "role": message.get("role", ""),
                    "content": content,
                }
                for message, content in zip(messages, contents, strict=True)
            ]
        family = json.dumps(
            {"source": source, "prompt": prompt},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return normalized, _sha256_bytes(family.encode("utf-8"))
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
        return raw, _sha256_bytes(f"{source}\0{raw}".encode())


def _token_digest(token_ids: tuple[int, ...]) -> str:
    values = torch.tensor(token_ids, dtype=torch.int32)
    return _sha256_bytes(values.numpy().tobytes())


def _sample_documents(
    corpus: Path,
    tokenizer: Any,
    *,
    documents_per_axis: int,
    fit_per_axis: int,
    confirmation_per_axis: int,
    min_tokens: int,
    max_tokens: int,
) -> tuple[list[_CorpusDocument], str]:
    if documents_per_axis <= 0:
        raise ValueError("documents_per_axis must be positive")
    if fit_per_axis < 1 or confirmation_per_axis < 1:
        raise ValueError("fit and confirmation quotas must be positive")
    if fit_per_axis + confirmation_per_axis >= documents_per_axis:
        raise ValueError("document quotas leave no untouched validation fold")
    if not 2 <= min_tokens <= max_tokens:
        raise ValueError("invalid calibration token bounds")

    corpus_sha256 = _sha256(corpus)
    by_axis: dict[str, list[_CorpusCandidate]] = defaultdict(list)
    document_axes: dict[str, str] = {}
    with corpus.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed calibration JSONL line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(f"calibration line {line_number} is not an object")
            axis = value.get("axis")
            source = value.get("source")
            raw = value.get("text")
            if not all(isinstance(item, str) and item for item in (axis, source, raw)):
                raise ValueError(
                    f"calibration line {line_number} lacks axis/source/text"
                )
            text, family_digest = _normalized_corpus_text(raw, source)
            if not text:
                continue
            digest = _sha256_bytes(text.encode("utf-8"))
            previous_axis = document_axes.get(digest)
            if previous_axis is not None:
                if previous_axis != axis:
                    raise ValueError("normalized calibration text spans multiple axes")
                continue
            document_axes[digest] = axis
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_tokens,
                return_attention_mask=False,
            )["input_ids"]
            token_ids = tuple(map(int, encoded))
            if len(token_ids) < min_tokens:
                continue
            by_axis[axis].append(
                _CorpusCandidate(
                    axis=axis,
                    source=source,
                    text=text,
                    line=line_number,
                    digest=digest,
                    family_digest=family_digest,
                    token_ids=token_ids,
                    token_digest=_token_digest(token_ids),
                )
            )
    if len(by_axis) < 4:
        raise ValueError("Fruit calibration corpus must cover at least four axes")

    selected: list[_CorpusDocument] = []
    selected_token_digests: set[str] = set()
    for axis, candidates in sorted(by_axis.items()):
        ordered = sorted(
            candidates,
            key=lambda item: _sha256_bytes(
                f"sample:{item.family_digest}:{item.digest}".encode("ascii")
            ),
        )
        eligible: list[_CorpusCandidate] = []
        selected_families: set[str] = set()
        for candidate in ordered:
            if (
                candidate.family_digest in selected_families
                or candidate.token_digest in selected_token_digests
            ):
                continue
            eligible.append(candidate)
            selected_families.add(candidate.family_digest)
            selected_token_digests.add(candidate.token_digest)
            if len(eligible) == documents_per_axis:
                break
        if len(eligible) != documents_per_axis:
            raise ValueError(
                f"axis {axis!r} has {len(eligible)} disjoint eligible documents, "
                f"expected {documents_per_axis}"
            )
        split_order = sorted(
            eligible,
            key=lambda item: _sha256_bytes(
                f"split:{item.family_digest}:{item.token_digest}".encode("ascii")
            ),
        )
        split_by_digest = {
            item.digest: (
                "fit"
                if index < fit_per_axis
                else "confirmation"
                if index < fit_per_axis + confirmation_per_axis
                else "validation"
            )
            for index, item in enumerate(split_order)
        }
        selected.extend(
            _CorpusDocument(
                axis=item.axis,
                source=item.source,
                text=item.text,
                line=item.line,
                digest=item.digest,
                family_digest=item.family_digest,
                token_ids=item.token_ids,
                token_digest=item.token_digest,
                split=split_by_digest[item.digest],
            )
            for item in sorted(eligible, key=lambda value: value.digest)
        )
    return selected, corpus_sha256


def _configure_trainer_environment(
    *,
    serve_native: bool,
    spec: FruitModelSpec,
) -> None:
    values = {
        "GEO_H": str(spec.hidden_size),
        "GEO_NL": str(spec.mtp_layer),
        "GEO_HEADS": "16",
        "GEO_QLORA": "1024",
        "GEO_DENSE_INTER": "2048",
        "GEO_MOE_INTER": str(spec.intermediate_size),
        "MOE_IMPL": "grouped",
        "ROPE_THETA": str(int(spec.trained_rope_theta)),
        "FRUIT_ROPE_THETA": str(int(spec.trained_rope_theta)),
        "SERVE_CONV": "1" if serve_native else "0",
        "GRAD_CKPT": "0",
        "FP8_LINEAR": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for name, value in values.items():
        previous = os.environ.get(name)
        if previous is not None and previous != value:
            raise ValueError(
                f"environment {name}={previous!r} conflicts with {value!r}"
            )
        os.environ[name] = value


def _load_trainer(
    trainer: Path,
    *,
    serve_native: bool,
    spec: FruitModelSpec,
):
    _configure_trainer_environment(serve_native=serve_native, spec=spec)
    trainer_root_path = trainer.parent.resolve()
    trainer_root = str(trainer_root_path)
    if trainer_root not in sys.path:
        sys.path.insert(0, trainer_root)
    for filename in FRUIT_CALIBRATION_TRAINER_FILES:
        sys.modules.pop(Path(filename).stem, None)
    module = importlib.import_module("train_fruit")
    if Path(module.__file__).resolve() != trainer.resolve():
        raise RuntimeError("imported the wrong Fruit trainer module")
    for filename in FRUIT_CALIBRATION_TRAINER_FILES:
        imported = sys.modules.get(Path(filename).stem)
        if imported is None:
            continue
        imported_path = getattr(imported, "__file__", None)
        if (
            not isinstance(imported_path, str)
            or Path(imported_path).resolve() != trainer_root_path / filename
        ):
            raise RuntimeError(
                f"imported the wrong Fruit trainer dependency: {filename}"
            )
    expected = {
        "H": spec.hidden_size,
        "NL": spec.mtp_layer,
        "N_EXP": spec.num_experts,
        "TOPK": FRUIT_CALIBRATION_TOPK,
        "MOE_INTER": spec.intermediate_size,
        "THETA": spec.trained_rope_theta,
    }
    for name, value in expected.items():
        if getattr(module, name) != value:
            raise ValueError(f"Fruit trainer geometry {name} mismatch")
    if bool(module.CONV["serve"]) != serve_native:
        raise ValueError("Fruit trainer serving convention mismatch")
    return module


@dataclass(frozen=True)
class _HFBinding:
    destination: str
    expert: int | None = None


def _hf_binding(name: str, spec: FruitModelSpec) -> _HFBinding:
    roots = {
        "lm_head.weight": _HFBinding("lm_head.weight"),
        "model.embed_tokens.weight": _HFBinding("embed_tokens.weight"),
        "model.norm.weight": _HFBinding("norm.weight"),
    }
    if name in roots:
        return roots[name]
    pieces = name.split(".")
    if len(pieces) < 5 or pieces[:2] != ["model", "layers"]:
        raise ValueError(f"unsupported Fruit BF16 tensor name: {name}")
    try:
        layer = int(pieces[2])
    except ValueError as exc:
        raise ValueError(f"invalid Fruit BF16 layer name: {name}") from exc
    if not 0 <= layer <= spec.mtp_layer:
        raise ValueError(f"Fruit BF16 layer is out of range: {name}")
    suffix = ".".join(pieces[3:])
    if layer == spec.mtp_layer:
        mtp_roots = {
            "eh_proj.weight": "mtp_eh_proj.weight",
            "enorm.weight": "mtp_enorm.weight",
            "hnorm.weight": "mtp_hnorm.weight",
            "shared_head.norm.weight": "norm.weight",
        }
        if suffix in mtp_roots:
            return _HFBinding(mtp_roots[suffix])
        prefix = "mtp_block."
    else:
        prefix = f"layers.{layer}."
    expert_prefix = "mlp.experts."
    if suffix.startswith(expert_prefix):
        expert_parts = suffix[len(expert_prefix) :].split(".")
        if len(expert_parts) != 3 or expert_parts[2] != "weight":
            raise ValueError(f"invalid Fruit BF16 expert tensor name: {name}")
        try:
            expert = int(expert_parts[0])
        except ValueError as exc:
            raise ValueError(f"invalid Fruit BF16 expert index: {name}") from exc
        if not 0 <= expert < spec.num_experts:
            raise ValueError(f"Fruit BF16 expert is out of range: {name}")
        projection = {
            "gate_proj": "w_gate",
            "up_proj": "w_up",
            "down_proj": "w_down",
        }.get(expert_parts[1])
        if projection is None:
            raise ValueError(f"invalid Fruit BF16 expert projection: {name}")
        return _HFBinding(f"{prefix}mlp.{projection}", expert)
    if suffix == "mlp.gate.e_score_correction_bias":
        suffix = "mlp.e_score_correction_bias"
    return _HFBinding(prefix + suffix)


def _stable_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_regular_at(directory_fd: int, name: str) -> tuple[int, tuple[int, ...]]:
    try:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ValueError(
            f"Fruit BF16 base file cannot be opened safely: {name}"
        ) from exc
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"Fruit BF16 base entry is not a regular file: {name}")
        return fd, _stable_file_identity(status)
    except BaseException:
        os.close(fd)
        raise


def _hash_open_file(
    fd: int,
    name: str,
    *,
    capture_bytes: bool,
) -> tuple[str, bytes | None, tuple[int, ...]]:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Fruit BF16 base entry is not a regular file: {name}")
    before_identity = _stable_file_identity(before)
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    captured = bytearray() if capture_bytes else None
    while chunk := os.read(fd, 1 << 20):
        digest.update(chunk)
        if captured is not None:
            captured.extend(chunk)
    after = os.fstat(fd)
    after_identity = _stable_file_identity(after)
    if after_identity != before_identity:
        raise ValueError(f"Fruit BF16 base file changed while authenticating: {name}")
    os.lseek(fd, 0, os.SEEK_SET)
    return (
        digest.hexdigest(),
        bytes(captured) if captured is not None else None,
        after_identity,
    )


@dataclass
class _AuthenticatedBf16Base:
    source_evidence: dict[str, object]
    weight_map: dict[str, str]
    _shard_fds: dict[str, int]
    _shard_identities: dict[str, tuple[int, ...]]

    def shard_path(self, name: str) -> str:
        self.assert_shard_stable(name)
        return f"/proc/self/fd/{self._shard_fds[name]}"

    def assert_shard_stable(
        self,
        name: str,
        *,
        expected_identity: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        if name not in self._shard_fds:
            raise ValueError(f"Fruit BF16 shard is not authenticated: {name}")
        try:
            status = os.fstat(self._shard_fds[name])
        except OSError as exc:
            raise RuntimeError(
                f"Fruit BF16 shard descriptor is closed: {name}"
            ) from exc
        identity = _stable_file_identity(status)
        authenticated_identity = self._shard_identities[name]
        if (
            not stat.S_ISREG(status.st_mode)
            or identity != authenticated_identity
            or (expected_identity is not None and identity != expected_identity)
        ):
            raise ValueError(f"Fruit BF16 shard changed after authentication: {name}")
        return identity

    def close(self) -> None:
        shard_fds, self._shard_fds = self._shard_fds, {}
        first_error: OSError | None = None
        for fd in shard_fds.values():
            try:
                os.close(fd)
            except OSError as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> _AuthenticatedBf16Base:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _authenticate_bf16_base(
    root: Path,
    authority: FruitCalibrationAuthority,
) -> _AuthenticatedBf16Base:
    if not sys.platform.startswith("linux") or not Path("/proc/self/fd").is_dir():
        raise RuntimeError(
            "Fruit BF16 calibration authentication requires Linux /proc/self/fd"
        )
    shard_fds: dict[str, int] = {}
    try:
        try:
            directory_fd = os.open(
                root,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
        except OSError as exc:
            raise ValueError("Fruit BF16 base must be a real directory") from exc
        try:
            directory_status = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_status.st_mode):
                raise ValueError("Fruit BF16 base must be a real directory")
            directory_identity = _stable_file_identity(directory_status)

            manifest_fd, _ = _open_regular_at(directory_fd, "MANIFEST.sha256")
            try:
                manifest_sha256, manifest_bytes, _ = _hash_open_file(
                    manifest_fd,
                    "MANIFEST.sha256",
                    capture_bytes=True,
                )
            finally:
                os.close(manifest_fd)
            assert manifest_bytes is not None
            if manifest_sha256 != authority.spec.safetensors_manifest_sha256:
                raise ValueError("Fruit BF16 base MANIFEST.sha256 identity mismatch")
            entries: dict[str, str] = {}
            try:
                manifest_text = manifest_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    "Fruit BF16 base MANIFEST.sha256 is not UTF-8"
                ) from exc
            for line in manifest_text.splitlines():
                fields = line.split()
                if (
                    len(fields) != 2
                    or len(fields[0]) != 64
                    or any(
                        character not in "0123456789abcdef" for character in fields[0]
                    )
                    or Path(fields[1]).name != fields[1]
                    or fields[1] in entries
                ):
                    raise ValueError("Fruit BF16 checksum manifest is malformed")
                entries[fields[1]] = fields[0]

            actual_names: set[str] = set()
            for name in os.listdir(directory_fd):
                if name == "MANIFEST.sha256":
                    continue
                try:
                    status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as exc:
                    raise ValueError(
                        "Fruit BF16 base changed while checking its inventory"
                    ) from exc
                if stat.S_ISREG(status.st_mode):
                    actual_names.add(name)
            unsealed_metadata = {"README.md", ".gitattributes"}
            if (
                not set(entries) <= actual_names
                or actual_names - set(entries) - unsealed_metadata
            ):
                raise ValueError(
                    "Fruit BF16 base file inventory differs from its manifest"
                )

            authenticated_bytes: dict[str, bytes] = {}
            shard_identities: dict[str, tuple[int, ...]] = {}
            for name, expected_sha256 in sorted(entries.items()):
                fd, _ = _open_regular_at(directory_fd, name)
                retain_fd = False
                try:
                    digest, content, identity = _hash_open_file(
                        fd,
                        name,
                        capture_bytes=name in {"config.json", _BF16_BASE_INDEX},
                    )
                    if digest != expected_sha256:
                        raise ValueError(f"Fruit BF16 base checksum mismatch: {name}")
                    if content is not None:
                        authenticated_bytes[name] = content
                    if name.endswith(".safetensors"):
                        shard_fds[name] = fd
                        shard_identities[name] = identity
                        retain_fd = True
                finally:
                    if not retain_fd:
                        os.close(fd)

            final_directory_status = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(final_directory_status.st_mode)
                or _stable_file_identity(final_directory_status) != directory_identity
            ):
                raise ValueError(
                    "Fruit BF16 base directory changed while authenticating"
                )
        finally:
            os.close(directory_fd)

        try:
            config = json.loads(authenticated_bytes["config.json"].decode("utf-8"))
            index = json.loads(authenticated_bytes[_BF16_BASE_INDEX].decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Fruit BF16 base config or weight index is malformed"
            ) from exc
        expected_config = {
            "dtype": "bfloat16",
            "hidden_size": authority.spec.hidden_size,
            "moe_intermediate_size": authority.spec.intermediate_size,
            "n_routed_experts": authority.spec.num_experts,
            "num_experts_per_tok": FRUIT_CALIBRATION_TOPK,
            "num_hidden_layers": authority.spec.mtp_layer,
            "num_nextn_predict_layers": 1,
            "rope_interleave": True,
            "rope_theta": authority.spec.trained_rope_theta,
            "tie_word_embeddings": False,
        }
        if any(config.get(name) != value for name, value in expected_config.items()):
            raise ValueError(
                "Fruit BF16 base config differs from the calibration contract"
            )
        if "quantization_config" in config:
            raise ValueError(
                "Fruit calibration source must be an unquantized BF16 export"
            )
        weight_map = index.get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(not isinstance(name, str) for name in weight_map)
            or any(not isinstance(filename, str) for filename in weight_map.values())
        ):
            raise ValueError("Fruit BF16 weight index is malformed")
        shard_names = {name for name in entries if name.endswith(".safetensors")}
        if set(weight_map.values()) != shard_names:
            raise ValueError("Fruit BF16 weight index shard inventory mismatch")
        source_evidence = {
            "kind": "authenticated_bf16_export",
            "directory": root.name,
            "manifest_sha256": manifest_sha256,
            "index_sha256": entries[_BF16_BASE_INDEX],
            "file_count": len(entries),
        }
        if any(
            source_evidence.get(name) != value
            for name, value in authority.source.items()
        ):
            raise ValueError("Fruit calibration source identity mismatch")
        authenticated = _AuthenticatedBf16Base(
            source_evidence=source_evidence,
            weight_map=weight_map,
            _shard_fds=shard_fds,
            _shard_identities=shard_identities,
        )
        shard_fds = {}
        return authenticated
    except BaseException:
        for fd in shard_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def _load_bf16_model(
    source_base: _AuthenticatedBf16Base,
    trainer: Any,
    device: torch.device,
    spec: FruitModelSpec,
) -> torch.nn.Module:
    model = trainer.Fruit()
    state = model.state_dict(keep_vars=True)
    bindings = {name: _hf_binding(name, spec) for name in source_base.weight_map}
    destination_keys = {binding.destination for binding in bindings.values()}
    expected_keys = set(state) - _MODEL_MARKER_KEYS
    if destination_keys != expected_keys:
        missing = sorted(expected_keys - destination_keys)
        extra = sorted(destination_keys - expected_keys)
        raise ValueError(
            f"Fruit BF16/trainer tensor inventory mismatch: missing={missing}, extra={extra}"
        )
    indexed_destinations: dict[str, set[int]] = defaultdict(set)
    for binding in bindings.values():
        if binding.expert is not None:
            indexed_destinations[binding.destination].add(binding.expert)
    expected_experts = set(range(spec.num_experts))
    if any(indices != expected_experts for indices in indexed_destinations.values()):
        raise ValueError("Fruit BF16 expert stack is incomplete")

    by_shard: dict[str, set[str]] = defaultdict(set)
    for name, filename in source_base.weight_map.items():
        by_shard[filename].add(name)
    loaded: set[tuple[str, int | None]] = set()
    with torch.no_grad():
        for filename, expected_names in sorted(by_shard.items()):
            path = source_base.shard_path(filename)
            load_identity = source_base.assert_shard_stable(filename)
            try:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    if set(handle.keys()) != expected_names:
                        raise ValueError(
                            f"Fruit BF16 shard tensor inventory mismatch: {filename}"
                        )
                    for name in sorted(expected_names):
                        source = handle.get_tensor(name)
                        binding = bindings[name]
                        destination = state[binding.destination]
                        if binding.expert is not None:
                            destination = destination[binding.expert]
                        if tuple(source.shape) != tuple(destination.shape):
                            raise ValueError(
                                f"Fruit BF16 tensor shape mismatch: {name}"
                            )
                        target_id = (binding.destination, binding.expert)
                        if target_id in loaded:
                            if not torch.equal(
                                source, destination.to(dtype=source.dtype)
                            ):
                                raise ValueError(
                                    f"Fruit BF16 duplicate tensor disagrees: {name}"
                                )
                        else:
                            destination.copy_(source)
                            loaded.add(target_id)
            finally:
                source_base.assert_shard_stable(
                    filename,
                    expected_identity=load_identity,
                )
    expected_loaded = {
        (binding.destination, binding.expert) for binding in bindings.values()
    }
    if loaded != expected_loaded:
        raise ValueError("Fruit BF16 model load did not cover every trainer tensor")
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _verify_source_closure(
    model: torch.nn.Module,
    reference_path: Path,
    *,
    device: torch.device,
    authority: FruitCalibrationAuthority,
) -> dict[str, object]:
    reference_sha256 = _sha256(reference_path)
    if reference_sha256 != authority.reference_sha256:
        raise ValueError("Fruit calibration closure reference identity mismatch")
    with torch.serialization.safe_globals([TorchVersion]):
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(reference, dict)
        or reference.get("kind") != "fruit_full_vocabulary_kld_reference"
        or reference.get("schema_version") != 1
        or not isinstance(reference.get("metadata"), dict)
        or reference["metadata"].get("checkpoint_sha256")
        != authority.spec.checkpoint_sha256
    ):
        raise ValueError("Fruit calibration closure reference contract mismatch")
    token_ids = reference.get("token_ids")
    expected = reference.get("log_probs")
    positions = reference.get("positions")
    vocab_size = reference.get("vocab_size")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(token_id) is not int for token_id in token_ids)
        or type(positions) is not int
        or type(vocab_size) is not int
        or not isinstance(expected, torch.Tensor)
        or expected.dtype != torch.float32
        or tuple(expected.shape) != (positions, vocab_size)
        or positions != len(token_ids) - 1
    ):
        raise ValueError("Fruit calibration closure reference payload mismatch")
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    torch.use_deterministic_algorithms(True)
    with (
        torch.inference_mode(),
        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
    ):
        hidden, mtp_hidden = model(inputs)
        logits = model.lm_head(hidden[0, :-1]).float()
        actual = torch.log_softmax(logits, dim=-1).cpu()
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError("Fruit calibration source closure output shape mismatch")
    teacher = expected.double()
    teacher -= torch.logsumexp(teacher, dim=1, keepdim=True)
    candidate = actual.double()
    candidate -= torch.logsumexp(candidate, dim=1, keepdim=True)
    delta = candidate - teacher
    forward_kl = (teacher.exp() * -delta).sum(dim=1)
    if not bool(torch.all(torch.isfinite(delta))) or bool(
        torch.any(forward_kl < -1e-8)
    ):
        raise ValueError("Fruit calibration source closure produced invalid metrics")
    metrics: dict[str, object] = {
        "reference_sha256": reference_sha256,
        "positions": positions,
        "vocab_size": vocab_size,
        "max_abs_logprob": float(delta.abs().max()),
        "rms_logprob": float(delta.square().mean().sqrt()),
        "mean_forward_kl": float(forward_kl.clamp_min(0).mean()),
        "max_forward_kl": float(forward_kl.clamp_min(0).max()),
        "top1_matches": int(
            torch.count_nonzero(teacher.argmax(dim=1) == candidate.argmax(dim=1))
        ),
        "mtp_positions": int(mtp_hidden.shape[1]),
        "mtp_finite": bool(torch.all(torch.isfinite(mtp_hidden))),
    }
    if (
        metrics["top1_matches"] != positions
        or any(
            float(metrics[name]) > limit
            for name, limit in authority.closure_limits.items()
        )
        or metrics["mtp_positions"] != positions
        or not metrics["mtp_finite"]
    ):
        raise ValueError(f"Fruit calibration source closure failed: {metrics}")
    return metrics


def _capture(
    model: torch.nn.Module,
    trainer: Any,
    documents: list[_CorpusDocument],
    *,
    device: torch.device,
    spec: FruitModelSpec,
) -> dict[int, dict[str, torch.Tensor]]:
    chunks = {layer: _LayerChunks() for layer in FRUIT_CALIBRATION_LAYERS}
    current_document = {"id": -1}
    handles: list[Any] = []

    def install(layer: int, module: torch.nn.Module) -> None:
        def pre_hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            if current_document["id"] < 0 or len(args) != 1:
                raise RuntimeError("Fruit calibration hook has no active document")
            inputs = args[0].detach().reshape(-1, trainer.H)
            logits = module.gate(inputs).float()
            scores = torch.sigmoid(logits)
            route_ids = (
                (scores + module.e_score_correction_bias)
                .topk(trainer.TOPK, dim=-1)
                .indices
            )
            route_weights = scores.gather(-1, route_ids)
            route_weights = (
                route_weights / route_weights.sum(dim=-1, keepdim=True)
            ) * trainer.ROUTED_SCALE
            chunks[layer].append(
                inputs,
                route_ids,
                route_weights,
                current_document["id"],
            )

        handles.append(module.register_forward_pre_hook(pre_hook))

    for layer in spec.layers:
        install(layer, model.layers[layer].mlp)
    install(spec.mtp_layer, model.mtp_block.mlp)

    torch.use_deterministic_algorithms(True)
    try:
        with (
            torch.inference_mode(),
            torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
        ):
            for document_id, document in enumerate(documents):
                current_document["id"] = document_id
                inputs = torch.tensor(
                    [document.token_ids], dtype=torch.long, device=device
                )
                output = model(inputs)
                del inputs, output
                print(
                    f"[{document_id + 1}/{len(documents)}] "
                    f"{document.axis} {document.split} {len(document.token_ids)} tokens",
                    flush=True,
                )
    finally:
        current_document["id"] = -1
        for handle in handles:
            handle.remove()
    return {layer: value.finish() for layer, value in chunks.items()}


def _capture_identity(
    *,
    authority: FruitCalibrationAuthority,
    source: dict[str, object],
    source_closure: dict[str, object],
    documents_per_axis: int,
    fit_per_axis: int,
    confirmation_per_axis: int,
    corpus: Path,
    corpus_sha256: str,
    tokenizer: Path,
    tokenizer_hashes: dict[str, str],
    trainer: Path,
    trainer_hashes: dict[str, str],
    documents: list[dict[str, object]],
    routed_scale: float,
    max_tokens: int,
    min_tokens: int,
) -> dict[str, object]:
    return {
        "schema": FRUIT_CALIBRATION_SCHEMA,
        "version": FRUIT_CALIBRATION_VERSION,
        "kind": "fruit_qsrt_activation_calibration",
        "checkpoint_sha256": authority.spec.checkpoint_sha256,
        "source": source,
        "source_closure": source_closure,
        "corpus": {
            "filename": corpus.name,
            "sha256": corpus_sha256,
        },
        "tokenizer": {
            "directory": tokenizer.name,
            "files": tokenizer_hashes,
        },
        "trainer": {
            "filename": trainer.name,
            "files": trainer_hashes,
        },
        "geometry": {
            "hidden_size": authority.spec.hidden_size,
            "intermediate_size": authority.spec.intermediate_size,
            "experts": authority.spec.num_experts,
            "topk": FRUIT_CALIBRATION_TOPK,
            "layers": [*authority.spec.layers, authority.spec.mtp_layer],
        },
        "routed_scale": routed_scale,
        "conventions": authority.conventions,
        "token_bounds": {"minimum": min_tokens, "maximum": max_tokens},
        "protocol": {
            "normalization": "fruit_data_prep_calib_jsonl_v1",
            "partition": "prompt_family_and_truncated_token_sha256_v1",
            "documents_per_axis": documents_per_axis,
            "fit_per_axis": fit_per_axis,
            "confirmation_per_axis": confirmation_per_axis,
            "validation_per_axis": (
                documents_per_axis - fit_per_axis - confirmation_per_axis
            ),
        },
        "documents": documents,
    }


def _load_tokenizer(root: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Fruit calibration tokenizer support is not installed; install KQuant "
            "with `kquant[fruit-calibration]` (for a checkout: "
            "`pip install -e '.[fruit-calibration]'`)."
        ) from exc
    return AutoTokenizer.from_pretrained(
        root,
        trust_remote_code=False,
        local_files_only=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("trainer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--variant",
        choices=tuple(FRUIT_CALIBRATION_AUTHORITIES),
        default="annealed",
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--documents-per-axis", type=int, default=64)
    parser.add_argument("--fit-per-axis", type=int, default=40)
    parser.add_argument("--confirmation-per-axis", type=int, default=16)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authority = fruit_calibration_authority(args.variant)
    for path in (args.corpus, args.reference, args.trainer):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.source.is_dir() or args.source.is_symlink():
        raise ValueError(
            "Fruit calibration source must be the authenticated BF16 directory"
        )
    if not args.tokenizer.is_dir():
        raise FileNotFoundError(args.tokenizer)
    requested_protocol = {
        "normalization": "fruit_data_prep_calib_jsonl_v1",
        "partition": "prompt_family_and_truncated_token_sha256_v1",
        "documents_per_axis": args.documents_per_axis,
        "fit_per_axis": args.fit_per_axis,
        "confirmation_per_axis": args.confirmation_per_axis,
        "validation_per_axis": (
            args.documents_per_axis - args.fit_per_axis - args.confirmation_per_axis
        ),
    }
    if (
        requested_protocol != FRUIT_CALIBRATION_PROTOCOL
        or {
            "minimum": args.min_tokens,
            "maximum": args.max_tokens,
        }
        != FRUIT_CALIBRATION_TOKEN_BOUNDS
    ):
        raise ValueError("capture arguments differ from the production protocol")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("calibration output directory must be absent or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Fruit calibration capture requires CUDA")
    torch.cuda.set_device(device)

    source_evidence: dict[str, object]
    input_snapshot = _snapshot_calibration_inputs(args, authority)
    tokenizer_hashes = input_snapshot.tokenizer_hashes
    trainer_hashes = input_snapshot.trainer_hashes

    tokenizer = _load_tokenizer(input_snapshot.tokenizer)
    documents, corpus_sha256 = _sample_documents(
        input_snapshot.corpus,
        tokenizer,
        documents_per_axis=args.documents_per_axis,
        fit_per_axis=args.fit_per_axis,
        confirmation_per_axis=args.confirmation_per_axis,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    if (
        corpus_sha256 != FRUIT_CALIBRATION_CORPUS["sha256"]
        or tuple(sorted({document.axis for document in documents}))
        != FRUIT_CALIBRATION_AXES
    ):
        raise ValueError("Fruit calibration corpus identity mismatch")
    document_ledger = [
        {
            "id": document_id,
            "axis": document.axis,
            "source": document.source,
            "line": document.line,
            "sha256": document.digest,
            "family_sha256": document.family_digest,
            "split": document.split,
            "tokens": len(document.token_ids),
            "token_ids_sha256": document.token_digest,
        }
        for document_id, document in enumerate(documents)
    ]
    split_counts = Counter(document.split for document in documents)
    axis_counts = Counter(document.axis for document in documents)
    print(
        "Fruit calibration plan "
        + json.dumps(
            {
                "documents": len(documents),
                "splits": dict(sorted(split_counts.items())),
                "axes": dict(sorted(axis_counts.items())),
                "tokens": sum(len(document.token_ids) for document in documents),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    trainer = _load_trainer(
        input_snapshot.trainer,
        serve_native=True,
        spec=authority.spec,
    )
    source_base = _authenticate_bf16_base(args.source, authority)
    source_evidence = source_base.source_evidence
    try:
        model = _load_bf16_model(
            source_base,
            trainer,
            device,
            authority.spec,
        )
    finally:
        source_base.close()
    source_closure = _verify_source_closure(
        model,
        input_snapshot.reference,
        device=device,
        authority=authority,
    )
    captured = _capture(
        model,
        trainer,
        documents,
        device=device,
        spec=authority.spec,
    )
    routed_scale = float(trainer.ROUTED_SCALE)
    del model
    torch.cuda.empty_cache()

    identity = _capture_identity(
        authority=authority,
        source=source_evidence,
        source_closure=source_closure,
        documents_per_axis=args.documents_per_axis,
        fit_per_axis=args.fit_per_axis,
        confirmation_per_axis=args.confirmation_per_axis,
        corpus=args.corpus,
        corpus_sha256=corpus_sha256,
        tokenizer=args.tokenizer,
        tokenizer_hashes=tokenizer_hashes,
        trainer=args.trainer,
        trainer_hashes=trainer_hashes,
        documents=document_ledger,
        routed_scale=routed_scale,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )
    capture_id = _sha256_bytes(_canonical_json(identity).encode("utf-8"))
    fit_ids = torch.tensor(
        [
            document_id
            for document_id, document in enumerate(documents)
            if document.split == "fit"
        ],
        dtype=torch.int32,
    )
    layers: dict[str, dict[str, object]] = {}
    for layer in FRUIT_CALIBRATION_LAYERS:
        tensors = captured.pop(layer)
        fit_mask = torch.isin(tensors["document_ids"], fit_ids)
        if not bool(torch.any(fit_mask)):
            raise ValueError(f"Fruit calibration layer {layer} has no fit rows")
        fit_inputs = tensors["inputs"][fit_mask]
        fit_route_weights = tensors["route_weights"][fit_mask].float()
        h13_weights = fit_route_weights.square().sum(dim=1)
        global_h13, denominator = weighted_covariance(
            fit_inputs,
            h13_weights,
            device=device,
            chunk_rows=512,
            return_cpu=True,
        )
        tensors["global_h13"] = global_h13.contiguous()
        filename = f"layer-{layer:03d}.safetensors"
        path = args.output / filename
        _atomic_safetensors(
            path,
            tensors,
            {
                "schema": FRUIT_CALIBRATION_SCHEMA,
                "version": str(FRUIT_CALIBRATION_VERSION),
                "capture_id": capture_id,
                "layer": str(layer),
            },
        )
        layers[str(layer)] = {
            "file": filename,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "rows": int(tensors["inputs"].shape[0]),
            "fit_h13_gate_square_sum": denominator,
        }
        print(
            f"layer {layer}: {tensors['inputs'].shape[0]} rows, "
            f"H13 weight={denominator:.6g}, {path.stat().st_size / (1 << 20):.2f} MiB",
            flush=True,
        )

    manifest: dict[str, object] = {
        **identity,
        "capture_id": capture_id,
        "layers": layers,
        "complete": True,
    }
    manifest["fingerprint"] = calibration_fingerprint(manifest)
    _atomic_text(
        args.output / "calibration-manifest.json",
        _canonical_json(manifest),
    )
    input_snapshot.close()
    print(
        f"complete Fruit calibration capture: {args.output} "
        f"fingerprint={manifest['fingerprint']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
