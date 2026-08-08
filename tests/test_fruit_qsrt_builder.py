from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.build_fruit_qsrt_model as builder
import scripts.encode_fruit_qsrt as encoder


def _commit_test_checkout(root: Path) -> str:
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(("git", "add", "."), cwd=root, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=KQuant Test",
            "-c",
            "user.email=kquant@example.invalid",
            "commit",
            "-qm",
            "test source",
        ),
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_current_encoder_provenance_requires_committed_source(
    monkeypatch, tmp_path: Path
) -> None:
    exllamav3_root = tmp_path / "exllamav3-root"
    package = exllamav3_root / "exllamav3"
    package.mkdir(parents=True)
    source = package / "codec.py"
    source.write_text("CODEBOOK = 1\n", encoding="utf-8")
    revision = _commit_test_checkout(exllamav3_root)
    monkeypatch.setattr(builder, "EXLLAMAV3_REVISION", revision)
    calibration = SimpleNamespace(
        fingerprint="calibration-fingerprint",
        capture_id="capture-id",
        manifest_sha256="manifest-sha256",
    )

    before = builder.current_encoder_provenance(
        exllamav3_root=exllamav3_root,
        calibration=calibration,
    )
    source.write_text("CODEBOOK = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="uncommitted"):
        builder.current_encoder_provenance(
            exllamav3_root=exllamav3_root,
            calibration=calibration,
        )
    assert before["exllamav3_revision"] == revision


def test_encoder_fingerprint_binds_source_revisions() -> None:
    encoder = {
        "kquant_revision": "1" * 40,
        "kquant_source_sha256": "2" * 64,
        "exllamav3_revision": "3" * 40,
        "exllamav3_source_sha256": "4" * 64,
        "calibration_fingerprint": "5" * 64,
        "calibration_capture_id": "6" * 64,
        "calibration_manifest_sha256": "7" * 64,
    }

    first = hashlib.sha256(
        builder._canonical_json(builder._encoder_fingerprint_payload(encoder)).encode(
            "utf-8"
        )
    ).hexdigest()
    encoder["kquant_revision"] = "8" * 40
    second = hashlib.sha256(
        builder._canonical_json(builder._encoder_fingerprint_payload(encoder)).encode(
            "utf-8"
        )
    ).hexdigest()

    assert first != second


def test_encoder_run_manifests_are_isolated_by_shard_and_assignment(
    tmp_path: Path,
) -> None:
    first = encoder._run_manifest_path(
        tmp_path,
        selection="all_assignments",
        shard_count=8,
        shard_index=0,
        assignments=((3, 0), (3, 8)),
    )
    second = encoder._run_manifest_path(
        tmp_path,
        selection="all_assignments",
        shard_count=8,
        shard_index=1,
        assignments=((3, 1), (3, 9)),
    )

    assert first.parent == tmp_path / "run-manifests"
    assert second.parent == first.parent
    assert first != second


def _rate_sweep(
    *,
    source: dict[str, object] | None = None,
    calibration: dict[str, str] | None = None,
    encoder: dict[str, object] | None = None,
) -> dict[str, object]:
    def endpoint(bits: int) -> dict[str, object]:
        error = float(5 - bits)
        matrices = [
            {
                "matrix": matrix,
                "weight_squared_error": error,
                "weight_reference_energy": 10.0,
                "captured_dense_h_numerator": error * 2,
                "captured_dense_h_denominator": 20.0,
            }
            for matrix in ("w1", "w3", "w2")
        ]
        return {
            "bits": bits,
            "aggregate": {
                "bytes_before_layer_deduplication": bits * 100,
                "bpw_before_layer_deduplication": bits + 0.046875,
            },
            "matrices": matrices,
            "routed_function": {
                "validation": {
                    "routed_sse": error * 3,
                    "reference_energy": 30.0,
                }
            },
        }

    return {
        "schema": builder._RATE_SWEEP_SCHEMA,
        "complete": True,
        "signature": {
            "schema": builder._RATE_SWEEP_SCHEMA,
            "source": source or {},
            "calibration": calibration or {},
            "encoder": encoder or {},
            "rates": [2, 3, 4],
            "assignments": [[3, 0]],
        },
        "results": {
            "3:0": {
                "status": "measured",
                "rates": {
                    "K2": endpoint(2),
                    "K3": endpoint(3),
                    "K4": endpoint(4),
                },
            }
        },
    }


def test_model_card_uses_sealed_calibration_and_layer_evidence(monkeypatch) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 2)
    calibration = SimpleNamespace(
        capture_id="capture-id",
        manifest_sha256="calibration-manifest",
        manifest={"documents": [{"tokens": 7}, {"tokens": 11}]},
    )
    layers = {
        "3": {
            "format_counts": {"R13=0,R2=0": 1, "R13=1,R2=0": 1},
            "experts": [
                {"encode_elapsed_seconds": 1.25, "peak_cuda_bytes": 1 << 30},
                {"encode_elapsed_seconds": 2.75, "peak_cuda_bytes": 2 << 30},
            ],
        }
    }
    producer = {
        "encoder": {"kquant_revision": "kquant-revision"},
        "runtime": {
            "b12x_revision": "b12x-revision",
            "vllm_revision": "vllm-revision",
        },
    }

    card = builder._render_model_card(
        source_evidence={
            "source_kind": "safetensors_manifest",
            "source_sha256": "source-digest",
            "source_repository": "owner/source",
            "source_revision": "1" * 40,
        },
        calibration=calibration,
        producer=producer,
        rate_sweep=_rate_sweep(),
        layers=layers,
    )

    assert "| `R13=0,R2=0` | 1 |" in card
    assert "| `R13=1,R2=0` | 1 |" in card
    assert "capture-id" in card
    assert "calibration-manifest" in card
    assert "https://huggingface.co/owner/source" in card
    assert "1" * 40 in card
    assert "2 documents /" in card and "18 tokens" in card
    assert "2 experts, 4.00 GPU-seconds" in card
    assert "2.000 GiB peak CUDA allocation" in card
    assert "Adjacent-rate evidence" in card
    assert "1 of 1 predeclared assignments" in card
    assert "__" not in card


def test_rate_sweep_validation_binds_build_provenance(tmp_path: Path) -> None:
    source = {"source_kind": "manifest", "source_sha256": "a" * 64}
    calibration_identity = {
        "capture_id": "capture-id",
        "fingerprint": "calibration-fingerprint",
        "manifest_sha256": "b" * 64,
    }
    calibration = SimpleNamespace(
        capture_id=calibration_identity["capture_id"],
        fingerprint=calibration_identity["fingerprint"],
        manifest_sha256=calibration_identity["manifest_sha256"],
    )
    encoder = {
        "fingerprint_schema": builder._ENCODER_FINGERPRINT_SCHEMA,
        "fingerprint": "c" * 64,
    }
    producer = {"encoder": encoder}
    payload = _rate_sweep(
        source=source,
        calibration=calibration_identity,
        encoder=encoder,
    )
    path = tmp_path / "rates.json"
    path.write_text(builder._canonical_json(payload), encoding="utf-8")

    assert (
        builder._validate_rate_sweep(
            path,
            source_evidence=source,
            calibration=calibration,
            producer=producer,
        )
        == payload
    )

    mixed = json.loads(builder._canonical_json(payload))
    mixed["signature"]["encoder"]["kquant_revision"] = "mixed"
    path.write_text(builder._canonical_json(mixed), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        builder._validate_rate_sweep(
            path,
            source_evidence=source,
            calibration=calibration,
            producer=producer,
        )
    path.write_text(builder._canonical_json(payload), encoding="utf-8")

    producer["encoder"] = {
        "fingerprint_schema": builder._ENCODER_FINGERPRINT_SCHEMA,
        "fingerprint": "d" * 64,
    }
    with pytest.raises(ValueError, match="provenance"):
        builder._validate_rate_sweep(
            path,
            source_evidence=source,
            calibration=calibration,
            producer=producer,
        )


def test_package_files_reject_resume_cache_before_sealing(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "report.json").write_text("{}", encoding="utf-8")
    parts = tmp_path / ".qsrt-parts/layer-003"
    parts.mkdir(parents=True)
    (parts / "expert-000.json").write_bytes(b"resume")

    with pytest.raises(ValueError, match="unexpected Fruit package directory"):
        builder._package_files(tmp_path)

    builder._remove_part_cache(tmp_path)
    files = builder._package_files(tmp_path)
    assert set(files) == {"config.json", "evaluation/report.json"}


def test_validated_package_files_reject_unknown_top_level_file(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "LAYERS", ())
    base_provenance = {"files": {"config.json": "a" * 64}}
    for relative in builder._expected_package_inventory(base_provenance):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed")
    (tmp_path / "model.safetensors").write_bytes(b"stale competing model")

    with pytest.raises(ValueError, match="unexpected=.*model.safetensors"):
        builder._validated_package_files(tmp_path, base_provenance)


def test_part_cache_rejects_nested_layer_symlink(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 1)
    cache = tmp_path / ".qsrt-parts"
    cache.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (cache / "layer-003").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="cache path"):
        builder._parts_need_encoder(
            tmp_path,
            source_sha256="a" * 64,
            encoder_fingerprint="b" * 64,
        )


def test_prepare_output_root_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "output"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="package root"):
        builder._prepare_output_root(output)


def test_staged_part_cache_restores_resume_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 1)
    output = tmp_path / "output"
    part = output / ".qsrt-parts" / "layer-003" / "expert-000.json"
    part.parent.mkdir(parents=True)
    part.write_text("{}", encoding="utf-8")

    staged = builder._stage_part_cache(output)
    assert staged is not None and staged.is_dir()
    assert not (output / ".qsrt-parts").exists()

    builder._restore_staged_part_cache(output, staged)
    assert part.read_text(encoding="utf-8") == "{}"


def test_remove_part_cache_preserves_package_files(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    parts = tmp_path / ".qsrt-parts" / "layer-003"
    parts.mkdir(parents=True)
    (parts / "expert-000.safetensors").write_bytes(b"resume")

    builder._remove_part_cache(tmp_path)

    assert not (tmp_path / ".qsrt-parts").exists()
    assert config.read_text(encoding="utf-8") == "{}"


def test_copy_authenticated_breaks_source_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    content = b"authenticated source"
    source.write_bytes(content)
    target.hardlink_to(source)

    builder._copy_authenticated(source, target, hashlib.sha256(content).hexdigest())
    source.write_bytes(b"mutated after copy")

    assert not target.samefile(source)
    assert target.read_bytes() == content


def test_copy_authenticated_breaks_third_party_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    external = tmp_path / "external.bin"
    target = tmp_path / "target.bin"
    content = b"authenticated source"
    source.write_bytes(content)
    external.write_bytes(content)
    target.hardlink_to(external)

    builder._copy_authenticated(source, target, hashlib.sha256(content).hexdigest())
    external.write_bytes(b"mutated external inode")

    assert not target.samefile(external)
    assert target.read_bytes() == content


def test_materialize_base_model_reauthenticates_index(tmp_path: Path) -> None:
    base_model = tmp_path / "base"
    output = tmp_path / "output"
    base_model.mkdir()
    output.mkdir()
    index_path = base_model / "model.safetensors.index.json"
    original = b'{"weight_map": {}}\n'
    index_path.write_bytes(original)
    provenance = {
        "files": {
            "model.safetensors.index.json": hashlib.sha256(original).hexdigest(),
        }
    }
    index_path.write_bytes(b'{"weight_map": {"changed": "shard.safetensors"}}\n')

    with pytest.raises(ValueError, match="base hash mismatch"):
        builder._materialize_base_model(base_model, output, provenance)


def test_seed_parts_validate_source_read_only_before_copy(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 1)
    seed = tmp_path / "seed"
    seed_layer = seed / "layer-003"
    seed_layer.mkdir(parents=True)
    source_tensor = seed_layer / "expert-000.safetensors"
    source_manifest = seed_layer / "expert-000.json"
    source_tensor.write_bytes(b"authenticated tensor")
    source_manifest.write_text('{"authenticated": true}', encoding="utf-8")
    calls = []

    def validate(tensor_path, manifest_path, **kwargs):
        calls.append((tensor_path, manifest_path, kwargs))
        return {
            "status": "valid",
            "safetensors_sha256": hashlib.sha256(
                source_tensor.read_bytes()
            ).hexdigest(),
        }

    monkeypatch.setattr(builder, "_validate_part", validate)
    output = tmp_path / "output"

    assert builder._seed_parts(output, seed, "a" * 64, "b" * 64) == 1
    target_tensor = output / ".qsrt-parts/layer-003/expert-000.safetensors"
    target_manifest = output / ".qsrt-parts/layer-003/expert-000.json"
    expected_kwargs = {
        "layer": 3,
        "expert": 0,
        "source_sha256": "a" * 64,
        "encoder_fingerprint": "b" * 64,
        "repair": False,
    }
    assert calls == [
        (source_tensor, source_manifest, expected_kwargs),
        (target_tensor, target_manifest, expected_kwargs),
    ]
    assert source_tensor.read_bytes() == b"authenticated tensor"
    assert source_manifest.read_text(encoding="utf-8") == '{"authenticated": true}'
    assert (
        output / ".qsrt-parts/layer-003/expert-000.safetensors"
    ).read_bytes() == b"authenticated tensor"
