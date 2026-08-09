from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.build_fruit_qsrt_model as builder
import scripts.encode_fruit_qsrt as encoder
from scripts.tracked_worktree import snapshot_tracked_worktree


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


def test_tracked_worktree_sha256_is_deterministic_for_clean_checkout(
    tmp_path: Path,
) -> None:
    source = tmp_path / "package" / "codec.py"
    source.parent.mkdir()
    source.write_text("CODEBOOK = 1\n", encoding="utf-8")
    executable = tmp_path / "build.sh"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    _commit_test_checkout(tmp_path)

    first = builder.tracked_worktree_sha256(tmp_path)
    second = builder.tracked_worktree_sha256(tmp_path)

    assert first == second
    assert len(first) == 64
    assert all(character in "0123456789abcdef" for character in first)


def test_tracked_worktree_snapshot_is_private_and_immutable_from_original(
    tmp_path: Path,
) -> None:
    source = tmp_path / "package" / "codec.py"
    source.parent.mkdir()
    source.write_text("CODEBOOK = 1\n", encoding="utf-8")
    revision = _commit_test_checkout(tmp_path)

    snapshot = snapshot_tracked_worktree(tmp_path)
    source.write_text("CODEBOOK = 2\n", encoding="utf-8")
    snapshot_root = snapshot.root

    assert snapshot.revision == revision
    assert (snapshot.root / "package" / "codec.py").read_text(encoding="utf-8") == (
        "CODEBOOK = 1\n"
    )
    snapshot.close()
    assert not snapshot_root.exists()


def test_tracked_worktree_snapshot_rejects_intermediate_symlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "package" / "codec.py"
    source.parent.mkdir()
    source.write_text("CODEBOOK = 1\n", encoding="utf-8")
    _commit_test_checkout(tmp_path)
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.mkdir()
    (external / "codec.py").write_text("CODEBOOK = 1\n", encoding="utf-8")
    source.unlink()
    source.parent.rmdir()
    source.parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="securely open"):
        snapshot_tracked_worktree(tmp_path, require_clean=False)


def test_tracked_worktree_sha256_rejects_hidden_assume_unchanged_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "package" / "codec.py"
    source.parent.mkdir()
    source.write_text("CODEBOOK = 1\n", encoding="utf-8")
    _commit_test_checkout(tmp_path)
    subprocess.run(
        ("git", "update-index", "--assume-unchanged", "--", "package/codec.py"),
        cwd=tmp_path,
        check=True,
    )
    source.write_text("CODEBOOK = 2\n", encoding="utf-8")
    status = subprocess.run(
        ("git", "status", "--porcelain=v1"),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    with pytest.raises(ValueError, match="assume-unchanged"):
        builder.tracked_worktree_sha256(tmp_path)


def test_current_encoder_provenance_uses_process_lifetime_source_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    kquant_root = tmp_path / "kquant-root"
    kquant_source = kquant_root / "kquant" / "codec.py"
    kquant_source.parent.mkdir(parents=True)
    kquant_source.write_text("CODEBOOK = 1\n", encoding="utf-8")
    _commit_test_checkout(kquant_root)
    monkeypatch.setattr(builder, "_KQUANT_IMPORT_IDENTITY", None)
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
        kquant_root=kquant_root,
    )
    source.write_text("CODEBOOK = 2\n", encoding="utf-8")
    after = builder.current_encoder_provenance(
        exllamav3_root=exllamav3_root,
        calibration=calibration,
        kquant_root=kquant_root,
    )

    assert after == before
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


def test_model_card_uses_sealed_calibration_and_layer_evidence(
    monkeypatch, tmp_path: Path
) -> None:
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
    receipt_path, receipt_output, _, receipt_producer, receipt_source = (
        _runtime_qualification_fixture(tmp_path)
    )
    runtime_qualification = _validate_runtime_fixture(
        receipt_path,
        receipt_output,
        receipt_producer,
        receipt_source,
    )

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
        publication=builder.fruit_publication_spec("instruct"),
        runtime_qualification=runtime_qualification,
    )

    assert "| `R13=0,R2=0` | 1 |" in card
    assert "| `R13=1,R2=0` | 1 |" in card
    assert "capture-id" in card
    assert "calibration-manifest" in card
    assert "https://huggingface.co/owner/source" in card
    assert "hf download malaiwah/GLM-5.2-QSRT-Fruit-Instruct" in card
    assert "1" * 40 in card
    assert "2 documents /" in card and "18 tokens" in card
    assert "2 experts, 4.00 GPU-seconds" in card
    assert "2.000 GiB peak CUDA allocation" in card
    assert "Adjacent-rate evidence" in card
    assert "1 of 1 predeclared assignments" in card
    assert "The sealed receipt" in card
    assert 'sha256sum -- "${MODEL_DIR}/QSRT_COMPLETE.json"' not in card
    assert (
        '--env FRUIT_QSRT_EXPECTED_COMPLETE_SHA256="${FRUIT_QSRT_EXPECTED_COMPLETE_SHA256}"'
        in card
    )
    assert card.index("set an independently supplied completion digest") < card.index(
        "hf download malaiwah/GLM-5.2-QSRT-Fruit-Instruct"
    )
    assert "__" not in card


def test_instruct_publication_uses_variant_specific_repositories() -> None:
    publication = builder.fruit_publication_spec("instruct")
    assert publication.repository == "malaiwah/GLM-5.2-QSRT-Fruit-Instruct"
    assert "Fruit Instruct BF16" in publication.fruit_audit_rows
    assert "GLM-5.2-SIQ-Fruit-Instruct/tree/48452ef3" in publication.fruit_audit_rows
    assert "GLM-5.2-QSRT-Fruit/tree/c1a0c62d" not in publication.fruit_audit_rows
    assert "Not assistant-quality" in publication.quality_limitations
    assert "four-prompt" not in publication.quality_limitations
    assert "63.54" not in publication.quality_limitations


def test_only_instruct_is_a_production_publication() -> None:
    assert tuple(builder.FRUIT_PUBLICATIONS) == ("instruct",)
    with pytest.raises(ValueError, match="unsupported"):
        builder.fruit_publication_spec("annealed")


def test_generated_recipe_never_self_derives_completion_anchor() -> None:
    recipe = builder.MODEL_CARD_TEMPLATE
    assert 'sha256sum -- "${MODEL_DIR}/QSRT_COMPLETE.json"' not in recipe
    prerequisite = recipe.index(
        "${FRUIT_QSRT_EXPECTED_COMPLETE_SHA256:?set an independently supplied completion digest}"
    )
    download = recipe.index("hf download __MODEL_REPOSITORY__")
    assert prerequisite < download


def _runtime_qualification_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, object], dict[str, object]]:
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "config.json").write_text("{}\n", encoding="utf-8")
    (output / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    (output / "model.safetensors").write_bytes(b"candidate tensors")
    revisions = {
        "vllm_revision": "1" * 40,
        "b12x_revision": "2" * 40,
        "kquant_revision": "3" * 40,
    }
    producer = {
        "schema": "kquant_fruit_qsrt_producer_v1",
        "encoder": {
            "fingerprint": "a" * 64,
            "kquant_revision": revisions["kquant_revision"],
        },
        "runtime": {
            "fingerprint": "b" * 64,
            "vllm_revision": revisions["vllm_revision"],
            "b12x_revision": revisions["b12x_revision"],
        },
        "fingerprint": "c" * 64,
    }
    source = {
        "source_kind": "safetensors_manifest",
        "source_sha256": "d" * 64,
        "source_repository": "owner/instruct-source",
        "source_revision": "e" * 40,
    }

    def loader(arm: str) -> dict[str, object]:
        argv = [
            "vllm",
            "serve",
            "/model",
            "--tensor-parallel-size",
            "1",
            "--pipeline-parallel-size=1",
            "--max-num-seqs",
            "1",
            "--max-model-len",
            "4096",
            "--max-num-batched-tokens",
            "4096",
            "--compilation-config",
            json.dumps(builder._FIXED_COMPILATION_CONFIG, separators=(",", ":")),
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":1}',
            "--attention-backend",
            "B12X_MLA_SPARSE",
            "--moe-backend",
            "b12x",
            "--kv-cache-dtype",
            "nvfp4_ds_mla",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--quantization",
            "kquant_hybrid",
            "--load-format",
            "fastsafetensors",
        ]
        return {
            "runtime": {
                "image": f"registry.invalid/{arm}@sha256:{'f' * 64}",
                **revisions,
                "argv": argv,
                "environment": {"CUDA_VISIBLE_DEVICES": "0"},
                "software": {"cuda": "13.2.1", "torch": "2.12.0"},
                "compilation_backend": "inductor",
                "cudagraph_mode": "FULL_AND_PIECEWISE",
            },
            "log_line": f"{arm} parsed loader statistics",
            "weight_bytes": 1 << 30,
            "peak_activation_bytes": 2 << 30,
            "non_torch_bytes": 3 << 30,
            "cudagraph_bytes": 4 << 20,
            "kv_cache_bytes": 5 << 20,
            "torch_allocated_bytes": 6 << 20,
            "torch_reserved_bytes": 7 << 20,
            "nvml_used_bytes": 8 << 20,
            "load_seconds": 4.0,
        }

    def runs(rate: float) -> list[dict[str, object]]:
        return [
            {
                "prompt_id": "decode-prompt",
                "repetition": repetition,
                "http_status": 200,
                "elapsed_seconds": 8.0 / rate,
                "completion_tokens": 8,
                "tokens_per_second": rate,
                "finish_reason": "length",
                "content": f"completion {repetition}",
            }
            for repetition in (1, 2, 3)
        ]

    def fidelity(mean: float) -> dict[str, object]:
        first = mean / 2
        second = mean * 1.5
        return {
            "mean_forward_kl": mean,
            "max_forward_kl": second,
            "top1_agreement": 0.5,
            "top10_agreement": 1.0,
            "per_position": [
                {
                    "position": 0,
                    "forward_kl": first,
                    "top1_agreement": True,
                    "top10_agreement": True,
                },
                {
                    "position": 1,
                    "forward_kl": second,
                    "top1_agreement": False,
                    "top10_agreement": True,
                },
            ],
        }

    prompts = [
        {"id": "generation-1", "prompt": "First prompt", "prompt_token_ids": [1, 2]},
        {"id": "generation-2", "prompt": "Second prompt", "prompt_token_ids": [3, 4]},
    ]
    tensors = {"model.safetensors": builder._sha256(output / "model.safetensors")}
    tensor_set_sha256 = hashlib.sha256(
        builder._canonical_json(tensors).encode("utf-8")
    ).hexdigest()
    index_sha256 = builder._sha256(output / "model.safetensors.index.json")
    config_sha256 = builder._sha256(output / "config.json")
    (output / builder._CANDIDATE_MARKER_NAME).write_text(
        builder._canonical_json({"schema": "kquant_qsrt_candidate_v1"}),
        encoding="utf-8",
    )
    candidate_marker_sha256 = builder._sha256(output / builder._CANDIDATE_MARKER_NAME)

    def model(arm: str) -> dict[str, object]:
        if arm in builder._INSTRUCT_COMPARATOR_MODELS:
            return dict(builder._INSTRUCT_COMPARATOR_MODELS[arm])
        return {
            "repository": builder.fruit_publication_spec("instruct").repository,
            "revision": "4" * 40,
            "manifest_sha256": "5" * 64,
            "config_sha256": config_sha256,
            "model_index_sha256": index_sha256,
            "safetensors_bytes": (output / "model.safetensors").stat().st_size,
            "safetensors_sha256": tensor_set_sha256,
        }

    runtime_layers = {
        str(layer): {
            "prefill": {"mode": "w4a16", "calls": 1},
            "decode": {
                "mode": "w4a8",
                "calls": 2,
                "part_count": 2,
                "capture_calls": 1,
                "replay_calls": 1,
            },
        }
        for layer in range(3, 13)
    }
    runtime_layers["13"] = {
        "mtp_decode": {
            "mode": "w4a8",
            "calls": 2,
            "part_count": 2,
            "capture_calls": 1,
            "replay_calls": 1,
        }
    }

    payload: dict[str, object] = {
        "schema": builder._RUNTIME_QUALIFICATION_SCHEMA,
        "version": 1,
        "complete": True,
        "publication": {
            "variant": "instruct",
            "repository": builder.fruit_publication_spec("instruct").repository,
        },
        "producer": producer,
        "source": source,
        "candidate": {
            "marker_sha256": candidate_marker_sha256,
            "model_index_sha256": index_sha256,
            "safetensors_sha256": tensors,
        },
        "models": {arm: model(arm) for arm in ("bf16", "siq", "qsrt")},
        "environment": {
            "gpu_model": "RTX test GPU",
            "gpu_driver": "999.0",
            "host": "test host",
        },
        "protocol": {
            "tensor_parallel_size": 1,
            "max_num_seqs": 1,
            "max_tokens": 8,
            "temperature": 0.0,
            "repetitions": 3,
            "prompt_id": "decode-prompt",
            "prompt": "Matched decode prompt",
            "prompt_token_ids": [7, 8, 9],
            "launch_order": ["bf16", "siq", "qsrt"],
        },
        "loaders": {arm: loader(arm) for arm in ("bf16", "siq", "qsrt")},
        "runtime_paths": {
            "schema": builder._RUNTIME_PATHS_SCHEMA,
            "version": 1,
            "layers": runtime_layers,
            "cudagraph": {
                "mode": "FULL_AND_PIECEWISE",
                "capture_count": 1,
                "replay_count": 1,
            },
            "speculative": {
                "method": "mtp",
                "num_speculative_tokens": 1,
                "draft_tokens": 1,
            },
        },
        "decode": {
            "bf16": runs(8.0),
            "siq": runs(6.0),
            "qsrt": runs(7.0),
        },
        "generation": {
            "prompts": prompts,
            "results": {
                arm: [
                    {"prompt_id": prompt["id"], "content": f"{arm} output"}
                    for prompt in prompts
                ]
                for arm in ("bf16", "siq", "qsrt")
            },
        },
        "fidelity": {
            "full_vocabulary": True,
            "positions": [0, 1],
            "vocab_size": 128,
            "candidates": {"siq": fidelity(0.2), "qsrt": fidelity(0.1)},
        },
    }
    path = tmp_path / "runtime-qualification.json"
    path.write_text(builder._canonical_json(payload), encoding="utf-8")
    return path, output, payload, producer, source


def _validate_runtime_fixture(
    path: Path,
    output: Path,
    producer: dict[str, object],
    source: dict[str, object],
) -> dict[str, object]:
    return builder._validate_runtime_qualification(
        path,
        output=output,
        variant="instruct",
        publication=builder.fruit_publication_spec("instruct"),
        producer=producer,
        source_evidence=source,
    )


def test_runtime_qualification_validates_seals_and_renders(tmp_path: Path) -> None:
    path, output, payload, producer, source = _runtime_qualification_fixture(tmp_path)

    validated = _validate_runtime_fixture(path, output, producer, source)
    receipt = output / builder._RUNTIME_QUALIFICATION_NAME
    receipt.parent.mkdir()
    receipt.write_text(builder._canonical_json(validated), encoding="utf-8")
    builder._validate_sealed_runtime_qualification(output, validated)
    section = builder._runtime_qualification_section(validated)

    assert "client-observed end-to-end generated-token rate" in section
    assert "legacy SIQ comparator" in section
    assert (
        "| QSRT | 7.00 | 0.875x | 1.000 | `inductor` / `FULL_AND_PIECEWISE` |"
        in section
    )
    assert "not identical software" in section
    assert builder._RUNTIME_QUALIFICATION_NAME in (
        builder._expected_package_inventory({"files": {}})
    )
    assert payload == validated


def test_runtime_qualification_copy_and_manifests(monkeypatch, tmp_path: Path) -> None:
    path, output, _, producer, source = _runtime_qualification_fixture(tmp_path)
    validated = _validate_runtime_fixture(path, output, producer, source)
    monkeypatch.setattr(
        builder,
        "_render_model_card",
        lambda **_arguments: "sealed card\n",
    )
    monkeypatch.setattr(
        builder,
        "_validated_package_files",
        lambda root, _base, **_arguments: builder._package_files(
            root, allow_part_cache=True
        ),
    )

    builder._write_package_manifests(
        output,
        source_evidence=source,
        base_provenance={"files": {}},
        producer=producer,
        calibration=SimpleNamespace(),
        rate_sweep={"sealed": True},
        layers={},
        publication=builder.fruit_publication_spec("instruct"),
        runtime_qualification=validated,
    )

    receipt = output / builder._RUNTIME_QUALIFICATION_NAME
    assert receipt.read_text(encoding="utf-8") == builder._canonical_json(validated)
    manifest = json.loads((output / "qsrt-manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluation"]["runtime_qualification"] == {
        "file": builder._RUNTIME_QUALIFICATION_NAME,
        "sha256": builder._sha256(receipt),
    }
    assert (
        f"{builder._sha256(receipt)}  {builder._RUNTIME_QUALIFICATION_NAME}"
        in (output / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    )


def test_candidate_seal_precedes_and_binds_runtime_qualification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "LAYERS", ())
    output = tmp_path / "candidate"
    output.mkdir()
    base = {
        "manifest_sha256": "a" * 64,
        "files": {
            "config.json": "ignored-by-focused-test",
            "model.safetensors.index.json": "ignored-by-focused-test",
            "model.safetensors": "ignored-by-focused-test",
        },
    }
    for name, content in (
        ("config.json", "{}\n"),
        ("model.safetensors.index.json", "{}\n"),
        (builder._SOURCE_EVIDENCE_NAME, "{}\n"),
        (builder._SOURCE_EVIDENCE_SHA_NAME, "seal\n"),
        (builder._CALIBRATION_EVIDENCE_NAME, "{}\n"),
    ):
        (output / name).write_text(content, encoding="utf-8")
    (output / "model.safetensors").write_bytes(b"weights")
    producer = {
        "fingerprint": "b" * 64,
        "encoder": {"fingerprint": "c" * 64},
    }
    source = {
        "source_kind": "safetensors_manifest",
        "source_sha256": "d" * 64,
    }
    publication = builder.fruit_publication_spec("instruct")

    builder._write_candidate_package(
        output,
        source_evidence=source,
        base_provenance=base,
        producer=producer,
        rate_sweep={"sealed": True},
        layers={},
        publication=publication,
    )
    builder._validate_candidate_package(
        output,
        source_evidence=source,
        base_provenance=base,
        producer=producer,
        rate_sweep={"sealed": True},
        layers={},
        publication=publication,
        allow_part_cache=False,
    )
    assert not (output / builder._COMPLETE_MARKER_NAME).exists()
    marker_digest = builder._sha256(output / builder._CANDIDATE_MARKER_NAME)

    receipt = {
        "candidate": {"marker_sha256": marker_digest},
    }
    qualification_path = output / builder._RUNTIME_QUALIFICATION_NAME
    qualification_path.write_text(builder._canonical_json(receipt), encoding="utf-8")
    complete = builder._completion_record(
        output,
        source_evidence=source,
        base_provenance=base,
        producer=producer,
        runtime_qualification=receipt,
    )
    assert complete["schema"] == "kquant_qsrt_complete_v3"
    assert complete["qualified_candidate_sha256"] == marker_digest
    qualification_path.unlink()

    (output / "model.safetensors").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="hash mismatch"):
        builder._validate_candidate_package(
            output,
            source_evidence=source,
            base_provenance=base,
            producer=producer,
            rate_sweep={"sealed": True},
            layers={},
            publication=publication,
            allow_part_cache=False,
        )


def test_runtime_qualification_requires_canonical_receipt(tmp_path: Path) -> None:
    path, output, payload, producer, source = _runtime_qualification_fixture(tmp_path)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical"):
        _validate_runtime_fixture(path, output, producer, source)


def test_runtime_qualification_argument_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_fruit_qsrt_model.py",
            "base",
            "output",
            "--exllamav3-root",
            "exllamav3",
            "--b12x-root",
            "b12x",
            "--vllm-root",
            "vllm",
            "--calibration",
            "calibration",
            "--rate-sweep",
            "rate-sweep",
        ],
    )

    with pytest.raises(SystemExit):
        builder.parse_args()


def test_runtime_qualification_rejects_stale_build_identities(tmp_path: Path) -> None:
    path, output, payload, producer, source = _runtime_qualification_fixture(tmp_path)
    mutations = (
        lambda value: value["producer"].update({"fingerprint": "0" * 64}),
        lambda value: value["source"].update({"source_sha256": "0" * 64}),
        lambda value: value["publication"].update({"repository": "owner/stale"}),
        lambda value: value["candidate"].update({"model_index_sha256": "0" * 64}),
        lambda value: value["candidate"]["safetensors_sha256"].update(
            {"model.safetensors": "0" * 64}
        ),
        lambda value: value["candidate"].pop("marker_sha256"),
        lambda value: value["models"]["siq"].pop("manifest_sha256"),
        lambda value: value["models"]["bf16"].update(
            {"model_index_sha256": "not-a-digest"}
        ),
        lambda value: value["models"]["bf16"].update({"manifest_sha256": "0" * 64}),
        lambda value: value["loaders"]["qsrt"]["runtime"].update(
            {"vllm_revision": "9" * 40}
        ),
    )
    for mutate in mutations:
        malformed = json.loads(builder._canonical_json(payload))
        mutate(malformed)
        path.write_text(builder._canonical_json(malformed), encoding="utf-8")
        with pytest.raises((TypeError, ValueError)):
            _validate_runtime_fixture(path, output, producer, source)


def test_runtime_qualification_rejects_argv_contradictions_and_omissions(
    tmp_path: Path,
) -> None:
    path, output, payload, producer, source = _runtime_qualification_fixture(tmp_path)
    malformed_values = []

    contradictory = json.loads(builder._canonical_json(payload))
    contradictory["loaders"]["bf16"]["runtime"]["argv"].extend(
        ["--tensor-parallel-size", "2"]
    )
    malformed_values.append(contradictory)

    misleading_eager = json.loads(builder._canonical_json(payload))
    misleading_eager["loaders"]["siq"]["runtime"]["argv"].append("--enforce-eager")
    malformed_values.append(misleading_eager)

    for flag in ("--speculative-config", "--compilation-config"):
        omitted = json.loads(builder._canonical_json(payload))
        argv = omitted["loaders"]["qsrt"]["runtime"]["argv"]
        option_index = argv.index(flag)
        del argv[option_index : option_index + 2]
        malformed_values.append(omitted)

    wrong_backend = json.loads(builder._canonical_json(payload))
    argv = wrong_backend["loaders"]["qsrt"]["runtime"]["argv"]
    argv[argv.index("--moe-backend") + 1] = "torch"
    malformed_values.append(wrong_backend)

    forged_graph = json.loads(builder._canonical_json(payload))

    for arm in builder._RUNTIME_ARMS:
        for flag in (
            "--kv-cache-dtype",
            "--attention-backend",
            "--moe-backend",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
        ):
            omitted = json.loads(builder._canonical_json(payload))
            argv = omitted["loaders"][arm]["runtime"]["argv"]
            option_index = argv.index(flag)
            del argv[
                option_index : option_index + (1 if flag.startswith("--enable-") else 2)
            ]
            malformed_values.append(omitted)

    mutated_custom_ops = json.loads(builder._canonical_json(payload))
    config = dict(builder._FIXED_COMPILATION_CONFIG)
    config["custom_ops"] = []
    mutated_custom_ops["loaders"]["bf16"]["runtime"]["argv"][
        mutated_custom_ops["loaders"]["bf16"]["runtime"]["argv"].index(
            "--compilation-config"
        )
        + 1
    ] = json.dumps(config, separators=(",", ":"))
    malformed_values.append(mutated_custom_ops)

    mutated_capture_sizes = json.loads(builder._canonical_json(payload))
    config = dict(builder._FIXED_COMPILATION_CONFIG)
    config["cudagraph_capture_sizes"] = [1, 2, 4]
    mutated_capture_sizes["loaders"]["siq"]["runtime"]["argv"][
        mutated_capture_sizes["loaders"]["siq"]["runtime"]["argv"].index(
            "--compilation-config"
        )
        + 1
    ] = json.dumps(config, separators=(",", ":"))
    malformed_values.append(mutated_capture_sizes)

    eager_arm = json.loads(builder._canonical_json(payload))
    eager_arm["loaders"]["bf16"]["runtime"]["compilation_backend"] = "eager"
    malformed_values.append(eager_arm)

    no_graph_arm = json.loads(builder._canonical_json(payload))
    no_graph_arm["loaders"]["siq"]["runtime"]["cudagraph_mode"] = "NONE"
    malformed_values.append(no_graph_arm)
    argv = forged_graph["loaders"]["qsrt"]["runtime"]["argv"]
    argv[argv.index("--compilation-config") + 1] = (
        '{"backend":"inductor","cudagraph_mode":"NONE"}'
    )
    malformed_values.append(forged_graph)

    disabled_compilation = json.loads(builder._canonical_json(payload))
    argv = disabled_compilation["loaders"]["qsrt"]["runtime"]["argv"]
    argv[argv.index("--compilation-config") + 1] = (
        '{"backend":"inductor","cudagraph_mode":"FULL_AND_PIECEWISE","mode":0}'
    )
    malformed_values.append(disabled_compilation)

    for malformed in malformed_values:
        path.write_text(builder._canonical_json(malformed), encoding="utf-8")
        with pytest.raises(ValueError):
            _validate_runtime_fixture(path, output, producer, source)


def test_runtime_qualification_rejects_incomplete_runtime_paths(
    tmp_path: Path,
) -> None:
    path, output, payload, producer, source = _runtime_qualification_fixture(tmp_path)
    malformed_values = []

    missing_layer = json.loads(builder._canonical_json(payload))
    missing_layer["runtime_paths"]["layers"].pop("12")
    malformed_values.append(missing_layer)

    wrong_mode = json.loads(builder._canonical_json(payload))
    wrong_mode["runtime_paths"]["layers"]["3"]["prefill"]["mode"] = "w4a8"
    malformed_values.append(wrong_mode)

    wrong_parts = json.loads(builder._canonical_json(payload))
    wrong_parts["runtime_paths"]["layers"]["7"]["decode"]["part_count"] = 1
    malformed_values.append(wrong_parts)

    no_replay = json.loads(builder._canonical_json(payload))
    no_replay["runtime_paths"]["layers"]["9"]["decode"]["replay_calls"] = 0
    malformed_values.append(no_replay)

    no_mtp_capture = json.loads(builder._canonical_json(payload))
    no_mtp_capture["runtime_paths"]["layers"]["13"]["mtp_decode"]["capture_calls"] = 0
    malformed_values.append(no_mtp_capture)

    no_graph = json.loads(builder._canonical_json(payload))
    no_graph["runtime_paths"]["cudagraph"]["capture_count"] = 0
    malformed_values.append(no_graph)

    no_drafts = json.loads(builder._canonical_json(payload))
    no_drafts["runtime_paths"]["speculative"]["draft_tokens"] = 0
    malformed_values.append(no_drafts)

    extra = json.loads(builder._canonical_json(payload))
    extra["runtime_paths"]["layers"]["3"]["decode"]["unsealed"] = True
    malformed_values.append(extra)

    for malformed in malformed_values:
        path.write_text(builder._canonical_json(malformed), encoding="utf-8")
        with pytest.raises(ValueError):
            _validate_runtime_fixture(path, output, producer, source)


def test_runtime_qualification_rejects_run_coverage_and_fidelity_errors(
    tmp_path: Path,
) -> None:
    path, output, payload, producer, source = _runtime_qualification_fixture(tmp_path)
    malformed_values = []
    bad_math = json.loads(builder._canonical_json(payload))
    bad_math["decode"]["qsrt"][0]["tokens_per_second"] = 999.0
    malformed_values.append(bad_math)
    bad_coverage = json.loads(builder._canonical_json(payload))
    bad_coverage["generation"]["results"]["siq"].pop()
    malformed_values.append(bad_coverage)
    bad_fidelity = json.loads(builder._canonical_json(payload))
    bad_fidelity["fidelity"]["candidates"]["qsrt"]["mean_forward_kl"] = 0.9
    malformed_values.append(bad_fidelity)
    unknown = json.loads(builder._canonical_json(payload))
    unknown["summary"] = {}
    malformed_values.append(unknown)

    for malformed in malformed_values:
        path.write_text(builder._canonical_json(malformed), encoding="utf-8")
        with pytest.raises(ValueError):
            _validate_runtime_fixture(path, output, producer, source)


def test_runtime_qualification_rejects_post_copy_mutation(tmp_path: Path) -> None:
    path, output, _payload, producer, source = _runtime_qualification_fixture(tmp_path)
    validated = _validate_runtime_fixture(path, output, producer, source)
    receipt = output / builder._RUNTIME_QUALIFICATION_NAME
    receipt.parent.mkdir()
    receipt.write_text(
        builder._canonical_json(validated) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed"):
        builder._validate_sealed_runtime_qualification(output, validated)


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

    with pytest.raises(ValueError, match=r"unexpected=.*model\.safetensors"):
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


def test_encoder_cache_allows_only_private_active_temporaries(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 1)
    cache = tmp_path / ".qsrt-parts"
    layer = cache / "layer-003"
    run_manifests = cache / "run-manifests"
    layer.mkdir(parents=True)
    run_manifests.mkdir()
    assert builder._is_encoder_temporary(layer / ".tmp-disappeared")
    safetensors_temporary = layer / ".tmpXWW6j5"
    safetensors_temporary.write_bytes(b"active")
    (layer / ".expert-000.json.tmp-123").write_bytes(b"active")
    (run_manifests / ".run-0000.json.tmp-123").write_bytes(b"active")

    builder._validate_part_cache_root(
        cache,
        create=False,
        allow_run_manifests=True,
    )

    safetensors_temporary.unlink()
    external = tmp_path / "external"
    external.write_bytes(b"external")
    safetensors_temporary.symlink_to(external)
    with pytest.raises(ValueError, match="part path"):
        builder._validate_part_cache_root(
            cache,
            create=False,
            allow_run_manifests=True,
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


def test_seed_parts_rejects_manifest_format_that_disagrees_with_tensor(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 1)
    monkeypatch.setattr(builder, "HIDDEN_SIZE", 2)
    monkeypatch.setattr(builder, "INTERMEDIATE_SIZE", 2)
    monkeypatch.setattr(builder, "PAIR_COUNT", 2)
    monkeypatch.setattr(builder, "PAIR_WORDS", 1)
    seed = tmp_path / "seed"
    seed_layer = seed / "layer-003"
    seed_layer.mkdir(parents=True)
    tensor_path = seed_layer / "expert-000.safetensors"
    manifest_path = seed_layer / "expert-000.json"
    torch = builder.torch
    tensors = {
        "expert_ids": torch.tensor([0], dtype=torch.int32),
        "formats": torch.tensor([[1, 2]], dtype=torch.int8),
        "permutations": torch.tensor([[0, 1]], dtype=torch.int16),
        "w13_trellis": torch.zeros((2, 1, 2, 1), dtype=torch.int16),
        "w2_trellis": torch.zeros((1, 2, 1), dtype=torch.int16),
        "fc1_pair_modes": torch.tensor([[1, 0]], dtype=torch.int32),
        "fc2_pair_modes": torch.tensor([[1, 1]], dtype=torch.int32),
        "gate_suh": torch.ones((1, 2), dtype=torch.float16),
        "up_suh": torch.ones((1, 2), dtype=torch.float16),
        "intermediate_rotations": torch.ones((1, 6), dtype=torch.float16),
        "down_svh": torch.ones((1, 2), dtype=torch.float16),
    }
    source_sha256 = "a" * 64
    encoder_fingerprint = "b" * 64
    builder._atomic_safetensors(
        tensor_path,
        tensors,
        {
            "schema": builder.FRUIT_QSRT_SCHEMA,
            "version": "1",
            "profile_id": str(builder.FRUIT_QSRT_PROFILE_ID),
            "codebook": builder.FRUIT_QSRT_CODEBOOK,
            "layer": "3",
            "expert": "0",
            "source_sha256": source_sha256,
            "encoder_fingerprint": encoder_fingerprint,
        },
    )
    manifest = {
        "schema": builder.FRUIT_QSRT_SCHEMA,
        "version": 1,
        "profile_id": builder.FRUIT_QSRT_PROFILE_ID,
        "codebook": builder.FRUIT_QSRT_CODEBOOK,
        "layer": 3,
        "expert": 0,
        "format": {"r13": 0, "r2": 0},
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
        "safetensors_bytes": tensor_path.stat().st_size,
        "safetensors_sha256": builder._sha256(tensor_path),
    }
    manifest_path.write_text(builder._canonical_json(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="formats disagree with manifest"):
        builder._validate_part(
            tensor_path,
            manifest_path,
            layer=3,
            expert=0,
            source_sha256=source_sha256,
            encoder_fingerprint=encoder_fingerprint,
            repair=False,
        )

    output = tmp_path / "output"
    assert (
        builder._seed_parts(
            output,
            seed,
            source_sha256,
            encoder_fingerprint,
        )
        == 0
    )
    assert not (output / ".qsrt-parts/layer-003/expert-000.safetensors").exists()


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
