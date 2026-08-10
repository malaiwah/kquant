from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest

import scripts.build_fruit_qsrt_model as builder
import scripts.encode_fruit_qsrt as encoder
from kquant import fruit_calibration as calibration_module
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


def test_tracked_worktree_disables_repository_fsmonitor_hook(
    tmp_path: Path,
) -> None:
    source = tmp_path / "package" / "codec.py"
    source.parent.mkdir()
    source.write_text("CODEBOOK = 1\n", encoding="utf-8")
    _commit_test_checkout(tmp_path)
    marker = tmp_path.parent / f"{tmp_path.name}-fsmonitor-executed"
    hook = tmp_path.parent / f"{tmp_path.name}-fsmonitor-hook"
    hook.write_text(
        f"#!/bin/sh\n/usr/bin/touch -- {marker!s}\nexit 0\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    subprocess.run(
        ("git", "config", "core.fsmonitor", str(hook)),
        cwd=tmp_path,
        check=True,
    )

    digest = builder.tracked_worktree_sha256(tmp_path)

    assert len(digest) == 64
    assert not marker.exists()


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


def test_calibration_snapshot_binds_layer_bytes_before_loading(tmp_path: Path) -> None:
    source = tmp_path / "calibration"
    source.mkdir()
    layer_bytes = b"authenticated calibration layer"
    layer = source / "layer-003.safetensors"
    layer.write_bytes(layer_bytes)
    manifest = {
        "layers": {
            "3": {
                "bytes": len(layer_bytes),
                "file": layer.name,
                "sha256": hashlib.sha256(layer_bytes).hexdigest(),
            }
        }
    }
    manifest_path = source / "calibration-manifest.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)

    snapshot, snapshot_root, captured_manifest = (
        calibration_module._snapshot_calibration_source(
            source,
            expected_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        )
    )
    try:
        layer.write_bytes(b"post-authentication mutation")
        assert captured_manifest == manifest_bytes
        assert (snapshot_root / layer.name).read_bytes() == layer_bytes
        assert snapshot_root != source
    finally:
        snapshot.cleanup()


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
    anchored = builder.current_encoder_provenance(
        exllamav3_root=exllamav3_root,
        calibration=calibration,
        kquant_identity=(
            str(before["kquant_revision"]),
            str(before["kquant_source_sha256"]),
        ),
    )
    source.write_text("CODEBOOK = 2\n", encoding="utf-8")
    after = builder.current_encoder_provenance(
        exllamav3_root=exllamav3_root,
        calibration=calibration,
        kquant_root=kquant_root,
    )

    assert after == before
    assert anchored == before
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
        "encoding_runtime": {
            "schema": "test-bootstrap",
            "runtime": {"external_oci_image_id": "sha256:" + "8" * 64},
        },
        "fingerprint_schema": builder._ENCODER_FINGERPRINT_SCHEMA,
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
    encoder["kquant_revision"] = "1" * 40
    encoder["encoding_runtime"]["runtime"]["external_oci_image_id"] = (
        "sha256:" + "9" * 64
    )
    third = hashlib.sha256(
        builder._canonical_json(builder._encoder_fingerprint_payload(encoder)).encode(
            "utf-8"
        )
    ).hexdigest()

    assert first != third


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
    assert "trained for conversational instruction following" in card
    assert "targeted\nassistant behavior" in card
    assert "Not assistant-quality" not in card
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
    assert "Compact proxy, not the 754B teacher" in publication.quality_limitations
    assert "each passed 0/8 behavior contracts" in publication.quality_limitations
    assert "Not assistant-quality" not in publication.quality_limitations
    assert "four-prompt" not in publication.quality_limitations


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
    (output / "qsrt-manifest.json").write_text("{}\n", encoding="utf-8")
    (output / "MANIFEST.sha256").write_text("fixture\n", encoding="utf-8")
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
        ports = {"bf16": "8101", "siq": "8102", "qsrt": "8103"}
        model_options = {
            "bf16": ["--load-format", "fastsafetensors"],
            "siq": ["--load-format", "fastsafetensors"],
            "qsrt": [
                "--quantization",
                "kquant_hybrid",
                "--load-format",
                "fastsafetensors",
            ],
        }
        runtime_revisions = revisions
        argv = [
            "vllm",
            "serve",
            f"/models/{arm}",
            "--served-model-name",
            f"fruit-{arm}",
            "--host=127.0.0.1",
            "--port",
            ports[arm],
            "--tensor-parallel-size",
            "1",
            "--pipeline-parallel-size=1",
            *model_options[arm],
            "--attention-backend",
            "B12X_MLA_SPARSE",
            "--moe-backend",
            "b12x",
            "--kv-cache-dtype",
            "nvfp4_ds_mla",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--compilation-config",
            json.dumps(builder._FIXED_COMPILATION_CONFIG, separators=(",", ":")),
            "--speculative-config",
            json.dumps(builder._FIXED_SPECULATIVE_CONFIG, separators=(",", ":")),
            "--gpu-memory-utilization",
            "0.80",
            "--max-model-len",
            "4096",
            "--max-num-batched-tokens",
            "4096",
            "--max-num-seqs",
            "1",
            "--tool-call-parser",
            "glm47",
            "--enable-auto-tool-choice",
            "--reasoning-parser",
            "glm45",
            "--generation-config",
            "vllm",
        ]
        return {
            "runtime": {
                "image": f"registry.invalid/fruit-final@sha256:{'f' * 64}",
                **runtime_revisions,
                "argv": argv,
                "environment": dict(builder._FIXED_RUNTIME_ENVIRONMENT),
                "software": {
                    "cuda": "13.2.1",
                    "torch": "2.12.0",
                },
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
    qsrt_manifest_sha256 = builder._sha256(output / "qsrt-manifest.json")
    checksum_manifest_sha256 = builder._sha256(output / "MANIFEST.sha256")
    (output / builder._CANDIDATE_MARKER_NAME).write_text(
        builder._canonical_json(
            {
                "schema": "kquant_qsrt_candidate_v1",
                "checksum_manifest_sha256": checksum_manifest_sha256,
            }
        ),
        encoding="utf-8",
    )
    candidate_marker_sha256 = builder._sha256(output / builder._CANDIDATE_MARKER_NAME)

    def model(arm: str) -> dict[str, object]:
        if arm in builder._INSTRUCT_COMPARATOR_MODELS:
            return dict(builder._INSTRUCT_COMPARATOR_MODELS[arm])
        return {
            "repository": builder.fruit_publication_spec("instruct").repository,
            "revision": "4" * 40,
            "manifest_sha256": qsrt_manifest_sha256,
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
        "mtp_prefill": {
            "mode": "w4a16",
            "calls": 1,
            "capture_calls": 1,
            "replay_calls": 1,
        },
        "mtp_decode": {
            "mode": "w4a8",
            "calls": 2,
            "part_count": 2,
            "capture_calls": 1,
            "replay_calls": 1,
        },
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
            "version": 2,
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
            "vocab_size": builder._FRUIT_VOCAB_SIZE,
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
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    return builder._validate_runtime_qualification(
        path,
        expected_sha256=expected_sha256 or builder._sha256(path),
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
    assert "same immutable image, software stack" in section
    assert "spanning instruction following" not in section
    assert "targeted prompts across BF16" in section
    assert "non-representative" not in section
    assert builder._RUNTIME_QUALIFICATION_NAME in (
        builder._expected_package_inventory({"files": {}})
    )
    assert payload == validated


def test_runtime_qualification_requires_external_digest_authority(
    tmp_path: Path,
) -> None:
    path, output, payload, producer, source = _runtime_qualification_fixture(tmp_path)
    authorized_sha256 = builder._sha256(path)
    payload["complete"] = False
    path.write_text(builder._canonical_json(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="external SHA-256 authority"):
        _validate_runtime_fixture(
            path,
            output,
            producer,
            source,
            expected_sha256=authorized_sha256,
        )


def test_runtime_qualification_rejects_cross_arm_runtime_drift(
    tmp_path: Path,
) -> None:
    path, output, payload, producer, source = _runtime_qualification_fixture(tmp_path)
    payload["loaders"]["siq"]["runtime"]["software"]["torch"] = "different"
    path.write_text(builder._canonical_json(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="same immutable runtime identity"):
        _validate_runtime_fixture(path, output, producer, source)


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
        runtime_qualification_sha256=builder._sha256(qualification_path),
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


def test_completion_failure_restores_entire_candidate_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate"
    candidate_files = {
        "README.md": b"candidate card\n",
        "qsrt-manifest.json": b'{"candidate":true}\n',
        "MANIFEST.sha256": b"candidate checksums\n",
        builder._CANDIDATE_MARKER_NAME: b'{"candidate":"marker"}\n',
        builder._RATE_SWEEP_NAME: b'{"rates":"sealed"}\n',
    }
    for relative, content in candidate_files.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    expected = builder._snapshot_candidate_metadata(output)

    def write_completion_metadata(root: Path, **_kwargs: object) -> None:
        builder.shutil.rmtree(root / "evaluation")
        (root / "evaluation").mkdir()
        (root / builder._RATE_SWEEP_NAME).write_text("final rates\n")
        (root / builder._RUNTIME_QUALIFICATION_NAME).write_text("receipt\n")
        (root / "README.md").write_text("final card\n")
        (root / "qsrt-manifest.json").write_text("final manifest\n")
        (root / "MANIFEST.sha256").write_text("final checksums\n")

    def fail_complete_marker(root: Path, **_kwargs: object) -> None:
        (root / builder._COMPLETE_MARKER_NAME).write_text("partial marker\n")
        raise OSError("injected complete-marker failure")

    monkeypatch.setattr(builder, "_write_package_manifests", write_completion_metadata)
    monkeypatch.setattr(builder, "_validate_output_package", lambda *_a, **_k: None)
    monkeypatch.setattr(builder, "_write_complete_marker", fail_complete_marker)

    with pytest.raises(OSError, match="injected complete-marker failure"):
        builder._complete_candidate_package(
            output,
            source_evidence={},
            base_provenance={},
            producer={},
            calibration=SimpleNamespace(),
            rate_sweep={},
            layers={},
            publication=builder.fruit_publication_spec("instruct"),
            runtime_qualification={},
            runtime_qualification_sha256="a" * 64,
            spec=SimpleNamespace(),
        )

    assert builder._snapshot_candidate_metadata(output) == expected
    assert not (output / builder._COMPLETE_MARKER_NAME).exists()
    assert not (output / builder._RUNTIME_QUALIFICATION_NAME).exists()


def test_atomic_text_failure_removes_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.json"
    temporary = path.with_name(f".{path.name}.tmp-{builder.os.getpid()}")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(builder.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected atomic replace failure"):
        builder._atomic_text(path, "{}\n")

    assert not path.exists()
    assert not temporary.exists()


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
        lambda value: value["fidelity"].update({"vocab_size": 11}),
        lambda value: value["candidate"].pop("marker_sha256"),
        lambda value: value["models"]["siq"].pop("manifest_sha256"),
        lambda value: value["models"]["bf16"].update(
            {"model_index_sha256": "not-a-digest"}
        ),
        lambda value: value["models"]["bf16"].update({"manifest_sha256": "0" * 64}),
        lambda value: value["loaders"]["qsrt"]["runtime"].update(
            {"vllm_revision": "9" * 40}
        ),
        lambda value: value["models"]["qsrt"].update({"manifest_sha256": "0" * 64}),
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
    wrong_draft_backend = json.loads(builder._canonical_json(payload))
    argv = wrong_draft_backend["loaders"]["qsrt"]["runtime"]["argv"]
    option_index = argv.index("--speculative-config")
    speculative = json.loads(argv[option_index + 1])
    speculative["attention_backend"] = "TRITON_MLA"
    argv[option_index + 1] = json.dumps(speculative, separators=(",", ":"))
    malformed_values.append(wrong_draft_backend)

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
    eager_arm["loaders"]["qsrt"]["runtime"]["compilation_backend"] = "eager"
    malformed_values.append(eager_arm)

    no_graph_arm = json.loads(builder._canonical_json(payload))
    no_graph_arm["loaders"]["qsrt"]["runtime"]["cudagraph_mode"] = "NONE"
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


def test_runtime_qualification_rejects_unlisted_duplicate_and_negative_argv(
    tmp_path: Path,
) -> None:
    path, output, payload, producer, source = _runtime_qualification_fixture(tmp_path)
    base_argv = payload["loaders"]["bf16"]["runtime"]["argv"]
    malformed_values = []

    for extra in (
        ["--no-enable-prefix-caching"],
        ["--no-enable-chunked-prefill"],
        ["--unknown-qualification-flag"],
        ["unexpected-positional"],
    ):
        malformed = json.loads(builder._canonical_json(payload))
        malformed["loaders"]["bf16"]["runtime"]["argv"].extend(extra)
        malformed_values.append(malformed)

    for index, argument in enumerate(base_argv):
        if not argument.startswith("--"):
            continue
        duplicate = [argument]
        flag = argument.partition("=")[0]
        if "=" not in argument and flag not in builder._FIXED_RUNTIME_SWITCHES:
            duplicate.append(base_argv[index + 1])
        malformed = json.loads(builder._canonical_json(payload))
        malformed["loaders"]["siq"]["runtime"]["argv"].extend(duplicate)
        malformed_values.append(malformed)

    for malformed in malformed_values:
        path.write_text(builder._canonical_json(malformed), encoding="utf-8")
        with pytest.raises(ValueError):
            _validate_runtime_fixture(path, output, producer, source)


def test_runtime_qualification_requires_exact_runtime_environment(
    tmp_path: Path,
) -> None:
    path, output, payload, producer, source = _runtime_qualification_fixture(tmp_path)
    malformed_values = []

    drift = json.loads(builder._canonical_json(payload))
    drift["loaders"]["siq"]["runtime"]["environment"]["CUDA_VISIBLE_DEVICES"] = "1"
    malformed_values.append(drift)

    same_extra = json.loads(builder._canonical_json(payload))
    for arm in builder._RUNTIME_ARMS:
        same_extra["loaders"][arm]["runtime"]["environment"][
            "VLLM_FAKE_PRODUCTION_TOGGLE"
        ] = "1"
    malformed_values.append(same_extra)

    same_production_drift = json.loads(builder._canonical_json(payload))
    for arm in builder._RUNTIME_ARMS:
        same_production_drift["loaders"][arm]["runtime"]["environment"][
            "PYTHONSAFEPATH"
        ] = "0"
    malformed_values.append(same_production_drift)

    for malformed in malformed_values:
        path.write_text(builder._canonical_json(malformed), encoding="utf-8")
        with pytest.raises(ValueError, match="sanitized production environment"):
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
    no_mtp_prefill = json.loads(builder._canonical_json(payload))
    no_mtp_prefill["runtime_paths"]["layers"]["13"].pop("mtp_prefill")
    malformed_values.append(no_mtp_prefill)

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
    production_encoder = {
        "kquant_revision": "1" * 40,
        "kquant_source_sha256": "2" * 64,
        "exllamav3_revision": "3" * 40,
        "exllamav3_source_sha256": "4" * 64,
        "calibration_fingerprint": calibration.fingerprint,
        "calibration_capture_id": calibration.capture_id,
        "calibration_manifest_sha256": calibration.manifest_sha256,
        "encoding_runtime": {"schema": "authenticated-builder"},
        "fingerprint_schema": builder._ENCODER_FINGERPRINT_SCHEMA,
    }
    production_encoder["fingerprint"] = hashlib.sha256(
        builder._canonical_json(
            builder._encoder_fingerprint_payload(production_encoder)
        ).encode("utf-8")
    ).hexdigest()
    sweep_encoder = json.loads(builder._canonical_json(production_encoder))
    producer = {"encoder": production_encoder}
    payload = _rate_sweep(
        source=source,
        calibration=calibration_identity,
        encoder=sweep_encoder,
    )
    path = tmp_path / "rates.json"
    path.write_text(builder._canonical_json(payload), encoding="utf-8")

    assert (
        builder._validate_rate_sweep(
            path,
            expected_sha256=builder._sha256(path),
            source_evidence=source,
            calibration=calibration,
            producer=producer,
        )
        == payload
    )

    repinned_sweep = json.loads(builder._canonical_json(payload))
    repinned_encoder = repinned_sweep["signature"]["encoder"]
    repinned_encoder["kquant_revision"] = "8" * 40
    repinned_encoder["fingerprint"] = hashlib.sha256(
        builder._canonical_json(
            builder._encoder_fingerprint_payload(repinned_encoder)
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(builder._canonical_json(repinned_sweep), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        builder._validate_rate_sweep(
            path,
            expected_sha256=builder._sha256(path),
            source_evidence=source,
            calibration=calibration,
            producer=producer,
        )

    repinned_encoder["exllamav3_revision"] = "7" * 40
    repinned_encoder["fingerprint"] = hashlib.sha256(
        builder._canonical_json(
            builder._encoder_fingerprint_payload(repinned_encoder)
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(builder._canonical_json(repinned_sweep), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        builder._validate_rate_sweep(
            path,
            expected_sha256=builder._sha256(path),
            source_evidence=source,
            calibration=calibration,
            producer=producer,
        )
    path.write_text(builder._canonical_json(payload), encoding="utf-8")

    changed_producer = json.loads(builder._canonical_json(producer))
    changed_encoder = changed_producer["encoder"]
    changed_encoder["kquant_revision"] = "9" * 40
    changed_encoder["fingerprint"] = hashlib.sha256(
        builder._canonical_json(
            builder._encoder_fingerprint_payload(changed_encoder)
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="provenance"):
        builder._validate_rate_sweep(
            path,
            expected_sha256=builder._sha256(path),
            source_evidence=source,
            calibration=calibration,
            producer=changed_producer,
        )

    changed_encoder["exllamav3_source_sha256"] = "6" * 64
    changed_encoder["fingerprint"] = hashlib.sha256(
        builder._canonical_json(
            builder._encoder_fingerprint_payload(changed_encoder)
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="provenance"):
        builder._validate_rate_sweep(
            path,
            expected_sha256=builder._sha256(path),
            source_evidence=source,
            calibration=calibration,
            producer=changed_producer,
        )
    authorized_sha256 = builder._sha256(path)
    payload["complete"] = False
    path.write_text(builder._canonical_json(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="external SHA-256 authority"):
        builder._validate_rate_sweep(
            path,
            expected_sha256=authorized_sha256,
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
        builder._validate_part_cache_root(cache, create=False)


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
        builder._prepare_fresh_candidate_output(output)


def test_fresh_candidate_output_rejects_resume_state(tmp_path: Path) -> None:
    output = tmp_path / "output"
    parts = output / ".qsrt-parts"
    parts.mkdir(parents=True)
    with pytest.raises(ValueError, match="fresh empty output"):
        builder._prepare_fresh_candidate_output(output)

    parts.rmdir()
    output.rmdir()
    staged = builder._staged_part_cache_path(output)
    staged.mkdir()
    with pytest.raises(ValueError, match="no pre-existing staged part cache"):
        builder._prepare_fresh_candidate_output(output)


def test_finalize_part_cache_removes_active_and_staged_trees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 1)
    output = tmp_path / "candidate"
    layer = output / ".qsrt-parts" / "layer-003"
    layer.mkdir(parents=True)

    builder._finalize_part_cache(output)

    assert not (output / ".qsrt-parts").exists()
    assert not builder._staged_part_cache_path(output).exists()


def test_finalize_part_cache_retries_transient_directory_not_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 1)
    output = tmp_path / "candidate"
    layer = output / ".qsrt-parts" / "layer-003"
    layer.mkdir(parents=True)
    staged = builder._staged_part_cache_path(output)
    real_rmtree = builder.shutil.rmtree
    removals: list[Path] = []
    delays: list[float] = []

    def transient_removal(path: Path) -> None:
        removals.append(path)
        if len(removals) == 1:
            raise OSError(builder.errno.ENOTEMPTY, "injected transient directory race")
        real_rmtree(path)

    monkeypatch.setattr(builder.shutil, "rmtree", transient_removal)
    monkeypatch.setattr(builder.time, "sleep", delays.append)
    builder._finalize_part_cache(output)

    assert removals == [staged, staged]
    assert delays == [0.05]
    assert not staged.exists()


def test_layer_assembly_releases_part_views_before_cache_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 1)
    monkeypatch.setattr(builder, "FRUIT_QSRT_ARTIFACT_TENSORS", ("payload",))
    output = tmp_path / "candidate"
    (output / ".qsrt-parts").mkdir(parents=True)
    source_refs: list[weakref.ReferenceType[object]] = []
    handle_refs: list[weakref.ReferenceType[object]] = []
    finalized = False

    class SourceTensor:
        pass

    class IndependentTensor:
        shape = (1,)

        def contiguous(self) -> IndependentTensor:
            return self

    class FakeSafeOpen:
        def __init__(self) -> None:
            handle_refs.append(weakref.ref(self))

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_tensor(self, _name: str) -> SourceTensor:
            tensor = SourceTensor()
            source_refs.append(weakref.ref(tensor))
            return tensor

    def write_safetensors(
        path: Path, _tensors: dict[str, object], _metadata: dict[str, str]
    ) -> None:
        path.write_bytes(b"assembled")

    def finalize_part_cache(_output: Path) -> None:
        nonlocal finalized
        assert source_refs and all(reference() is None for reference in source_refs)
        assert handle_refs and all(reference() is None for reference in handle_refs)
        finalized = True

    monkeypatch.setattr(builder, "_validate_part_cache_root", lambda *_a, **_k: None)
    monkeypatch.setattr(builder, "_validate_layer", lambda *_a, **_k: None)
    monkeypatch.setattr(
        builder,
        "_validate_part",
        lambda *_a, **_k: {
            "layer": 3,
            "expert": 0,
            "format": {"r13": 0, "r2": 0},
        },
    )
    monkeypatch.setattr(builder, "safe_open", lambda *_a, **_k: FakeSafeOpen())
    monkeypatch.setattr(
        builder.torch,
        "cat",
        lambda _values, *, dim: IndependentTensor(),
    )
    monkeypatch.setattr(
        builder,
        "pack_fruit_atom_layer",
        lambda _pairs, *, layer: {"packed": IndependentTensor()},
    )
    monkeypatch.setattr(builder, "_atomic_safetensors", write_safetensors)
    monkeypatch.setattr(builder, "_fruit_atom_metadata", lambda **_k: {})
    monkeypatch.setattr(builder, "_finalize_part_cache", finalize_part_cache)

    layers = builder._assemble_layers(
        output,
        source_sha256="a" * 64,
        encoder_fingerprint="b" * 64,
    )

    assert finalized
    assert layers["3"]["expert_count"] == 1


def test_empty_layer_assembly_consumes_part_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "LAYERS", ())
    output = tmp_path / "candidate"
    (output / ".qsrt-parts").mkdir(parents=True)

    layers = builder._assemble_layers(
        output,
        source_sha256="a" * 64,
        encoder_fingerprint="b" * 64,
    )

    assert layers == {}
    assert not (output / ".qsrt-parts").exists()
    assert not builder._staged_part_cache_path(output).exists()
    assert not (output / builder._CANDIDATE_MARKER_NAME).exists()


def test_finalize_part_cache_failure_is_nonresumable_and_unsealed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(builder, "LAYERS", (3,))
    monkeypatch.setattr(builder, "EXPERTS", 1)
    output = tmp_path / "candidate"
    layer = output / ".qsrt-parts" / "layer-003"
    layer.mkdir(parents=True)
    staged = builder._staged_part_cache_path(output)

    def fail_removal(path: Path) -> None:
        assert path == staged
        raise OSError("injected staged-cache cleanup failure")

    monkeypatch.setattr(builder.shutil, "rmtree", fail_removal)
    with pytest.raises(OSError, match="injected staged-cache cleanup failure"):
        builder._finalize_part_cache(output)

    assert not (output / ".qsrt-parts").exists()
    assert staged.is_dir()
    assert not (output / builder._CANDIDATE_MARKER_NAME).exists()
    with pytest.raises(ValueError, match="no pre-existing staged part cache"):
        builder._prepare_fresh_candidate_output(output)


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


def test_strip_routed_experts_reads_private_authenticated_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(builder, "EXPERTS", 1)
    source = tmp_path / "model-layer-003.safetensors"
    target = tmp_path / "filtered.safetensors"
    alternate = tmp_path / "alternate.safetensors"
    prefix = "model.layers.3.mlp.experts.0"
    original_tensors = {
        f"{prefix}.{projection}.weight": builder.torch.tensor([1.0])
        for projection in ("down_proj", "gate_proj", "up_proj")
    }
    original_tensors["model.layers.3.input_layernorm.weight"] = builder.torch.tensor(
        [1.0]
    )
    altered_tensors = {
        name: builder.torch.full_like(tensor, 9.0)
        for name, tensor in original_tensors.items()
    }
    builder.save_file(original_tensors, source)
    builder.save_file(altered_tensors, alternate)
    original_bytes = source.read_bytes()
    altered_bytes = alternate.read_bytes()
    expected_sha256 = hashlib.sha256(original_bytes).hexdigest()
    real_safe_open = builder.safe_open

    class SwappingSafeOpen:
        def __init__(self, path: Path, *args: object, **kwargs: object) -> None:
            self.path = path
            self.args = args
            self.kwargs = kwargs
            self.inner = None

        def __enter__(self):
            source.write_bytes(altered_bytes)
            self.inner = real_safe_open(self.path, *self.args, **self.kwargs)
            return self.inner.__enter__()

        def __exit__(self, *args: object) -> object:
            assert self.inner is not None
            try:
                return self.inner.__exit__(*args)
            finally:
                source.write_bytes(original_bytes)

    monkeypatch.setattr(builder, "safe_open", SwappingSafeOpen)

    removed, _ = builder._strip_routed_experts(
        source,
        target,
        expected_sha256,
    )

    assert removed == {
        f"{prefix}.down_proj.weight",
        f"{prefix}.gate_proj.weight",
        f"{prefix}.up_proj.weight",
    }
    with real_safe_open(target, framework="pt", device="cpu") as handle:
        kept = handle.get_tensor("model.layers.3.input_layernorm.weight")
    builder.torch.testing.assert_close(kept, builder.torch.tensor([1.0]))


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


def test_part_validation_rejects_manifest_format_that_disagrees_with_tensor(
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
