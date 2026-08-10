from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "capture_fruit_qsrt_calibration.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_capture_fruit_qsrt_calibration_test", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_CAPTURE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CAPTURE
_SPEC.loader.exec_module(_CAPTURE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bf16_base(root: Path) -> Any:
    root.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    save_file(
        {"lm_head.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        str(root / shard_name),
    )
    config = {
        "dtype": "bfloat16",
        "hidden_size": 2,
        "moe_intermediate_size": 3,
        "n_routed_experts": 0,
        "num_experts_per_tok": _CAPTURE.FRUIT_CALIBRATION_TOPK,
        "num_hidden_layers": 0,
        "num_nextn_predict_layers": 1,
        "rope_interleave": True,
        "rope_theta": 500_000.0,
        "tie_word_embeddings": False,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    index = {"weight_map": {"lm_head.weight": shard_name}}
    (root / _CAPTURE._BF16_BASE_INDEX).write_text(json.dumps(index), encoding="utf-8")
    entries = {
        name: _sha256(root / name)
        for name in ("config.json", _CAPTURE._BF16_BASE_INDEX, shard_name)
    }
    (root / "MANIFEST.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items())),
        encoding="utf-8",
    )
    manifest_sha256 = _sha256(root / "MANIFEST.sha256")
    spec = SimpleNamespace(
        safetensors_manifest_sha256=manifest_sha256,
        hidden_size=2,
        intermediate_size=3,
        num_experts=0,
        mtp_layer=0,
        trained_rope_theta=500_000.0,
    )
    source = {
        "kind": "authenticated_bf16_export",
        "directory": root.name,
        "manifest_sha256": manifest_sha256,
        "index_sha256": entries[_CAPTURE._BF16_BASE_INDEX],
        "file_count": len(entries),
    }
    return SimpleNamespace(spec=spec, source=source)


class _TinyFruit(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(2, 2, bias=False)


def test_authenticated_shard_load_rejects_pathname_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bf16"
    authority = _write_bf16_base(root)
    source = _CAPTURE._authenticate_bf16_base(root, authority)
    shard_name = "model-00001-of-00001.safetensors"
    descriptor = source._shard_fds[shard_name]
    replacement = root / "replacement.safetensors"
    save_file(
        {"lm_head.weight": torch.full((2, 2), 99.0)},
        str(replacement),
    )
    os.replace(replacement, root / shard_name)

    try:
        with pytest.raises(ValueError, match="changed after authentication"):
            _CAPTURE._load_bf16_model(
                source,
                SimpleNamespace(Fruit=_TinyFruit),
                torch.device("cpu"),
                authority.spec,
            )
    finally:
        source.close()

    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_calibration_inputs_are_consumed_from_pinned_private_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    trainer = tmp_path / "trainer.py"
    reference = tmp_path / "reference.pt"
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    corpus.write_text('{"axis":"a","source":"s","text":"content"}\n', encoding="utf-8")
    trainer.write_text("ROUTED_SCALE = 1.0\n", encoding="utf-8")
    trainer_dependencies = {
        "dependency.py": b"VALUE = 1\n",
        trainer.name: trainer.read_bytes(),
    }
    (tmp_path / "dependency.py").write_bytes(trainer_dependencies["dependency.py"])
    reference.write_bytes(b"reference")
    tokenizer_payloads = {
        "config.json": b"{}",
        "tokenizer.json": b'{"version":"1.0"}',
        "tokenizer_config.json": b"{}",
    }
    for name, payload in tokenizer_payloads.items():
        (tokenizer / name).write_bytes(payload)
    (tokenizer / "special_tokens_map.json").write_text(
        '{"additional_special_tokens":["injected"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        _CAPTURE,
        "FRUIT_CALIBRATION_CORPUS",
        {"filename": corpus.name, "sha256": _sha256(corpus)},
    )
    monkeypatch.setattr(
        _CAPTURE,
        "FRUIT_CALIBRATION_TOKENIZER_FILES",
        {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in tokenizer_payloads.items()
        },
    )
    monkeypatch.setattr(
        _CAPTURE,
        "FRUIT_CALIBRATION_TRAINER_FILES",
        {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in trainer_dependencies.items()
        },
    )
    inputs = _CAPTURE._snapshot_calibration_inputs(
        SimpleNamespace(
            corpus=corpus,
            tokenizer=tokenizer,
            trainer=trainer,
            reference=reference,
        ),
        SimpleNamespace(reference_sha256=_sha256(reference)),
    )
    try:
        assert {path.name for path in inputs.tokenizer.iterdir()} == set(
            tokenizer_payloads
        )
        assert not (inputs.tokenizer / "special_tokens_map.json").exists()
        assert {path.name for path in inputs.trainer.parent.iterdir()} == set(
            trainer_dependencies
        )
        (tmp_path / "dependency.py").write_text("VALUE = 2\n", encoding="utf-8")
        assert (inputs.trainer.parent / "dependency.py").read_bytes() == (
            trainer_dependencies["dependency.py"]
        )
        corpus.write_text("attacker-controlled\n", encoding="utf-8")
        assert inputs.corpus.read_text(encoding="utf-8").startswith('{"axis"')
    finally:
        inputs.close()


def test_authentication_failure_closes_retained_shard_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bf16"
    authority = _write_bf16_base(root)
    authority.source = {"kind": "not-the-authenticated-source"}
    opened_shards: list[int] = []
    open_regular_at = _CAPTURE._open_regular_at

    def recording_open(directory_fd: int, name: str) -> tuple[int, tuple[int, ...]]:
        result = open_regular_at(directory_fd, name)
        if name.endswith(".safetensors"):
            opened_shards.append(result[0])
        return result

    monkeypatch.setattr(_CAPTURE, "_open_regular_at", recording_open)
    with pytest.raises(ValueError, match="source identity mismatch"):
        _CAPTURE._authenticate_bf16_base(root, authority)

    assert opened_shards
    for descriptor in opened_shards:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_missing_transformers_dependency_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def missing_transformers(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_transformers)
    spec = importlib.util.spec_from_file_location(
        "_capture_fruit_qsrt_calibration_without_transformers", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    capture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = capture
    try:
        spec.loader.exec_module(capture)
        with pytest.raises(RuntimeError, match=r"kquant\[fruit-calibration\]"):
            capture._load_tokenizer(tmp_path)
    finally:
        sys.modules.pop(spec.name, None)


def test_fruit_calibration_extra_declares_bounded_transformers_dependency() -> None:
    pyproject = tomllib.loads(
        (_SCRIPT_PATH.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "transformers" not in {
        dependency.split("[")[0].split("=")[0].split("<")[0].split(">")[0]
        for dependency in pyproject["project"]["dependencies"]
    }
    assert pyproject["project"]["optional-dependencies"]["fruit-calibration"] == [
        "transformers>=5.0,<6"
    ]
