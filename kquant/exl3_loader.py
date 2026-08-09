"""Load kquant's QSRT encoder atop an unmodified ExLlamaV3 dependency."""

from __future__ import annotations

import gc
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import torch

from scripts.tracked_worktree import (
    TrackedWorktreeSnapshot,
    snapshot_tracked_worktree,
)

_BACKEND_MODULE = "exllamav3.modules.quant.exl3_lib.kquant_qsrt_encoder"
_EXLLAMAV3_SNAPSHOT: TrackedWorktreeSnapshot | None = None
_EXLLAMAV3_SOURCE_ROOT: Path | None = None


def _exllamav3_snapshot(exllamav3_root: str | Path) -> TrackedWorktreeSnapshot:
    global _EXLLAMAV3_SNAPSHOT, _EXLLAMAV3_SOURCE_ROOT

    source_root = Path(exllamav3_root).resolve(strict=True)
    if _EXLLAMAV3_SNAPSHOT is None:
        preloaded = sorted(
            name
            for name in sys.modules
            if name == "exllamav3" or name.startswith("exllamav3.")
        )
        if preloaded:
            raise RuntimeError(
                "ExLlamaV3 modules were imported before source snapshot creation: "
                + ", ".join(preloaded[:3])
            )
        _EXLLAMAV3_SNAPSHOT = snapshot_tracked_worktree(source_root)
        _EXLLAMAV3_SOURCE_ROOT = source_root
    elif source_root != _EXLLAMAV3_SOURCE_ROOT:
        raise ValueError(
            "one process cannot load QSRT encoders from multiple ExLlamaV3 sources"
        )
    return _EXLLAMAV3_SNAPSHOT


def exllamav3_source_identity(exllamav3_root: str | Path) -> tuple[str, str]:
    """Return the immutable identity of the process-lifetime source snapshot."""

    snapshot = _exllamav3_snapshot(exllamav3_root)
    return snapshot.revision, snapshot.sha256


def load_qsrt_encoder(exllamav3_root: str | Path) -> types.ModuleType:
    """Load kquant's QSRT encoder using an unmodified ExLlamaV3 checkout.

    kquant owns the mixed-rate LDLQ implementation. ExLlamaV3 supplies its
    unmodified extension plus Hadamard and tensor utilities. Importing the
    package root would also require serving, tokenizer, and generation
    dependencies that are intentionally absent from kquant's environment.
    """

    source_snapshot = _exllamav3_snapshot(exllamav3_root)
    root = source_snapshot.root / "exllamav3"
    packages = (
        ("exllamav3", root),
        ("exllamav3.modules", root / "modules"),
        ("exllamav3.modules.quant", root / "modules" / "quant"),
        (
            "exllamav3.modules.quant.exl3_lib",
            root / "modules" / "quant" / "exl3_lib",
        ),
        ("exllamav3.util", root / "util"),
    )
    for name, path in packages:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = module.__path__
        module.__spec__ = spec
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)

    util = sys.modules["exllamav3.util"]

    def cuda_sync_active() -> None:
        for device_id in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{device_id}")
            if torch.cuda.memory_allocated(device) > 0:
                torch.cuda.synchronize(device)

    util.cuda_sync_active = cuda_sync_active

    memory = types.ModuleType("exllamav3.util.memory")

    def free_mem() -> None:
        gc.collect()
        torch.cuda.empty_cache()

    memory.free_mem = free_mem
    memory.list_gpu_tensors = lambda *args, **kwargs: None
    sys.modules[memory.__name__] = memory
    util.memory = memory

    progress = types.ModuleType("exllamav3.util.progress")

    class ProgressBar:
        def __init__(self, text: str, count: int, transient: bool = True):
            self.text = text
            self.count = count
            self.transient = transient

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def update(self, value: int) -> None:
            return None

        def new_task(self, text: str, count: int) -> None:
            self.text = text
            self.count = count

    progress.ProgressBar = ProgressBar
    sys.modules[progress.__name__] = progress
    util.progress = progress

    module = sys.modules.get(_BACKEND_MODULE)
    if module is None:
        backend_path = Path(__file__).with_name("exl3_encoder_backend.py")
        spec = importlib.util.spec_from_file_location(_BACKEND_MODULE, backend_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load QSRT encoder backend from {backend_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_BACKEND_MODULE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_BACKEND_MODULE, None)
            raise
        module.__kquant_exllamav3_revision__ = source_snapshot.revision
        module.__kquant_exllamav3_source_sha256__ = source_snapshot.sha256
        sys.modules["exllamav3.modules.quant.exl3_lib"].kquant_qsrt_encoder = module
    return module
