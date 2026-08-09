"""Capture the committed KQuant tree before encoder modules are imported."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.tracked_worktree import TrackedWorktreeSnapshot, snapshot_tracked_worktree


def _capture() -> tuple[tuple[str, str], TrackedWorktreeSnapshot, Path]:
    root = Path(__file__).resolve().parents[1]
    already_imported = sorted(
        name for name in sys.modules if name == "kquant" or name.startswith("kquant.")
    )
    if already_imported:
        raise RuntimeError(
            "KQuant modules were imported before source snapshot creation: "
            + ", ".join(already_imported[:3])
        )
    snapshot = snapshot_tracked_worktree(root, require_clean=False)
    snapshot_path = str(snapshot.root)
    sys.path[:] = [
        entry for entry in sys.path if not entry or Path(entry).resolve() != root
    ]
    sys.path.insert(0, snapshot_path)
    scripts = sys.modules.get("scripts")
    if scripts is not None and hasattr(scripts, "__path__"):
        scripts.__path__ = [str(snapshot.root / "scripts")]
    return (snapshot.revision, snapshot.sha256), snapshot, root


(
    KQUANT_IMPORT_IDENTITY,
    _KQUANT_IMPORT_SNAPSHOT,
    KQUANT_IMPORT_SOURCE_ROOT,
) = _capture()
