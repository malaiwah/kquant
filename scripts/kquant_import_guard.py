"""Capture the committed KQuant tree before encoder modules are imported."""

from __future__ import annotations

from pathlib import Path

from scripts.tracked_worktree import git_revision, tracked_worktree_sha256


def _capture() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[1]
    revision = git_revision(root)
    fingerprint = tracked_worktree_sha256(root, require_clean=False)
    if git_revision(root) != revision:
        raise RuntimeError("KQuant source revision changed during import attestation")
    return revision, fingerprint


KQUANT_IMPORT_IDENTITY = _capture()
