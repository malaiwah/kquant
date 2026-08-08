"""Capture the committed KQuant tree before encoder modules are imported."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _capture() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ("git", "-C", str(root), "ls-tree", "-r", "-z", "--full-tree", "HEAD"),
        check=True,
        capture_output=True,
    ).stdout
    if len(revision) != 40 or not tree:
        raise RuntimeError("cannot capture committed KQuant import identity")
    return revision, hashlib.sha256(b"git-ls-tree-v1\0" + tree).hexdigest()


KQUANT_IMPORT_IDENTITY = _capture()
