"""Deterministically attest the actual bytes in a tracked Git worktree."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Self

_FINGERPRINT_PREFIX = b"kquant-tracked-worktree-sha256-v1\0"
GIT_EXECUTABLE = "/usr/bin/git"
_GIT_CONFIG_OVERRIDES = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
)

_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY


def _git_output(root: Path, *args: str) -> bytes:
    try:
        environment = dict(os.environ)
        environment["GIT_CONFIG_GLOBAL"] = os.devnull
        environment["GIT_CONFIG_NOSYSTEM"] = "1"
        environment["GIT_TERMINAL_PROMPT"] = "0"
        return subprocess.run(
            (GIT_EXECUTABLE, *_GIT_CONFIG_OVERRIDES, "-C", str(root), *args),
            check=True,
            capture_output=True,
            env=environment,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect source tree: {root}") from exc


def git_revision(root: Path) -> str:
    revision = _git_output(root, "rev-parse", "HEAD").strip().decode("ascii")
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"source revision is not a commit digest: {root}")
    return revision


def _nul_records(output: bytes, *, description: str) -> list[bytes]:
    if not output or output[-1:] != b"\0":
        raise ValueError(f"source checkout has malformed {description}")
    records = output.split(b"\0")
    records.pop()
    if not records or any(not record for record in records):
        raise ValueError(f"source checkout has malformed {description}")
    return records


def _tracked_index(root: Path) -> dict[bytes, tuple[bytes, bytes]]:
    index: dict[bytes, tuple[bytes, bytes]] = {}
    records = _nul_records(
        _git_output(root, "ls-files", "--stage", "-v", "-z"),
        description="Git index",
    )
    for record in records:
        if len(record) < 3 or record[1:2] != b" " or b"\t" not in record:
            raise ValueError("source checkout has malformed Git index")
        tag = record[:1]
        if tag == b"h":
            raise ValueError("source checkout has assume-unchanged tracked files")
        if tag in (b"S", b"s"):
            raise ValueError("source checkout has skip-worktree tracked files")
        if tag != b"H":
            raise ValueError("source checkout has anomalous tracked files")
        metadata, relative_path = record[2:].split(b"\t", 1)
        fields = metadata.split(b" ")
        if (
            len(fields) != 3
            or fields[2] != b"0"
            or len(fields[1]) != 40
            or any(byte not in b"0123456789abcdef" for byte in fields[1])
        ):
            raise ValueError("source checkout has malformed or unmerged Git index")
        mode, object_id = fields[:2]
        if mode not in (b"100644", b"100755"):
            raise ValueError(
                f"source checkout tracks a non-regular entry: "
                f"{os.fsdecode(relative_path)!r}"
            )
        if (
            not relative_path
            or relative_path.startswith(b"/")
            or any(part in (b"", b".", b"..") for part in relative_path.split(b"/"))
            or relative_path in index
        ):
            raise ValueError("source checkout has anomalous tracked paths")
        index[relative_path] = (mode, object_id)
    return index


def _tracked_head(root: Path) -> dict[bytes, tuple[bytes, bytes]]:
    head: dict[bytes, tuple[bytes, bytes]] = {}
    records = _nul_records(
        _git_output(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD"),
        description="HEAD tree",
    )
    for record in records:
        if b"\t" not in record:
            raise ValueError("source checkout has malformed HEAD tree")
        metadata, relative_path = record.split(b"\t", 1)
        fields = metadata.split(b" ")
        if (
            len(fields) != 3
            or len(fields[2]) != 40
            or any(byte not in b"0123456789abcdef" for byte in fields[2])
        ):
            raise ValueError("source checkout has malformed HEAD tree")
        mode, object_type, object_id = fields
        if object_type != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError(
                f"source checkout tracks a symlink, submodule, or non-regular entry: "
                f"{os.fsdecode(relative_path)!r}"
            )
        if not relative_path or relative_path in head:
            raise ValueError("source checkout has anomalous HEAD paths")
        head[relative_path] = (mode, object_id)
    return head


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class TrackedWorktreeSnapshot:
    """A process-private copy of one attested tracked source tree."""

    def __init__(
        self,
        temporary_directory: tempfile.TemporaryDirectory[str],
        *,
        revision: str,
        sha256: str,
    ) -> None:
        self._temporary_directory = temporary_directory
        self.root = Path(temporary_directory.name)
        self.revision = revision
        self.sha256 = sha256

    def close(self) -> None:
        self._temporary_directory.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _write_all(file_fd: int, chunk: bytes) -> None:
    offset = 0
    while offset < len(chunk):
        written = os.write(file_fd, chunk[offset:])
        if written <= 0:
            raise OSError("short write while creating private source snapshot")
        offset += written


def _open_tracked_file(root_fd: int, relative_path: bytes) -> tuple[int, int]:
    """Open a tracked file without following any path component."""

    components = relative_path.split(b"/")
    parent_fd = os.dup(root_fd)
    try:
        for component in components[:-1]:
            child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        file_fd = os.open(components[-1], _READ_FLAGS, dir_fd=parent_fd)
    except BaseException:
        os.close(parent_fd)
        raise
    return parent_fd, file_fd


def _attest_tracked_worktree(
    root: Path,
    *,
    require_clean: bool,
    snapshot_root: Path | None,
) -> tuple[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    revision = git_revision(root)
    index = _tracked_index(root)
    head = _tracked_head(root)
    if index != head:
        raise ValueError(f"source checkout index does not match HEAD: {root}")
    if require_clean:
        _require_status_clean(root)

    tree_digest = hashlib.sha256(_FINGERPRINT_PREFIX)
    try:
        root_fd = os.open(root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise ValueError(f"cannot securely open source checkout: {root}") from exc
    try:
        for relative_path in sorted(index):
            expected_mode, expected_object_id = index[relative_path]
            try:
                parent_fd, file_fd = _open_tracked_file(root_fd, relative_path)
            except OSError as exc:
                raise ValueError(
                    f"cannot securely open tracked source file: "
                    f"{os.fsdecode(relative_path)!r}"
                ) from exc
            relative_name = relative_path.rsplit(b"/", 1)[-1]
            snapshot_fd: int | None = None
            try:
                before = os.fstat(file_fd)
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError(
                        f"tracked source is not a regular file: "
                        f"{os.fsdecode(relative_path)!r}"
                    )
                actual_mode = b"100755" if before.st_mode & 0o111 else b"100644"
                if actual_mode != expected_mode:
                    raise ValueError(
                        f"tracked source executable mode differs from Git: "
                        f"{os.fsdecode(relative_path)!r}"
                    )

                if snapshot_root is not None:
                    snapshot_path = snapshot_root / os.fsdecode(relative_path)
                    snapshot_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    snapshot_fd = os.open(
                        snapshot_path,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        0o600,
                    )

                tree_digest.update(actual_mode)
                tree_digest.update(len(relative_path).to_bytes(8, "big"))
                tree_digest.update(relative_path)
                tree_digest.update(before.st_size.to_bytes(8, "big"))
                blob_digest = hashlib.sha1(
                    b"blob " + str(before.st_size).encode("ascii") + b"\0"
                )
                bytes_read = 0
                while chunk := os.read(file_fd, 1 << 20):
                    bytes_read += len(chunk)
                    tree_digest.update(chunk)
                    blob_digest.update(chunk)
                    if snapshot_fd is not None:
                        _write_all(snapshot_fd, chunk)

                after = os.fstat(file_fd)
                try:
                    path_after = os.stat(
                        relative_name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise ValueError(
                        f"tracked source path changed while hashing: "
                        f"{os.fsdecode(relative_path)!r}"
                    ) from exc
                if (
                    bytes_read != before.st_size
                    or _stat_identity(before) != _stat_identity(after)
                    or _stat_identity(after) != _stat_identity(path_after)
                ):
                    raise ValueError(
                        f"tracked source changed while hashing: "
                        f"{os.fsdecode(relative_path)!r}"
                    )
                if (
                    require_clean
                    and blob_digest.hexdigest().encode("ascii") != expected_object_id
                ):
                    raise ValueError(
                        f"tracked source bytes differ from Git: "
                        f"{os.fsdecode(relative_path)!r}"
                    )
                if snapshot_fd is not None:
                    snapshot_after = os.fstat(snapshot_fd)
                    if (
                        not stat.S_ISREG(snapshot_after.st_mode)
                        or snapshot_after.st_nlink != 1
                        or snapshot_after.st_size != bytes_read
                    ):
                        raise ValueError(
                            f"cannot securely snapshot tracked source file: "
                            f"{os.fsdecode(relative_path)!r}"
                        )
            finally:
                if snapshot_fd is not None:
                    os.close(snapshot_fd)
                os.close(file_fd)
                os.close(parent_fd)
    finally:
        os.close(root_fd)

    if (
        git_revision(root) != revision
        or _tracked_index(root) != index
        or _tracked_head(root) != head
    ):
        raise ValueError(f"source checkout changed while hashing: {root}")
    if require_clean:
        _require_status_clean(root)
    return revision, tree_digest.hexdigest()


def snapshot_tracked_worktree(
    root: Path, *, require_clean: bool = True
) -> TrackedWorktreeSnapshot:
    """Copy tracked bytes once, retaining their revision and canonical digest."""

    temporary_directory = tempfile.TemporaryDirectory(prefix="kquant-source-")
    snapshot_root = Path(temporary_directory.name)
    try:
        revision, digest = _attest_tracked_worktree(
            root,
            require_clean=require_clean,
            snapshot_root=snapshot_root,
        )
    except BaseException:
        temporary_directory.cleanup()
        raise
    return TrackedWorktreeSnapshot(
        temporary_directory,
        revision=revision,
        sha256=digest,
    )


def _require_status_clean(root: Path) -> None:
    if _git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError(f"source checkout has uncommitted files: {root}")


def tracked_worktree_sha256(root: Path, *, require_clean: bool = True) -> str:
    """Hash tracked paths, executable modes, and actual bytes, failing on races."""

    _, digest = _attest_tracked_worktree(
        root,
        require_clean=require_clean,
        snapshot_root=None,
    )
    return digest
