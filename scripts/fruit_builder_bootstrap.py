"""Standalone trusted launcher for the production Fruit QSRT builder.

Invoke these externally authenticated bytes directly with a trusted Python
interpreter in isolated, no-site mode.  This file imports no repository module.
"""

from __future__ import annotations

import sys

if not sys.flags.isolated or not sys.flags.no_site:
    raise RuntimeError(
        "Fruit builder bootstrap requires direct trusted-python invocation with -I -S"
    )

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

_SOURCE_FINGERPRINT_PREFIX = b"kquant-tracked-worktree-sha256-v1\0"
_BOOTSTRAP_CONTEXT_ENV = "KQUANT_FRUIT_BUILDER_BOOTSTRAP_V3"
_CONTEXT_SCHEMA = "kquant_fruit_builder_bootstrap_v3"
_RUNTIME_IDENTITY_SCHEMA = "kquant_fruit_builder_external_oci_runtime_v1"
_PYTHON_EXECUTABLE = Path("/opt/venv/bin/python")
_PYTHON_SITE_PACKAGES = Path("/opt/venv/lib/python3.12/site-packages")
_GIT_EXECUTABLE = Path("/usr/bin/git")
_CC_EXECUTABLE = Path("/usr/bin/gcc")
_CXX_EXECUTABLE = Path("/usr/bin/g++")
_NINJA_EXECUTABLE = Path("/opt/venv/bin/ninja")
_CUDA_HOME = Path("/usr/local/cuda")
_NVCC_EXECUTABLE = _CUDA_HOME / "bin" / "nvcc"
_TOOLCHAIN_PATH = "/opt/venv/bin:/usr/local/cuda/bin:/usr/bin"
_HEX = frozenset("0123456789abcdef")
_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
}
_RUNNER = """\
import runpy
import shutil
import sys
root = sys.argv[1]
site_packages = sys.argv[2]
runtime_root = sys.argv[3]
script = root + "/scripts/build_fruit_qsrt_model.py"
sys.path.insert(0, root)
sys.path.append(site_packages)
sys.argv = [script, *sys.argv[4:]]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    try:
        shutil.rmtree(root)
    finally:
        shutil.rmtree(runtime_root)
"""


def _digest(value: str, *, length: int, name: str) -> str:
    if len(value) != length or any(character not in _HEX for character in value):
        raise ValueError(f"{name} must be a lowercase {length}-character digest")
    return value


def _oci_image_id(value: str) -> str:
    if not value.startswith("sha256:"):
        raise ValueError("external OCI image ID must use immutable sha256:<digest>")
    _digest(value.removeprefix("sha256:"), length=64, name="OCI image digest")
    return value


def _secure_file_sha256(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"trusted bootstrap is a symlink: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot securely open trusted bootstrap: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                f"trusted bootstrap is not an independent regular file: {path}"
            )
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(after) != identity(
            path_after
        ):
            raise ValueError("trusted bootstrap changed while it was authenticated")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def installed_bootstrap_sha256() -> str:
    """Return the digest an external launcher must pin for this installed file."""

    return _secure_file_sha256(Path(__file__).absolute())


def _executable_identity(path: Path, *, name: str) -> dict[str, str]:
    if not path.is_absolute():
        raise ValueError(f"trusted {name} path must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"trusted {name} is not executable: {path}")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "sha256": _secure_file_sha256(resolved),
    }


def _git_output(root: Path, *arguments: str) -> bytes:
    environment = {
        **_GIT_ENV,
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "PATH": "/usr/bin",
    }
    try:
        return subprocess.run(
            (str(_GIT_EXECUTABLE), "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            env=environment,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"cannot read authenticated Git objects from {root}"
        ) from exc


def _git_object(root: Path, object_type: str, object_id: str) -> bytes:
    content = _git_output(root, "cat-file", object_type, object_id)
    actual = hashlib.sha1(
        object_type.encode("ascii")
        + b" "
        + str(len(content)).encode("ascii")
        + b"\0"
        + content
    ).hexdigest()
    if actual != object_id:
        raise ValueError(
            f"Git {object_type} object failed hash verification: {object_id}"
        )
    return content


def _commit_tree(root: Path, revision: str) -> str:
    commit = _git_object(root, "commit", revision)
    first_line = commit.split(b"\n", 1)[0]
    if not first_line.startswith(b"tree "):
        raise ValueError("authenticated revision has a malformed Git commit")
    try:
        tree_id = first_line.removeprefix(b"tree ").decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("authenticated revision has a malformed Git tree ID") from exc
    return _digest(tree_id, length=40, name="Git tree object ID")


def _tree_records(
    root: Path,
    tree_id: str,
    *,
    prefix: bytes = b"",
    depth: int = 0,
) -> list[tuple[bytes, str, bytes]]:
    if depth > 256:
        raise ValueError("authenticated revision exceeds the Git tree depth limit")
    content = _git_object(root, "tree", tree_id)
    records: list[tuple[bytes, str, bytes]] = []
    offset = 0
    while offset < len(content):
        space = content.find(b" ", offset)
        nul = content.find(b"\0", space + 1)
        if space <= offset or nul <= space + 1 or nul + 21 > len(content):
            raise ValueError("authenticated revision has a malformed Git tree entry")
        mode = content[offset:space]
        name = content[space + 1 : nul]
        object_id = content[nul + 1 : nul + 21].hex()
        offset = nul + 21
        if not name or b"/" in name or name in (b".", b"..") or b"\0" in name:
            raise ValueError(
                "authenticated revision contains an anomalous tracked path"
            )
        relative_path = prefix + name
        if mode == b"40000":
            records.extend(
                _tree_records(
                    root,
                    object_id,
                    prefix=relative_path + b"/",
                    depth=depth + 1,
                )
            )
        elif mode in (b"100644", b"100755"):
            records.append((mode, object_id, relative_path))
        else:
            raise ValueError(
                "authenticated revision contains a non-regular tracked entry"
            )
    if offset != len(content):
        raise ValueError("authenticated revision has malformed Git tree bytes")
    return records


def snapshot_git_revision(
    root: Path,
    *,
    expected_revision: str,
    expected_sha256: str,
) -> Path:
    """Materialize authenticated committed bytes into a private source snapshot."""

    revision = _digest(expected_revision, length=40, name="KQuant revision")
    expected_digest = _digest(expected_sha256, length=64, name="KQuant source SHA-256")
    checkout = root.resolve(strict=True)
    if not checkout.is_dir():
        raise NotADirectoryError(checkout)
    root_tree_id = _commit_tree(checkout, revision)
    records = _tree_records(checkout, root_tree_id)
    if not records or len({record[2] for record in records}) != len(records):
        raise ValueError("authenticated revision has an anomalous Git tree")

    snapshot = Path(tempfile.mkdtemp(prefix="kquant-authenticated-source-", dir="/tmp"))
    tree_digest = hashlib.sha256(_SOURCE_FINGERPRINT_PREFIX)
    try:
        for mode, object_id, relative_path in sorted(records, key=lambda item: item[2]):
            content = _git_object(checkout, "blob", object_id)
            tree_digest.update(mode)
            tree_digest.update(len(relative_path).to_bytes(8, "big"))
            tree_digest.update(relative_path)
            tree_digest.update(len(content).to_bytes(8, "big"))
            tree_digest.update(content)
            destination = snapshot / os.fsdecode(relative_path)
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
            descriptor = os.open(
                destination, flags, 0o700 if mode == b"100755" else 0o600
            )
            try:
                offset = 0
                while offset < len(content):
                    written = os.write(descriptor, content[offset:])
                    if written <= 0:
                        raise OSError(
                            "short write while creating authenticated snapshot"
                        )
                    offset += written
            finally:
                os.close(descriptor)
        if tree_digest.hexdigest() != expected_digest:
            raise ValueError(
                "authenticated Git revision does not match KQuant source SHA-256"
            )
        if _tree_records(checkout, root_tree_id) != records:
            raise ValueError(
                "authenticated Git revision changed during snapshot creation"
            )
        if _commit_tree(checkout, revision) != root_tree_id:
            raise ValueError(
                "authenticated Git commit changed during snapshot creation"
            )
        builder = snapshot / "scripts" / "build_fruit_qsrt_model.py"
        guard = snapshot / "scripts" / "kquant_import_guard.py"
        if not builder.is_file() or not guard.is_file():
            raise ValueError("authenticated revision has no Fruit production builder")
        return snapshot
    except BaseException:
        shutil.rmtree(snapshot)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate a KQuant Git snapshot and launch the Fruit QSRT builder"
    )
    parser.add_argument("--show-bootstrap-sha256", action="store_true")
    parser.add_argument("--kquant-root", type=Path)
    parser.add_argument("--kquant-revision")
    parser.add_argument("--kquant-source-sha256")
    parser.add_argument("--bootstrap-sha256")
    parser.add_argument("--external-oci-image-id")
    parser.add_argument("builder_args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = _parser().parse_args()
    bootstrap_sha256 = installed_bootstrap_sha256()
    if args.show_bootstrap_sha256:
        if (
            any(
                value is not None
                for value in (
                    args.kquant_root,
                    args.kquant_revision,
                    args.kquant_source_sha256,
                    args.bootstrap_sha256,
                    args.external_oci_image_id,
                )
            )
            or args.builder_args
        ):
            raise ValueError("--show-bootstrap-sha256 cannot launch a builder")
        print(bootstrap_sha256)
        return
    if None in (
        args.kquant_root,
        args.kquant_revision,
        args.kquant_source_sha256,
        args.bootstrap_sha256,
        args.external_oci_image_id,
    ):
        raise ValueError("production launch requires all external KQuant trust anchors")
    expected_bootstrap = _digest(
        args.bootstrap_sha256, length=64, name="bootstrap SHA-256"
    )
    if bootstrap_sha256 != expected_bootstrap:
        raise ValueError(
            "installed Fruit builder bootstrap does not match its external anchor"
        )
    checkout = args.kquant_root.resolve(strict=True)
    bootstrap_path = Path(__file__).resolve(strict=True)
    if bootstrap_path == checkout or checkout in bootstrap_path.parents:
        raise ValueError("production bootstrap must be independently installed")
    interpreter = Path(sys.executable).absolute()
    if interpreter != _PYTHON_EXECUTABLE:
        raise ValueError(
            f"production bootstrap requires trusted interpreter {_PYTHON_EXECUTABLE}"
        )
    site_packages = _PYTHON_SITE_PACKAGES.resolve(strict=True)
    if not site_packages.is_dir() or checkout in site_packages.parents:
        raise ValueError("trusted Python site-packages are unavailable or mutable")
    external_oci_image_id = _oci_image_id(args.external_oci_image_id)
    runtime_identity = {
        "schema": _RUNTIME_IDENTITY_SCHEMA,
        "verification_boundary": "external_container_runtime",
        "external_oci_image_id": external_oci_image_id,
        "python": _executable_identity(_PYTHON_EXECUTABLE, name="Python"),
        "python_site_packages": str(site_packages),
        "git": _executable_identity(_GIT_EXECUTABLE, name="Git"),
        "cc": _executable_identity(_CC_EXECUTABLE, name="C compiler"),
        "cxx": _executable_identity(_CXX_EXECUTABLE, name="C++ compiler"),
        "ninja": _executable_identity(_NINJA_EXECUTABLE, name="Ninja"),
        "cuda_home": str(_CUDA_HOME.resolve(strict=True)),
        "nvcc": _executable_identity(_NVCC_EXECUTABLE, name="NVCC"),
        "path": _TOOLCHAIN_PATH,
    }
    builder_args = list(args.builder_args)
    if not builder_args or builder_args.pop(0) != "--":
        raise ValueError(
            "production builder arguments must follow an explicit -- separator"
        )

    snapshot = snapshot_git_revision(
        checkout,
        expected_revision=args.kquant_revision,
        expected_sha256=args.kquant_source_sha256,
    )
    builder_sha256 = _secure_file_sha256(
        snapshot / "scripts" / "build_fruit_qsrt_model.py"
    )
    context = {
        "schema": _CONTEXT_SCHEMA,
        "kquant_revision": args.kquant_revision,
        "kquant_source_sha256": args.kquant_source_sha256,
        "bootstrap_sha256": bootstrap_sha256,
        "builder_sha256": builder_sha256,
        "runtime": runtime_identity,
        "snapshot_root": str(snapshot),
    }
    runtime_root = Path(tempfile.mkdtemp(prefix="kquant-fruit-runtime-", dir="/tmp"))
    private_directories = {
        name: runtime_root / name
        for name in (
            "home",
            "tmp",
            "xdg-cache",
            "torch-extensions",
            "cuda-cache",
            "triton",
        )
    }
    for directory in private_directories.values():
        directory.mkdir(mode=0o700)
    environment = {
        _BOOTSTRAP_CONTEXT_ENV: json.dumps(
            context, allow_nan=False, separators=(",", ":"), sort_keys=True
        ),
        "CC": str(_CC_EXECUTABLE),
        "CUDA_CACHE_PATH": str(private_directories["cuda-cache"]),
        "CUDA_HOME": runtime_identity["cuda_home"],
        "CXX": str(_CXX_EXECUTABLE),
        "LC_ALL": "C.UTF-8",
        "HF_HUB_OFFLINE": "1",
        "HOME": str(private_directories["home"]),
        "NVCC": str(_NVCC_EXECUTABLE),
        "PATH": _TOOLCHAIN_PATH,
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(private_directories["tmp"]),
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_EXTENSIONS_DIR": str(private_directories["torch-extensions"]),
        "TRANSFORMERS_OFFLINE": "1",
        "TRITON_CACHE_DIR": str(private_directories["triton"]),
        "TZ": "UTC",
        "XDG_CACHE_HOME": str(private_directories["xdg-cache"]),
    }
    arguments = (
        sys.executable,
        "-I",
        "-S",
        "-c",
        _RUNNER,
        str(snapshot),
        str(site_packages),
        str(runtime_root),
        *builder_args,
    )
    try:
        os.execve(sys.executable, arguments, environment)
    except BaseException:
        shutil.rmtree(snapshot)
        shutil.rmtree(runtime_root)
        raise


if __name__ == "__main__":
    main()
