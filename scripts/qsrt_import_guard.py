"""Authenticate the externally created source snapshot before QSRT imports."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

_BOOTSTRAP_CONTEXT_ENV = "QSRT_FRUIT_BUILDER_BOOTSTRAP_V5"
_CONTEXT_SCHEMA = "qsrt_fruit_builder_bootstrap_v5"
_HEX = frozenset("0123456789abcdef")


def _digest(value: object, *, length: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase {length}-character digest")
    return value


def _secure_sha256(path: Path) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot securely open authenticated builder: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                "authenticated Fruit builder is not a private regular file"
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
            raise ValueError("authenticated Fruit builder changed during bootstrap")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _runtime_executable(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "resolved_path",
        "sha256",
    }:
        raise ValueError(f"Fruit builder runtime {name} identity is invalid")
    path = value.get("path")
    resolved_path = value.get("resolved_path")
    sha256 = _digest(value.get("sha256"), length=64, name=f"runtime {name} SHA-256")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or not isinstance(resolved_path, str)
        or not Path(resolved_path).is_absolute()
        or Path(path).resolve(strict=True) != Path(resolved_path)
        or _secure_sha256(Path(resolved_path)) != sha256
    ):
        raise ValueError(f"Fruit builder runtime {name} executable changed")
    return {"path": path, "resolved_path": resolved_path, "sha256": sha256}


def _runtime_identity(value: object) -> dict[str, object]:
    expected_keys = {
        "schema",
        "verification_boundary",
        "external_oci_image_id",
        "python",
        "python_site_packages",
        "git",
        "cc",
        "cxx",
        "ninja",
        "cuda_home",
        "nvcc",
        "path",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("Fruit builder runtime identity has an invalid schema")
    if value.get("schema") != "qsrt_fruit_builder_external_oci_runtime_v1":
        raise ValueError("Fruit builder runtime identity has an unsupported schema")
    external_oci_image_id = value.get("external_oci_image_id")
    if (
        value.get("verification_boundary") != "external_container_runtime"
        or not isinstance(external_oci_image_id, str)
        or not external_oci_image_id.startswith("sha256:")
    ):
        raise ValueError("Fruit builder external OCI identity is invalid")
    _digest(
        external_oci_image_id.removeprefix("sha256:"),
        length=64,
        name="external OCI image digest",
    )
    runtime: dict[str, object] = {
        "schema": value["schema"],
        "verification_boundary": value["verification_boundary"],
        "external_oci_image_id": external_oci_image_id,
    }
    for name in ("python", "git", "cc", "cxx", "ninja", "nvcc"):
        runtime[name] = _runtime_executable(value.get(name), name=name)
    for name in ("python_site_packages", "cuda_home"):
        path = value.get(name)
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"Fruit builder runtime {name} path is invalid")
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_dir() or str(resolved) != path:
            raise ValueError(f"Fruit builder runtime {name} path changed")
        runtime[name] = path
    toolchain_path = value.get("path")
    if (
        not isinstance(toolchain_path, str)
        or not toolchain_path
        or any(not Path(entry).is_absolute() for entry in toolchain_path.split(":"))
    ):
        raise ValueError("Fruit builder runtime PATH is invalid")
    runtime["path"] = toolchain_path
    return runtime


def authenticate_production_builder(
    builder_file: Path,
) -> tuple[tuple[str, str], Path, dict[str, object], str | None, str | None]:
    """Validate the trusted-launch context before any QSRT module is imported."""

    already_imported = sorted(
        name for name in sys.modules if name == "qsrt" or name.startswith("qsrt.")
    )
    if already_imported:
        raise RuntimeError(
            "QSRT modules were imported before production bootstrap authentication: "
            + ", ".join(already_imported[:3])
        )
    raw_context = os.environ.pop(_BOOTSTRAP_CONTEXT_ENV, None)
    if raw_context is None:
        raise RuntimeError(
            "direct production Fruit builder execution is forbidden; invoke an "
            "externally authenticated standalone bootstrap with trusted Python -I -S"
        )
    try:
        context = json.loads(raw_context)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Fruit builder bootstrap context is malformed") from exc
    expected_keys = {
        "schema",
        "qsrt_revision",
        "qsrt_source_sha256",
        "bootstrap_sha256",
        "builder_sha256",
        "runtime",
        "snapshot_root",
        "runtime_qualification_sha256",
        "rate_sweep_sha256",
    }
    if not isinstance(context, dict) or set(context) != expected_keys:
        raise ValueError("Fruit builder bootstrap context has an invalid schema")
    if context.get("schema") != _CONTEXT_SCHEMA:
        raise ValueError("Fruit builder bootstrap context has an unsupported schema")
    revision = _digest(
        context.get("qsrt_revision"), length=40, name="QSRT revision"
    )
    source_sha256 = _digest(
        context.get("qsrt_source_sha256"),
        length=64,
        name="QSRT source SHA-256",
    )
    bootstrap_sha256 = _digest(
        context.get("bootstrap_sha256"), length=64, name="bootstrap SHA-256"
    )
    builder_sha256 = _digest(
        context.get("builder_sha256"), length=64, name="builder SHA-256"
    )
    runtime_qualification_sha256 = context.get("runtime_qualification_sha256")
    if runtime_qualification_sha256 is not None:
        runtime_qualification_sha256 = _digest(
            runtime_qualification_sha256,
            length=64,
            name="runtime qualification SHA-256",
        )
    rate_sweep_sha256 = context.get("rate_sweep_sha256")
    if rate_sweep_sha256 is not None:
        rate_sweep_sha256 = _digest(
            rate_sweep_sha256,
            length=64,
            name="rate-sweep SHA-256",
        )
    runtime = _runtime_identity(context.get("runtime"))
    python = runtime["python"]
    if not isinstance(python, dict):
        raise TypeError("Fruit builder runtime Python identity is invalid")
    if (
        str(Path(sys.executable).absolute()) != python["path"]
        or not sys.flags.isolated
        or not sys.flags.no_site
    ):
        raise ValueError("Fruit builder lost trusted Python isolation during re-exec")
    expected_environment = {
        "CC": runtime["cc"]["path"],
        "CUDA_HOME": runtime["cuda_home"],
        "CXX": runtime["cxx"]["path"],
        "HF_HUB_OFFLINE": "1",
        "LC_ALL": "C.UTF-8",
        "NVCC": runtime["nvcc"]["path"],
        "PATH": runtime["path"],
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "TZ": "UTC",
    }
    for name, expected in expected_environment.items():
        if os.environ.get(name) != expected:
            raise ValueError(f"Fruit builder canonical environment changed: {name}")
    private_names = {
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR",
    }
    allowed_environment = set(expected_environment) | private_names
    if set(os.environ) != allowed_environment:
        raise ValueError("Fruit builder inherited a non-canonical environment")
    private_roots: set[Path] = set()
    for name in private_names:
        directory = Path(os.environ[name]).resolve(strict=True)
        status = directory.stat()
        if (
            not directory.is_dir()
            or status.st_uid != os.getuid()
            or status.st_mode & 0o077
            or any(directory.iterdir())
        ):
            raise ValueError(f"Fruit builder private environment is unsafe: {name}")
        private_roots.add(directory.parent)
    if len(private_roots) != 1:
        raise ValueError("Fruit builder private environment roots diverge")
    snapshot_value = context.get("snapshot_root")
    if not isinstance(snapshot_value, str) or not snapshot_value:
        raise ValueError("Fruit builder bootstrap snapshot root is invalid")
    snapshot_root = Path(snapshot_value).resolve(strict=True)
    if not snapshot_root.is_dir():
        raise NotADirectoryError(snapshot_root)
    root_status = snapshot_root.stat()
    if (
        root_status.st_uid != os.getuid()
        or not stat.S_ISDIR(root_status.st_mode)
        or root_status.st_mode & 0o077
    ):
        raise ValueError("Fruit builder bootstrap snapshot is not process-private")
    authenticated_builder = builder_file.resolve(strict=True)
    expected_builder = snapshot_root / "scripts" / "build_fruit_qsrt_model.py"
    authenticated_guard = Path(__file__).resolve(strict=True)
    expected_guard = snapshot_root / "scripts" / "qsrt_import_guard.py"
    if (
        authenticated_builder != expected_builder
        or authenticated_guard != expected_guard
        or (snapshot_root / ".git").exists()
    ):
        raise ValueError("Fruit production builder is not executing from its snapshot")
    if _secure_sha256(authenticated_builder) != builder_sha256:
        raise ValueError(
            "Fruit production builder does not match its bootstrap identity"
        )
    bootstrap = {
        "schema": "qsrt_fruit_builder_bootstrap_identity_v5",
        "bootstrap_sha256": bootstrap_sha256,
        "builder_sha256": builder_sha256,
        "qsrt_revision": revision,
        "qsrt_source_sha256": source_sha256,
        "rate_sweep_authority": "external_sha256",
        "runtime_qualification_authority": "external_sha256",
        "runtime": runtime,
    }
    return (
        (revision, source_sha256),
        snapshot_root,
        bootstrap,
        runtime_qualification_sha256,
        rate_sweep_sha256,
    )


def _capture_legacy_import_snapshot() -> tuple[tuple[str, str], object, Path]:
    """Retain non-production module/test imports without authorizing a builder."""

    from scripts.tracked_worktree import snapshot_tracked_worktree

    root = Path(__file__).resolve().parents[1]
    already_imported = sorted(
        name for name in sys.modules if name == "qsrt" or name.startswith("qsrt.")
    )
    if already_imported:
        raise RuntimeError(
            "QSRT modules were imported before source snapshot creation: "
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


if _BOOTSTRAP_CONTEXT_ENV in os.environ:
    QSRT_IMPORT_IDENTITY = None
    _QSRT_IMPORT_SNAPSHOT = None
    QSRT_IMPORT_SOURCE_ROOT = Path(__file__).resolve().parents[1]
else:
    (
        QSRT_IMPORT_IDENTITY,
        _QSRT_IMPORT_SNAPSHOT,
        QSRT_IMPORT_SOURCE_ROOT,
    ) = _capture_legacy_import_snapshot()
