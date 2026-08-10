from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.tracked_worktree import tracked_worktree_sha256


def _commit_checkout(root: Path) -> str:
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(("git", "add", "."), cwd=root, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=QSRT Test",
            "-c",
            "user.email=qsrt@example.invalid",
            "commit",
            "-qm",
            "trusted builder",
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


def _bootstrap_fixture(tmp_path: Path) -> tuple[Path, Path, str, str, str]:
    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    builder = scripts / "build_fruit_qsrt_model.py"
    builder.write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "output = Path(sys.argv[1])\n"
        "output.write_text('committed snapshot\\n', encoding='utf-8')\n"
        "output.with_suffix('.env.json').write_text("
        "json.dumps(dict(os.environ), sort_keys=True), encoding='utf-8')\n",
        encoding="utf-8",
    )
    (scripts / "qsrt_import_guard.py").write_text(
        "# committed guard\n", encoding="utf-8"
    )
    revision = _commit_checkout(checkout)
    source_sha256 = tracked_worktree_sha256(checkout)

    trusted_bootstrap = tmp_path / "installed" / "fruit_builder_bootstrap.py"
    trusted_bootstrap.parent.mkdir()
    bootstrap_source = (
        Path(__file__).parents[1] / "scripts" / "fruit_builder_bootstrap.py"
    ).read_text(encoding="utf-8")
    cuda_home = tmp_path / "runtime-image" / "usr" / "local" / "cuda"
    (cuda_home / "bin").mkdir(parents=True)
    nvcc = cuda_home / "bin" / "nvcc"
    shutil.copy2(sys.executable, nvcc)
    site_packages = tmp_path / "runtime-image" / "opt" / "venv" / "site-packages"
    site_packages.mkdir(parents=True)
    python = Path(sys.executable).absolute()
    git = Path(shutil.which("git") or "/usr/bin/git").resolve()
    toolchain_path = f"{python.parent}:{cuda_home / 'bin'}:{git.parent}"
    replacements = {
        '_PYTHON_EXECUTABLE = Path("/opt/venv/bin/python")': f"_PYTHON_EXECUTABLE = Path({str(python)!r})",
        '_PYTHON_SITE_PACKAGES = Path("/opt/venv/lib/python3.12/site-packages")': f"_PYTHON_SITE_PACKAGES = Path({str(site_packages)!r})",
        '_GIT_EXECUTABLE = Path("/usr/bin/git")': f"_GIT_EXECUTABLE = Path({str(git)!r})",
        '_CC_EXECUTABLE = Path("/usr/bin/gcc")': f"_CC_EXECUTABLE = Path({str(python)!r})",
        '_CXX_EXECUTABLE = Path("/usr/bin/g++")': f"_CXX_EXECUTABLE = Path({str(python)!r})",
        '_NINJA_EXECUTABLE = Path("/opt/venv/bin/ninja")': f"_NINJA_EXECUTABLE = Path({str(python)!r})",
        '_CUDA_HOME = Path("/usr/local/cuda")': f"_CUDA_HOME = Path({str(cuda_home)!r})",
        '_TOOLCHAIN_PATH = "/opt/venv/bin:/usr/local/cuda/bin:/usr/bin"': f"_TOOLCHAIN_PATH = {toolchain_path!r}",
    }
    for original, replacement in replacements.items():
        assert original in bootstrap_source
        bootstrap_source = bootstrap_source.replace(original, replacement)
    trusted_bootstrap.write_text(bootstrap_source, encoding="utf-8")
    bootstrap_sha256 = hashlib.sha256(trusted_bootstrap.read_bytes()).hexdigest()
    return checkout, trusted_bootstrap, revision, source_sha256, bootstrap_sha256


def _launch(
    trusted_bootstrap: Path,
    checkout: Path,
    revision: str,
    source_sha256: str,
    bootstrap_sha256: str,
    output: Path,
    *,
    environment: dict[str, str] | None = None,
    external_oci_image_id: str = "sha256:" + "f" * 64,
    candidate_only: bool = True,
    rate_sweep_only: bool = False,
    runtime_qualification_sha256: str | None = None,
    rate_sweep_sha256: str | None = (
        "95d3cb9f5dc66ec7615497d07ec0e42281594bfb20649e23b2c6dcbff34406f6"
    ),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            str(trusted_bootstrap),
            "--qsrt-root",
            str(checkout),
            "--qsrt-revision",
            revision,
            "--qsrt-source-sha256",
            source_sha256,
            "--bootstrap-sha256",
            bootstrap_sha256,
            "--external-oci-image-id",
            external_oci_image_id,
            *(
                ("--rate-sweep-sha256", rate_sweep_sha256)
                if rate_sweep_sha256 is not None
                else ()
            ),
            *(
                (
                    "--runtime-qualification-sha256",
                    runtime_qualification_sha256,
                )
                if runtime_qualification_sha256 is not None
                else ()
            ),
            "--",
            str(output),
            *(
                ("--rate-sweep-only",)
                if rate_sweep_only
                else (
                    ("--candidate-only",)
                    if candidate_only
                    else ("--runtime-qualification", "runtime-qualification.json")
                )
            ),
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_authenticated_bootstrap_executes_committed_builder_not_restoring_live_code(
    tmp_path: Path,
) -> None:
    checkout, trusted_bootstrap, revision, source_sha256, bootstrap_sha256 = (
        _bootstrap_fixture(tmp_path)
    )
    live_builder = checkout / "scripts" / "build_fruit_qsrt_model.py"
    live_builder.write_text(
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n"
        "restored = subprocess.run("
        "('git', 'show', 'HEAD:scripts/build_fruit_qsrt_model.py'), "
        "check=True, capture_output=True, cwd=Path(__file__).parents[1]).stdout\n"
        "Path(__file__).write_bytes(restored)\n"
        "Path(sys.argv[1]).write_text('malicious live tree\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    output = tmp_path / "artifact-write.txt"

    result = _launch(
        trusted_bootstrap,
        checkout,
        revision,
        source_sha256,
        bootstrap_sha256,
        output,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == "committed snapshot\n"
    assert "malicious live tree" in live_builder.read_text(encoding="utf-8")


def test_authenticated_bootstrap_requires_stage_specific_qualification_authority(
    tmp_path: Path,
) -> None:
    checkout, trusted_bootstrap, revision, source_sha256, bootstrap_sha256 = (
        _bootstrap_fixture(tmp_path)
    )
    output = tmp_path / "artifact-write.txt"
    authority = "a" * 64

    missing_runtime_authority = _launch(
        trusted_bootstrap,
        checkout,
        revision,
        source_sha256,
        bootstrap_sha256,
        output,
        candidate_only=False,
    )
    assert missing_runtime_authority.returncode != 0
    assert (
        "requires an external runtime qualification SHA-256"
        in missing_runtime_authority.stderr
    )
    assert not output.exists()

    missing_rate_authority = _launch(
        trusted_bootstrap,
        checkout,
        revision,
        source_sha256,
        bootstrap_sha256,
        output,
        rate_sweep_sha256=None,
    )
    assert missing_rate_authority.returncode != 0
    assert "requires an external rate-sweep SHA-256" in missing_rate_authority.stderr
    assert not output.exists()

    arbitrary_rate_output = tmp_path / "arbitrary-rate-authority.txt"
    arbitrary_rate_authority = _launch(
        trusted_bootstrap,
        checkout,
        revision,
        source_sha256,
        bootstrap_sha256,
        arbitrary_rate_output,
        rate_sweep_sha256="c" * 64,
    )
    assert arbitrary_rate_authority.returncode == 0, arbitrary_rate_authority.stderr
    assert arbitrary_rate_output.read_text(encoding="utf-8") == "committed snapshot\n"

    candidate = _launch(
        trusted_bootstrap,
        checkout,
        revision,
        source_sha256,
        bootstrap_sha256,
        output,
        runtime_qualification_sha256=authority,
    )
    assert candidate.returncode != 0
    assert "must not accept a runtime qualification anchor" in candidate.stderr
    assert not output.exists()

    final = _launch(
        trusted_bootstrap,
        checkout,
        revision,
        source_sha256,
        bootstrap_sha256,
        output,
        candidate_only=False,
        runtime_qualification_sha256=authority,
    )
    assert final.returncode == 0, final.stderr
    assert output.read_text(encoding="utf-8") == "committed snapshot\n"

    sweep_output = tmp_path / "rate-sweep.txt"
    sweep = _launch(
        trusted_bootstrap,
        checkout,
        revision,
        source_sha256,
        bootstrap_sha256,
        sweep_output,
        rate_sweep_only=True,
        rate_sweep_sha256=None,
    )
    assert sweep.returncode == 0, sweep.stderr
    assert sweep_output.read_text(encoding="utf-8") == "committed snapshot\n"


@pytest.mark.parametrize(
    ("revision", "source_sha256", "bootstrap_sha256"),
    (
        ("0" * 40, None, None),
        (None, "0" * 64, None),
        (None, None, "0" * 64),
    ),
)
def test_authenticated_bootstrap_rejects_wrong_external_anchors_before_writes(
    tmp_path: Path,
    revision: str | None,
    source_sha256: str | None,
    bootstrap_sha256: str | None,
) -> None:
    checkout, trusted_bootstrap, actual_revision, actual_source, actual_bootstrap = (
        _bootstrap_fixture(tmp_path)
    )
    output = tmp_path / "artifact-write.txt"

    result = _launch(
        trusted_bootstrap,
        checkout,
        revision or actual_revision,
        source_sha256 or actual_source,
        bootstrap_sha256 or actual_bootstrap,
        output,
    )

    assert result.returncode != 0
    assert not output.exists()


def test_authenticated_bootstrap_rejects_missing_anchors_before_writes(
    tmp_path: Path,
) -> None:
    _, trusted_bootstrap, _, _, _ = _bootstrap_fixture(tmp_path)
    output = tmp_path / "artifact-write.txt"

    result = subprocess.run(
        (sys.executable, "-I", "-S", str(trusted_bootstrap), "--", str(output)),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "requires all external QSRT trust anchors" in result.stderr
    assert not output.exists()


def test_authenticated_bootstrap_rejects_mutable_or_nondigest_image_identity(
    tmp_path: Path,
) -> None:
    checkout, trusted_bootstrap, revision, source_sha256, bootstrap_sha256 = (
        _bootstrap_fixture(tmp_path)
    )
    output = tmp_path / "artifact-write.txt"

    result = _launch(
        trusted_bootstrap,
        checkout,
        revision,
        source_sha256,
        bootstrap_sha256,
        output,
        external_oci_image_id="registry.example/qsrt:latest",
    )

    assert result.returncode != 0
    assert "must use immutable sha256:<digest>" in result.stderr
    assert not output.exists()


def test_standalone_bootstrap_rejects_nonisolated_python_before_writes(
    tmp_path: Path,
) -> None:
    _, trusted_bootstrap, _, _, _ = _bootstrap_fixture(tmp_path)
    output = tmp_path / "artifact-write.txt"

    result = subprocess.run(
        (sys.executable, str(trusted_bootstrap), "--show-bootstrap-sha256"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "requires direct trusted-python invocation with -I -S" in result.stderr
    assert not output.exists()


def test_isolated_standalone_bootstrap_does_not_execute_site_hooks(
    tmp_path: Path,
) -> None:
    _, trusted_bootstrap, _, _, _ = _bootstrap_fixture(tmp_path)
    hook_root = tmp_path / "site-hook"
    hook_root.mkdir()
    hook_output = tmp_path / "site-hook-executed"
    (hook_root / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(hook_output)!r}).touch()\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(hook_root)

    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            str(trusted_bootstrap),
            "--show-bootstrap-sha256",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip()) == 64
    assert not hook_output.exists()


def test_bootstrap_replaces_hostile_compiler_jit_cache_and_python_environment(
    tmp_path: Path,
) -> None:
    checkout, trusted_bootstrap, revision, source_sha256, bootstrap_sha256 = (
        _bootstrap_fixture(tmp_path)
    )
    output = tmp_path / "artifact-write.txt"
    hostile = os.environ.copy()
    hostile.update(
        {
            "CC": "/hostile/cc",
            "CXX": "/hostile/cxx",
            "NVCC": "/hostile/nvcc",
            "CUDA_HOME": "/hostile/cuda",
            "PATH": "/hostile/bin",
            "HOME": "/hostile/home",
            "PYTHONPATH": "/hostile/site",
            "TORCH_EXTENSIONS_DIR": "/hostile/torch",
            "TRITON_CACHE_DIR": "/hostile/triton",
            "CUDA_CACHE_PATH": "/hostile/cuda-cache",
            "XDG_CACHE_HOME": "/hostile/xdg",
        }
    )

    result = _launch(
        trusted_bootstrap,
        checkout,
        revision,
        source_sha256,
        bootstrap_sha256,
        output,
        environment=hostile,
    )

    assert result.returncode == 0, result.stderr
    launched_environment = json.loads(
        output.with_suffix(".env.json").read_text(encoding="utf-8")
    )
    assert launched_environment["CC"] == str(Path(sys.executable).absolute())
    assert launched_environment["CXX"] == str(Path(sys.executable).absolute())
    assert launched_environment["NVCC"].endswith("/usr/local/cuda/bin/nvcc")
    assert launched_environment["PATH"] != hostile["PATH"]
    assert "PYTHONPATH" not in launched_environment
    for name in (
        "HOME",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
    ):
        assert launched_environment[name].startswith("/tmp/qsrt-fruit-runtime-")
        assert not launched_environment[name].startswith("/hostile/")


def test_authenticated_bootstrap_rejects_mutated_installed_launcher_before_writes(
    tmp_path: Path,
) -> None:
    checkout, trusted_bootstrap, revision, source_sha256, bootstrap_sha256 = (
        _bootstrap_fixture(tmp_path)
    )
    trusted_bootstrap.write_text(
        trusted_bootstrap.read_text(encoding="utf-8") + "\n# mutation\n",
        encoding="utf-8",
    )
    output = tmp_path / "artifact-write.txt"

    result = _launch(
        trusted_bootstrap,
        checkout,
        revision,
        source_sha256,
        bootstrap_sha256,
        output,
    )

    assert result.returncode != 0
    assert "does not match its external anchor" in result.stderr
    assert not output.exists()


def test_checkout_local_bootstrap_is_not_a_production_trust_boundary(
    tmp_path: Path,
) -> None:
    checkout, trusted_bootstrap, revision, source_sha256, _ = _bootstrap_fixture(
        tmp_path
    )
    live_bootstrap = checkout / "fruit_builder_bootstrap.py"
    live_bootstrap.write_bytes(trusted_bootstrap.read_bytes())
    live_digest = hashlib.sha256(live_bootstrap.read_bytes()).hexdigest()
    output = tmp_path / "artifact-write.txt"

    result = _launch(
        live_bootstrap,
        checkout,
        revision,
        source_sha256,
        live_digest,
        output,
    )

    assert result.returncode != 0
    assert "must be independently installed" in result.stderr
    assert not output.exists()


def test_live_production_builder_without_bootstrap_context_fails_before_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact-output"
    builder = Path(__file__).parents[1] / "scripts" / "build_fruit_qsrt_model.py"

    result = subprocess.run(
        (sys.executable, str(builder), str(tmp_path / "base"), str(output)),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "direct production Fruit builder execution is forbidden" in result.stderr
    assert not output.exists()
