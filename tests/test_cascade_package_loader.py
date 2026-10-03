"""Optional dependency failures and exact source provenance checks."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from cali import _cascade_package as loader

if TYPE_CHECKING:
    from pathlib import Path

    pass


@pytest.fixture
def package_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    contents = b'"""Generated package stand-in."""\n'
    source = tmp_path / "__init__.py"
    source.write_bytes(contents)
    monkeypatch.setattr(
        loader,
        "_SOURCE_SHA256",
        {
            "__init__.py": hashlib.sha256(contents).hexdigest(),
        },
    )
    monkeypatch.setattr(loader, "version", lambda name: "2.0")
    modules = {
        name: ModuleType(name)
        for name in (
            "cascade2p",
            "cascade2p.cascade",
            "cascade2p.config",
            "cascade2p.utils",
            "torch",
        )
    }
    modules["cascade2p"].__file__ = str(source)
    for module, names in (
        ("cascade2p.cascade", ["predict"]),
        ("cascade2p.config", ["read_config"]),
        ("cascade2p.utils", ["define_model", "calculate_noise_levels"]),
    ):
        for name in names:
            setattr(modules[module], name, lambda: None)
    calls: list[str] = []

    def import_package(name: str) -> ModuleType:
        calls.append(name)
        return modules[name]

    monkeypatch.setattr(loader, "import_module", import_package)
    return source, modules, calls


def test_module_import_never_requires_optional_dependencies(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; import cali._cascade_package; "
                "assert 'torch' not in sys.modules; "
                "assert 'cascade2p' not in sys.modules"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_missing_distribution_reports_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(loader, "version", missing)
    with pytest.raises(ImportError, match=r"Install cali\[cascade\]"):
        loader.load_cascade_package()


@pytest.mark.parametrize("version", ["1.0", "3.0"])
def test_incompatible_package_version_fails_before_heavy_imports(
    package_modules: tuple,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    _, _, calls = package_modules
    monkeypatch.setattr(loader, "version", lambda name: version)
    with pytest.raises(ImportError, match="pinned package"):
        loader.load_cascade_package()
    assert calls == ["cascade2p"]


def test_missing_init_package_is_rejected(package_modules: tuple) -> None:
    _, modules, calls = package_modules
    modules["cascade2p"].__file__ = None
    with pytest.raises(ImportError, match="pinned package"):
        loader.load_cascade_package()
    assert calls == ["cascade2p"]


def test_modified_source_is_rejected_before_torch_import(
    package_modules: tuple,
) -> None:
    source, _, calls = package_modules
    source.write_bytes(b"# Different inference implementation\n")
    with pytest.raises(ImportError, match="differs from pinned commit"):
        loader.load_cascade_package()
    assert calls == ["cascade2p"]


def test_incomplete_wheel_is_reported_without_fallback(package_modules: tuple) -> None:
    source, _, calls = package_modules
    source.unlink()
    with pytest.raises(ImportError, match="installed package is incomplete"):
        loader.load_cascade_package()
    assert calls == ["cascade2p"]


def test_windows_git_line_endings_preserve_the_same_source_commit(
    package_modules: tuple,
) -> None:
    source, modules, _ = package_modules
    source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    package = loader.load_cascade_package()
    assert package.cascade is modules["cascade2p.cascade"]
    assert package.package_revision == loader.CASCADE_PACKAGE_REVISION
    assert len(package.source_manifest_sha256) == 64


def test_missing_upstream_api_is_actionable(package_modules: tuple) -> None:
    _, modules, _ = package_modules
    del modules["cascade2p.cascade"].predict
    with pytest.raises(ImportError, match="upstream inference API is missing"):
        loader.load_cascade_package()


def test_missing_transitive_dependency_is_actionable(
    package_modules: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, modules, _ = package_modules

    def missing_torch(name: str) -> ModuleType:
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return modules[name]

    monkeypatch.setattr(loader, "import_module", missing_torch)
    with pytest.raises(ImportError, match="required dependency is unavailable"):
        loader.load_cascade_package()
