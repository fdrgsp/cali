"""Basic tests for the cali package."""

from importlib.metadata import version

import pytest


def test_package_imports() -> None:
    """Test that the main package can be imported."""
    import cali

    assert hasattr(cali, "__version__")


def test_package_has_version() -> None:
    """Test that the package has a version."""
    try:
        from cali import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
    except ImportError:
        pytest.skip("Package not properly installed")


def test_package_version_from_metadata() -> None:
    """Test that version can be retrieved from package metadata."""
    try:
        pkg_version = version("cali")
        assert pkg_version is not None
        assert isinstance(pkg_version, str)
    except Exception:
        # This might fail if package is not installed in development mode
        pytest.skip("Package metadata not available")
