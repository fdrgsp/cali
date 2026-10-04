"""Opt-in installed-wheel checks in an environment without inference extras."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("CALI_CASCADE_BASE_TESTS") != "1",
    reason="Requires the dedicated base-wheel environment without Torch/CASCADE",
)


def test_base_wheel_imports_and_reports_missing_cascade(tmp_path: Path) -> None:
    code = textwrap.dedent("""
        import importlib.util
        import sys
        from importlib.metadata import PackageNotFoundError, distribution
        from pathlib import Path

        for name in ('torch', 'cascade2p'):
            assert importlib.util.find_spec(name) is None, name
        for name in ('torch', 'CascadeTorch'):
            try:
                distribution(name)
            except PackageNotFoundError:
                pass
            else:
                raise AssertionError(f'{name} must be absent from this environment')

        import cali
        import cali._cascade_models
        import cali.extraction._spike_inference._cascade_reference
        from cali._cascade_package import load_cascade_package

        assert Path(cali.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert 'torch' not in sys.modules
        assert 'cascade2p' not in sys.modules
        try:
            load_cascade_package()
        except ImportError as error:
            assert 'Install cali[cascade]' in str(error)
        else:
            raise AssertionError('Missing CASCADE must fail without a fallback')
        assert 'torch' not in sys.modules
        assert 'cascade2p' not in sys.modules
    """)
    subprocess.run(
        [sys.executable, "-I", "-Werror", "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
