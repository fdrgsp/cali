from unittest.mock import MagicMock, patch

import pytest

from cali.detection._detection_runner import DetectionRunner
from cali.readers import TiffCollectionReader
from cali.sqlmodel import DetectionSettings


def test_caiman_import_error() -> None:
    """Test that ModuleNotFoundError is raised when caiman is not installed."""
    runner = DetectionRunner()
    settings = DetectionSettings(method="caiman")

    mock_dataset = MagicMock(spec=TiffCollectionReader)

    with patch.dict("sys.modules", {"caiman": None}):
        with pytest.raises(
            ModuleNotFoundError, match="CaImAn detection requires the CaImAn package"
        ):
            runner.run(mock_dataset, settings, [])
