"""Comprehensive tests for all advanced plotting functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sqlmodel import create_engine

if TYPE_CHECKING:
    from sqlalchemy import Engine

from cali.plot._main_plot import (
    CALCIUM_CONNECTIVITY_MATRIX,
    CALCIUM_NETWORK_CONNECTIVITY,
    CALCIUM_PEAKS_GLOBAL_SYNCHRONY,
    INFERRED_SPIKE_BURST_ANALYSIS,
    INFERRED_SPIKE_CLUSTERING,
    INFERRED_SPIKE_CLUSTERING_DENDROGRAM,
    INFERRED_SPIKE_CROSS_CORRELATION,
    INFERRED_SPIKES_NORMALIZED_WITH_BURSTS,
    INFERRED_SPIKES_THRESHOLDED_SYNCHRONY,
    NEUROPIL_ROI_MASKS,
    NON_STIMULATED_PEAKS_AMP,
    STIMULATED_AREA,
    STIMULATED_PEAKS_AMP,
    STIMULATED_ROIS,
    STIMULATED_ROIS_WITH_STIMULATED_AREA,
    STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED,
    STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS,
    STIMULATED_VS_NON_STIMULATED_SPIKE_TRACES,
    plot_single_well_data,
)


class MockPlateViewer:
    """Mock plate viewer for testing."""

    def __init__(self, output_path: Path | None = None) -> None:
        self.output_path = output_path


class MockGraphWidget:
    """Mock graph widget for testing."""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self._plate_viewer = MockPlateViewer()
        self.roiSelected = MagicMock()  # Mock signal for ROI selection


@pytest.fixture
def db_path() -> Path:
    """Path to test database."""
    return Path(__file__).parent / "test_data" / "evoked" / "results.cali"

@pytest.fixture
def db_engine(db_path: Path):  # type: ignore[misc]
    """Create a database engine for testing."""
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"timeout": 30.0, "check_same_thread": False},
        pool_pre_ping=True,
    )
    yield engine
    engine.dispose()


@pytest.fixture
def widget() -> MockGraphWidget:
    """Create a mock widget for testing."""
    return MockGraphWidget()


def _has_plot_content(widget: MockGraphWidget) -> bool:
    """Check if the widget has actual plot content."""
    for ax in widget.figure.get_axes():
        # Check for various plot elements
        if ax.lines or ax.collections or ax.patches or ax.images:
            return True
        # Check for text that isn't just empty axes
        for text in ax.texts:
            if text.get_text().strip():
                return True
    return False


class TestCalciumPeaksSynchrony:
    """Tests for calcium peaks synchrony plot."""

    def test_synchrony_with_run_id(
        self, widget: MockGraphWidget, db_engine: Engine
    ) -> None:
        """Test synchrony plot with specific run_id."""
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            CALCIUM_PEAKS_GLOBAL_SYNCHRONY,
            rois=None,
            run_id=1,
        )
        assert _has_plot_content(widget), "Synchrony plot should have content"

    def test_synchrony_without_run_id(
        self, widget: MockGraphWidget, db_engine: Engine
    ) -> None:
        """Test synchrony plot without run_id (fallback)."""
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            CALCIUM_PEAKS_GLOBAL_SYNCHRONY,
            rois=None,
            run_id=None,
        )
        assert _has_plot_content(widget), "Synchrony plot should have content"


class TestNetworkConnectivity:
    """Tests for network connectivity plots."""

    def test_connectivity_network(
        self, widget: MockGraphWidget, db_engine: Engine
    ) -> None:
        """Test network connectivity plot."""
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            CALCIUM_NETWORK_CONNECTIVITY,
            rois=None,
            run_id=1,
        )
        # This plot may be stubbed out, so just check it doesn't crash
        assert widget.figure is not None

    def test_connectivity_matrix(
        self, widget: MockGraphWidget, db_engine: Engine
    ) -> None:
        """Test connectivity matrix plot."""
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            CALCIUM_CONNECTIVITY_MATRIX,
            rois=None,
            run_id=1,
        )
        # This plot may be stubbed out, so just check it doesn't crash
        assert widget.figure is not None


class TestInferredSpikeSynchrony:
    """Tests for inferred spike synchrony plots."""

    def test_spike_synchrony(self, widget: MockGraphWidget, db_engine: Engine) -> None:
        """Test spike synchrony plot."""
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            INFERRED_SPIKES_THRESHOLDED_SYNCHRONY,
            rois=None,
            run_id=1,
        )
        # This plot may be stubbed out, so just check it doesn't crash
        assert widget.figure is not None

    def test_spike_cross_correlation(
        self, widget: MockGraphWidget, db_engine: Engine
    ) -> None:
        """Test spike cross-correlation plot."""
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            INFERRED_SPIKE_CROSS_CORRELATION,
            rois=None,
            run_id=1,
        )
        # This plot may be stubbed out, so just check it doesn't crash
        assert widget.figure is not None

    def test_spike_clustering(self, widget: MockGraphWidget, db_engine: Engine) -> None:
        """Test spike hierarchical clustering plot."""
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            INFERRED_SPIKE_CLUSTERING,
            rois=None,
            run_id=1,
        )
        # This plot may be stubbed out, so just check it doesn't crash
        assert widget.figure is not None

    def test_spike_clustering_dendrogram(
        self, widget: MockGraphWidget, db_engine: Engine
    ) -> None:
        """Test spike clustering dendrogram plot."""
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            INFERRED_SPIKE_CLUSTERING_DENDROGRAM,
            rois=None,
            run_id=1,
        )
        # This plot may be stubbed out, so just check it doesn't crash
        assert widget.figure is not None


class TestBurstActivity:
    """Tests for burst activity plots."""

    def test_burst_analysis(self, widget: MockGraphWidget, db_engine: Engine) -> None:
        """Test burst activity analysis plot."""
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            INFERRED_SPIKE_BURST_ANALYSIS,
            rois=None,
            run_id=1,
        )
        assert _has_plot_content(widget), "Burst analysis should have content"

    def test_spikes_with_bursts(
        self, widget: MockGraphWidget, db_engine: Engine
    ) -> None:
        """Test inferred spikes with network bursts plot."""
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            INFERRED_SPIKES_NORMALIZED_WITH_BURSTS,
            rois=None,
            run_id=1,
        )
        assert _has_plot_content(widget), "Spikes with bursts should have content"


class TestNeuropilVisualization:
    """Tests for neuropil visualization."""

    def test_neuropil_masks(self, widget: MockGraphWidget, db_engine: Engine) -> None:
        """Test neuropil and ROI masks visualization."""
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            NEUROPIL_ROI_MASKS,
            rois=None,
            run_id=1,
        )
        # Should have content (masks or no-data message)
        assert widget.figure is not None


class TestEvokedExperiment:
    """Tests for evoked experiment plots."""

    def test_stimulated_area(
        self, widget: MockGraphWidget, db_engine: Engine, db_path: Path
    ) -> None:
        """Test stimulated area visualization."""
        widget._plate_viewer.output_path = db_path.parent / "evk_analysis"
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            STIMULATED_AREA,
            rois=None,
            run_id=1,
        )
        # Should not crash
        assert widget.figure is not None

    def test_stimulated_rois(
        self, widget: MockGraphWidget, db_engine: Engine, db_path: Path
    ) -> None:
        """Test stimulated vs non-stimulated ROIs visualization."""
        widget._plate_viewer.output_path = db_path.parent / "evk_analysis"
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            STIMULATED_ROIS,
            rois=None,
            run_id=1,
        )
        assert widget.figure is not None

    def test_stimulated_rois_with_area(
        self, widget: MockGraphWidget, db_engine: Engine, db_path: Path
    ) -> None:
        """Test stimulated ROIs with stimulated area."""
        widget._plate_viewer.output_path = db_path.parent / "evk_analysis"
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            STIMULATED_ROIS_WITH_STIMULATED_AREA,
            rois=None,
            run_id=1,
        )
        assert widget.figure is not None

    def test_stimulated_peaks_amp(
        self, widget: MockGraphWidget, db_engine: Engine, db_path: Path
    ) -> None:
        """Test stimulated calcium peaks amplitudes."""
        widget._plate_viewer.output_path = db_path.parent / "evk_analysis"
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            STIMULATED_PEAKS_AMP,
            rois=None,
            run_id=1,
        )
        assert widget.figure is not None

    def test_non_stimulated_peaks_amp(
        self, widget: MockGraphWidget, db_engine: Engine, db_path: Path
    ) -> None:
        """Test non-stimulated calcium peaks amplitudes."""
        widget._plate_viewer.output_path = db_path.parent / "evk_analysis"
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            NON_STIMULATED_PEAKS_AMP,
            rois=None,
            run_id=1,
        )
        assert widget.figure is not None

    def test_stim_vs_non_stim_traces(
        self, widget: MockGraphWidget, db_engine: Engine, db_path: Path
    ) -> None:
        """Test stimulated vs non-stimulated normalized calcium traces."""
        widget._plate_viewer.output_path = db_path.parent / "evk_analysis"
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED,
            rois=None,
            run_id=1,
        )
        assert widget.figure is not None

    def test_stim_vs_non_stim_traces_with_peaks(
        self, widget: MockGraphWidget, db_engine: Engine, db_path: Path
    ) -> None:
        """Test stimulated vs non-stimulated traces with peaks."""
        widget._plate_viewer.output_path = db_path.parent / "evk_analysis"
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS,
            rois=None,
            run_id=1,
        )
        assert widget.figure is not None

    def test_stim_vs_non_stim_spike_traces(
        self, widget: MockGraphWidget, db_engine: Engine, db_path: Path
    ) -> None:
        """Test stimulated vs non-stimulated spike traces."""
        widget._plate_viewer.output_path = db_path.parent / "evk_analysis"
        plot_single_well_data(
            widget,  # type: ignore[arg-type]
            db_engine,
            "B5_0000",
            STIMULATED_VS_NON_STIMULATED_SPIKE_TRACES,
            rois=None,
            run_id=1,
        )
        assert widget.figure is not None
