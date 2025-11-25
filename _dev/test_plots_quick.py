"""Quick script to test which advanced plots work."""

import sys
sys.path.insert(0, "/Users/fdrgsp/Documents/git/cali/src")

import matplotlib
matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from unittest.mock import MagicMock
from pathlib import Path

class MockPlateViewer:
    def __init__(self):
        self.output_path = Path("/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis")

class MockGraphWidget:
    def __init__(self):
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self._plate_viewer = MockPlateViewer()
        self.roiSelected = MagicMock()

from cali.plot._main_plot import (
    CALCIUM_PEAKS_GLOBAL_SYNCHRONY,
    INFERRED_SPIKE_BURST_ANALYSIS,
    INFERRED_SPIKES_NORMALIZED_WITH_BURSTS,
    NEUROPIL_ROI_MASKS,
    plot_single_well_data,
)

db_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/results.cali"
fov_name = "B5_0000"

plots_to_test = [
    ("Calcium Peaks Synchrony", CALCIUM_PEAKS_GLOBAL_SYNCHRONY),
    ("Inferred Spike Burst Analysis", INFERRED_SPIKE_BURST_ANALYSIS),
    ("Spikes with Bursts", INFERRED_SPIKES_NORMALIZED_WITH_BURSTS),
    ("Neuropil ROI Masks", NEUROPIL_ROI_MASKS),
]

for name, plot_name in plots_to_test:
    widget = MockGraphWidget()
    try:
        plot_single_well_data(widget, db_path, fov_name, plot_name, rois=None, run_id=1)
        axes = widget.figure.get_axes()
        if axes:
            ax = axes[0]
            has_content = bool(ax.lines or ax.collections or ax.patches or ax.images)
            print(f"✓ {name}: {has_content} (axes: {len(axes)})")
        else:
            print(f"✗ {name}: No axes")
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}: {e}")
