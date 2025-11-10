from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from typing_extensions import TypeAlias

from cali._plate_viewer._util import ROIData
from cali.cali_logger import LOGGER

from ._multi_wells_plots._csv_bar_plot import plot_csv_bar_plot
from ._single_wells_plots._plolt_evoked_experiment_data_plots import (
    _plot_stim_or_not_stim_peaks_amplitude,
    _plot_stimulated_vs_non_stimulated_roi_amp,
    _plot_stimulated_vs_non_stimulated_spike_traces,
    _visualize_stimulated_area,
)
from ._single_wells_plots._plot_calcium_amplitudes_and_frequencies_data import (
    _plot_amplitude_and_frequency_data,
)
from ._single_wells_plots._plot_calcium_network_connectivity import (
    _plot_connectivity_matrix_data,
    _plot_connectivity_network_data,
)
from ._single_wells_plots._plot_calcium_peaks_correlation import (
    _plot_cross_correlation_data,
    _plot_hierarchical_clustering_data,
)
from ._single_wells_plots._plot_calcium_peaks_iei_data import _plot_iei_data
from ._single_wells_plots._plot_calcium_peaks_raster_plots import _generate_raster_plot
from ._single_wells_plots._plot_calcium_peaks_synchrony import (
    _plot_peak_event_synchrony_data,
)
from ._single_wells_plots._plot_calcium_traces_data import _plot_traces_data
from ._single_wells_plots._plot_cell_size import _plot_cell_size_data
from ._single_wells_plots._plot_inferred_spike_burst_activity import (
    _plot_inferred_spike_burst_activity,
)
from ._single_wells_plots._plot_inferred_spike_correlation import (
    _plot_spike_cross_correlation_data,
    _plot_spike_hierarchical_clustering_data,
)
from ._single_wells_plots._plot_inferred_spike_raster_plots import (
    _generate_spike_raster_plot,
)
from ._single_wells_plots._plot_inferred_spike_synchrony import (
    _plot_spike_synchrony_data,
)
from ._single_wells_plots._plot_inferred_spikes import (
    _plot_inferred_spikes,
    _plot_inferred_spikes_normalized_with_bursts,
)
from ._single_wells_plots._plot_neuropil_traces import (
    _plot_neuropil_traces,
)
from ._single_wells_plots._plot_neuropil_visualization import _plot_neuropil_masks

if TYPE_CHECKING:
    from cali._plate_viewer._graph_widgets import (
        _MultilWellGraphWidget,
        _SingleWellGraphWidget,
    )
    from cali.sqlmodel._models import FOV

logger = LOGGER


# ANALYSIS PRODUCT REGISTRY ===========================================================


class AnalysisGroup(Enum):
    """Enum for grouping analysis products."""

    SINGLE_WELL = "single_well"
    MULTI_WELL = "multi_well"


# Type aliases for better type hints
SingleWellAnalyzer: TypeAlias = (
    "Callable[[_SingleWellGraphWidget, dict, list[int] | None], Any]"
)
MultiWellAnalyzer: TypeAlias = "Callable[[_MultilWellGraphWidget, str, str], None]"
AnyAnalyzer: TypeAlias = "SingleWellAnalyzer | MultiWellAnalyzer"


@dataclass
class AnalysisProduct:
    """Represents a single analysis/plot type with its configuration.

    Attributes
    ----------
    name : str
        Display name shown in the UI combobox
    group : AnalysisGroup
        Whether this is a single-well or multi-well analysis
    analyzer : AnyAnalyzer
        The plotting function to call
    category : str
        Category for grouping in the UI (e.g., "Calcium Traces", "Evoked Experiment")
    """

    name: str
    group: AnalysisGroup
    analyzer: AnyAnalyzer
    category: str = "General"

    def __post_init__(self) -> None:
        """Register this product in the global registry."""
        if any(self.name == product.name for product in ANALYSIS_PRODUCTS):
            raise ValueError(f"AnalysisProduct '{self.name}' already registered.")
        ANALYSIS_PRODUCTS.append(self)


# Global registry of all analysis products
ANALYSIS_PRODUCTS: list[AnalysisProduct] = []


# TITLES FOR THE PLOTS THAT WILL BE SHOWN IN THE COMBOBOX
# fmt: off
RAW_TRACES = "Calcium Raw Traces"
CORRECTED_TRACES = "Calcium Neuropil Corrected Traces"
NORMALIZED_TRACES = "Calcium Normalized Traces"
DFF = "Calcium ΔF/F0 Traces"
DFF_NORMALIZED = "Calcium ΔF/F0 Normalized  Traces "
DEC_DFF_NORMALIZED_ACTIVE_ONLY = "Calcium Deconvolved ΔF/F0 Traces Normalized (Active Only)"  # noqa: E501
DEC_DFF = "Calcium Deconvolved ΔF/F0 Traces"
DEC_DFF_WITH_PEAKS = "Calcium Deconvolved ΔF/F0 Traces with Peaks"
DEC_DFF_WITH_PEAKS_AND_THRESHOLDS = "Calcium Deconvolved ΔF/F0 Traces with Peaks and Thresholds (If 1 ROI selected)"  # noqa: E501
DEC_DFF_NORMALIZED = "Calcium Deconvolved ΔF/F0 Normalized Traces "
DEC_DFF_NORMALIZED_WITH_PEAKS = "Calcium Deconvolved ΔF/F0 Normalized Traces with Peaks"
DEC_DFF_AMPLITUDE = "Calcium Peaks Amplitudes (Deconvolved ΔF/F0)"
DEC_DFF_FREQUENCY = "Calcium Peaks Frequencies (Deconvolved ΔF/F0)"
DEC_DFF_AMPLITUDE_VS_FREQUENCY = "Calcium Peaks Amplitudes vs Frequencies (Deconvolved ΔF/F0)"  # noqa: E501
DEC_DFF_IEI = "Calcium Peaks Inter-event Interval (Deconvolved ΔF/F0)"
INFERRED_SPIKES_RAW = "Inferred Spikes Raw"
INFERRED_SPIKES_THRESHOLDED = "Inferred Spikes (Thresholded)"
INFERRED_SPIKES_RAW_WITH_THRESHOLD = "Inferred Spikes Raw (with Thresholds - If 1 ROI selected)"  # noqa: E501
INFERRED_SPIKES_THRESHOLDED_WITH_DEC_DFF = "Inferred Spikes (Thresholded) with Deconvolved ΔF/F0 Traces"  # noqa: E501
INFERRED_SPIKES_THRESHOLDED_NORMALIZED = "Inferred Spikes (Thresholded) Normalized"
INFERRED_SPIKES_THRESHOLDED_ACTIVE_ONLY = "Inferred Spikes (Thresholded) Normalized (Active Only)"  # noqa: E501
INFERRED_SPIKES_NORMALIZED_WITH_BURSTS = "Inferred Spikes (Thresholded) Normalized with Network Bursts"  # noqa: E501
INFERRED_SPIKES_THRESHOLDED_SYNCHRONY = "Inferred Spikes (Thresholded) Global Synchrony"
INFERRED_SPIKE_CROSS_CORRELATION = "Inferred Spikes (Thresholded) Cross-Correlation"
INFERRED_SPIKE_CLUSTERING = "Inferred Spikes (Thresholded) Hierarchical Clustering"
INFERRED_SPIKE_CLUSTERING_DENDROGRAM = "Inferred Spikes (Thresholded) Hierarchical Clustering (Dendrogram)"  # noqa: E501
INFERRED_SPIKE_BURST_ANALYSIS = "Inferred Spikes (Thresholded) Burst Activity Analysis"
RASTER_PLOT = "Calcium Peaks Raster plot Colored by ROI"
RASTER_PLOT_AMP = "Calcium Peaks Raster plot Colored by Amplitude"
RASTER_PLOT_AMP_WITH_COLORBAR = "Calcium Peaks Raster plot Colored by Amplitude with Colorbar"  # noqa: E501
INFERRED_SPIKE_RASTER_PLOT = "Inferred Spikes Raster plot Colored by ROI"
INFERRED_SPIKE_RASTER_PLOT_AMP = "Inferred Spikes Raster plot Colored by Amplitude"
INFERRED_SPIKE_RASTER_PLOT_AMP_WITH_COLORBAR = "Inferred Spikes Raster plot Colored by Amplitude with Colorbar"  # noqa: E501
STIMULATED_VS_NON_STIMULATED_SPIKE_TRACES = "Stimulated vs Non-Stimulated Spike Traces"
CALCIUM_PEAKS_GLOBAL_SYNCHRONY = "Calcium Peaks Global Synchrony"
CALCIUM_NETWORK_CONNECTIVITY = "Calcium Network Connectivity"
CALCIUM_CONNECTIVITY_MATRIX = "Calcium Network Connectivity Matrix"
CROSS_CORRELATION = "Calcium Peaks Cross-Correlation"
CLUSTERING = "Calcium Peaks Hierarchical Clustering"
CLUSTERING_DENDROGRAM = "Calcium Peaks Hierarchical Clustering (Dendrogram)"
CELL_SIZE = "Cell Size"
STIMULATED_AREA = "Stim Area"
STIMULATED_ROIS = "Stim vs Non-Stim ROIs"
STIMULATED_ROIS_WITH_STIMULATED_AREA = "Stim vs Non-Stim ROIs with Stim Area"
STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED = "Stim vs Non-Stim Normalized Calcium Traces (Deconvolved ΔF/F0)"  # noqa: E501
STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS = "Stim vs Non-Stim Normalized Calcium Traces with Peaks (Deconvolved ΔF/F0)"  # noqa: E501
STIMULATED_PEAKS_AMP = "Stim Calcium Peaks Amplitudes"
NON_STIMULATED_PEAKS_AMP = "Non-Stim Calcium Peaks Amplitudes"
STIMULATED_PEAKS_FREQ = "Stim Calcium Peaks Frequencies"
NON_STIMULATED_PEAKS_FREQ = "Non-Stim Calcium Peaks Frequencies"
NEUROPIL_ROI_MASKS = "Neuropil and ROI Masks Visualization"
NEUROPIL_TRACES = "Neuropil and Raw Traces"


# REGISTER SINGLE WELL ANALYSIS PRODUCTS ==============================================
# Define all analysis products using the AnalysisProduct dataclass
# This replaces the old dictionary-based approach with a more structured pattern

# Calcium Traces Group
AnalysisProduct(
    name=RAW_TRACES,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, raw=True),
    category="Calcium Traces",
)
AnalysisProduct(
    name=CORRECTED_TRACES,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_traces_data,
    category="Calcium Traces",
)
AnalysisProduct(
    name=NORMALIZED_TRACES,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, normalize=True),
    category="Calcium Traces",
)
AnalysisProduct(
    name=DFF,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dff=True),
    category="Calcium Traces",
)
AnalysisProduct(
    name=DFF_NORMALIZED,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dff=True, normalize=True),
    category="Calcium Traces",
)
AnalysisProduct(
    name=DEC_DFF,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True),
    category="Calcium Traces",
)
AnalysisProduct(
    name=DEC_DFF_WITH_PEAKS,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, with_peaks=True),
    category="Calcium Traces",
)
AnalysisProduct(
    name=DEC_DFF_WITH_PEAKS_AND_THRESHOLDS,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, with_peaks=True, thresholds=True),
    category="Calcium Traces",
)
AnalysisProduct(
    name=DEC_DFF_NORMALIZED,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, normalize=True),
    category="Calcium Traces",
)
AnalysisProduct(
    name=DEC_DFF_NORMALIZED_ACTIVE_ONLY,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, normalize=True, active_only=True),
    category="Calcium Traces",
)
AnalysisProduct(
    name=DEC_DFF_NORMALIZED_WITH_PEAKS,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, normalize=True, with_peaks=True),
    category="Calcium Traces",
)

# Neuropil Group
AnalysisProduct(
    name=NEUROPIL_ROI_MASKS,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_neuropil_masks,
    category="Neuropil Correction",
)
AnalysisProduct(
    name=NEUROPIL_TRACES,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_neuropil_traces,
    category="Neuropil Correction",
)

# Inferred Spikes Group
AnalysisProduct(
    name=INFERRED_SPIKES_RAW,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, raw=True),
    category="Inferred Spikes Traces",
)
AnalysisProduct(
    name=INFERRED_SPIKES_THRESHOLDED,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_inferred_spikes,
    category="Inferred Spikes Traces",
)
AnalysisProduct(
    name=INFERRED_SPIKES_RAW_WITH_THRESHOLD,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, raw=True, thresholds=True),
    category="Inferred Spikes Traces",
)
AnalysisProduct(
    name=INFERRED_SPIKES_THRESHOLDED_NORMALIZED,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, normalize=True),
    category="Inferred Spikes Traces",
)
AnalysisProduct(
    name=INFERRED_SPIKES_THRESHOLDED_ACTIVE_ONLY,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, normalize=True, active_only=True),
    category="Inferred Spikes Traces",
)
AnalysisProduct(
    name=INFERRED_SPIKES_NORMALIZED_WITH_BURSTS,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_inferred_spikes_normalized_with_bursts,
    category="Inferred Spikes Traces",
)
AnalysisProduct(
    name=INFERRED_SPIKES_THRESHOLDED_WITH_DEC_DFF,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, dec_dff=True),
    category="Inferred Spikes Traces",
)

# Amplitude and Frequency Group
AnalysisProduct(
    name=DEC_DFF_AMPLITUDE,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_amplitude_and_frequency_data, amp=True),
    category="Calcium Peaks Amplitude and Frequency",
)
AnalysisProduct(
    name=DEC_DFF_FREQUENCY,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_amplitude_and_frequency_data, freq=True),
    category="Calcium Peaks Amplitude and Frequency",
)
AnalysisProduct(
    name=DEC_DFF_AMPLITUDE_VS_FREQUENCY,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_amplitude_and_frequency_data, amp=True, freq=True),
    category="Calcium Peaks Amplitude and Frequency",
)

# Raster Plots Group
AnalysisProduct(
    name=RASTER_PLOT,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_generate_raster_plot,
    category="Raster Plots",
)
AnalysisProduct(
    name=RASTER_PLOT_AMP,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_generate_raster_plot, amplitude_colors=True, colorbar=False),
    category="Raster Plots",
)
AnalysisProduct(
    name=RASTER_PLOT_AMP_WITH_COLORBAR,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_generate_raster_plot, amplitude_colors=True, colorbar=True),
    category="Raster Plots",
)
AnalysisProduct(
    name=INFERRED_SPIKE_RASTER_PLOT,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_generate_spike_raster_plot,
    category="Raster Plots",
)
AnalysisProduct(
    name=INFERRED_SPIKE_RASTER_PLOT_AMP,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(
        _generate_spike_raster_plot, amplitude_colors=True, colorbar=False
    ),
    category="Raster Plots",
)
AnalysisProduct(
    name=INFERRED_SPIKE_RASTER_PLOT_AMP_WITH_COLORBAR,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_generate_spike_raster_plot, amplitude_colors=True, colorbar=True),
    category="Raster Plots",
)

# Inter-event Interval Group
AnalysisProduct(
    name=DEC_DFF_IEI,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_iei_data,
    category="Calcium Peaks Interevent Interval",
)

# Cell Size Group
AnalysisProduct(
    name=CELL_SIZE,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_cell_size_data,
    category="Cell Size",
)

# Correlation Analysis Group
AnalysisProduct(
    name=CALCIUM_PEAKS_GLOBAL_SYNCHRONY,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_peak_event_synchrony_data,
    category="Correlation Analysis",
)
AnalysisProduct(
    name=CALCIUM_NETWORK_CONNECTIVITY,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_connectivity_network_data,
    category="Correlation Analysis",
)
AnalysisProduct(
    name=CALCIUM_CONNECTIVITY_MATRIX,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_connectivity_matrix_data,
    category="Correlation Analysis",
)
AnalysisProduct(
    name=CROSS_CORRELATION,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_cross_correlation_data,
    category="Correlation Analysis",
)
AnalysisProduct(
    name=CLUSTERING,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_hierarchical_clustering_data,
    category="Correlation Analysis",
)
AnalysisProduct(
    name=CLUSTERING_DENDROGRAM,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_hierarchical_clustering_data, use_dendrogram=True),
    category="Correlation Analysis",
)
AnalysisProduct(
    name=INFERRED_SPIKES_THRESHOLDED_SYNCHRONY,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_spike_synchrony_data,
    category="Correlation Analysis",
)
AnalysisProduct(
    name=INFERRED_SPIKE_CROSS_CORRELATION,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_spike_cross_correlation_data,
    category="Correlation Analysis",
)
AnalysisProduct(
    name=INFERRED_SPIKE_CLUSTERING,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_spike_hierarchical_clustering_data,
    category="Correlation Analysis",
)
AnalysisProduct(
    name=INFERRED_SPIKE_CLUSTERING_DENDROGRAM,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_spike_hierarchical_clustering_data, use_dendrogram=True),
    category="Correlation Analysis",
)
AnalysisProduct(
    name=INFERRED_SPIKE_BURST_ANALYSIS,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_inferred_spike_burst_activity,
    category="Correlation Analysis",
)

# Evoked Experiment Group
AnalysisProduct(
    name=STIMULATED_AREA,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_visualize_stimulated_area, stimulated_area=False),
    category="Evoked Experiment",
)
AnalysisProduct(
    name=STIMULATED_ROIS,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_visualize_stimulated_area, with_rois=True),
    category="Evoked Experiment",
)
AnalysisProduct(
    name=STIMULATED_ROIS_WITH_STIMULATED_AREA,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_visualize_stimulated_area, with_rois=True, stimulated_area=True),
    category="Evoked Experiment",
)
AnalysisProduct(
    name=STIMULATED_PEAKS_AMP,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_stim_or_not_stim_peaks_amplitude, stimulated=True),
    category="Evoked Experiment",
)
AnalysisProduct(
    name=NON_STIMULATED_PEAKS_AMP,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stim_or_not_stim_peaks_amplitude,
    category="Evoked Experiment",
)
AnalysisProduct(
    name=STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stimulated_vs_non_stimulated_roi_amp,
    category="Evoked Experiment",
)
AnalysisProduct(
    name=STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_stimulated_vs_non_stimulated_roi_amp, with_peaks=True),
    category="Evoked Experiment",
)
AnalysisProduct(
    name=STIMULATED_VS_NON_STIMULATED_SPIKE_TRACES,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stimulated_vs_non_stimulated_spike_traces,
    category="Evoked Experiment",
)

# Multi-Well Analysis Products --------------------------------------------------------
# These plot CSV bar plots from grouped analysis data

# Helper function that all multi-well products use
def _plot_csv_bar_plot_wrapper(
    widget: _MultilWellGraphWidget, text: str, analysis_path: str, **kwargs: Any
) -> None:
    """Wrapper for CSV bar plot that matches MultiWellAnalyzer signature."""
    suffix = kwargs.get("suffix")
    if not suffix:
        widget.figure.clear()
        return

    # Determine CSV path based on whether it's stimulated data
    stimulated = kwargs.get("stimulated", False)
    if stimulated or "stimulated" in suffix:
        csv_path = Path(analysis_path) / "grouped_evk"
    else:
        csv_path = Path(analysis_path) / "grouped"

    # Find the CSV file
    csv_files = list(csv_path.glob(f"*{suffix}*.csv"))
    if not csv_files:
        widget.figure.clear()
        return

    # Prepare info dict for plot_csv_bar_plot
    info = {k: v for k, v in kwargs.items() if k != "suffix" and k != "stimulated"}
    plot_csv_bar_plot(widget, csv_files[0], info)


# General Multi-Well Products
AnalysisProduct(
    name="Cell Size Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        parameter="Cell Size",
        suffix="cell_size",
        units="μm²",
    ),
    category="General",
)
AnalysisProduct(
    name="Percentage of Active Cells (Based on Calcium Peaks) Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        parameter="Percentage of Active Cells",
        suffix="percentage_active",
        add_to_title="Based on Calcium Peaks",
    ),
    category="General",
)
AnalysisProduct(
    name="Calcium Peaks Amplitude Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        parameter="Calcium Peaks Amplitude",
        suffix="amplitude",
        add_to_title=" (Deconvolved ΔF/F0)",
    ),
    category="General",
)
AnalysisProduct(
    name="Calcium Peaks Frequency Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        parameter="Calcium Peaks Frequency",
        suffix="frequency",
        add_to_title=" (Deconvolved ΔF/F0)",
        units="Hz",
    ),
    category="General",
)
AnalysisProduct(
    name="Calcium Peaks Inter-Event Interval Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        parameter="Calcium Peaks Inter-Event Interval",
        suffix="iei",
        add_to_title=" (Deconvolved ΔF/F0)",
        units="Sec",
    ),
    category="General",
)
AnalysisProduct(
    name="Calcium Peak Events Global Synchrony Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        parameter="Calcium Peak Events Global Synchrony",
        suffix="calcium_peaks_synchrony",
        add_to_title="(Median)",
        units="Index",
    ),
    category="General",
)
AnalysisProduct(
    name="Inferred Spikes Global Synchrony Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        parameter="Inferred Spikes Global Synchrony",
        suffix="spike_synchrony",
        add_to_title="(Median - Thresholded Data)",
        units="Index",
    ),
    category="General",
)
AnalysisProduct(
    name="Burst Count Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        parameter="Burst Count",
        suffix="burst_activity",
        burst_metric="count",
        add_to_title="(Inferred Spikes)",
        units="Count",
    ),
    category="General",
)
AnalysisProduct(
    name="Burst Average Duration Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        parameter="Burst Average Duration",
        suffix="burst_activity",
        burst_metric="avg_duration_sec",
        add_to_title="(Inferred Spikes)",
        units="Sec",
    ),
    category="General",
)
AnalysisProduct(
    name="Burst Average Interval Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        parameter="Burst Average Interval",
        suffix="burst_activity",
        burst_metric="avg_interval_sec",
        add_to_title="(Inferred Spikes)",
        units="Sec",
    ),
    category="General",
)
AnalysisProduct(
    name="Burst Rate Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        parameter="Burst Rate",
        suffix="burst_activity",
        burst_metric="rate_burst_per_min",
        add_to_title="(Inferred Spikes)",
        units="Bursts/min",
    ),
    category="General",
)

# Evoked Experiment Multi-Well Products
AnalysisProduct(
    name="Stimulated Calcium Peaks Amplitude Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        stimulated=True,
        parameter="Calcium Peaks Amplitude",
        suffix="calcium_peaks_amplitudes_stimulated",
        add_to_title=" (Stimulated - Deconvolved ΔF/F0)",
    ),
    category="Evoked Experiment",
)
AnalysisProduct(
    name="Non-Stimulated Calcium Peaks Amplitude Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        _plot_csv_bar_plot_wrapper,
        stimulated=False,
        parameter="Calcium Peaks Amplitude",
        suffix="calcium_peaks_amplitudes_non_stimulated",
        add_to_title=" (Non-Stimulated - Deconvolved ΔF/F0)",
    ),
    category="Evoked Experiment",
)


# DATABASE HELPERS ====================================================================
# Helper functions to extract plotting data from database models


# NOPE, not good, use database directly!!!
def get_fov_data_from_db(fov: FOV) -> dict:
    """Extract ROI data from database FOV model.

    Converts database models (ROI, Traces, DataAnalysis, Mask) into the dict
    format expected by plotting functions.

    Parameters
    ----------
    fov : FOV
        FOV database model containing ROIs with traces, masks, and analysis results

    Returns
    -------
    dict
        Dictionary mapping ROI label to ROIData objects
    """
    roi_data: dict = {}

    for roi in fov.rois:
        # Create ROIData object with all fields set individually
        roi_obj = ROIData()

        # Set traces data
        if roi.traces:
            roi_obj.raw_trace = roi.traces.raw_trace
            roi_obj.corrected_trace = roi.traces.corrected_trace
            roi_obj.neuropil_trace = roi.traces.neuropil_trace
            roi_obj.dff = roi.traces.dff
            roi_obj.dec_dff = roi.traces.dec_dff
            roi_obj.elapsed_time_list_ms = roi.traces.x_axis

        # Set mask data (only if all components are not None)
        if roi.roi_mask and all(
            x is not None
            for x in [
                roi.roi_mask.coords_y,
                roi.roi_mask.coords_x,
                roi.roi_mask.height,
                roi.roi_mask.width,
            ]
        ):
            roi_obj.mask_coord_and_shape = (
                (roi.roi_mask.coords_y, roi.roi_mask.coords_x),  # type: ignore
                (roi.roi_mask.height, roi.roi_mask.width),  # type: ignore
            )

        # Set neuropil mask data (only if all components are not None)
        if roi.neuropil_mask and all(
            x is not None
            for x in [
                roi.neuropil_mask.coords_y,
                roi.neuropil_mask.coords_x,
                roi.neuropil_mask.height,
                roi.neuropil_mask.width,
            ]
        ):
            roi_obj.neuropil_mask_coord_and_shape = (
                (roi.neuropil_mask.coords_y, roi.neuropil_mask.coords_x),  # type: ignore
                (roi.neuropil_mask.height, roi.neuropil_mask.width),  # type: ignore
            )

        # Set analysis results
        if roi.data_analysis:
            da = roi.data_analysis
            roi_obj.cell_size = da.cell_size
            roi_obj.dec_dff_frequency = da.dec_dff_frequency
            roi_obj.peaks_dec_dff = da.peaks_dec_dff
            roi_obj.peaks_amplitudes_dec_dff = da.peaks_amplitudes_dec_dff
            roi_obj.iei = da.iei
            roi_obj.inferred_spikes = da.inferred_spikes
            roi_obj.inferred_spikes_threshold = da.inferred_spikes_threshold
            roi_obj.peaks_prominence_dec_dff = da.peaks_prominence_dec_dff
            roi_obj.peaks_height_dec_dff = da.peaks_height_dec_dff

        # Set activity status
        roi_obj.active = False
        if roi.data_analysis and roi.data_analysis.peaks_dec_dff:
            roi_obj.active = len(roi.data_analysis.peaks_dec_dff) > 0

        # Set ROI metadata
        roi_obj.stimulated = roi.stimulated

        roi_data[str(roi.label_value)] = roi_obj

    return roi_data


# COMBO BOX OPTIONS ===================================================================
# Generate combobox options dynamically from the registry


def _get_combo_options_dict(group: AnalysisGroup) -> dict[str, list[str]]:
    """Generate combobox options grouped by category.

    Returns a dictionary where keys are category headers (with dividers)
    and values are lists of analysis names in that category.
    """
    # Group products by category
    categories: dict[str, list[str]] = {}
    for product in ANALYSIS_PRODUCTS:
        if product.group == group:
            if product.category not in categories:
                categories[product.category] = []
            categories[product.category].append(product.name)

    # Format with dividers for the combobox
    result = {}
    for category, names in sorted(categories.items()):
        # Create a divider key that won't be selectable
        divider_key = f"----------{category}".ljust(60, "-")
        result[divider_key] = names

    return result


# Generate the dictionaries on module load
SINGLE_WELL_COMBO_OPTIONS_DICT = _get_combo_options_dict(AnalysisGroup.SINGLE_WELL)
MULTI_WELL_COMBO_OPTIONS_DICT = _get_combo_options_dict(AnalysisGroup.MULTI_WELL)


# PLOTTING DISPATCH FUNCTIONS =========================================================


def plot_single_well_data(

    widget: _SingleWellGraphWidget,
    data: dict,
    text: str,
    rois: list[int] | None = None,
) -> None:
    """Plot single-well analysis data using registry pattern.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        The widget to plot into
    data : dict
        The analysis data dictionary
    text : str
        The name of the analysis to plot (matches AnalysisProduct.name)
    rois : list[int] | None, optional
        List of ROI indices to plot, by default None
    """
    try:
        # Look up the analysis in the registry
        for product in ANALYSIS_PRODUCTS:
            if product.name == text and product.group == AnalysisGroup.SINGLE_WELL:
                # Type narrowing: we know this is a SingleWellAnalyzer
                analyzer = cast("SingleWellAnalyzer", product.analyzer)
                return analyzer(widget, data, rois)

        # If we get here, analysis was not found
        logger.warning(f"Analysis '{text}' not found in registry")

    except Exception as e:
        logger.error(f"Error plotting single well data for '{text}': {e}")
        raise


def plot_multi_well_data(
    widget: _MultilWellGraphWidget,
    text: str,
    analysis_path: str | None,
) -> None:
    """Plot multi-well data using registry pattern.

    Parameters
    ----------
    widget : _MultilWellGraphWidget
        The widget to plot into
    text : str
        The name of the analysis to plot (matches AnalysisProduct.name)
    analysis_path : str | None
        Path to the analysis directory containing grouped CSV files
    """
    # Handle empty/invalid selection
    if not text or text == "None" or text in MULTI_WELL_COMBO_OPTIONS_DICT.keys():
        widget.figure.clear()
        return

    if not analysis_path:
        widget.figure.clear()
        return

    try:
        # Look up the analysis in the registry
        for product in ANALYSIS_PRODUCTS:
            if product.name == text and product.group == AnalysisGroup.MULTI_WELL:
                # Type narrowing: we know this is a MultiWellAnalyzer
                analyzer = cast("MultiWellAnalyzer", product.analyzer)
                return analyzer(widget, text, analysis_path)

        # If we get here, analysis was not found
        logger.warning(f"Multi-well analysis '{text}' not found in registry")
        widget.figure.clear()

    except Exception as e:
        logger.error(f"Error plotting multi-well data for '{text}': {e}")
        widget.figure.clear()
        raise


def _plot_csv_bar_plot_data_legacy(
    widget: _MultilWellGraphWidget, text: str, analysis_path: str, **kwargs: Any
) -> None:
    """Legacy helper function - kept for reference during transition."""
    suffix = kwargs.get("suffix")
    if not suffix:
        print(f"No suffix provided for {text}.")
        widget.figure.clear()
        return

    # Determine CSV path based on whether it's stimulated data
    stimulated = kwargs.get("stimulated", False)
    if stimulated or "stimulated" in suffix:
        csv_path = Path(analysis_path) / "grouped_evk"
    else:
        csv_path = Path(analysis_path) / "grouped"

    if not csv_path.exists():
        LOGGER.error(f"CSV path {csv_path} does not exist.")
        widget.figure.clear()
        return

    csv_file = next(
        (f for f in csv_path.glob("*.csv") if f.name.endswith(f"_{suffix}.csv")),
        None,
    )

    if not csv_file:
        LOGGER.error(f"CSV file for suffix '{suffix}' not found in {csv_path}.")
        widget.figure.clear()
        return

    # Create plot options from kwargs, filtering out non-plot parameters
    plot_options = {
        k: v
        for k, v in kwargs.items()
        if k not in ["stimulated", "per_led_power", "burst_metric"]
    }

    # Special handling for burst activity plots
    burst_metric = kwargs.get("burst_metric")
    if suffix == "burst_activity" and burst_metric:
        # Add burst_metric to plot_options for handling in plot_csv_bar_plot
        plot_options["burst_metric"] = burst_metric
        return plot_csv_bar_plot(
            widget,
            csv_file,
            plot_options,
            mean_n_sem=False,
        )

    # Special handling for certain plot types that don't use mean_n_sem
    synchrony_suffixes = [
        "synchrony",
        "spike_synchrony",
        "calcium_network_density",
        "calcium_peaks_synchrony",
    ]
    if any(sync_suffix in suffix for sync_suffix in synchrony_suffixes):
        return plot_csv_bar_plot(
            widget,
            csv_file,
            plot_options,
            mean_n_sem=False,
        )

    if "percentage_active" in suffix:
        return plot_csv_bar_plot(
            widget,
            csv_file,
            plot_options,
            value_n=True,
        )

    return plot_csv_bar_plot(widget, csv_file, plot_options)
