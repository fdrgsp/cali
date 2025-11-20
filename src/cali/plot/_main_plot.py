from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, cast

from typing_extensions import TypeAlias

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
    from pathlib import Path

    from cali.gui._graph_widgets import _MultilWellGraphWidget, _SingleWellGraphWidget


cali_logger = logging.getLogger("cali_logger")

# ANALYSIS PRODUCT REGISTRY ===========================================================


class AnalysisGroup(Enum):
    """Enum for grouping analysis products."""

    SINGLE_WELL = "single_well"
    MULTI_WELL = "multi_well"


# Type aliases for better type hints
# Single-well analyzers now accept (widget, db_path, fov_name, rois)
SingleWellAnalyzer: TypeAlias = (
    "Callable[[_SingleWellGraphWidget, str | Path, str, list[int] | None], Any]"
)
# Multi-well analyzers accept (widget, text, db_path)
MultiWellAnalyzer: TypeAlias = (
    "Callable[[_MultilWellGraphWidget, str, str | Path], None]"
)
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

# # Multi-Well Analysis Products --------------------------------------------------------
# # These plot CSV bar plots from grouped analysis data

# # General Multi-Well Products
# AnalysisProduct(
#     name="Cell Size Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         parameter="Cell Size",
#         suffix="cell_size",
#         units="μm²",
#     ),
#     category="General",
# )
# AnalysisProduct(
#     name="Percentage of Active Cells (Based on Calcium Peaks) Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         parameter="Percentage of Active Cells",
#         suffix="percentage_active",
#         add_to_title="Based on Calcium Peaks",
#     ),
#     category="General",
# )
# AnalysisProduct(
#     name="Calcium Peaks Amplitude Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         parameter="Calcium Peaks Amplitude",
#         suffix="amplitude",
#         add_to_title=" (Deconvolved ΔF/F0)",
#     ),
#     category="General",
# )
# AnalysisProduct(
#     name="Calcium Peaks Frequency Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         parameter="Calcium Peaks Frequency",
#         suffix="frequency",
#         add_to_title=" (Deconvolved ΔF/F0)",
#         units="Hz",
#     ),
#     category="General",
# )
# AnalysisProduct(
#     name="Calcium Peaks Inter-Event Interval Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         parameter="Calcium Peaks Inter-Event Interval",
#         suffix="iei",
#         add_to_title=" (Deconvolved ΔF/F0)",
#         units="Sec",
#     ),
#     category="General",
# )
# AnalysisProduct(
#     name="Calcium Peak Events Global Synchrony Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         parameter="Calcium Peak Events Global Synchrony",
#         suffix="calcium_peaks_synchrony",
#         add_to_title="(Median)",
#         units="Index",
#     ),
#     category="General",
# )
# AnalysisProduct(
#     name="Inferred Spikes Global Synchrony Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         parameter="Inferred Spikes Global Synchrony",
#         suffix="spike_synchrony",
#         add_to_title="(Median - Thresholded Data)",
#         units="Index",
#     ),
#     category="General",
# )
# AnalysisProduct(
#     name="Burst Count Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         parameter="Burst Count",
#         suffix="burst_activity",
#         burst_metric="count",
#         add_to_title="(Inferred Spikes)",
#         units="Count",
#     ),
#     category="General",
# )
# AnalysisProduct(
#     name="Burst Average Duration Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         parameter="Burst Average Duration",
#         suffix="burst_activity",
#         burst_metric="avg_duration_sec",
#         add_to_title="(Inferred Spikes)",
#         units="Sec",
#     ),
#     category="General",
# )
# AnalysisProduct(
#     name="Burst Average Interval Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         parameter="Burst Average Interval",
#         suffix="burst_activity",
#         burst_metric="avg_interval_sec",
#         add_to_title="(Inferred Spikes)",
#         units="Sec",
#     ),
#     category="General",
# )
# AnalysisProduct(
#     name="Burst Rate Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         parameter="Burst Rate",
#         suffix="burst_activity",
#         burst_metric="rate_burst_per_min",
#         add_to_title="(Inferred Spikes)",
#         units="Bursts/min",
#     ),
#     category="General",
# )

# # Evoked Experiment Multi-Well Products
# AnalysisProduct(
#     name="Stimulated Calcium Peaks Amplitude Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         stimulated=True,
#         parameter="Calcium Peaks Amplitude",
#         suffix="calcium_peaks_amplitudes_stimulated",
#         add_to_title=" (Stimulated - Deconvolved ΔF/F0)",
#     ),
#     category="Evoked Experiment",
# )
# AnalysisProduct(
#     name="Non-Stimulated Calcium Peaks Amplitude Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=partial(
#         _plot_csv_bar_plot_wrapper,
#         stimulated=False,
#         parameter="Calcium Peaks Amplitude",
#         suffix="calcium_peaks_amplitudes_non_stimulated",
#         add_to_title=" (Non-Stimulated - Deconvolved ΔF/F0)",
#     ),
#     category="Evoked Experiment",
# )


# DATABASE HELPERS ====================================================================
# Helper functions to extract plotting data from database models


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


# Plots that require active ROIs only (for random selection filtering)
# Centralized configuration - easier to maintain than scattered logic
# Used by _graph_widgets.py to filter random ROI selection
ACTIVE_ONLY_PLOTS: set[str] = {
    DEC_DFF_WITH_PEAKS,
    DEC_DFF_WITH_PEAKS_AND_THRESHOLDS,
    DEC_DFF_NORMALIZED_WITH_PEAKS,
    DEC_DFF_AMPLITUDE,
    DEC_DFF_FREQUENCY,
    DEC_DFF_AMPLITUDE_VS_FREQUENCY,
    DEC_DFF_IEI,
    RASTER_PLOT,
    RASTER_PLOT_AMP,
    RASTER_PLOT_AMP_WITH_COLORBAR,
    CALCIUM_PEAKS_GLOBAL_SYNCHRONY,
    CROSS_CORRELATION,
    CLUSTERING,
    CLUSTERING_DENDROGRAM,
    INFERRED_SPIKES_THRESHOLDED,
    INFERRED_SPIKES_RAW_WITH_THRESHOLD,
    INFERRED_SPIKES_THRESHOLDED_WITH_DEC_DFF,
    INFERRED_SPIKES_THRESHOLDED_NORMALIZED,
    INFERRED_SPIKES_THRESHOLDED_ACTIVE_ONLY,
    INFERRED_SPIKES_NORMALIZED_WITH_BURSTS,
    INFERRED_SPIKES_THRESHOLDED_SYNCHRONY,
    INFERRED_SPIKE_CROSS_CORRELATION,
    INFERRED_SPIKE_CLUSTERING,
    INFERRED_SPIKE_CLUSTERING_DENDROGRAM,
    INFERRED_SPIKE_BURST_ANALYSIS,
    INFERRED_SPIKE_RASTER_PLOT,
    INFERRED_SPIKE_RASTER_PLOT_AMP,
    INFERRED_SPIKE_RASTER_PLOT_AMP_WITH_COLORBAR,
    CALCIUM_CONNECTIVITY_MATRIX,
    CALCIUM_NETWORK_CONNECTIVITY,
}


def requires_active_rois(plot_name: str) -> bool:
    """Check if a plot requires only active ROIs.

    Parameters
    ----------
    plot_name : str
        The name of the plot (from combo box selection)

    Returns
    -------
    bool
        True if the plot requires active ROIs only
    """
    return plot_name in ACTIVE_ONLY_PLOTS


# PLOTTING DISPATCH FUNCTIONS =========================================================


def plot_single_well_data(
    widget: _SingleWellGraphWidget,
    db_path: str | Path,
    fov_name: str,
    text: str,
    rois: list[int] | None = None,
) -> None:
    """Plot single-well analysis data using registry pattern with database queries.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        The widget to plot into
    db_path : str | Path
        Path to the SQLite database file
    fov_name : str
        Name of the FOV to query (e.g., "B5_0000")
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
                return analyzer(widget, db_path, fov_name, rois)

        # If we get here, analysis was not found
        cali_logger.warning(f"Analysis '{text}' not found in registry")

    except Exception as e:
        cali_logger.error(f"Error plotting single well data for '{text}': {e}")
        raise


def plot_multi_well_data(
    widget: _MultilWellGraphWidget,
    text: str,
    db_path: str | Path,
) -> None:
    """Plot multi-well data using registry pattern with database queries.

    Parameters
    ----------
    widget : _MultilWellGraphWidget
        The widget to plot into
    text : str
        The name of the analysis to plot (matches AnalysisProduct.name)
    db_path : str | Path
        Path to the SQLite database file
    """
    # Handle empty/invalid selection
    if not text or text == "None" or text in MULTI_WELL_COMBO_OPTIONS_DICT.keys():
        widget.figure.clear()
        return

    try:
        # Look up the analysis in the registry
        for product in ANALYSIS_PRODUCTS:
            if product.name == text and product.group == AnalysisGroup.MULTI_WELL:
                # Type narrowing: we know this is a MultiWellAnalyzer
                analyzer = cast("MultiWellAnalyzer", product.analyzer)
                return analyzer(widget, text, db_path)

        # If we get here, analysis was not found
        cali_logger.warning(f"Multi-well analysis '{text}' not found in registry")
        widget.figure.clear()

    except Exception as e:
        cali_logger.error(f"Error plotting multi-well data for '{text}': {e}")
        widget.figure.clear()
        raise
