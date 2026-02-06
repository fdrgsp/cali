from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, cast

from typing_extensions import TypeAlias

from cali._constants import EVOKED

from ._multi_wells_plots import (
    plot_cell_size_bar_plot,
)
from ._single_wells_plots.burst import (
    _plot_calcium_burst_activity,
    _plot_calcium_normalized_with_bursts,
    _plot_calcium_raster_with_bursts,
    _plot_inferred_spike_burst_activity,
    _plot_inferred_spike_raster_with_bursts,
    _plot_inferred_spikes_normalized_with_bursts,
)
from ._single_wells_plots.calcium_traces._plot_calcium_traces_data import (
    _plot_traces_data,
)
from ._single_wells_plots.calcium_traces._plot_neuropil_traces import (
    _plot_neuropil_traces,
)
from ._single_wells_plots.correlation._plot_calcium_traces_correlation import (
    _plot_dec_dff_correlation_data,
    _plot_dff_correlation_data,
)
from ._single_wells_plots.correlation._plot_connectivity import (
    _plot_connectivity_network_data,
)
from ._single_wells_plots.correlation._plot_evoked_correlation_synchrony import (
    _plot_sorted_dec_dff_correlation,
    _plot_sorted_dec_dff_correlation_windowed_by_stim,
    _plot_sorted_dec_dff_correlation_windowed_non_stim,
    _plot_sorted_spike_ccg_zscore,
    _plot_sorted_spike_max_lag_correlation,
    _plot_sorted_spike_max_lag_values,
    _plot_sorted_spike_synchrony,
)
from ._single_wells_plots.correlation._plot_inferred_spike_synchrony import (
    _plot_spike_synchrony_data,
)
from ._single_wells_plots.correlation._plot_spike_max_lag_correlation import (
    _plot_ccg_zscore_data,
    _plot_spike_max_lag_correlation_data,
)
from ._single_wells_plots.correlation._plot_spike_max_lag_values import (
    _plot_spike_max_lag_values_data,
)
from ._single_wells_plots.evoked._plot_evoked_experiment_data_plots import (
    _plot_stim_and_non_stim_peaks_amplitude,
    _plot_stimulated_vs_non_stimulated_calcium_peaks_raster,
    _plot_stimulated_vs_non_stimulated_roi_traces,
    _plot_stimulated_vs_non_stimulated_spike_raster,
    _plot_stimulated_vs_non_stimulated_spike_traces,
)
from ._single_wells_plots.evoked._stimulation_area import _visualize_stimulated_area
from ._single_wells_plots.metrics._plot_calcium_amplitudes_and_frequencies_data import (
    _plot_amplitude_and_frequency_data,
)
from ._single_wells_plots.metrics._plot_calcium_peaks_iei_data import _plot_iei_data
from ._single_wells_plots.metrics._plot_cell_size import _plot_cell_size_data
from ._single_wells_plots.metrics._plot_inferred_spikes_frequency_data import (
    _plot_inferred_spikes_frequency_data,
)
from ._single_wells_plots.raster._plot_calcium_peaks_raster_plots import (
    _generate_intensity_heatmap,
    _generate_raster_plot,
)
from ._single_wells_plots.raster._plot_inferred_spike_raster_plots import (
    _generate_spike_raster_plot,
)
from ._single_wells_plots.spikes._plot_inferred_spikes import (
    _plot_inferred_spikes,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import (
        _MultilWellGraphWidget,
        _SingleWellGraphWidget,
    )


from cali.logger import cali_logger

# ANALYSIS PRODUCT REGISTRY ===========================================================


class AnalysisGroup(Enum):
    """Enum for grouping analysis products."""

    SINGLE_WELL = "single_well"
    MULTI_WELL = "multi_well"


class PipelineStage(Enum):
    """Enum for three-stage pipeline stages."""

    DETECTION = "detection"  # Requires Detection only (ROI masks, cell size)
    EXTRACTION = "extraction"  # Requires Detection + Extraction (traces, neuropil)
    ANALYSIS = "analysis"  # Requires Detection + Extraction + Analysis (peaks, spikes)


# Type aliases for better type hints
# Single-well analyzers accept (widget, engine, fov_name, rois, run_id)
# Using Any for analyzer since functions may have additional keyword arguments
SingleWellAnalyzer: TypeAlias = (
    "Callable[..., Any]"  # Flexible signature for partial functions
)
# Multi-well analyzers accept (widget, text, engine, run_id)
MultiWellAnalyzer: TypeAlias = (
    "Callable[..., None]"  # Flexible signature for partial functions
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
    pipeline_stage : PipelineStage
        Minimum pipeline stage required to generate this plot
    experiment_type : str | None
        Required experiment type ("evoked" or "spontaneous"), None for all types
    """

    name: str
    group: AnalysisGroup
    analyzer: AnyAnalyzer
    category: str = "General"
    pipeline_stage: PipelineStage = PipelineStage.ANALYSIS
    experiment_type: str | None = None  # "evoked", "spontaneous", or None for all

    def __post_init__(self) -> None:
        """Register this product in the global registry."""
        if any(self.name == product.name for product in ANALYSIS_PRODUCTS):
            raise ValueError(f"AnalysisProduct '{self.name}' already registered.")
        ANALYSIS_PRODUCTS.append(self)


# Global registry of all analysis products
ANALYSIS_PRODUCTS: list[AnalysisProduct] = []


# REGISTER SINGLE WELL ANALYSIS PRODUCTS ==============================================
# Define all analysis products using the AnalysisProduct dataclass

# Calcium Traces Group
AnalysisProduct(
    name="Calcium Raw Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, raw=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    name="Calcium Raw Normalized Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, raw=True, normalize=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    name="Calcium Neuropil Corrected Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_neuropil_traces, corrected=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    name="Neuropil and Raw Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_neuropil_traces,
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    name="Calcium ΔF/F0 Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dff=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    name="Calcium ΔF/F0 Normalized  Traces ",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dff=True, normalize=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    name="Calcium Deconvolved ΔF/F0 Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    name="Calcium Deconvolved ΔF/F0 Traces with Peaks",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, with_peaks=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Deconvolved ΔF/F0 Traces with Peaks and Thresholds (1 ROI)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, with_peaks=True, thresholds=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Deconvolved ΔF/F0 Normalized Traces ",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, normalize=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    name="Calcium Deconvolved ΔF/F0 Traces Normalized (Active Only)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, normalize=True, active_only=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Deconvolved ΔF/F0 Normalized Traces with Peaks",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, normalize=True, with_peaks=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Inferred Spikes Group
AnalysisProduct(
    name="Inferred Spikes",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, raw=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Normalized",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, normalize=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Normalized (Active Only)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, normalize=True, active_only=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes (with Thresholds if 1 ROI)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, raw=True, thresholds=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes with Deconvolved ΔF/F0 Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, dec_dff=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, thresholded=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded Normalized",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, thresholded=True, normalize=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded Normalized (Active Only)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(
        _plot_inferred_spikes, thresholded=True, normalize=True, active_only=True
    ),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Raster Plots Group
AnalysisProduct(
    name="Calcium Peaks Raster",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_generate_raster_plot,
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Peaks Raster plot Colored by Amplitude",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_generate_raster_plot, amplitude_colors=True, colorbar=False),
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Peaks Raster plot Colored by Amplitude with Colorbar",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_generate_raster_plot, amplitude_colors=True, colorbar=True),
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Intensity Heatmap",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_generate_intensity_heatmap,
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Raster Thresholded",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_generate_spike_raster_plot,
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Raster Thresholded (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_generate_spike_raster_plot, edges=True),
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Calcium Amplitude and Frequency Group
AnalysisProduct(
    name="Calcium Peaks Amplitudes (Deconvolved ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_amplitude_and_frequency_data, amp=True),
    category="Calcium Peaks Amplitude, Frequency and Event Interval",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Peaks Frequencies (Deconvolved ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_amplitude_and_frequency_data, freq=True),
    category="Calcium Peaks Amplitude, Frequency and Event Interval",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Peaks Amplitudes vs Frequencies (Deconvolved ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_amplitude_and_frequency_data, amp=True, freq=True),
    category="Calcium Peaks Amplitude, Frequency and Event Interval",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Peaks Inter-event Interval (Deconvolved ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_iei_data,
    category="Calcium Peaks Amplitude, Frequency and Event Interval",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Inferred Spikes Frequency Group
AnalysisProduct(
    name="Inferred Spikes Thresholded Frequency",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes_frequency_data, rising_edge=False),
    category="Inferred Spikes Frequency",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded Frequency (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes_frequency_data, rising_edge=True),
    category="Inferred Spikes Frequency",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Calcium Burst Analysis Group
AnalysisProduct(
    name="Calcium Burst Activity Analysis",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_calcium_burst_activity,
    category="Calcium Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Traces Normalized with Network Bursts",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_calcium_normalized_with_bursts,
    category="Calcium Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Raster with Network Bursts",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_calcium_raster_with_bursts,
    category="Calcium Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
# Inferred Spike Burst Analysis Group
AnalysisProduct(
    name="Inferred Spikes Thresholded Burst Activity Analysis",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_inferred_spike_burst_activity,
    category="Inferred Spike Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded Normalized with Network Bursts",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_inferred_spikes_normalized_with_bursts,
    category="Inferred Spike Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spike Raster with Network Bursts",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_inferred_spike_raster_with_bursts,
    category="Inferred Spike Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)


# Correlation Analysis Group
AnalysisProduct(
    name="Calcium ΔF/F0 Correlation",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_dff_correlation_data,
    category="Calcium Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Deconvolved ΔF/F0 Correlation",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_dec_dff_correlation_data,
    category="Calcium Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Calcium Functional Connectivity",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_connectivity_network_data,
    category="Calcium Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)


# Inferred Spikes Correlation Analysis Group
AnalysisProduct(
    name="Inferred Spikes Thresholded Max Lag Correlation",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_spike_max_lag_correlation_data,
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded Max Lag Correlation (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_spike_max_lag_correlation_data, rising_edges=True),
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded CCG Z-Score",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_ccg_zscore_data,
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded CCG Z-Score (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_ccg_zscore_data, rising_edges=True),
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded Max Lag Values",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_spike_max_lag_values_data,
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded Max Lag Values (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_spike_max_lag_values_data, rising_edges=True),
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded Global Synchrony",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_spike_synchrony_data,
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    name="Inferred Spikes Thresholded Global Synchrony (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_spike_synchrony_data, rising_edges=True),
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Evoked Experiment Group
AnalysisProduct(
    name="Stim Area",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_visualize_stimulated_area, stimulated_area=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Stim vs Non-Stim ROIs",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_visualize_stimulated_area, with_rois=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Stim vs Non-Stim ROIs with Stim Area",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_visualize_stimulated_area, with_rois=True, stimulated_area=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)

AnalysisProduct(
    name="Stim vs Non-Stim Normalized Calcium Traces (Deconvolved ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stimulated_vs_non_stimulated_roi_traces,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Stim vs Non-Stim Normalized Calcium Traces with Peaks (Deconvolved ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_stimulated_vs_non_stimulated_roi_traces, with_peaks=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Stimulated vs Non-Stimulated Spike Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stimulated_vs_non_stimulated_spike_traces,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Stimulated vs Non-Stimulated Raster Calcium Peaks",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stimulated_vs_non_stimulated_calcium_peaks_raster,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name=(
        "Stimulated vs Non-Stimulated Raster Inferred Spikes Thresholded (Rising Edges)"
    ),
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stimulated_vs_non_stimulated_spike_raster,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Sorted Calcium Deconvolved ΔF/F0 Correlation",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_dec_dff_correlation,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Sorted Calcium Deconvolved ΔF/F0 Correlation (Stim Windows ±250ms)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_dec_dff_correlation_windowed_by_stim,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Sorted Calcium Deconvolved ΔF/F0 Correlation (Non-Stim Periods)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_dec_dff_correlation_windowed_non_stim,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Sorted Inferred Spikes Thresholded Global Synchrony",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_spike_synchrony,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Sorted Inferred Spikes Thresholded Global Synchrony (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_sorted_spike_synchrony, rising_edges=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Sorted Inferred Spikes Thresholded Max Lag Correlation",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_spike_max_lag_correlation,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Sorted Inferred Spikes Thresholded Max Lag Correlation (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_sorted_spike_max_lag_correlation, rising_edges=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Sorted Inferred Spikes Thresholded Max Lag Values",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_spike_max_lag_values,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Sorted Inferred Spikes Thresholded Max Lag Values (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_sorted_spike_max_lag_values, rising_edges=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Sorted Inferred Spikes Thresholded CCG Z-Score",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_spike_ccg_zscore,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Sorted Inferred Spikes Thresholded CCG Z-Score (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_sorted_spike_ccg_zscore, rising_edges=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    name="Stim vs Non-Stim Calcium Peaks Amplitudes",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stim_and_non_stim_peaks_amplitude,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)

# Cell Size Group
AnalysisProduct(
    name="Cell Size",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_cell_size_data,
    category="Cell Size",
    pipeline_stage=PipelineStage.DETECTION,
)

# Multi-Well Analysis Products --------------------------------------------------------
# These plot bar plots from database queries across multiple wells

# General Multi-Well Products
AnalysisProduct(
    name="Cell Size Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_cell_size_bar_plot,
    category="General",
    pipeline_stage=PipelineStage.DETECTION,
)
# AnalysisProduct(
#     name="Percentage of Active Cells Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_percentage_active_bar_plot,
#     category="General",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )
# AnalysisProduct(
#     name="Calcium Peaks Amplitude Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_calcium_peaks_amplitude_bar_plot,
#     category="General",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )
# AnalysisProduct(
#     name="Calcium Peaks Frequency Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_calcium_peaks_frequency_bar_plot,
#     category="General",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )
# AnalysisProduct(
#     name="Calcium Peaks Inter-Event Interval Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_calcium_peaks_iei_bar_plot,
#     category="General",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )
# AnalysisProduct(
#     name="Inferred Spikes Global Synchrony Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_spike_synchrony_bar_plot,
#     category="General",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )
# AnalysisProduct(
#     name="Burst Count Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_burst_count_bar_plot,
#     category="General",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )
# AnalysisProduct(
#     name="Burst Average Duration Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_burst_avg_duration_bar_plot,
#     category="General",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )
# AnalysisProduct(
#     name="Burst Average Interval Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_burst_avg_interval_bar_plot,
#     category="General",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )
# AnalysisProduct(
#     name="Burst Rate Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_burst_rate_bar_plot,
#     category="General",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )

# Evoked Multi-Well Products
# AnalysisProduct(
#     name="Stimulated Peaks Amplitude Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_stimulated_peaks_amplitude_bar_plot,
#     category="Evoked",
#     pipeline_stage=PipelineStage.ANALYSIS,
#     experiment_type=EVOKED,
# )
# AnalysisProduct(
#     name="Non-Stimulated Peaks Amplitude Bar Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_non_stimulated_peaks_amplitude_bar_plot,
#     category="Evoked",
#     pipeline_stage=PipelineStage.ANALYSIS,
#     experiment_type=EVOKED,
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
    for category, names in categories.items():
        # Create a divider key that won't be selectable
        divider_key = f"----------{category}".ljust(60, "-")
        result[divider_key] = names

    return result


# Generate the dictionaries on module load
SINGLE_WELL_COMBO_OPTIONS_DICT = _get_combo_options_dict(AnalysisGroup.SINGLE_WELL)
MULTI_WELL_COMBO_OPTIONS_DICT = _get_combo_options_dict(AnalysisGroup.MULTI_WELL)


def get_available_plots(
    group: AnalysisGroup,
    has_detection: bool = False,
    has_extraction: bool = False,
    has_analysis: bool = False,
    experiment_type: str | None = None,
) -> dict[str, list[str]]:
    """Filter available plots based on completed pipeline stages and experiment type.

    Parameters
    ----------
    group : AnalysisGroup
        Whether to return single-well or multi-well plots
    has_detection : bool
        Whether detection has been completed
    has_extraction : bool
        Whether extraction has been completed
    has_analysis : bool
        Whether analysis has been completed
    experiment_type : str | None
        Experiment type (use EVOKED constant for evoked experiments), None to show all

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping category headers to list of available plot names
    """
    # Group products by category, filtering by pipeline stage and experiment type
    categories: dict[str, list[str]] = {}
    for product in ANALYSIS_PRODUCTS:
        if product.group != group:
            continue

        # Check if this plot is available based on pipeline stages
        if product.pipeline_stage == PipelineStage.DETECTION and not has_detection:
            continue
        if product.pipeline_stage == PipelineStage.EXTRACTION and not has_extraction:
            continue
        if product.pipeline_stage == PipelineStage.ANALYSIS and not has_analysis:
            continue

        # Filter by experiment type
        if (
            product.experiment_type is not None
            and experiment_type != product.experiment_type
        ):
            continue

        # Add to categories
        if product.category not in categories:
            categories[product.category] = []
        categories[product.category].append(product.name)

    # Format with dividers for the combobox
    result = {}
    for category, names in categories.items():
        # Create a divider key that won't be selectable
        divider_key = f"---------------------{category}".ljust(60, "-")
        result[divider_key] = names

    return result


# Plots that require active ROIs only (for random selection filtering)
# Centralized configuration - easier to maintain than scattered logic
# Used by _graph_widgets.py to filter random ROI selection
ACTIVE_ONLY_PLOTS: set[str] = {
    "Calcium Deconvolved ΔF/F0 Traces with Peaks",
    "Calcium Deconvolved ΔF/F0 Traces with Peaks and Thresholds (1 ROI)",
    "Calcium Deconvolved ΔF/F0 Normalized Traces with Peaks",
    "Calcium Deconvolved ΔF/F0 Traces Normalized (Active Only)"
    "Calcium Peaks Amplitudes (Deconvolved ΔF/F0)",
    "Calcium Peaks Frequencies (Deconvolved ΔF/F0)",
    "Calcium Peaks Amplitudes vs Frequencies (Deconvolved ΔF/F0)",
    "Calcium Peaks Inter-event Interval (Deconvolved ΔF/F0)",
    "Calcium Peaks Raster plot Colored by ROI",
    "Calcium Peaks Raster plot Colored by Amplitude",
    "Calcium Peaks Raster plot Colored by Amplitude with Colorbar",
    "Calcium Burst Activity Analysis",
    "Calcium Functional Connectivity",
    "Inferred Spikes (Thresholded)",
    "Inferred Spikes Raw (with Thresholds - 1 ROI)",
    "Inferred Spikes (Thresholded) with Deconvolved ΔF/F0 Traces",
    "Inferred Spikes (Thresholded) Normalized",
    "Inferred Spikes (Thresholded) Normalized (Active Only)",
    "Inferred Spikes (Thresholded) Normalized with Network Bursts",
    "Inferred Spikes (Thresholded) Global Synchrony",
    "Inferred Spikes (Thresholded) Cross-Correlation",
    "Inferred Spikes (Thresholded) Burst Activity Analysis",
    "Inferred Spikes Thresholded",
    "Inferred Spikes Thresholded Normalized",
    "Inferred Spikes Thresholded (Active Only)",
    "Inferred Spikes Thresholded Normalized (Active Only)",
    "Inferred Spikes Thresholded Frequency",
    "Inferred Spikes Rising Edge Frequency",
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
    engine: Engine,
    fov_name: str,
    text: str,
    run_id: int | None,
    rois: list[int] | None = None,
) -> None:
    """Plot single-well analysis data using registry pattern with database queries.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        The widget to plot into
    engine : Engine
        SQLAlchemy Engine connected to the database
    fov_name : str
        Name of the FOV to query (e.g., "B5_0000")
    text : str
        The name of the analysis to plot (matches AnalysisProduct.name)
    run_id : int | None
        The CaliResult.id of the selected run to filter by, or None for default
    rois : list[int] | None, optional
        List of ROI indices to plot, by default None
    """
    try:
        # Look up the analysis in the registry
        for product in ANALYSIS_PRODUCTS:
            if product.name == text and product.group == AnalysisGroup.SINGLE_WELL:
                # Type narrowing: we know this is a SingleWellAnalyzer
                analyzer = cast("SingleWellAnalyzer", product.analyzer)
                # Pass run_id as keyword argument to avoid positional conflicts
                # with other keyword args in the analyzer functions
                return analyzer(widget, engine, fov_name, rois, run_id=run_id)  # type: ignore[no-any-return]

        # If we get here, analysis was not found
        cali_logger.warning(f"Analysis '{text}' not found in registry")

    except Exception as e:
        cali_logger.error(f"Error plotting single well data for '{text}': {e}")
        raise


def plot_multi_well_data(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
) -> None:
    """Plot multi-well data using registry pattern with database queries.

    Parameters
    ----------
    widget : _MultilWellGraphWidget
        The widget to plot into
    text : str
        The name of the analysis to plot (matches AnalysisProduct.name)
    engine : Engine
        SQLAlchemy Engine connected to the database
    run_id : int | None, optional
        The CaliResult.id of the selected run to filter by, by default None
    """
    # Handle empty/invalid selection
    if not text or text == "None" or text in MULTI_WELL_COMBO_OPTIONS_DICT.keys():
        widget.clear_plot()
        return

    try:
        # Look up the analysis in the registry
        for product in ANALYSIS_PRODUCTS:
            if product.name == text and product.group == AnalysisGroup.MULTI_WELL:
                # Type narrowing: we know this is a MultiWellAnalyzer
                analyzer = cast("MultiWellAnalyzer", product.analyzer)
                return analyzer(widget, text, engine, run_id)

        # If we get here, analysis was not found
        cali_logger.warning(f"Multi-well analysis '{text}' not found in registry")
        widget.clear_plot()

    except Exception as e:
        cali_logger.error(f"Error plotting multi-well data for '{text}': {e}")
        widget.clear_plot()
        raise
