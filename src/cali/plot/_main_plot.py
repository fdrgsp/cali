from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, cast

from sqlmodel import Session, col, select
from typing_extensions import TypeAlias

from cali._constants import EVOKED
from cali.sqlmodel._engine import ensure_schema_current
from cali.sqlmodel._model import FOV, ROI, Traces
from cali.sqlmodel._spike_settings import canonical_spike_methods
from cali.sqlmodel._trace_provenance import ExtractionFrameWindow

from ._multi_wells_plots import (
    compute_burst_avg_duration_data,
    compute_burst_avg_interval_data,
    compute_burst_count_data,
    compute_burst_rate_data,
    compute_calcium_amplitude_stim_split_data,
    compute_calcium_burst_avg_duration_data,
    compute_calcium_burst_avg_interval_data,
    compute_calcium_burst_count_data,
    compute_calcium_den_dff_correlation_data,
    compute_calcium_dff_correlation_data,
    compute_cell_size_data,
    compute_fraction_significant_ccg_pairs_data,
    compute_fraction_significant_ccg_pairs_rising_edges_data,
    compute_percentage_active_data,
    compute_percentage_active_stim_split_data,
    compute_spike_correlation_data,
    compute_spike_correlation_rising_edges_data,
    compute_spike_synchrony_data,
    compute_spike_synchrony_rising_edges_data,
    make_parameter_compute_fn,
    plot_burst_avg_duration_bar_plot,
    plot_burst_avg_interval_bar_plot,
    plot_burst_count_bar_plot,
    plot_burst_rate_bar_plot,
    plot_calcium_burst_avg_duration_bar_plot,
    plot_calcium_burst_avg_interval_bar_plot,
    plot_calcium_burst_count_bar_plot,
    plot_calcium_den_dff_correlation_bar_plot,
    plot_calcium_dff_correlation_bar_plot,
    plot_calcium_peaks_amplitude_bar_plot,
    plot_calcium_peaks_amplitude_stim_split_bar_plot,
    plot_calcium_peaks_frequency_bar_plot,
    plot_calcium_peaks_frequency_stim_split_bar_plot,
    plot_calcium_peaks_iei_bar_plot,
    plot_cell_size_bar_plot,
    plot_fraction_significant_ccg_pairs_bar_plot,
    plot_fraction_significant_ccg_pairs_rising_edges_bar_plot,
    plot_inferred_spikes_frequency_bar_plot,
    plot_inferred_spikes_frequency_stim_split_bar_plot,
    plot_inferred_spikes_rising_edge_frequency_bar_plot,
    plot_inferred_spikes_rising_edge_frequency_stim_split_bar_plot,
    # plot_pca_loadings,
    # plot_pca_scatter,
    # plot_pca_scatter_stim_split,
    # plot_pca_scree,
    plot_percentage_active_bar_plot,
    plot_percentage_active_stim_split_bar_plot,
    plot_spike_correlation_bar_plot,
    plot_spike_correlation_rising_edges_bar_plot,
    plot_spike_synchrony_bar_plot,
    plot_spike_synchrony_rising_edges_bar_plot,
)
from ._multi_wells_plots._util import plot_parameter_bar_plot
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
from ._single_wells_plots.cluster._plot_cluster_analysis import (
    _plot_cluster_colored_raster,
    _plot_cluster_colored_traces,
    _plot_cluster_connectivity_graph,
    _plot_cluster_sorted_correlation_heatmap,
)
from ._single_wells_plots.correlation._plot_calcium_traces_correlation import (
    _plot_den_dff_correlation_data,
    _plot_dff_correlation_data,
)
from ._single_wells_plots.correlation._plot_connectivity import (
    _plot_connectivity_network_data,
)
from ._single_wells_plots.correlation._plot_evoked_correlation_synchrony import (
    _plot_sorted_den_dff_correlation,
    _plot_sorted_den_dff_correlation_windowed_by_stim,
    _plot_sorted_den_dff_correlation_windowed_non_stim,
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
from ._spike_data import get_stored_spike_capabilities

if TYPE_CHECKING:
    from collections.abc import Collection

    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import (
        _MultilWellGraphWidget,
        _SingleWellGraphWidget,
    )
    from cali.sqlmodel._spike_settings import SpikeMethod

    from ._multi_wells_plots._util import BarPlotData


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
# Headless compute function: (engine, run_id) -> (BarPlotData, name, units) | None
ComputeFn: TypeAlias = "Callable[..., tuple[BarPlotData, str, str] | None]"


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
    product_id : str
        Stable identifier; display names remain accepted as legacy lookup aliases.
    supported_spike_methods : tuple | None
        Allowed results methods; None describes a shared calcium/general product.
    required_metrics : tuple[str, ...]
        Stored fields needed to offer this product for the selected method.
    category : str
        Category for grouping in the UI (e.g., "Calcium Traces", "Evoked Experiment")
    pipeline_stage : PipelineStage
        Minimum pipeline stage required to generate this plot
    experiment_type : str | None
        Required experiment type ("evoked" or "spontaneous"), None for all types
    compute_fn : ComputeFn | None
        Headless function that computes bar plot data without rendering.
        Signature: (engine, run_id) -> (BarPlotData, name, units) | None.
        Used for batch CSV export of multi-well aggregated data.
    """

    name: str
    group: AnalysisGroup
    analyzer: AnyAnalyzer
    product_id: str
    category: str = "General"
    pipeline_stage: PipelineStage = PipelineStage.ANALYSIS
    experiment_type: str | None = None  # "evoked", "spontaneous", or None for all
    compute_fn: ComputeFn | None = None
    supported_spike_methods: tuple[SpikeMethod, ...] | None = None
    required_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Register this product in the global registry."""
        if not self.product_id or any(
            self.name == product.name or self.product_id == product.product_id
            for product in ANALYSIS_PRODUCTS
        ):
            raise ValueError(f"AnalysisProduct '{self.name}' already registered.")
        if self.supported_spike_methods is not None:
            self.supported_spike_methods = canonical_spike_methods(
                self.supported_spike_methods
            )
        ANALYSIS_PRODUCTS.append(self)

    def selected_method(self, method: SpikeMethod | None) -> SpikeMethod | None:
        """Resolve one method and reject unsupported requests before dispatch."""
        if method is not None:
            canonical_spike_methods((method,))
        if self.supported_spike_methods is None:
            return None
        method = method or self.supported_spike_methods[0]
        if method not in self.supported_spike_methods:
            raise ValueError(f"Plot {self.product_id!r} does not support {method}.")
        return method

    def compute_data(
        self,
        engine: Engine,
        run_id: int | None,
        *,
        spike_method: SpikeMethod | None = None,
    ) -> tuple[BarPlotData, str, str] | None:
        """Compute a method-qualified product without depending on GUI filtering."""
        method = self.selected_method(spike_method)
        if self.compute_fn is None:
            return None
        if method is not None and run_id is not None:
            metrics = get_stored_spike_capabilities(engine, run_id).get(method, set())
            if not set(self.required_metrics) <= metrics:
                return None
        if self.supported_spike_methods == ("oasis", "cascade"):
            return self.compute_fn(engine, run_id, spike_method=method)
        return self.compute_fn(engine, run_id)


# Global registry of all analysis products
ANALYSIS_PRODUCTS: list[AnalysisProduct] = []


# REGISTER SINGLE WELL ANALYSIS PRODUCTS ==============================================
# Define all analysis products using the AnalysisProduct dataclass

# Calcium Traces Group
AnalysisProduct(
    product_id="single_well.calcium_raw_traces",
    name="Calcium Raw Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, raw=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    product_id="single_well.calcium_raw_normalized_traces",
    name="Calcium Raw Normalized Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, raw=True, normalize=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    product_id="single_well.calcium_neuropil_corrected_traces",
    name="Calcium Neuropil Corrected Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_neuropil_traces, corrected=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    product_id="single_well.neuropil_and_raw_traces",
    name="Neuropil and Raw Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_neuropil_traces,
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    product_id="single_well.calcium_dff_traces",
    name="Calcium ΔF/F0 Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dff=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    product_id="single_well.calcium_dff_normalized_traces",
    name="Calcium ΔF/F0 Normalized  Traces ",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dff=True, normalize=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    product_id="single_well.calcium_denoised_dff_traces",
    name="Calcium Denoised ΔF/F0 Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    product_id="single_well.calcium_denoised_dff_traces_with_peaks",
    name="Calcium Denoised ΔF/F0 Traces with Peaks",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, with_peaks=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_denoised_dff_traces_with_peaks_and_thresholds_1_roi",
    name="Calcium Denoised ΔF/F0 Traces with Peaks and Thresholds (1 ROI)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, with_peaks=True, thresholds=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_denoised_dff_normalized_traces",
    name="Calcium Denoised ΔF/F0 Normalized Traces ",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, normalize=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    product_id="single_well.calcium_denoised_dff_traces_normalized_active_only",
    name="Calcium Denoised ΔF/F0 Traces Normalized (Active Only)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, normalize=True, active_only=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_denoised_dff_normalized_traces_with_peaks",
    name="Calcium Denoised ΔF/F0 Normalized Traces with Peaks",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, dec=True, normalize=True, with_peaks=True),
    category="Calcium Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Inferred Spikes Group
AnalysisProduct(
    product_id="single_well.inferred_spikes",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_trace",),
    name="Inferred Spikes",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, raw=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_normalized",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_trace",),
    name="Inferred Spikes Normalized",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, normalize=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_normalized_active_only",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_trace",),
    name="Inferred Spikes Normalized (Active Only)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, normalize=True, active_only=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_with_thresholds_if_1_roi",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_trace", "threshold"),
    name="Inferred Spikes (with Thresholds if 1 ROI)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, raw=True, thresholds=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_with_denoised_dff_traces",
    supported_spike_methods=("oasis",),
    required_metrics=("spike_trace",),
    name="Inferred Spikes with Denoised ΔF/F0 Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, den_dff=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.EXTRACTION,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_trace", "threshold"),
    name="Inferred Spikes Thresholded",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, thresholded=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_normalized",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_trace", "threshold"),
    name="Inferred Spikes Thresholded Normalized",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes, thresholded=True, normalize=True),
    category="Inferred Spikes Traces",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_normalized_active_only",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_trace", "threshold"),
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
    product_id="single_well.calcium_peaks_raster",
    name="Calcium Peaks Raster",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_generate_raster_plot,
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_peaks_raster_plot_colored_by_amplitude",
    name="Calcium Peaks Raster plot Colored by Amplitude",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_generate_raster_plot, amplitude_colors=True, colorbar=False),
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_peaks_raster_plot_colored_by_amplitude_with_colorbar",
    name="Calcium Peaks Raster plot Colored by Amplitude with Colorbar",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_generate_raster_plot, amplitude_colors=True, colorbar=True),
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_intensity_heatmap",
    name="Calcium Intensity Heatmap",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_generate_intensity_heatmap,
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_raster_thresholded",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_trace", "threshold"),
    name="Inferred Spikes Raster Thresholded",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_generate_spike_raster_plot,
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_raster_thresholded_rising_edges",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_trace", "threshold"),
    name="Inferred Spikes Raster Thresholded (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_generate_spike_raster_plot, edges=True),
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Calcium Amplitude and Frequency Group
AnalysisProduct(
    product_id="single_well.calcium_peaks_amplitudes_denoised_dff",
    name="Calcium Peaks Amplitudes (Denoised ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_amplitude_and_frequency_data, amp=True),
    category="Calcium Peaks Amplitude, Frequency and Event Interval",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_peaks_frequencies_denoised_dff",
    name="Calcium Peaks Frequencies (Denoised ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_amplitude_and_frequency_data, freq=True),
    category="Calcium Peaks Amplitude, Frequency and Event Interval",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_peaks_amplitudes_vs_frequencies_denoised_dff",
    name="Calcium Peaks Amplitudes vs Frequencies (Denoised ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_amplitude_and_frequency_data, amp=True, freq=True),
    category="Calcium Peaks Amplitude, Frequency and Event Interval",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_peaks_inter_event_interval_denoised_dff",
    name="Calcium Peaks Inter-event Interval (Denoised ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_iei_data,
    category="Calcium Peaks Amplitude, Frequency and Event Interval",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Inferred Spikes Frequency Group
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_frequency",
    supported_spike_methods=("oasis",),
    required_metrics=("suprathreshold_sample_rate_hz",),
    name="Inferred Spikes Thresholded Frequency",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes_frequency_data, rising_edge=False),
    category="Inferred Spikes Frequency",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_frequency_rising_edges",
    supported_spike_methods=("oasis",),
    required_metrics=("suprathreshold_rising_edge_rate_hz",),
    name="Inferred Spikes Thresholded Frequency (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_inferred_spikes_frequency_data, rising_edge=True),
    category="Inferred Spikes Frequency",
    pipeline_stage=PipelineStage.ANALYSIS,
)

AnalysisProduct(
    product_id="single_well.cascade_expected_spike_rate_hz",
    name="CASCADE Expected Spike Rate",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(
        _plot_inferred_spikes_frequency_data,
        spike_method="cascade",
        metric="expected_spike_rate_hz",
    ),
    category="Inferred Spikes Frequency",
    supported_spike_methods=("cascade",),
    required_metrics=("expected_spike_rate_hz",),
)
AnalysisProduct(
    product_id="multi_well.cascade_expected_spike_rate_hz",
    name="CASCADE Expected Spike Rate Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        plot_parameter_bar_plot,
        parameter="expected_spike_rate_hz",
        units="Hz",
        spike_method="cascade",
    ),
    compute_fn=make_parameter_compute_fn(
        "expected_spike_rate_hz",
        "Hz",
        "CASCADE Expected Spike Rate",
        spike_method="cascade",
    ),
    category="Inferred Spikes",
    supported_spike_methods=("cascade",),
    required_metrics=("expected_spike_rate_hz",),
)
AnalysisProduct(
    product_id="single_well.cascade_expected_spike_count",
    name="CASCADE Expected Spike Count",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(
        _plot_inferred_spikes_frequency_data,
        spike_method="cascade",
        metric="expected_spike_count",
    ),
    category="Inferred Spikes Frequency",
    supported_spike_methods=("cascade",),
    required_metrics=("expected_spike_count",),
)
AnalysisProduct(
    product_id="multi_well.cascade_expected_spike_count",
    name="CASCADE Expected Spike Count Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        plot_parameter_bar_plot,
        parameter="expected_spike_count",
        units="spikes",
        spike_method="cascade",
    ),
    compute_fn=make_parameter_compute_fn(
        "expected_spike_count",
        "spikes",
        "CASCADE Expected Spike Count",
        spike_method="cascade",
    ),
    category="Inferred Spikes",
    supported_spike_methods=("cascade",),
    required_metrics=("expected_spike_count",),
)
AnalysisProduct(
    product_id="single_well.cascade_suprathreshold_excursion_rate_hz",
    name="CASCADE Threshold Excursion Rate",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(
        _plot_inferred_spikes_frequency_data,
        spike_method="cascade",
        metric="suprathreshold_excursion_rate_hz",
    ),
    category="Inferred Spikes Frequency",
    supported_spike_methods=("cascade",),
    required_metrics=("suprathreshold_excursion_rate_hz",),
)
AnalysisProduct(
    product_id="multi_well.cascade_suprathreshold_excursion_rate_hz",
    name="CASCADE Threshold Excursion Rate Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=partial(
        plot_parameter_bar_plot,
        parameter="suprathreshold_excursion_rate_hz",
        units="Hz",
        spike_method="cascade",
    ),
    compute_fn=make_parameter_compute_fn(
        "suprathreshold_excursion_rate_hz",
        "Hz",
        "CASCADE Threshold Excursion Rate",
        spike_method="cascade",
    ),
    category="Inferred Spikes",
    supported_spike_methods=("cascade",),
    required_metrics=("suprathreshold_excursion_rate_hz",),
)

# Calcium Burst Analysis Group
AnalysisProduct(
    product_id="single_well.calcium_burst_activity_analysis",
    name="Calcium Burst Activity Analysis",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_calcium_burst_activity,
    category="Calcium Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_traces_normalized_with_network_bursts",
    name="Calcium Traces Normalized with Network Bursts",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_calcium_normalized_with_bursts,
    category="Calcium Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_raster_with_network_bursts",
    name="Calcium Raster with Network Bursts",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_calcium_raster_with_bursts,
    category="Calcium Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
# Inferred Spike Burst Analysis Group
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_burst_activity_analysis",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_population_activity",),
    name="Inferred Spikes Thresholded Burst Activity Analysis",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_inferred_spike_burst_activity,
    category="Inferred Spike Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_normalized_with_network_bursts",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_population_activity", "spike_trace", "threshold"),
    name="Inferred Spikes Thresholded Normalized with Network Bursts",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_inferred_spikes_normalized_with_bursts,
    category="Inferred Spike Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spike_raster_with_network_bursts",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_population_activity", "spike_trace", "threshold"),
    name="Inferred Spike Raster with Network Bursts",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_inferred_spike_raster_with_bursts,
    category="Inferred Spike Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Correlation Analysis Group
AnalysisProduct(
    product_id="single_well.calcium_dff_correlation",
    name="Calcium ΔF/F0 Correlation",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_dff_correlation_data,
    category="Calcium Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_denoised_dff_correlation",
    name="Calcium Denoised ΔF/F0 Correlation",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_den_dff_correlation_data,
    category="Calcium Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.calcium_functional_connectivity_pearson_correlation",
    name="Calcium Functional Connectivity (Pearson Correlation)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_connectivity_network_data,
    category="Calcium Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Cluster Analysis Group
AnalysisProduct(
    product_id="single_well.calcium_functional_connectivity_clustering",
    name="Calcium Functional Connectivity (Clustering)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_cluster_connectivity_graph,
    category="Cluster Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.cluster_sorted_correlation_heatmap_denoised_dff",
    name="Cluster-Sorted Correlation Heatmap (Denoised ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_cluster_sorted_correlation_heatmap,
    category="Cluster Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.cluster_colored_calcium_peaks_raster_denoised_dff",
    name="Cluster-Colored Calcium Peaks Raster (Denoised ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_cluster_colored_raster,
    category="Cluster Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.cluster_colored_denoised_dff_traces",
    name="Cluster-Colored Denoised ΔF/F0 Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_cluster_colored_traces,
    category="Cluster Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Inferred Spikes Correlation Analysis Group
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_max_lag_correlation",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_max_lag_correlation_matrix",),
    name="Inferred Spikes Thresholded Max Lag Correlation",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_spike_max_lag_correlation_data,
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_max_lag_correlation_rising_edges",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_max_lag_correlation_matrix_rising_edges",),
    name="Inferred Spikes Thresholded Max Lag Correlation (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_spike_max_lag_correlation_data, rising_edges=True),
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    required_metrics=("spike_ccg_zscore_matrix",),
    supported_spike_methods=("oasis", "cascade"),
    product_id="single_well.inferred_spikes_thresholded_ccg_z_score",
    name="Inferred Spikes Thresholded CCG Z-Score",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_ccg_zscore_data,
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    required_metrics=("spike_ccg_zscore_matrix_rising_edges",),
    supported_spike_methods=("oasis", "cascade"),
    product_id="single_well.inferred_spikes_thresholded_ccg_z_score_rising_edges",
    name="Inferred Spikes Thresholded CCG Z-Score (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_ccg_zscore_data, rising_edges=True),
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_max_lag_values",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_max_lag_values_matrix",),
    name="Inferred Spikes Thresholded Max Lag Values",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_spike_max_lag_values_data,
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_max_lag_values_rising_edges",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_max_lag_values_matrix_rising_edges",),
    name="Inferred Spikes Thresholded Max Lag Values (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_spike_max_lag_values_data, rising_edges=True),
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_global_synchrony",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_jitter_synchrony_matrix",),
    name="Inferred Spikes Thresholded Global Synchrony",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_spike_synchrony_data,
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)
AnalysisProduct(
    product_id="single_well.inferred_spikes_thresholded_global_synchrony_rising_edges",
    supported_spike_methods=("oasis", "cascade"),
    required_metrics=("spike_jitter_synchrony_matrix_rising_edges",),
    name="Inferred Spikes Thresholded Global Synchrony (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_spike_synchrony_data, rising_edges=True),
    category="Inferred Spikes Correlation Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
)

# Evoked Experiment Group
AnalysisProduct(
    product_id="single_well.stim_area",
    name="Stim Area",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_visualize_stimulated_area, stimulated_area=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.stim_vs_non_stim_rois",
    name="Stim vs Non-Stim ROIs",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_visualize_stimulated_area, with_rois=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.stim_vs_non_stim_rois_with_stim_area",
    name="Stim vs Non-Stim ROIs with Stim Area",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_visualize_stimulated_area, with_rois=True, stimulated_area=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)

AnalysisProduct(
    product_id="single_well.stim_vs_non_stim_normalized_calcium_traces_denoised_dff",
    name="Stim vs Non-Stim Normalized Calcium Traces (Denoised ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stimulated_vs_non_stimulated_roi_traces,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.stim_vs_non_stim_normalized_calcium_traces_with_peaks_denoised_dff",
    name="Stim vs Non-Stim Normalized Calcium Traces with Peaks (Denoised ΔF/F0)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_stimulated_vs_non_stimulated_roi_traces, with_peaks=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.stimulated_vs_non_stimulated_spike_traces",
    supported_spike_methods=("oasis",),
    name="Stimulated vs Non-Stimulated Spike Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stimulated_vs_non_stimulated_spike_traces,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.stimulated_vs_non_stimulated_raster_calcium_peaks",
    name="Stimulated vs Non-Stimulated Raster Calcium Peaks",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stimulated_vs_non_stimulated_calcium_peaks_raster,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.stimulated_vs_non_stimulated_raster_inferred_spikes_thresholded_rising_edges",
    supported_spike_methods=("oasis",),
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
    product_id="single_well.sorted_calcium_denoised_dff_correlation",
    name="Sorted Calcium Denoised ΔF/F0 Correlation",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_den_dff_correlation,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.sorted_calcium_denoised_dff_correlation_stim_windows_250ms",
    name="Sorted Calcium Denoised ΔF/F0 Correlation (Stim Windows ±250ms)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_den_dff_correlation_windowed_by_stim,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.sorted_calcium_denoised_dff_correlation_non_stim_periods",
    name="Sorted Calcium Denoised ΔF/F0 Correlation (Non-Stim Periods)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_den_dff_correlation_windowed_non_stim,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.sorted_inferred_spikes_thresholded_global_synchrony",
    supported_spike_methods=("oasis",),
    required_metrics=("spike_jitter_synchrony_matrix",),
    name="Sorted Inferred Spikes Thresholded Global Synchrony",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_spike_synchrony,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.sorted_inferred_spikes_thresholded_global_synchrony_rising_edges",
    supported_spike_methods=("oasis",),
    required_metrics=("spike_jitter_synchrony_matrix_rising_edges",),
    name="Sorted Inferred Spikes Thresholded Global Synchrony (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_sorted_spike_synchrony, rising_edges=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.sorted_inferred_spikes_thresholded_max_lag_correlation",
    supported_spike_methods=("oasis",),
    required_metrics=("spike_max_lag_correlation_matrix",),
    name="Sorted Inferred Spikes Thresholded Max Lag Correlation",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_spike_max_lag_correlation,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.sorted_inferred_spikes_thresholded_max_lag_correlation_rising_edges",
    supported_spike_methods=("oasis",),
    required_metrics=("spike_max_lag_correlation_matrix_rising_edges",),
    name="Sorted Inferred Spikes Thresholded Max Lag Correlation (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_sorted_spike_max_lag_correlation, rising_edges=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.sorted_inferred_spikes_thresholded_max_lag_values",
    supported_spike_methods=("oasis",),
    required_metrics=("spike_max_lag_values_matrix",),
    name="Sorted Inferred Spikes Thresholded Max Lag Values",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_spike_max_lag_values,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.sorted_inferred_spikes_thresholded_max_lag_values_rising_edges",
    supported_spike_methods=("oasis",),
    required_metrics=("spike_max_lag_values_matrix_rising_edges",),
    name="Sorted Inferred Spikes Thresholded Max Lag Values (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_sorted_spike_max_lag_values, rising_edges=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.sorted_inferred_spikes_thresholded_ccg_z_score",
    supported_spike_methods=("oasis",),
    required_metrics=("spike_ccg_zscore_matrix",),
    name="Sorted Inferred Spikes Thresholded CCG Z-Score",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_sorted_spike_ccg_zscore,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.sorted_inferred_spikes_thresholded_ccg_z_score_rising_edges",
    supported_spike_methods=("oasis",),
    required_metrics=("spike_ccg_zscore_matrix_rising_edges",),
    name="Sorted Inferred Spikes Thresholded CCG Z-Score (Rising Edges)",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_sorted_spike_ccg_zscore, rising_edges=True),
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)
AnalysisProduct(
    product_id="single_well.stim_vs_non_stim_calcium_peaks_amplitudes",
    name="Stim vs Non-Stim Calcium Peaks Amplitudes",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_stim_and_non_stim_peaks_amplitude,
    category="Evoked Experiment",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
)

# Cell Size Group
AnalysisProduct(
    product_id="single_well.cell_size",
    name="Cell Size",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_cell_size_data,
    category="Cell Size",
    pipeline_stage=PipelineStage.DETECTION,
)

# Multi-Well Analysis Products --------------------------------------------------------
# These plot bar plots from database queries across multiple wells

# General Multi-Well Products — scalar per-ROI metrics
AnalysisProduct(
    product_id="multi_well.cell_size_bar_plot",
    name="Cell Size Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_cell_size_bar_plot,
    category="General",
    pipeline_stage=PipelineStage.DETECTION,
    compute_fn=compute_cell_size_data,
)
AnalysisProduct(
    product_id="multi_well.percentage_of_active_cells_bar_plot",
    name="Percentage of Active Cells Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_percentage_active_bar_plot,
    category="General",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_percentage_active_data,
)
AnalysisProduct(
    product_id="multi_well.calcium_peaks_amplitude_bar_plot",
    name="Calcium Peaks Amplitude Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_calcium_peaks_amplitude_bar_plot,
    category="Calcium Peaks",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=make_parameter_compute_fn(
        "peaks_amplitudes_den_dff", "ΔF/F0", "Calcium Peaks Amplitude"
    ),
)
AnalysisProduct(
    product_id="multi_well.calcium_peaks_frequency_bar_plot",
    name="Calcium Peaks Frequency Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_calcium_peaks_frequency_bar_plot,
    category="Calcium Peaks",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=make_parameter_compute_fn(
        "den_dff_frequency", "Hz", "Calcium Peaks Frequency"
    ),
)
AnalysisProduct(
    product_id="multi_well.calcium_peaks_inter_event_interval_bar_plot",
    name="Calcium Peaks Inter-Event Interval Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_calcium_peaks_iei_bar_plot,
    category="Calcium Peaks",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=make_parameter_compute_fn("iei", "s", "Calcium Peaks IEI"),
)

# Multi-Well Products — calcium burst metrics
AnalysisProduct(
    product_id="multi_well.calcium_burst_count_bar_plot",
    name="Calcium Burst Count Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_calcium_burst_count_bar_plot,
    category="Calcium Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_calcium_burst_count_data,
)
AnalysisProduct(
    product_id="multi_well.calcium_burst_average_duration_bar_plot",
    name="Calcium Burst Average Duration Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_calcium_burst_avg_duration_bar_plot,
    category="Calcium Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_calcium_burst_avg_duration_data,
)
AnalysisProduct(
    product_id="multi_well.calcium_burst_average_interval_bar_plot",
    name="Calcium Burst Average Interval Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_calcium_burst_avg_interval_bar_plot,
    category="Calcium Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_calcium_burst_avg_interval_data,
)

AnalysisProduct(
    product_id="multi_well.inferred_spikes_frequency_bar_plot",
    supported_spike_methods=("oasis",),
    required_metrics=("suprathreshold_sample_rate_hz",),
    name="Inferred Spikes Frequency Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_inferred_spikes_frequency_bar_plot,
    category="Inferred Spikes",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=make_parameter_compute_fn(
        "inferred_spikes_frequency", "Hz", "Inferred Spikes Frequency"
    ),
)
AnalysisProduct(
    product_id="multi_well.inferred_spikes_rising_edge_frequency_bar_plot",
    supported_spike_methods=("oasis",),
    required_metrics=("suprathreshold_rising_edge_rate_hz",),
    name="Inferred Spikes Rising Edge Frequency Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_inferred_spikes_rising_edge_frequency_bar_plot,
    category="Inferred Spikes",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=make_parameter_compute_fn(
        "inferred_spikes_rising_edge_frequency",
        "Hz",
        "Inferred Spikes Rising Edge Frequency",
    ),
)

# Multi-Well Products — inferred spikes burst metrics
AnalysisProduct(
    required_metrics=("spike_burst_count",),
    supported_spike_methods=("oasis", "cascade"),
    product_id="multi_well.inferred_spikes_burst_count_bar_plot",
    name="Inferred Spikes Burst Count Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_burst_count_bar_plot,
    category="Inferred Spikes Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_burst_count_data,
)
AnalysisProduct(
    required_metrics=("spike_burst_avg_duration",),
    supported_spike_methods=("oasis", "cascade"),
    product_id="multi_well.inferred_spikes_burst_average_duration_bar_plot",
    name="Inferred Spikes Burst Average Duration Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_burst_avg_duration_bar_plot,
    category="Inferred Spikes Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_burst_avg_duration_data,
)
AnalysisProduct(
    required_metrics=("spike_burst_avg_interval",),
    supported_spike_methods=("oasis", "cascade"),
    product_id="multi_well.inferred_spikes_burst_average_interval_bar_plot",
    name="Inferred Spikes Burst Average Interval Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_burst_avg_interval_bar_plot,
    category="Inferred Spikes Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_burst_avg_interval_data,
)
AnalysisProduct(
    required_metrics=("spike_burst_count",),
    supported_spike_methods=("oasis", "cascade"),
    product_id="multi_well.inferred_spikes_burst_rate_bar_plot",
    name="Inferred Spikes Burst Rate Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_burst_rate_bar_plot,
    category="Inferred Spikes Burst Analysis",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_burst_rate_data,
)

# Multi-Well Products — network metrics (FOV-level scalars)
AnalysisProduct(
    product_id="multi_well.calcium_dff_correlation_bar_plot",
    name="Calcium ΔF/F Correlation Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_calcium_dff_correlation_bar_plot,
    category="Calcium Correlation",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_calcium_dff_correlation_data,
)
AnalysisProduct(
    product_id="multi_well.calcium_denoised_dff_correlation_bar_plot",
    name="Calcium Denoised ΔF/F Correlation Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_calcium_den_dff_correlation_bar_plot,
    category="Calcium Correlation",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_calcium_den_dff_correlation_data,
)
AnalysisProduct(
    required_metrics=("global_spike_jitter_synchrony",),
    product_id="multi_well.spike_jitter_synchrony_bar_plot",
    supported_spike_methods=("oasis", "cascade"),
    name="Spike Jitter Synchrony Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_spike_synchrony_bar_plot,
    category="Spike Synchrony",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_spike_synchrony_data,
)
AnalysisProduct(
    required_metrics=("global_spike_jitter_synchrony_rising_edges",),
    product_id="multi_well.spike_jitter_synchrony_bar_plot_rising_edges",
    supported_spike_methods=("oasis", "cascade"),
    name="Spike Jitter Synchrony Bar Plot (Rising Edges)",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_spike_synchrony_rising_edges_bar_plot,
    category="Spike Synchrony",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_spike_synchrony_rising_edges_data,
)
AnalysisProduct(
    required_metrics=("global_spike_max_lag_correlation",),
    product_id="multi_well.spike_max_lag_correlation_bar_plot",
    supported_spike_methods=("oasis", "cascade"),
    name="Spike Max-Lag Correlation Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_spike_correlation_bar_plot,
    category="Spike Synchrony",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_spike_correlation_data,
)
AnalysisProduct(
    required_metrics=("global_spike_max_lag_correlation_rising_edges",),
    product_id="multi_well.spike_max_lag_correlation_bar_plot_rising_edges",
    supported_spike_methods=("oasis", "cascade"),
    name="Spike Max-Lag Correlation Bar Plot (Rising Edges)",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_spike_correlation_rising_edges_bar_plot,
    category="Spike Synchrony",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_spike_correlation_rising_edges_data,
)
AnalysisProduct(
    required_metrics=("fraction_significant_ccg_pairs",),
    supported_spike_methods=("oasis", "cascade"),
    product_id="multi_well.fraction_significant_ccg_pairs_bar_plot",
    name="Fraction Significant CCG Pairs Bar Plot",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_fraction_significant_ccg_pairs_bar_plot,
    category="Spike Synchrony",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_fraction_significant_ccg_pairs_data,
)
AnalysisProduct(
    required_metrics=("fraction_significant_ccg_pairs_rising_edges",),
    supported_spike_methods=("oasis", "cascade"),
    product_id="multi_well.fraction_significant_ccg_pairs_bar_plot_rising_edges",
    name="Fraction Significant CCG Pairs Bar Plot (Rising Edges)",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_fraction_significant_ccg_pairs_rising_edges_bar_plot,
    category="Spike Synchrony",
    pipeline_stage=PipelineStage.ANALYSIS,
    compute_fn=compute_fraction_significant_ccg_pairs_rising_edges_data,
)

# Multi-Well Products — PCA
# AnalysisProduct(
#     name="PCA Scatter (FOV Feature Space)",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_pca_scatter,
#     category="PCA",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )
# AnalysisProduct(
#     name="PCA Scatter (Stim vs NonStim)",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_pca_scatter_stim_split,
#     category="PCA",
#     pipeline_stage=PipelineStage.ANALYSIS,
#     experiment_type=EVOKED,
# )
# AnalysisProduct(
#     name="PCA Loadings (PC1)",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_pca_loadings,
#     category="PCA",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )
# AnalysisProduct(
#     name="PCA Scree Plot",
#     group=AnalysisGroup.MULTI_WELL,
#     analyzer=plot_pca_scree,
#     category="PCA",
#     pipeline_stage=PipelineStage.ANALYSIS,
# )

# Evoked Multi-Well Products
AnalysisProduct(
    product_id="multi_well.calcium_peaks_amplitude_bar_plot_stim_vs_nonstim",
    name="Calcium Peaks Amplitude Bar Plot (Stim vs NonStim)",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_calcium_peaks_amplitude_stim_split_bar_plot,
    category="Evoked",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
    compute_fn=compute_calcium_amplitude_stim_split_data,
)
AnalysisProduct(
    product_id="multi_well.calcium_peaks_frequency_bar_plot_stim_vs_nonstim",
    name="Calcium Peaks Frequency Bar Plot (Stim vs NonStim)",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_calcium_peaks_frequency_stim_split_bar_plot,
    category="Evoked",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
    compute_fn=make_parameter_compute_fn(
        "den_dff_frequency",
        "Hz",
        "Calcium Peaks Frequency (Stim vs NonStim)",
        include_stim_status=True,
    ),
)
AnalysisProduct(
    product_id="multi_well.percentage_of_active_cells_bar_plot_stim_vs_nonstim",
    name="Percentage of Active Cells Bar Plot (Stim vs NonStim)",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_percentage_active_stim_split_bar_plot,
    category="Evoked",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
    compute_fn=compute_percentage_active_stim_split_data,
)
AnalysisProduct(
    product_id="multi_well.inferred_spikes_frequency_bar_plot_stim_vs_nonstim",
    supported_spike_methods=("oasis",),
    required_metrics=("suprathreshold_sample_rate_hz",),
    name="Inferred Spikes Frequency Bar Plot (Stim vs NonStim)",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_inferred_spikes_frequency_stim_split_bar_plot,
    category="Evoked",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
    compute_fn=make_parameter_compute_fn(
        "inferred_spikes_frequency",
        "Hz",
        "Inferred Spikes Frequency (Stim vs NonStim)",
        include_stim_status=True,
    ),
)
AnalysisProduct(
    product_id="multi_well.inferred_spikes_rising_edge_frequency_bar_plot_stim_vs_nonstim",
    supported_spike_methods=("oasis",),
    required_metrics=("suprathreshold_rising_edge_rate_hz",),
    name="Inferred Spikes Rising Edge Frequency Bar Plot (Stim vs NonStim)",
    group=AnalysisGroup.MULTI_WELL,
    analyzer=plot_inferred_spikes_rising_edge_frequency_stim_split_bar_plot,
    category="Evoked",
    pipeline_stage=PipelineStage.ANALYSIS,
    experiment_type=EVOKED,
    compute_fn=make_parameter_compute_fn(
        "inferred_spikes_rising_edge_frequency",
        "Hz",
        "Inferred Spikes Rising Edge Frequency (Stim vs NonStim)",
        include_stim_status=True,
    ),
)

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
    *,
    stored_spike_methods: tuple[SpikeMethod, ...] | None = None,
    spike_method: SpikeMethod = "oasis",
    available_metrics: Collection[str] | None = None,
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
    stored_spike_methods : tuple | None
        Methods stored on the selected run; None preserves stage-only filtering.
    spike_method : {"oasis", "cascade"}
        Active results method, independent of extraction settings.
    available_metrics : Collection[str] | None
        Non-NULL fields stored for that method, from get_stored_spike_capabilities().

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping category headers to list of available plot names
    """
    canonical_spike_methods((spike_method,))
    if stored_spike_methods:
        canonical_spike_methods(stored_spike_methods)
    # Group products by category, filtering by pipeline stage and experiment type
    categories: dict[str, list[str]] = {}
    for product in ANALYSIS_PRODUCTS:
        if product.group != group:
            continue

        if product.supported_spike_methods is not None:
            if spike_method not in product.supported_spike_methods:
                continue
            if (
                stored_spike_methods is not None
                and spike_method not in stored_spike_methods
            ):
                continue
            if available_metrics is not None and not set(
                product.required_metrics
            ) <= set(available_metrics):
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
    "Calcium Denoised ΔF/F0 Traces with Peaks",
    "Calcium Denoised ΔF/F0 Traces with Peaks and Thresholds (1 ROI)",
    "Calcium Denoised ΔF/F0 Normalized Traces with Peaks",
    "Calcium Denoised ΔF/F0 Traces Normalized (Active Only)"
    "Calcium Peaks Amplitudes (Denoised ΔF/F0)",
    "Calcium Peaks Frequencies (Denoised ΔF/F0)",
    "Calcium Peaks Amplitudes vs Frequencies (Denoised ΔF/F0)",
    "Calcium Peaks Inter-event Interval (Denoised ΔF/F0)",
    "Calcium Peaks Raster plot Colored by ROI",
    "Calcium Peaks Raster plot Colored by Amplitude",
    "Calcium Peaks Raster plot Colored by Amplitude with Colorbar",
    "Calcium Burst Activity Analysis",
    "Calcium Functional Connectivity",
    "Inferred Spikes (Thresholded)",
    "Inferred Spikes Raw (with Thresholds - 1 ROI)",
    "Inferred Spikes (Thresholded) with Denoised ΔF/F0 Traces",
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
    *,
    spike_method: SpikeMethod | None = None,
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
    spike_method : {"oasis", "cascade"} | None
        Explicit stored method; None uses the product's legacy/default method.
    """
    try:
        # Look up the analysis in the registry
        for product in ANALYSIS_PRODUCTS:
            if (
                text in {product.name, product.product_id}
                and product.group == AnalysisGroup.SINGLE_WELL
            ):
                method = product.selected_method(spike_method)
                # Type narrowing: we know this is a SingleWellAnalyzer
                analyzer = cast("SingleWellAnalyzer", product.analyzer)
                # Pass run_id as keyword argument to avoid positional conflicts
                # with other keyword args in the analyzer functions
                if method is not None and product.supported_spike_methods == (
                    "oasis",
                    "cascade",
                ):
                    analyzer(
                        widget,
                        engine,
                        fov_name,
                        rois,
                        run_id=run_id,
                        spike_method=method,
                    )
                else:
                    analyzer(widget, engine, fov_name, rois, run_id=run_id)
                widget.plot_item.setToolTip(
                    _source_coordinate_tooltip(engine, fov_name, run_id, rois)
                )
                return

        # If we get here, analysis was not found
        cali_logger.warning(f"Analysis '{text}' not found in registry")

    except Exception as e:
        cali_logger.error(f"Error plotting single well data for '{text}': {e}")
        raise


def _source_coordinate_tooltip(
    engine: Engine, fov_name: str, run_id: int | None, rois: list[int] | None
) -> str:
    """Describe the selected plot's stored source offset without guessing timing."""
    ensure_schema_current(engine)
    with Session(engine) as session:
        stmt = (
            select(
                ExtractionFrameWindow.source_start_frame,
                ExtractionFrameWindow.source_start_time_ms,
                ExtractionFrameWindow.timing_source,
                ExtractionFrameWindow.schema_version,
            )
            .join(
                Traces,
                col(Traces.extraction_frame_window_id) == col(ExtractionFrameWindow.id),
            )
            .join(ROI, col(Traces.roi_id) == col(ROI.id))
            .join(FOV, col(ROI.fov_id) == col(FOV.id))
            .where(FOV.name == fov_name)
            .distinct()
        )
        if run_id is not None:
            stmt = stmt.where(Traces.analysis_result_id == run_id)
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        windows = session.exec(stmt).all()
    details = ["Frame indices are relative to the retained trace (first frame = 0)."]
    for start, time_ms, timing, coordinate_version in windows:
        details.append(
            f"Discarded source prefix: {start} frames ({time_ms:g} ms); "
            f"timing: {timing or 'unknown'}; "
            + (
                "source pulse inputs are one-based."
                if coordinate_version >= 3
                else "historical pulse convention is preserved."
            )
        )
    return "\n".join(details)


def plot_multi_well_data(
    widget: _MultilWellGraphWidget,
    text: str,
    engine: Engine,
    run_id: int | None = None,
    *,
    spike_method: SpikeMethod | None = None,
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
    spike_method : {"oasis", "cascade"} | None
        Requested stored method, validated independently of GUI visibility.
    """
    # Handle empty/invalid selection
    if not text or text == "None" or text in MULTI_WELL_COMBO_OPTIONS_DICT.keys():
        widget.clear_plot()
        return

    try:
        # Look up the analysis in the registry
        for product in ANALYSIS_PRODUCTS:
            if (
                text in {product.name, product.product_id}
                and product.group == AnalysisGroup.MULTI_WELL
            ):
                method = product.selected_method(spike_method)
                # Type narrowing: we know this is a MultiWellAnalyzer
                analyzer = cast("MultiWellAnalyzer", product.analyzer)
                if product.supported_spike_methods == ("oasis", "cascade"):
                    return analyzer(widget, text, engine, run_id, spike_method=method)
                return analyzer(widget, text, engine, run_id)

        # If we get here, analysis was not found
        cali_logger.warning(f"Multi-well analysis '{text}' not found in registry")
        widget.clear_plot()

    except Exception as e:
        cali_logger.error(f"Error plotting multi-well data for '{text}': {e}")
        widget.clear_plot()
        raise
