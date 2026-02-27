"""Multi-well plotting functions."""

from ._calcium_peaks import (
    plot_calcium_burst_avg_duration_bar_plot,
    plot_calcium_burst_avg_interval_bar_plot,
    plot_calcium_burst_count_bar_plot,
    plot_calcium_peaks_amplitude_bar_plot,
    plot_calcium_peaks_amplitude_stim_split_bar_plot,
    plot_calcium_peaks_frequency_bar_plot,
    plot_calcium_peaks_frequency_stim_split_bar_plot,
    plot_calcium_peaks_iei_bar_plot,
)
from ._cell_properties import (
    plot_cell_size_bar_plot,
    plot_percentage_active_bar_plot,
    plot_percentage_active_stim_split_bar_plot,
)
from ._dimensionality_reduction import (
    plot_pca_loadings,
    plot_pca_scatter,
    plot_pca_scatter_stim_split,
    plot_pca_scree,
)
from ._inferred_spikes import (
    plot_burst_avg_duration_bar_plot,
    plot_burst_avg_interval_bar_plot,
    plot_burst_count_bar_plot,
    plot_burst_rate_bar_plot,
    plot_inferred_spikes_frequency_bar_plot,
    plot_inferred_spikes_frequency_stim_split_bar_plot,
    plot_inferred_spikes_rising_edge_frequency_bar_plot,
    plot_inferred_spikes_rising_edge_frequency_stim_split_bar_plot,
    plot_spike_correlation_bar_plot,
    plot_spike_synchrony_bar_plot,
)
from ._util import plot_parameter_bar_plot

__all__ = [
    "plot_burst_avg_duration_bar_plot",
    "plot_burst_avg_interval_bar_plot",
    "plot_burst_count_bar_plot",
    "plot_burst_rate_bar_plot",
    "plot_calcium_burst_avg_duration_bar_plot",
    "plot_calcium_burst_avg_interval_bar_plot",
    "plot_calcium_burst_count_bar_plot",
    "plot_calcium_peaks_amplitude_bar_plot",
    "plot_calcium_peaks_amplitude_stim_split_bar_plot",
    "plot_calcium_peaks_frequency_bar_plot",
    "plot_calcium_peaks_frequency_stim_split_bar_plot",
    "plot_calcium_peaks_iei_bar_plot",
    "plot_cell_size_bar_plot",
    "plot_inferred_spikes_frequency_bar_plot",
    "plot_inferred_spikes_frequency_stim_split_bar_plot",
    "plot_inferred_spikes_rising_edge_frequency_bar_plot",
    "plot_inferred_spikes_rising_edge_frequency_stim_split_bar_plot",
    "plot_parameter_bar_plot",
    "plot_pca_loadings",
    "plot_pca_scatter",
    "plot_pca_scatter_stim_split",
    "plot_pca_scree",
    "plot_percentage_active_bar_plot",
    "plot_percentage_active_stim_split_bar_plot",
    "plot_spike_correlation_bar_plot",
    "plot_spike_synchrony_bar_plot",
]
