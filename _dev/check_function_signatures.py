"""Temporary script to identify correct function signatures for tests."""

import inspect
from pathlib import Path

# Import modules to inspect
from cali.plot._single_wells_plots.correlation import (
    _plot_inferred_spike_synchrony,
)
from cali.plot._single_wells_plots.burst import _plot_inferred_spike_burst_activity
from cali.plot._single_wells_plots.calcium_traces import _plot_neuropil_traces
from cali.plot._single_wells_plots.correlation import _plot_calcium_network_connectivity, _plot_calcium_peaks_correlation, _plot_calcium_peaks_synchrony, _plot_inferred_spike_correlation
from cali.plot._single_wells_plots.metrics import _plot_calcium_amplitudes_and_frequencies_data, _plot_calcium_peaks_iei_data, _plot_cell_size
from cali.plot._single_wells_plots.raster import _plot_calcium_peaks_raster_plots, _plot_inferred_spike_raster_plots

modules = [
    ("calcium_peaks_raster", _plot_calcium_peaks_raster_plots),
    ("inferred_spike_raster", _plot_inferred_spike_raster_plots),
    ("calcium_amplitudes_frequencies", _plot_calcium_amplitudes_and_frequencies_data),
    ("calcium_peaks_iei", _plot_calcium_peaks_iei_data),
    ("calcium_network", _plot_calcium_network_connectivity),
    ("calcium_correlation", _plot_calcium_peaks_correlation),
    ("spike_correlation", _plot_inferred_spike_correlation),
    ("calcium_synchrony", _plot_calcium_peaks_synchrony),
    ("spike_synchrony", _plot_inferred_spike_synchrony),
    ("spike_burst", _plot_inferred_spike_burst_activity),
    ("cell_size", _plot_cell_size),
    ("neuropil_traces", _plot_neuropil_traces),
]

print("Function signatures for plotting modules:\n")
for name, module in modules:
    print(f"\n{name}:")
    print("-" * 80)
    for func_name in dir(module):
        if func_name.startswith("_plot") or func_name.startswith("_generate"):
            func = getattr(module, func_name)
            if callable(func):
                try:
                    sig = inspect.signature(func)
                    print(f"  {func_name}{sig}")
                except:
                    pass
