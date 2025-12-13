"""Learning..."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from sqlmodel import Session, select

from cali.analysis._util import (
    _compute_zero_lag_corr_matrix,
    _get_calcium_peaks_event_correlations_matrix,
    _get_spike_correlations_matrix,
)
from cali.sqlmodel import FOV, ROI, Traces
from cali.sqlmodel._model import DataAnalysis

database_path = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"
engine = create_engine(f"sqlite:///{database_path}")


def query_traces(
    run_id: int,
    selected_roi_labels: list[str] | None,
    trace_type: Literal["raw_trace", "dff", "dec_dff", "inferred_spikes"] = "dec_dff",
    position_index: int | None = None,
    plot: bool = False,
) -> tuple[list[np.ndarray], dict[str, list[float]], dict[str, list[float]]]:
    """Query traces from the database and optionally plot them.

    Returns
    -------
        tuple: (traces_arrays, roi_dict) where roi_dict maps ROI labels to trace data
    """
    if trace_type not in ["raw_trace", "dff", "dec_dff", "inferred_spikes"]:
        raise ValueError(f"Invalid trace_type: {trace_type}")

    with Session(engine) as session:
        # Control variables
        # Change this to select different analysis runs
        run_id = run_id

        statement = select(Traces).join(ROI).join(FOV)
        statement1 = select(DataAnalysis).join(ROI).join(FOV)

        # Filter by run/analysis result ID
        if run_id is not None:
            statement = statement.where(Traces.analysis_result_id == run_id)
            statement1 = statement1.where(DataAnalysis.analysis_result_id == run_id)

        # Filter by position index
        if position_index is not None:
            statement = statement.where(FOV.position_index == position_index)
            statement1 = statement1.where(FOV.position_index == position_index)

        # Filter by specific ROI labels if provided
        if selected_roi_labels is not None:
            statement = statement.where(ROI.label_value.in_(selected_roi_labels))
            statement1 = statement1.where(ROI.label_value.in_(selected_roi_labels))

        traces_list = session.exec(statement).all()
        data_analysis_list = session.exec(statement1).all()

        # roi label: trace value list of float
        roi_dict_spikes: dict[str, list[float]] = {}
        roi_dict_calcium_peaks: dict[str, list[float]] = {}

        # Get calcium peaks data
        for data_analysis in data_analysis_list:
            roi_label = str(data_analysis.roi.label_value)
            if data_analysis.peaks_dec_dff is not None:
                roi_dict_calcium_peaks[roi_label] = data_analysis.peaks_dec_dff

        # Plot all inferred spikes traces
        if plot:
            _fig, ax = plt.subplots(figsize=(12, 6))
        traces_arrays = []
        for trace in traces_list:
            tr = None
            roi_label = str(trace.roi.label_value)
            if trace_type == "raw_trace" and trace.raw_trace is not None:
                traces_arrays.append(np.array(trace.raw_trace))
                tr = trace.raw_trace
            elif trace_type == "dff" and trace.dff is not None:
                traces_arrays.append(np.array(trace.dff))
                tr = trace.dff
            elif trace_type == "dec_dff" and trace.dec_dff is not None:
                traces_arrays.append(np.array(trace.dec_dff))
                tr = trace.dec_dff
            elif trace_type == "inferred_spikes" and trace.inferred_spikes is not None:
                traces_arrays.append(np.array(trace.inferred_spikes))
                tr = trace.inferred_spikes
                roi_dict_spikes[roi_label] = tr
            else:
                continue

            if not plot:
                continue
            x_data = trace.x_axis if trace.x_axis is not None else list(range(len(tr)))
            ax.plot(x_data, tr, alpha=0.7)
        if plot:
            plt.tight_layout()
            plt.show()

        return traces_arrays, roi_dict_spikes, roi_dict_calcium_peaks


run_id = 4
pos = 5
rois = None
# rois = [str(x) for x in [1, 60, 34, 67, 7]]  # Convert to strings for ROI labels

# raw, _, _ = query_traces(5, rois, "raw_trace", position_index=18)
dff, _, _ = query_traces(run_id, rois, "dff", pos)
dec_dff, _, roi_dict_calcium_peaks = query_traces(run_id, rois, "dec_dff", pos)
spikes, roi_dict_spikes, _ = query_traces(run_id, rois, "inferred_spikes", pos)

# pc_matrix_raw = _compute_zero_lag_corr_matrix(raw)
# pc_matrix_dff = _compute_zero_lag_corr_matrix(dff)
pc_matrix_dec_dff = _compute_zero_lag_corr_matrix(dec_dff)  # ***
pc_matrix_spikes = _compute_zero_lag_corr_matrix(spikes)

len_rois = len(rois) if rois is not None else len(roi_dict_spikes.keys())
assert len_rois == len(roi_dict_spikes.keys()) == len(roi_dict_calcium_peaks.keys())

print("\n\nPEARSON'S CORRELATION------------------------------------")
# print("Raw Calcium ----- Median Pearson's correlation:", np.median(pc_matrix_raw))
# print("DFF Calcium ----- Median Pearson's correlation:", np.median(pc_matrix_dff))
print("Dec DFF Calcium - Median Pearson's correlation:", np.median(pc_matrix_dec_dff))
print("Spikes ---------- Median Pearson's correlation:", np.median(pc_matrix_spikes))

# pc_matrix_spikes_1, _ = _get_spike_correlations_matrix(
#     roi_dict_spikes, method="correlation"
# )
# print(
#     f"Spikes Binary --- Median Pearson's correlation: {np.median(pc_matrix_spikes_1)}"
# )

# Convert peak indices to binary arrays for correlation analysis
# Get the trace length from one of the traces
trace_length = len(dec_dff[0]) if dec_dff else 1000  # fallback if no traces

roi_dict_calcium_peaks_binary = {}
for roi_label, peak_indices in roi_dict_calcium_peaks.items():
    # Create binary array: 1 where peaks occur, 0 elsewhere
    binary_peaks = np.zeros(trace_length, dtype=int)
    # Ensure peak indices are within bounds
    valid_peaks = [int(p) for p in peak_indices if 0 <= int(p) < trace_length]
    binary_peaks[valid_peaks] = 1
    roi_dict_calcium_peaks_binary[roi_label] = binary_peaks.tolist()


print("\n\nSYNCHRONY ANALYSIS------------------------------------")
sync_cal_peaks, _ = _get_calcium_peaks_event_correlations_matrix(
    roi_dict_calcium_peaks_binary, method="jitter_window", jitter_window=4
)
if sync_cal_peaks is not None:
    print(
        f"Calcium Peaks --- Median Synchrony (jitter_window=4): "
        f"{np.median(sync_cal_peaks)}"
    )

sync_spikes, _ = _get_spike_correlations_matrix(
    roi_dict_spikes, method="jitter_window", jitter_window=4
)
if sync_spikes is not None:
    print(
        f"Spikes Events --- Median Synchrony (jitter_window=4): "
        f"{np.median(sync_spikes)}"
    )

print("\n\nCROSS-CORRELATION ANALYSIS------------------------------------")
cc_spikes, _ = _get_spike_correlations_matrix(
    roi_dict_spikes, method="cross_correlation", max_lag=10
)
if cc_spikes is not None:
    print(
        f"Spikes Binary --- Median Cross-Correlation (cross_correlation=5): "
        f"{np.median(cc_spikes)}"
    )


# p18
# PEARSON'S CORRELATION------------------------------------
# Dec DFF Calcium - Median Pearson's correlation: 0.8316392000875181
# Spikes ---------- Median Pearson's correlation: 0.5202445015519529
# Spikes Binary --- Median Pearson's correlation: 0.050302881477159785
# SYNCHRONY ANALYSIS------------------------------------
# Calcium Peaks --- Median Synchrony (jitter_window=4): 0.4727272727272727
# Spikes Events --- Median Synchrony (jitter_window=4): 0.5625
# CROSS-CORRELATION ANALYSIS------------------------------------
# Spikes Binary --- Median Cross-Correlation (cross_correlation=5): 0.14866702524809083

# p12
# PEARSON'S CORRELATION------------------------------------
# Dec DFF Calcium - Median Pearson's correlation: 0.010107764479549022
# Spikes ---------- Median Pearson's correlation: 0.0027730646327039747
# Spikes Binary --- Median Pearson's correlation: 0.01648742715456141
# SYNCHRONY ANALYSIS------------------------------------
# Calcium Peaks --- Median Synchrony (jitter_window=4): 0.17142857142857143
# Spikes Events --- Median Synchrony (jitter_window=4): 0.5887096774193549
# CROSS-CORRELATION ANALYSIS------------------------------------
# Spikes Binary --- Median Cross-Correlation (cross_correlation=5): 0.14280855448284688

# p5

# PEARSON'S CORRELATION------------------------------------
# Dec DFF Calcium - Median Pearson's correlation: 0.025673762168785818
# Spikes ---------- Median Pearson's correlation: 0.0014042426349259334
# Spikes Binary --- Median Pearson's correlation: 0.016267940790323163
# SYNCHRONY ANALYSIS------------------------------------
# Calcium Peaks --- Median Synchrony (jitter_window=4): 0.1744186046511628
# Spikes Events --- Median Synchrony (jitter_window=4): 0.5573770491803278
# CROSS-CORRELATION ANALYSIS------------------------------------
# Spikes Binary --- Median Cross-Correlation (cross_correlation=5): 0.13625041996196363
