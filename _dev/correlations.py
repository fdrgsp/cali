"""Learning..."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from sqlmodel import Session, select

from cali.analysis._util import (
    _compute_zero_lag_corr_matrix,
    _get_spike_correlations_matrix,
)
from cali.sqlmodel import FOV, ROI, Traces
from cali.sqlmodel._model import DataAnalysis

database_path = (
    "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results_new.cali"
)
engine = create_engine(f"sqlite:///{database_path}")


def query_traces(
    run_id: int,
    selected_roi_labels: list[str] | None,
    trace_type: Literal["raw_trace", "dff", "dec_dff", "inferred_spikes"] = "dec_dff",
    position_index: int | None = None,
    plot: bool = False,
) -> tuple[
    list[np.ndarray], dict[str, tuple[list[float], float]], dict[str, list[float]]
]:
    """Query traces from the database and optionally plot them.

    Returns
    -------
        tuple: (traces_arrays, roi_dict_spikes, roi_dict_calcium_peaks)
            - traces_arrays: List of numpy arrays of the requested trace type
            - roi_dict_spikes: Dictionary mapping ROI labels to inferred spikes traces
            and thresholds
            - roi_dict_calcium_peaks: Dictionary mapping ROI labels to calcium peaks
            traces
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
        roi_dict_spikes: dict[str, tuple[list[float], float]] = {}
        roi_dict_calcium_peaks: dict[str, list[float]] = {}

        # Create mapping from ROI label to DataAnalysis for threshold access
        roi_to_data_analysis: dict[str, DataAnalysis] = {}
        for data_analysis in data_analysis_list:
            roi_label = str(data_analysis.roi.label_value)
            roi_to_data_analysis[roi_label] = data_analysis
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
                # Get threshold from corresponding DataAnalysis
                threshold = (
                    roi_to_data_analysis[roi_label].inferred_spikes_threshold
                    if roi_label in roi_to_data_analysis
                    else 0.0
                )
                roi_dict_spikes[roi_label] = (tr, threshold)
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


run_id = 8
pos = 18
rois = None
# rois = [str(x) for x in [1, 60, 34, 67, 7]]  # Convert to strings for ROI labels

# Query traces
dff, _, _ = query_traces(run_id, rois, "dff", pos)
dec_dff, _, roi_dict_calcium_peaks = query_traces(run_id, rois, "dec_dff", pos)
spikes, roi_dict_spikes, _ = query_traces(run_id, rois, "inferred_spikes", pos)

len_rois = len(rois) if rois is not None else len(roi_dict_spikes.keys())
assert len_rois == len(roi_dict_spikes.keys()) == len(roi_dict_calcium_peaks.keys())

# Pearson's correlation analysis
pc_matrix_dec_dff = _compute_zero_lag_corr_matrix(dec_dff)
print("\n\nPEARSON'S CORRELATION------------------------------------")
print("Dec DFF Calcium -- Median Pearson's correlation:", np.median(pc_matrix_dec_dff))

# Binarize spike traces based on thresholds
roi_dict_spikes_binary = {}
for roi_label, (spike_trace, threshold) in roi_dict_spikes.items():
    binary_trace = [1 if val >= threshold else 0 for val in spike_trace]
    roi_dict_spikes_binary[roi_label] = binary_trace

print("\nSYNCHRONY ANALYSIS------------------------------------")
jitter_window = 5
sync_spikes, _ = _get_spike_correlations_matrix(
    roi_dict_spikes_binary, method="jitter_window", jitter_window=jitter_window
)
if sync_spikes is not None:
    print(
        f"Spikes Events ---- Median Synchrony (jitter_window={jitter_window}): "
        f"{np.median(sync_spikes)}"
    )

print("\nCROSS-CORRELATION ANALYSIS------------------------------------")
cc_lag = 10
cc_spikes, _ = _get_spike_correlations_matrix(
    roi_dict_spikes_binary, method="cross_correlation", max_lag=cc_lag
)
if cc_spikes is not None:
    print(
        f"Spikes Binary ---- Median Cross-Correlation (cross_correlation={cc_lag}): "
        f"{np.median(cc_spikes)}"
    )
