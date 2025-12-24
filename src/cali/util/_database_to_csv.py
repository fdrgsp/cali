"""Export calcium imaging analysis data from database to CSV files.

This module provides efficient methods to export various data types from the
database to CSV format, including traces, correlation matrices, and more.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sqlmodel import Session, col, select

from cali._constants import (
    DEC_DFF_TRACES,
    DFF_TRACES,
    INFERRED_SPIKES_THRESHOLDED_TRACES,
    INFERRED_SPIKES_TRACES,
    NEUROPIL_CORRECTED_TRACES,
    NEUROPIL_TRACES,
    RAW_CALCIUM_TRACES,
    TraceDataType,
)
from cali.sqlmodel._model import (
    FOV,
    ROI,
    DataAnalysis,
    FOVAnalysis,
    Traces,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


def export_raw_traces_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
) -> None:
    """Export raw fluorescence traces to CSV.

    For evoked experiments, creates separate columns for stimulated and
    non-stimulated ROIs, with stimulated ROIs listed first.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output CSV file path
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=RAW_CALCIUM_TRACES,
        fov_name=fov_name,
        run_id=run_id,
    )


def export_neuropil_traces_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
) -> None:
    """Export neuropil fluorescence traces to CSV.

    For evoked experiments, creates separate columns for stimulated and
    non-stimulated ROIs, with stimulated ROIs listed first.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output CSV file path
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=NEUROPIL_TRACES,
        fov_name=fov_name,
        run_id=run_id,
    )


def export_neuropil_corrected_traces_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
) -> None:
    """Export neuropil-corrected fluorescence traces to CSV.

    For evoked experiments, creates separate columns for stimulated and
    non-stimulated ROIs, with stimulated ROIs listed first.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output CSV file path
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=NEUROPIL_CORRECTED_TRACES,
        fov_name=fov_name,
        run_id=run_id,
    )


def export_dff_traces_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
) -> None:
    """Export ΔF/F traces to CSV.

    For evoked experiments, creates separate columns for stimulated and
    non-stimulated ROIs, with stimulated ROIs listed first.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output CSV file path
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=DFF_TRACES,
        fov_name=fov_name,
        run_id=run_id,
    )


def export_deconvolved_dff_traces_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
) -> None:
    """Export deconvolved ΔF/F traces to CSV.

    For evoked experiments, creates separate columns for stimulated and
    non-stimulated ROIs, with stimulated ROIs listed first.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output CSV file path
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=DEC_DFF_TRACES,
        fov_name=fov_name,
        run_id=run_id,
    )


def export_inferred_spikes_raw_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
) -> None:
    """Export raw inferred spike traces to CSV.

    For evoked experiments, creates separate columns for stimulated and
    non-stimulated ROIs, with stimulated ROIs listed first.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output CSV file path
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=INFERRED_SPIKES_TRACES,
        fov_name=fov_name,
        run_id=run_id,
    )


def export_inferred_spikes_thresholded_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
) -> None:
    """Export thresholded inferred spike traces to CSV (binary).

    Spikes are thresholded using the spike threshold stored in DataAnalysis.
    For evoked experiments, creates separate columns for stimulated and
    non-stimulated ROIs, with stimulated ROIs listed first.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output CSV file path
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=INFERRED_SPIKES_THRESHOLDED_TRACES,
        fov_name=fov_name,
        run_id=run_id,
    )


def export_correlation_matrices_to_csv(
    engine: Engine,
    output_dir: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
) -> None:
    """Export all correlation matrices to CSV files.

    Creates separate CSV files for each correlation type:
    - calcium_dff_correlation.csv
    - calcium_dec_dff_correlation.csv
    - spike_max_lag_correlation.csv
    - spike_max_lag_values.csv
    - spike_jitter_synchrony.csv

    ROIs are ordered with stimulated first (if evoked experiment), then non-stimulated.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_dir : str | Path
        Output directory for CSV files
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get run_id if not provided
    if run_id is None:
        run_id = _get_default_run_id(engine)

    # Query FOV analysis data
    with Session(engine) as session:
        stmt = (
            select(FOVAnalysis, FOV)
            .join(FOV, FOVAnalysis.fov_id == FOV.id)
            .where(FOVAnalysis.analysis_result_id == run_id)
        )

        if fov_name is not None:
            stmt = stmt.where(col(FOV.name) == fov_name)

        results = session.exec(stmt).all()

        if not results:
            msg = "No FOV analysis data found"
            raise ValueError(msg)

        for fov_analysis, fov in results:
            fov_prefix = f"{fov.name}_" if len(results) > 1 else ""

            # Get ROI labels to check if evoked experiment
            roi_labels = fov_analysis.active_roi_labels
            if not roi_labels:
                continue

            # Query ROIs to get stimulation status
            roi_stmt = (
                select(ROI)
                .where(ROI.fov_id == fov.id)
                .where(col(ROI.label_value).in_(roi_labels))
            )
            rois = session.exec(roi_stmt).all()

            # Sort ROIs: stimulated first, then non-stimulated
            roi_dict = {roi.label_value: roi for roi in rois}
            sorted_labels = sorted(
                roi_labels,
                key=lambda lbl: (
                    (not roi_dict[lbl].stimulated, lbl)
                    if roi_dict[lbl].stimulated is not None
                    else (False, lbl)
                ),
            )
            sorted_roi_names = [f"ROI_{lbl}" for lbl in sorted_labels]

            # Export each correlation matrix
            _export_matrix_to_csv(
                fov_analysis.calcium_dff_correlation_matrix,
                sorted_roi_names,
                output_dir / f"{fov_prefix}calcium_dff_correlation.csv",
            )

            _export_matrix_to_csv(
                fov_analysis.calcium_dec_dff_corr_matrix,
                sorted_roi_names,
                output_dir / f"{fov_prefix}calcium_dec_dff_correlation.csv",
            )

            _export_matrix_to_csv(
                fov_analysis.spike_max_lag_correlation_matrix,
                sorted_roi_names,
                output_dir / f"{fov_prefix}spike_max_lag_correlation.csv",
            )

            _export_matrix_to_csv(
                fov_analysis.spike_max_lag_values_matrix,
                sorted_roi_names,
                output_dir / f"{fov_prefix}spike_max_lag_values.csv",
            )

            _export_matrix_to_csv(
                fov_analysis.spike_jitter_synchrony_matrix,
                sorted_roi_names,
                output_dir / f"{fov_prefix}spike_jitter_synchrony.csv",
            )


# ==================== Helper Functions ====================


def _get_default_run_id(engine: Engine) -> int:
    """Get the first available analysis run ID."""
    with Session(engine) as session:
        stmt = select(Traces.analysis_result_id).limit(1)
        run_id = session.exec(stmt).first()
        if run_id is None:
            msg = "No analysis runs found in database"
            raise ValueError(msg)
        return int(run_id)


def _export_trace_data(
    engine: Engine,
    output_path: str | Path,
    trace_type: TraceDataType,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
) -> None:
    """Export trace data to CSV (internal helper)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get run_id if not provided
    if run_id is None:
        run_id = _get_default_run_id(engine)

    # Query traces data
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .join(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
        )

        if fov_name is not None:
            stmt = stmt.where(col(FOV.name) == fov_name)

        stmt = stmt.order_by(col(FOV.name), col(ROI.label_value))
        results = session.exec(stmt).all()

        if not results:
            msg = f"No trace data found for run_id={run_id}"
            raise ValueError(msg)

        # Group by FOV and stimulation status
        fov_data: dict[str, dict[Literal["stim", "non_stim"], list]] = {}

        for roi, traces, data_analysis in results:
            # Map display names to Traces model attribute names
            trace_attr_map = {
                RAW_CALCIUM_TRACES: "raw_trace",
                NEUROPIL_TRACES: "neuropil_trace",
                NEUROPIL_CORRECTED_TRACES: "corrected_trace",
                DFF_TRACES: "dff",
                DEC_DFF_TRACES: "dec_dff",
                INFERRED_SPIKES_TRACES: "inferred_spikes",
                INFERRED_SPIKES_THRESHOLDED_TRACES: "inferred_spikes",
            }

            # Get trace data
            if trace_type == INFERRED_SPIKES_THRESHOLDED_TRACES:
                # Binarize inferred spikes based on threshold
                trace_data = traces.inferred_spikes
                if trace_data is not None and data_analysis.inferred_spikes_threshold:
                    trace_data = [
                        1.0 if val >= data_analysis.inferred_spikes_threshold else 0.0
                        for val in trace_data
                    ]
            else:
                attr_name = trace_attr_map.get(trace_type)
                trace_data = getattr(traces, attr_name, None) if attr_name else None

            if trace_data is None:
                continue

            # Determine FOV key
            fov_key = roi.fov.name

            # Initialize FOV data structure
            if fov_key not in fov_data:
                fov_data[fov_key] = {"stim": [], "non_stim": []}

            # Add to appropriate group based on stimulation status
            is_evoked = roi.stimulated is not None
            if is_evoked:
                group: Literal["stim", "non_stim"] = (
                    "stim" if roi.stimulated else "non_stim"
                )
            else:
                # For non-evoked experiments, put all in non_stim group
                group = "non_stim"

            fov_data[fov_key][group].append(
                {
                    "roi_label": roi.label_value,
                    "fov_name": fov_key,
                    "trace": trace_data,
                }
            )

        # Create DataFrame
        if not fov_data:
            msg = f"No valid trace data found for trace_type={trace_type}"
            raise ValueError(msg)

        # Build columns in order: stimulated first, then non-stimulated
        all_data = []
        column_names = []

        for fov_key in sorted(fov_data.keys()):
            # Add stimulated ROIs first
            for roi_info in sorted(
                fov_data[fov_key]["stim"], key=lambda x: x["roi_label"]
            ):
                all_data.append(roi_info["trace"])
                if len(fov_data) > 1:
                    column_names.append(
                        f"{roi_info['fov_name']}_ROI_{roi_info['roi_label']}_stim"
                    )
                else:
                    column_names.append(f"ROI_{roi_info['roi_label']}_stim")

            # Then add non-stimulated ROIs
            for roi_info in sorted(
                fov_data[fov_key]["non_stim"], key=lambda x: x["roi_label"]
            ):
                all_data.append(roi_info["trace"])
                suffix = (
                    "_non_stim"
                    if fov_data[fov_key]["stim"]
                    else ""  # Only add suffix if there are stim ROIs
                )
                if len(fov_data) > 1:
                    column_names.append(
                        f"{roi_info['fov_name']}_ROI_{roi_info['roi_label']}{suffix}"
                    )
                else:
                    column_names.append(f"ROI_{roi_info['roi_label']}{suffix}")

        # Create DataFrame with traces as columns
        df = pd.DataFrame(np.array(all_data).T, columns=column_names)

        # Save to CSV
        df.to_csv(output_path, index=False)


def _export_matrix_to_csv(
    matrix: list[list[float]] | list[list[int]] | None,
    labels: list[str],
    output_path: Path,
) -> None:
    """Export a correlation/synchrony matrix to CSV with row/column labels."""
    if matrix is None:
        return

    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(output_path, index=True)
