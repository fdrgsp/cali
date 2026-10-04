"""Export calcium imaging analysis data from database to CSV files.

This module provides efficient methods to export various data types from the
database to CSV format, including traces, correlation matrices, and more.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sqlmodel import Session, col, select

from cali._constants import (
    CASCADE_EXPECTED_SPIKES_TRACES,
    DEN_DFF_TRACES,
    DFF_TRACES,
    INFERRED_SPIKES_THRESHOLDED_BINARY,
    INFERRED_SPIKES_TRACES,
    NEUROPIL_CORRECTED_TRACES,
    NEUROPIL_TRACES,
    RAW_CALCIUM_TRACES,
    CorrelationDataType,
    TraceDataType,
    natural_sort_key,
)
from cali.analysis._roi_analysis import valid_spike_events
from cali.sqlmodel._engine import ensure_schema_current
from cali.sqlmodel._model import (
    FOV,
    ROI,
    DataAnalysis,
    FOVAnalysis,
    Traces,
)
from cali.sqlmodel._spike_settings import canonical_spike_methods

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.extraction._frame_window import SourceFrameTransform
    from cali.sqlmodel._spike_settings import SpikeMethod


def export_raw_traces_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
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
    position_indices : list[int] | None, optional
        Position indices to filter exports. If provided, only exports data
        from these positions.
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=RAW_CALCIUM_TRACES,
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
    )


def export_neuropil_traces_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
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
    position_indices : list[int] | None, optional
        Position indices to filter exports. If provided, only exports data
        from these positions.
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=NEUROPIL_TRACES,
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
    )


def export_neuropil_corrected_traces_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
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
    position_indices : list[int] | None, optional
        Position indices to filter exports. If provided, only exports data
        from these positions.
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=NEUROPIL_CORRECTED_TRACES,
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
    )


def export_dff_traces_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
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
    position_indices : list[int] | None, optional
        Position indices to filter exports. If provided, only exports data
        from these positions.
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=DFF_TRACES,
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
    )


def export_denoised_dff_traces_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
) -> None:
    """Export denoised ΔF/F traces to CSV.

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
    position_indices : list[int] | None, optional
        Position indices to filter exports. If provided, only exports data
        from these positions.
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=DEN_DFF_TRACES,
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
    )


def export_inferred_spikes_raw_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
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
    position_indices : list[int] | None, optional
        Position indices to filter exports. If provided, only exports data
        from these positions.
    spike_method : {"oasis", "cascade"}
        Stored output to export; each method uses its own population and threshold.
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=INFERRED_SPIKES_TRACES,
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
        spike_method=spike_method,
    )


def export_inferred_spikes_thresholded_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
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
    position_indices : list[int] | None, optional
        Position indices to filter exports. If provided, only exports data
        from these positions.
    spike_method : {"oasis", "cascade"}
        Stored output to export; each method uses its own population and threshold.
    """
    _export_trace_data(
        engine=engine,
        output_path=output_path,
        trace_type=INFERRED_SPIKES_THRESHOLDED_BINARY,
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
        spike_method=spike_method,
    )


def export_correlation_matrices_to_csv(
    engine: Engine,
    output_dir: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_methods: tuple[SpikeMethod, ...] | None = None,
) -> None:
    """Export all correlation matrices to CSV files.

    Creates separate CSV files for each correlation type:
    - calcium_dff_correlation.csv
    - calcium_den_dff_correlation.csv
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
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    spike_methods : tuple | None
        Stored spike methods to export separately; None selects every stored method.
    """
    from ._spike_export import _selected_methods, available_spike_result_methods

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get run_id if not provided
    if run_id is None:
        run_id = _get_default_run_id(engine)

    available = available_spike_result_methods(
        engine,
        run_id=run_id,
        fov_name=fov_name,
        position_indices=position_indices,
    )
    methods = _selected_methods(available, spike_methods)

    # Query FOV analysis data
    ensure_schema_current(engine)
    with Session(engine) as session:
        stmt = (
            select(FOVAnalysis, FOV)
            .join(FOV, FOVAnalysis.fov_id == FOV.id)
            .where(FOVAnalysis.analysis_result_id == run_id)
        )

        if fov_name is not None:
            stmt = stmt.where(col(FOV.name) == fov_name)

        if position_indices is not None:
            stmt = stmt.where(col(FOV.position_index).in_(position_indices))

        results = session.exec(stmt).all()

        if not results:
            msg = "No FOV analysis data found"
            raise ValueError(msg)

        for fov_analysis, fov in results:
            fov_prefix = f"{fov.name}_" if len(results) > 1 else ""

            for metric, filename in (
                ("calcium_dff_correlation_matrix", "calcium_dff_correlation.csv"),
                ("calcium_den_dff_corr_matrix", "calcium_den_dff_correlation.csv"),
                ("spike_max_lag_correlation_matrix", "spike_max_lag_correlation.csv"),
                ("spike_max_lag_values_matrix", "spike_max_lag_values.csv"),
                ("spike_jitter_synchrony_matrix", "spike_jitter_synchrony.csv"),
            ):
                if metric.startswith("calcium_"):
                    matrix = getattr(fov_analysis, metric)
                    roi_labels = fov_analysis.calcium_active_roi_labels
                    _export_ordered_fov_matrix(
                        session,
                        fov,
                        matrix,
                        roi_labels,
                        output_dir / f"{fov_prefix}{filename}",
                    )
                else:
                    for method in methods:
                        qualified = (
                            f"{method}_{filename}"
                            if available != {"oasis"}
                            else filename
                        )
                        _export_spike_fov_matrix(
                            session,
                            fov_analysis,
                            fov,
                            method,
                            metric,
                            output_dir / f"{fov_prefix}{qualified}",
                        )


def export_calcium_dff_correlation_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
) -> None:
    """Export ΔF/F correlation matrix to CSV.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output file path for CSV
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    """
    _export_single_correlation_matrix(
        engine,
        output_path,
        "calcium_dff_correlation_matrix",
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
    )


def export_calcium_den_dff_correlation_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
) -> None:
    """Export denoised ΔF/F correlation matrix to CSV.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output file path for CSV
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    """
    _export_single_correlation_matrix(
        engine,
        output_path,
        "calcium_den_dff_corr_matrix",
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
    )


def export_inferred_spikes_synchrony_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Export inferred spikes synchrony matrix to CSV.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output file path for CSV
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    spike_method : {"oasis", "cascade"}
        Stored output to export; each method uses its own population and threshold.
    """
    _export_single_correlation_matrix(
        engine,
        output_path,
        "spike_jitter_synchrony_matrix",
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
        spike_method=spike_method,
    )


def export_inferred_spikes_cross_correlation_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Export inferred spikes cross-correlation matrix to CSV.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output file path for CSV
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    spike_method : {"oasis", "cascade"}
        Stored output to export; each method uses its own population and threshold.
    """
    _export_single_correlation_matrix(
        engine,
        output_path,
        "spike_max_lag_correlation_matrix",
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
        spike_method=spike_method,
    )


def export_inferred_spikes_cross_correlation_lags_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Export inferred spikes cross-correlation lags matrix to CSV.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output file path for CSV
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    spike_method : {"oasis", "cascade"}
        Stored output to export; each method uses its own population and threshold.
    """
    _export_single_correlation_matrix(
        engine,
        output_path,
        "spike_max_lag_values_matrix",
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
        spike_method=spike_method,
    )


def export_inferred_spikes_ccg_zscore_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Export inferred spikes CCG z-score matrix to CSV.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output file path for CSV
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    spike_method : {"oasis", "cascade"}
        Stored output to export; each method uses its own population and threshold.
    """
    _export_single_correlation_matrix(
        engine,
        output_path,
        "spike_ccg_zscore_matrix",
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
        spike_method=spike_method,
    )


def export_inferred_spikes_synchrony_rising_edges_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Export inferred spikes synchrony matrix (rising edges) to CSV.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output file path for CSV
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    spike_method : {"oasis", "cascade"}
        Stored output to export; each method uses its own population and threshold.
    """
    _export_single_correlation_matrix(
        engine,
        output_path,
        "spike_jitter_synchrony_matrix_rising_edges",
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
        spike_method=spike_method,
    )


def export_inferred_spikes_cross_correlation_rising_edges_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Export inferred spikes cross-correlation matrix (rising edges) to CSV.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output file path for CSV
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    spike_method : {"oasis", "cascade"}
        Stored output to export; each method uses its own population and threshold.
    """
    _export_single_correlation_matrix(
        engine,
        output_path,
        "spike_max_lag_correlation_matrix_rising_edges",
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
        spike_method=spike_method,
    )


def export_inferred_spikes_cross_correlation_lags_rising_edges_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Export inferred spikes cross-correlation lags matrix (rising edges) to CSV.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output file path for CSV
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    spike_method : {"oasis", "cascade"}
        Stored output to export; each method uses its own population and threshold.
    """
    _export_single_correlation_matrix(
        engine,
        output_path,
        "spike_max_lag_values_matrix_rising_edges",
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
        spike_method=spike_method,
    )


def export_inferred_spikes_ccg_zscore_rising_edges_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Export inferred spikes CCG z-score matrix (rising edges) to CSV.

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output file path for CSV
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    spike_method : {"oasis", "cascade"}
        Stored output to export; each method uses its own population and threshold.
    """
    _export_single_correlation_matrix(
        engine,
        output_path,
        "spike_ccg_zscore_matrix_rising_edges",
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
        spike_method=spike_method,
    )


def export_cluster_labels_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
) -> None:
    """Export cluster labels per ROI to CSV.

    Creates a CSV with one row per ROI containing the FOV name, ROI label,
    cluster assignment, and cluster metadata (method, k, silhouette score).

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
    position_indices : list[int] | None, optional
        Position indices to filter exports. If provided, only exports data
        from these positions.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = _get_default_run_id(engine)

    ensure_schema_current(engine)
    with Session(engine) as session:
        stmt = (
            select(FOVAnalysis, FOV)
            .join(FOV, FOVAnalysis.fov_id == FOV.id)
            .where(FOVAnalysis.analysis_result_id == run_id)
        )
        if fov_name is not None:
            stmt = stmt.where(col(FOV.name) == fov_name)
        if position_indices is not None:
            stmt = stmt.where(col(FOV.position_index).in_(position_indices))

        results = session.exec(stmt).all()

    if not results:
        msg = "No FOV analysis data found"
        raise ValueError(msg)

    rows = []
    for fov_analysis, fov in results:
        if (
            fov_analysis.cluster_labels is None
            or fov_analysis.calcium_active_roi_labels is None
        ):
            continue
        for roi_label, cluster_label in zip(
            fov_analysis.calcium_active_roi_labels, fov_analysis.cluster_labels
        ):
            rows.append(
                {
                    "fov": fov.name,
                    "roi_label": roi_label,
                    "cluster_label": cluster_label,
                    "cluster_method": fov_analysis.cluster_method,
                    "cluster_n_clusters": fov_analysis.cluster_n_clusters,
                    "cluster_silhouette_score": fov_analysis.cluster_silhouette_score,
                }
            )

    if not rows:
        msg = "No cluster label data found"
        raise ValueError(msg)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


# ==================== Helper Functions ====================


def _export_single_correlation_matrix(
    engine: Engine,
    output_path: str | Path,
    matrix_attr: str,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Export a single correlation matrix type to CSV (internal helper).

    Parameters
    ----------
    engine : Engine
        Database engine
    output_path : str | Path
        Output file path for CSV
    matrix_attr : str
        FOVAnalysis attribute name for the matrix to export
    fov_name : str | None, optional
        Specific FOV to export. If None, exports all FOVs (one file per FOV)
    run_id : int | None, optional
        Analysis run ID. If None, uses the first available run
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    spike_method : {"oasis", "cascade"}
        Stored output to export; each method uses its own population and threshold.
    """
    canonical_spike_methods((spike_method,))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get run_id if not provided
    if run_id is None:
        run_id = _get_default_run_id(engine)

    # Query FOV analysis data
    ensure_schema_current(engine)
    with Session(engine) as session:
        stmt = (
            select(FOVAnalysis, FOV)
            .join(FOV, FOVAnalysis.fov_id == FOV.id)
            .where(FOVAnalysis.analysis_result_id == run_id)
        )

        if fov_name is not None:
            stmt = stmt.where(col(FOV.name) == fov_name)

        if position_indices is not None:
            stmt = stmt.where(col(FOV.position_index).in_(position_indices))

        results = session.exec(stmt).all()

        if not results:
            msg = "No FOV analysis data found"
            raise ValueError(msg)

        if not matrix_attr.startswith("calcium_") and not any(
            parent.get_spike_analysis(spike_method) is not None for parent, _ in results
        ):
            raise ValueError(f"No FOV spike analysis results for {spike_method}.")
        for fov_analysis, fov in results:
            fov_output_path = output_path.parent / f"{fov.name}_{output_path.name}"
            if matrix_attr.startswith("calcium_"):
                _export_ordered_fov_matrix(
                    session,
                    fov,
                    getattr(fov_analysis, matrix_attr),
                    fov_analysis.calcium_active_roi_labels,
                    fov_output_path,
                )
            else:
                _export_spike_fov_matrix(
                    session,
                    fov_analysis,
                    fov,
                    spike_method,
                    matrix_attr,
                    fov_output_path,
                )


def _get_condition_groups(
    engine: Engine, *, run_id: int | None = None
) -> dict[str, list[int]]:
    """Map condition labels to FOV position indices.

    If run_id is provided, only includes FOVs that have trace data for that run.
    This prevents creating empty export subfolders for conditions without data.

    Parameters
    ----------
    engine : Engine
        Database engine
    run_id : int | None
        If provided, only include FOVs with trace data for this analysis run

    Returns
    -------
    dict[str, list[int]]
        Dict like `{"WT_Drug": [0, 1], "KO_Vehicle": [2, 3], "": [4]}`
        where the `""` key collects FOVs whose well has no conditions (or no well).
    """
    from cali.plot._multi_wells_plots._util import _get_condition_label

    groups: dict[str, list[int]] = {}
    ensure_schema_current(engine)
    with Session(engine) as session:
        # Query FOVs, optionally filtered by run_id
        if run_id is not None:
            # Only include FOVs that have trace data for this run
            stmt = (
                select(FOV)
                .join(ROI, FOV.id == ROI.fov_id)
                .join(Traces, ROI.id == Traces.roi_id)
                .where(Traces.analysis_result_id == run_id)
                .distinct()
            )
            fovs = session.exec(stmt).all()
        else:
            # Get all FOVs
            fovs = session.exec(select(FOV)).all()

        for fov in fovs:
            well = fov.well if fov.well_id is not None else None
            if well is not None and well.conditions:
                label = _get_condition_label(well)
            else:
                label = ""
            groups.setdefault(label, []).append(fov.position_index)
    return groups


def _get_default_run_id(engine: Engine) -> int:
    """Get the first available analysis run ID."""
    ensure_schema_current(engine)
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
    position_indices: list[int] | None = None,
    spike_method: SpikeMethod = "oasis",
) -> None:
    """Export trace data to CSV (internal helper)."""
    canonical_spike_methods((spike_method,))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get run_id if not provided
    if run_id is None:
        run_id = _get_default_run_id(engine)

    # Query traces data
    ensure_schema_current(engine)
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .outerjoin(
                DataAnalysis,
                (col(DataAnalysis.roi_id) == col(ROI.id))
                & (col(DataAnalysis.analysis_result_id) == run_id),
            )
        )

        if fov_name is not None:
            stmt = stmt.where(col(FOV.name) == fov_name)

        if position_indices is not None:
            stmt = stmt.where(col(FOV.position_index).in_(position_indices))

        stmt = stmt.order_by(col(FOV.name), col(ROI.label_value))
        results = session.exec(stmt).all()

        if not results:
            msg = f"No trace data found for run_id={run_id}"
            raise ValueError(msg)

        # Group by FOV and stimulation status
        fov_data: dict[str, dict[Literal["stim", "non_stim"], list]] = {}
        applied_thresholds = []

        for roi, traces, data_analysis in results:
            trace_data: list[float] | None
            # Map display names to Traces model attribute names
            trace_attr_map = {
                RAW_CALCIUM_TRACES: "raw_trace",
                NEUROPIL_TRACES: "neuropil_trace",
                NEUROPIL_CORRECTED_TRACES: "corrected_trace",
                DFF_TRACES: "dff",
                DEN_DFF_TRACES: "den_dff",
                INFERRED_SPIKES_TRACES: "inferred_spikes",
                CASCADE_EXPECTED_SPIKES_TRACES: "inferred_spikes",
                INFERRED_SPIKES_THRESHOLDED_BINARY: "inferred_spikes",
            }

            # Get trace data
            if trace_type == INFERRED_SPIKES_THRESHOLDED_BINARY:
                # Binarize inferred spikes based on threshold
                child = traces.get_spike_trace(spike_method)
                threshold = (
                    data_analysis.get_spike_metric(spike_method, "threshold")
                    if data_analysis is not None
                    else None
                )
                if child is None or threshold is None:
                    continue
                metric = data_analysis.get_spike_analysis(spike_method)
                assert metric is not None
                if metric.provenance_source == "legacy_unresolved":
                    continue
                if (
                    metric.spike_trace_id is not None
                    and metric.spike_trace_id != child.id
                ):
                    raise ValueError("Binary export must use the analyzed spike trace.")
                from ._spike_export import _threshold_record

                applied_thresholds.append(
                    {
                        **_threshold_record(metric, roi.label_value),
                        "fov_name": roi.fov.name,
                        "spike_trace_id": child.id,
                        "valid_start_frame_0based": child.valid_start,
                        "valid_stop_frame_exclusive": child.resolved_valid_stop,
                        "provenance": child.inference_run.model_dump(mode="json"),
                    }
                )
                binary, _ = valid_spike_events(child, threshold)
                trace_data = [math.nan] * len(child.values)
                trace_data[child.valid_start : child.resolved_valid_stop] = (
                    binary.tolist()
                )
            else:
                attr_name = trace_attr_map.get(trace_type)
                trace_data = (
                    traces.get_spike_values(spike_method)
                    if attr_name == "inferred_spikes"
                    else getattr(traces, attr_name, None)
                    if attr_name
                    else None
                )

            if trace_data is None:
                continue
            if trace_type in (INFERRED_SPIKES_TRACES, CASCADE_EXPECTED_SPIKES_TRACES):
                child = traces.get_spike_trace(spike_method)
                assert child is not None
                trace_data = [
                    value
                    if child.valid_start <= index < child.resolved_valid_stop
                    else math.nan
                    for index, value in enumerate(trace_data)
                ]

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
        # Always include FOV name in column to ensure uniqueness across positions
        all_data = []
        column_names = []

        for fov_key in sorted(fov_data.keys(), key=natural_sort_key):
            # Add stimulated ROIs first
            for roi_info in sorted(
                fov_data[fov_key]["stim"], key=lambda x: x["roi_label"]
            ):
                all_data.append(roi_info["trace"])
                column_names.append(
                    f"{roi_info['fov_name']}_ROI_{roi_info['roi_label']}_stim"
                )

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
                column_names.append(
                    f"{roi_info['fov_name']}_ROI_{roi_info['roi_label']}{suffix}"
                )

        # Create DataFrame with traces as columns
        # Seconds-mode discard can leave different lengths in different FOVs.
        # Align retained row indices and leave missing trailing samples empty.
        df = (
            pd.DataFrame(np.array(all_data).T, columns=column_names)
            if len({len(values) for values in all_data}) == 1
            else pd.DataFrame(
                {
                    name: pd.Series(values)
                    for name, values in zip(column_names, all_data)
                }
            )
        )

        # Save to CSV
        df.to_csv(output_path, index=False)
        if trace_type == INFERRED_SPIKES_THRESHOLDED_BINARY:
            from ._spike_export import write_spike_metadata

            write_spike_metadata(
                output_path.with_suffix(".metadata.json"),
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "method": spike_method,
                    "thresholds": applied_thresholds,
                },
            )


_COORDINATE_COLUMNS = [
    "fov_name",
    "position_index",
    "roi_label",
    "trace_id",
    "extraction_result_id",
    "coordinate_schema_version",
    "retained_frame_0based",
    "source_frame_0based",
    "source_frame_1based",
    "retained_time_ms",
    "source_time_ms",
    "source_timestamp_ms",
    "source_start_frame",
    "source_start_time_ms",
    "timing_source",
]


def _coordinate_row(
    roi: ROI, trace: Traces, frame: int, transform: SourceFrameTransform | None = None
) -> dict[str, object]:
    """Build an explicit, reversible sample coordinate without inferring timing."""
    transform = transform or trace.source_frame_transform()
    retained_time, source_time, timestamp = transform.frame_times(frame)
    window = trace.extraction_frame_window
    return dict(
        zip(
            _COORDINATE_COLUMNS,
            (
                roi.fov.name,
                roi.fov.position_index,
                roi.label_value,
                trace.id,
                window.extraction_result_id if window else None,
                window.schema_version if window else None,
                frame,
                int(transform.to_source(frame, one_based=False)),
                int(transform.to_source(frame)),
                retained_time,
                source_time,
                timestamp,
                transform.source_start_frame,
                transform.source_start_time_ms,
                window.timing_source if window else None,
            ),
        )
    )


def export_frame_coordinates_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    run_id: int,
    fov_name: str | None = None,
    position_indices: list[int] | None = None,
) -> None:
    """Export each trace's retained/source indices, offsets, and known timing.

    Unknown historical absolute timestamps remain empty. The existing wide trace
    CSV row number is ``retained_frame_0based``; source frames have both explicit
    bases so the file can be joined to one-based acquisition/stimulation inputs.
    """
    ensure_schema_current(engine)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces)
            .join(FOV, col(ROI.fov_id) == col(FOV.id))
            .join(Traces, col(Traces.roi_id) == col(ROI.id))
            .where(Traces.analysis_result_id == run_id)
            .order_by(col(FOV.name), col(ROI.label_value))
        )
        if fov_name is not None:
            stmt = stmt.where(FOV.name == fov_name)
        if position_indices is not None:
            stmt = stmt.where(col(FOV.position_index).in_(position_indices))
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=_COORDINATE_COLUMNS)
            writer.writeheader()
            for roi, trace in session.exec(stmt):
                transform = trace.source_frame_transform()
                writer.writerows(
                    _coordinate_row(roi, trace, frame, transform)
                    for frame in range(transform.retained_frame_count)
                )


def export_events_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    run_id: int,
    fov_name: str | None = None,
    position_indices: list[int] | None = None,
) -> None:
    """Export calcium peaks and threshold excursions with exact source coordinates.

    Excursion starts are threshold events, not discrete inferred action potentials.
    They use the applied method threshold and only that spike trace's valid interval.
    Unresolved historical spike metrics have no trace binding and are omitted.
    """
    ensure_schema_current(engine)
    rows: list[dict[str, object]] = []
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, col(ROI.fov_id) == col(FOV.id))
            .join(Traces, col(Traces.roi_id) == col(ROI.id))
            .join(DataAnalysis, col(DataAnalysis.roi_id) == col(ROI.id))
            .where(
                Traces.analysis_result_id == run_id,
                DataAnalysis.analysis_result_id == run_id,
            )
            .order_by(col(FOV.name), col(ROI.label_value))
        )
        if fov_name is not None:
            stmt = stmt.where(FOV.name == fov_name)
        if position_indices is not None:
            stmt = stmt.where(col(FOV.position_index).in_(position_indices))
        for roi, trace, analysis in session.exec(stmt).all():
            for peak in analysis.peaks_den_dff or []:
                if not math.isfinite(peak) or not float(peak).is_integer():
                    raise ValueError(
                        "Calcium event indices must be finite whole frames."
                    )
                frame = int(peak)
                row = _coordinate_row(roi, trace, frame)
                row.update(
                    event_type="calcium_peak",
                    method="oasis_denoising",
                    units="dF/F",
                    value=trace.den_dff[frame] if trace.den_dff else None,
                    threshold=analysis.peaks_height_den_dff,
                    threshold_mode=None,
                    threshold_units="dF/F",
                    valid_start_frame_0based=0,
                    valid_stop_frame_0based=trace.source_frame_transform().retained_frame_count,
                )
                rows.append(row)
            for child in sorted(
                analysis.spike_analyses, key=lambda child: child.method != "oasis"
            ):
                spike = child.spike_trace
                if spike is None or child.threshold is None:
                    continue
                source_trace = spike.trace
                _, onsets = valid_spike_events(spike, child.threshold)
                starts = np.flatnonzero(onsets)
                for start in starts:
                    frame = int(start) + spike.valid_start
                    row = _coordinate_row(roi, source_trace, frame)
                    row.update(
                        event_type="threshold_excursion_start",
                        method=child.method,
                        units=child.units,
                        value=spike.values[frame],
                        threshold=child.threshold,
                        threshold_mode=child.threshold_mode,
                        threshold_units=child.units,
                        valid_start_frame_0based=spike.valid_start,
                        valid_stop_frame_0based=spike.resolved_valid_stop,
                    )
                    rows.append(row)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        *_COORDINATE_COLUMNS,
        "event_type",
        "method",
        "units",
        "value",
        "threshold",
        "threshold_mode",
        "threshold_units",
        "valid_start_frame_0based",
        "valid_stop_frame_0based",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def export_cascade_expected_spikes_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    fov_name: str | None = None,
    run_id: int | None = None,
    position_indices: list[int] | None = None,
) -> None:
    """Export stored CASCADE spikes/frame, leaving invalid edge samples empty."""
    _export_trace_data(
        engine,
        output_path,
        CASCADE_EXPECTED_SPIKES_TRACES,
        fov_name=fov_name,
        run_id=run_id,
        position_indices=position_indices,
        spike_method="cascade",
    )
    export_trace_metadata(
        engine,
        Path(output_path).with_suffix(".metadata.json"),
        run_id=run_id,
        fov_name=fov_name,
        position_indices=position_indices,
    )


def export_trace_metadata(
    engine: Engine,
    output_path: str | Path,
    *,
    run_id: int | None = None,
    fov_name: str | None = None,
    position_indices: list[int] | None = None,
) -> set[str]:
    """Write stored provenance/coordinates and return the retained method IDs."""
    ensure_schema_current(engine)
    if run_id is None:
        run_id = _get_default_run_id(engine)
    records = []
    methods: set[str] = set()
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces)
            .join(FOV, col(ROI.fov_id) == col(FOV.id))
            .join(Traces, col(Traces.roi_id) == col(ROI.id))
            .where(col(Traces.analysis_result_id) == run_id)
            .order_by(col(FOV.position_index), col(ROI.label_value), col(Traces.id))
        )
        if fov_name is not None:
            stmt = stmt.where(col(FOV.name) == fov_name)
        if position_indices is not None:
            stmt = stmt.where(col(FOV.position_index).in_(position_indices))
        for roi, trace in session.exec(stmt):
            methods.update(child.inference_run.method for child in trace.spike_traces)
            window = trace.extraction_frame_window
            records.append(
                {
                    "trace_id": trace.id,
                    "fov_name": roi.fov.name,
                    "position_index": roi.fov.position_index,
                    "roi_label": roi.label_value,
                    "calcium_method": "oasis_denoising",
                    "calcium_noise": trace.calcium_noise,
                    "x_axis_units": trace.x_axis_units,
                    "frame_window": window.model_dump(mode="json") if window else None,
                    "spike_outputs": [
                        {
                            "spike_trace_id": child.id,
                            "provenance": child.inference_run.model_dump(mode="json"),
                            "valid_start_frame_0based": child.valid_start,
                            "valid_stop_frame_exclusive": child.resolved_valid_stop,
                            "noise": child.noise,
                            "selected_noise_level": child.selected_noise_level,
                        }
                        for child in sorted(
                            trace.spike_traces,
                            key=lambda child: (
                                child.inference_run.method != "oasis",
                                child.id or 0,
                            ),
                        )
                    ],
                }
            )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "run_id": run_id, "traces": records},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return methods


def export_traces_to_csv(
    engine: Engine,
    export_traces: dict[TraceDataType, bool],
    run_id: int,
    db_path: Path,
    *,
    position_indices: list[int] | None = None,
) -> None:
    """Export selected traces to CSV files.

    Parameters
    ----------
    engine : Engine
        Database engine
    export_traces : dict[TraceDataType, bool]
        Dictionary mapping trace type names to export status.
        Only TraceDataType literals are valid keys.
    run_id : int
        Analysis result ID to export
    db_path : Path
        Database path (used to determine output directory)
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    """
    # Map trace type names to export functions
    export_map = {
        RAW_CALCIUM_TRACES: (export_raw_traces_to_csv, "raw_traces.csv"),
        NEUROPIL_TRACES: (export_neuropil_traces_to_csv, "neuropil_traces.csv"),
        NEUROPIL_CORRECTED_TRACES: (
            export_neuropil_corrected_traces_to_csv,
            "neuropil_corrected_traces.csv",
        ),
        DFF_TRACES: (export_dff_traces_to_csv, "dff_traces.csv"),
        DEN_DFF_TRACES: (
            export_denoised_dff_traces_to_csv,
            "denoised_dff_traces.csv",
        ),
        INFERRED_SPIKES_TRACES: (
            export_inferred_spikes_raw_to_csv,
            "inferred_spikes_raw.csv",
        ),
        CASCADE_EXPECTED_SPIKES_TRACES: (
            export_cascade_expected_spikes_to_csv,
            "cascade_expected_spikes.csv",
        ),
        INFERRED_SPIKES_THRESHOLDED_BINARY: (
            export_inferred_spikes_thresholded_to_csv,
            "inferred_spikes_thresholded.csv",
        ),
    }

    # Create export directory next to database
    export_dir = db_path.parent / f"{db_path.stem}_exports" / f"run_{run_id}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Group FOVs by condition for subfolder organization
    # Only include conditions with data for this run_id
    condition_groups = _get_condition_groups(engine, run_id=run_id)
    has_conditions = any(label != "" for label in condition_groups)

    # Build list of (output_dir, position_indices) pairs to export
    if not has_conditions:
        # No conditions — flat export as before
        export_targets = [(export_dir, position_indices)]
    else:
        export_targets = []
        for label, group_indices in sorted(condition_groups.items()):
            # Filter by caller's position_indices if provided
            if position_indices is not None:
                group_indices = [i for i in group_indices if i in position_indices]
                if not group_indices:
                    continue
            # Sanitize label for use as folder name (e.g. "+/+" -> "+_+")
            safe_label = label.replace("/", "_")
            out_dir = export_dir / safe_label if safe_label else export_dir
            export_targets.append((out_dir, group_indices))

    # Export each selected trace type into each target directory
    for target_dir, target_indices in export_targets:
        if any(export_traces.values()):
            from ._noise_qc_export import export_noise_qc_to_csv

            export_noise_qc_to_csv(
                engine,
                target_dir / "noise_qc.csv",
                run_id=run_id,
                position_indices=target_indices,
            )
            stored_methods = export_trace_metadata(
                engine,
                target_dir / "trace_metadata.json",
                run_id=run_id,
                position_indices=target_indices,
            )
            export_frame_coordinates_to_csv(
                engine,
                target_dir / "frame_coordinates.csv",
                run_id=run_id,
                position_indices=target_indices,
            )
            export_events_to_csv(
                engine,
                target_dir / "events.csv",
                run_id=run_id,
                position_indices=target_indices,
            )
            from ._spike_export import (
                available_spike_result_methods,
                export_spike_results_to_csv,
            )

            if available_spike_result_methods(
                engine,
                run_id=run_id,
                position_indices=target_indices,
            ):
                export_spike_results_to_csv(
                    engine,
                    target_dir,
                    run_id=run_id,
                    position_indices=target_indices,
                    include_matrices=False,
                )
        for trace_type, should_export in export_traces.items():
            if should_export and trace_type in export_map:
                export_func, filename = export_map[trace_type]
                if trace_type == INFERRED_SPIKES_TRACES and "cascade" in stored_methods:
                    filename = "oasis_inferred_spikes_raw.csv"
                output_path = target_dir / filename
                try:
                    from cali.logger import cali_logger

                    cali_logger.info(f"📊 Exporting {trace_type} to {output_path}...")
                    export_func(
                        engine,
                        output_path,
                        run_id=run_id,
                        position_indices=target_indices,
                    )
                    cali_logger.info(f"✅ Exported {trace_type} successfully")
                except Exception as e:
                    from cali.logger import cali_logger

                    cali_logger.error(f"❌ Failed to export {trace_type}: {e}")


def export_correlations_to_csv(
    engine: Engine,
    export_correlations: dict[CorrelationDataType, bool],
    run_id: int,
    db_path: Path,
    *,
    position_indices: list[int] | None = None,
    spike_methods: tuple[SpikeMethod, ...] | None = None,
) -> None:
    """Export selected correlation data to CSV files.

    Parameters
    ----------
    engine : Engine
        Database engine
    export_correlations : dict
        Dictionary mapping correlation data type names to export status.
        Valid keys are the correlation matrix type literals.
    run_id : int
        Analysis result ID to export
    db_path : Path
        Database path (used to determine output directory)
    position_indices : list[int] | None
        Optional list of position indices to filter exports.
        If provided, only exports data from these positions.
    spike_methods : tuple | None
        Stored methods to export; None exports all available methods separately.
    """
    from cali._constants import (
        CALCIUM_DEN_DFF_CORRELATION,
        CALCIUM_DFF_CORRELATION,
        CLUSTER_LABELS,
        INFERRED_SPIKES_CCG_ZSCORE,
        INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES,
        INFERRED_SPIKES_CROSS_CORRELATION,
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS,
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES,
        INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES,
        INFERRED_SPIKES_SYNCHRONY,
        INFERRED_SPIKES_SYNCHRONY_RISING_EDGES,
    )

    # Map correlation type names to export functions
    export_map = {
        # Calcium correlations
        CALCIUM_DFF_CORRELATION: (
            export_calcium_dff_correlation_to_csv,
            "calcium_dff_correlation_matrix.csv",
        ),
        CALCIUM_DEN_DFF_CORRELATION: (
            export_calcium_den_dff_correlation_to_csv,
            "calcium_den_dff_correlation_matrix.csv",
        ),
        # Inferred Spikes - Thresholded Binary
        INFERRED_SPIKES_SYNCHRONY: (
            export_inferred_spikes_synchrony_to_csv,
            "inferred_spikes_synchrony_matrix.csv",
        ),
        INFERRED_SPIKES_CROSS_CORRELATION: (
            export_inferred_spikes_cross_correlation_to_csv,
            "inferred_spikes_cross_correlation_matrix.csv",
        ),
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS: (
            export_inferred_spikes_cross_correlation_lags_to_csv,
            "inferred_spikes_cross_correlation_lags_matrix.csv",
        ),
        INFERRED_SPIKES_CCG_ZSCORE: (
            export_inferred_spikes_ccg_zscore_to_csv,
            "inferred_spikes_ccg_zscore_matrix.csv",
        ),
        # Inferred Spikes - Thresholded Rising Edges
        INFERRED_SPIKES_SYNCHRONY_RISING_EDGES: (
            export_inferred_spikes_synchrony_rising_edges_to_csv,
            "inferred_spikes_synchrony_matrix_rising_edges.csv",
        ),
        INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES: (
            export_inferred_spikes_cross_correlation_rising_edges_to_csv,
            "inferred_spikes_cross_correlation_matrix_rising_edges.csv",
        ),
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES: (
            export_inferred_spikes_cross_correlation_lags_rising_edges_to_csv,
            "inferred_spikes_cross_correlation_lags_matrix_rising_edges.csv",
        ),
        INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES: (
            export_inferred_spikes_ccg_zscore_rising_edges_to_csv,
            "inferred_spikes_ccg_zscore_matrix_rising_edges.csv",
        ),
        # Cluster Analysis
        CLUSTER_LABELS: (
            export_cluster_labels_to_csv,
            "cluster_labels.csv",
        ),
    }

    from ._spike_export import (
        available_spike_result_methods,
        export_spike_results_to_csv,
    )

    spike_attrs = {
        INFERRED_SPIKES_SYNCHRONY: "spike_jitter_synchrony_matrix",
        INFERRED_SPIKES_CROSS_CORRELATION: "spike_max_lag_correlation_matrix",
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS: "spike_max_lag_values_matrix",
        INFERRED_SPIKES_CCG_ZSCORE: "spike_ccg_zscore_matrix",
        INFERRED_SPIKES_SYNCHRONY_RISING_EDGES: (
            "spike_jitter_synchrony_matrix_rising_edges"
        ),
        INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES: (
            "spike_max_lag_correlation_matrix_rising_edges"
        ),
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES: (
            "spike_max_lag_values_matrix_rising_edges"
        ),
        INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES: "spike_ccg_zscore_matrix_rising_edges",
    }
    has_spike_exports = any(
        selected and name in spike_attrs
        for name, selected in export_correlations.items()
    )
    available = (
        available_spike_result_methods(
            engine, run_id=run_id, position_indices=position_indices
        )
        if has_spike_exports
        else set()
    )
    methods = (
        canonical_spike_methods(spike_methods)
        if spike_methods is not None
        else canonical_spike_methods(list(available))
        if available
        else ()
    )
    if has_spike_exports and spike_methods is not None and set(methods) - available:
        raise ValueError("Requested spike export methods are not stored on this run.")

    # Create export directory next to database
    export_dir = db_path.parent / f"{db_path.stem}_exports" / f"run_{run_id}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Group FOVs by condition for subfolder organization
    # Only include conditions with data for this run_id
    condition_groups = _get_condition_groups(engine, run_id=run_id)
    has_conditions = any(label != "" for label in condition_groups)

    # Build list of (output_dir, position_indices) pairs to export
    if not has_conditions:
        export_targets = [(export_dir, position_indices)]
    else:
        export_targets = []
        for label, group_indices in sorted(condition_groups.items()):
            if position_indices is not None:
                group_indices = [i for i in group_indices if i in position_indices]
                if not group_indices:
                    continue
            # Sanitize label for use as folder name (e.g. "+/+" -> "+_+")
            safe_label = label.replace("/", "_")
            out_dir = export_dir / safe_label if safe_label else export_dir
            export_targets.append((out_dir, group_indices))

    # Export each selected correlation type into each target directory
    # Note: Correlation exports create separate files per FOV
    for target_dir, target_indices in export_targets:
        target_available = (
            available_spike_result_methods(
                engine, run_id=run_id, position_indices=target_indices
            )
            if has_spike_exports
            else set()
        )
        target_methods = tuple(
            method for method in methods if method in target_available
        )
        matrix_methods = (
            available_spike_result_methods(
                engine,
                run_id=run_id,
                position_indices=target_indices,
                fov_only=True,
            )
            if target_methods
            else set()
        )
        if target_methods:
            export_spike_results_to_csv(
                engine,
                target_dir,
                run_id=run_id,
                spike_methods=target_methods,
                position_indices=target_indices,
                include_matrices=False,
            )
        for corr_type, should_export in export_correlations.items():
            if not should_export or corr_type not in export_map:
                continue
            export_func, base_filename = export_map[corr_type]
            if corr_type in spike_attrs:
                for method in target_methods:
                    if method not in matrix_methods:
                        continue
                    filename = (
                        f"{method}_{base_filename}"
                        if available != {"oasis"}
                        else base_filename
                    )
                    _export_single_correlation_matrix(
                        engine,
                        target_dir / filename,
                        spike_attrs[corr_type],
                        run_id=run_id,
                        position_indices=target_indices,
                        spike_method=method,
                    )
            else:
                try:
                    export_func(
                        engine,
                        target_dir / base_filename,
                        run_id=run_id,
                        position_indices=target_indices,
                    )
                except Exception as error:
                    from cali.logger import cali_logger

                    cali_logger.error(f"❌ Failed to export {corr_type}: {error}")


def _export_spike_fov_matrix(
    session: Session,
    parent: FOVAnalysis,
    fov: FOV,
    method: SpikeMethod,
    metric: str,
    output_path: Path,
) -> None:
    """Write one method's matrix and its exact population/threshold metadata."""
    from ._spike_export import fov_spike_metadata, write_spike_metadata

    child = parent.get_spike_analysis(method)
    if child is None or getattr(child, metric) is None or not child.active_roi_labels:
        return
    metadata = fov_spike_metadata(session, parent, fov, child)
    metadata["matrix_metric"] = metric
    metadata["exported_roi_labels"] = _export_ordered_fov_matrix(
        session,
        fov,
        getattr(child, metric),
        child.active_roi_labels,
        output_path,
    )
    write_spike_metadata(output_path.with_suffix(".metadata.json"), metadata)


def _export_ordered_fov_matrix(
    session: Session,
    fov: FOV,
    matrix: list[list[float]] | list[list[int]] | None,
    roi_labels: list[int] | None,
    output_path: Path,
) -> list[int]:
    """Keep a pillar/method's matrix aligned when sorting its labels for export."""
    if matrix is None or not roi_labels:
        return []
    if len(roi_labels) != len(set(roi_labels)) or np.asarray(matrix).shape != (
        len(roi_labels),
        len(roi_labels),
    ):
        raise ValueError(
            "Matrix exports require unique labels and matching dimensions."
        )
    rois = session.exec(
        select(ROI).where(ROI.fov_id == fov.id, col(ROI.label_value).in_(roi_labels))
    ).all()
    roi_dict = {roi.label_value: roi for roi in rois}
    if set(roi_labels) - roi_dict.keys():
        raise ValueError("Matrix export labels must belong to the selected FOV.")
    sorted_labels = sorted(
        roi_labels,
        key=lambda label: (not bool(roi_dict[label].stimulated), label),
    )
    order = [roi_labels.index(label) for label in sorted_labels]
    ordered_matrix = np.asarray(matrix)[np.ix_(order, order)].tolist()
    _export_matrix_to_csv(
        ordered_matrix,
        [f"ROI_{label}" for label in sorted_labels],
        output_path,
    )
    return sorted_labels


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


def export_multi_well_to_csv(
    engine: Engine,
    run_id: int,
    db_path: Path,
    experiment_type: str | None = None,
    *,
    spike_methods: tuple[SpikeMethod, ...] | None = None,
) -> None:
    """Export all multi-well aggregated bar plot data to CSV files.

    Iterates over registered multi-well AnalysisProducts that have a
    `compute_fn`, calls each to compute aggregated condition-level
    statistics, and writes one CSV per analysis into a `multi_well/`
    subfolder.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine connected to the database.
    run_id : int
        The CaliResult.id to export data for.
    db_path : Path
        Path to the database file (used to determine export directory).
    experiment_type : str | None
        Experiment type (`"evoked"` or `"spontaneous"`).  Products
        whose `experiment_type` doesn't match are skipped.
    spike_methods : tuple | None
        Stored methods to export separately; None exports all available methods.
    """
    from cali.logger import cali_logger
    from cali.plot._main_plot import ANALYSIS_PRODUCTS, AnalysisGroup

    from ._spike_export import (
        _selected_methods,
        available_spike_result_methods,
        write_spike_metadata,
    )

    available = available_spike_result_methods(engine, run_id=run_id)
    methods = _selected_methods(available, spike_methods)
    # Resolve all spike products before writing, so invalid selected data cannot
    # partially replace previously exported method files.
    computed = []
    for product in ANALYSIS_PRODUCTS:
        if product.group != AnalysisGroup.MULTI_WELL or product.compute_fn is None:
            continue
        if (
            product.experiment_type is not None
            and product.experiment_type != experiment_type
        ):
            continue
        choices = (
            [method for method in methods if method in product.supported_spike_methods]
            if product.supported_spike_methods is not None
            else [None]
        )
        for method in choices:
            try:
                result = product.compute_data(engine, run_id, spike_method=method)
            except Exception as e:
                if method is not None:
                    raise
                cali_logger.debug(
                    f"Skipping multi-well export for '{product.name}': {e}"
                )
                continue
            if result is not None:
                computed.append((product, method, result))

    output_dir = (
        db_path.parent / f"{db_path.stem}_exports" / f"run_{run_id}" / "multi_well"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for product, method, result in computed:
        bar_data, _name, _units = result
        # Build DataFrame from BarPlotData
        well_values = bar_data["well_values_list"]
        well_names = bar_data["well_names_list"]
        # Collect all unique well names across conditions to use as columns
        all_well_names: list[str] = []
        seen_wells: set[str] = set()
        for names in well_names:
            for name in names:
                if name not in seen_wells:
                    seen_wells.add(name)
                    all_well_names.append(name)
        rows: list[dict[str, object]] = []
        for cond, mean, sem, wv, wn in zip(
            bar_data["conditions"],
            bar_data["means"],
            bar_data["sems"],
            well_values,
            well_names,
        ):
            row: dict[str, object] = {"condition": cond, "mean": mean, "sem": sem}
            # Map well names to values for this condition
            name_to_val = dict(zip(wn, wv))
            for name in all_well_names:
                row[name] = (
                    float(name_to_val[name]) if name in name_to_val else float("nan")
                )
            rows.append(row)

        df = pd.DataFrame(rows)
        # Sanitize filename from product name
        safe_name = product.name.replace(" ", "_").replace("/", "_").lower()
        if method is not None and (method != "oasis" or len(available) > 1):
            if not safe_name.startswith(method + "_"):
                safe_name = method + "_" + safe_name
        csv_path = output_dir / f"{safe_name}.csv"
        df.to_csv(csv_path, index=False)
        if product.supported_spike_methods is not None:
            write_spike_metadata(
                csv_path.with_suffix(".metadata.json"),
                {
                    "schema_version": 1,
                    "product_id": product.product_id,
                    "run_id": run_id,
                    "method": method,
                    "metric": _name,
                    "metric_units": _units,
                    "required_metrics": list(product.required_metrics),
                },
            )
        cali_logger.debug(f"Exported multi-well data: {csv_path}")


def export_multi_well_pca_to_csv(
    engine: Engine,
    run_id: int,
    db_path: Path,
    *,
    spike_methods: tuple[SpikeMethod, ...] | None = None,
) -> None:
    """Export PCA analysis data (feature matrix, coordinates, loadings) to CSV.

    Builds the FOV feature matrix, runs PCA, and exports three CSV files:
    - `pca_feature_matrix.csv`: raw per-FOV feature values
    - `pca_coordinates.csv`: PCA-transformed coordinates per FOV
    - `pca_loadings_and_scree.csv`: loadings and explained variance per component

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine connected to the database.
    run_id : int
        The CaliResult.id to export data for.
    db_path : Path
        Path to the database file (used to determine export directory).
    spike_methods : tuple | None
        Stored methods to export separately; None exports all available methods.
    """
    from cali.logger import cali_logger

    try:
        from cali.plot._multi_wells_plots._dimensionality_reduction import (
            build_fov_feature_matrix,
            compute_pca,
        )
    except ImportError:
        cali_logger.debug("sklearn not available, skipping PCA export")
        return

    from ._spike_export import (
        _selected_methods,
        available_spike_result_methods,
        write_spike_metadata,
    )

    available = available_spike_result_methods(engine, run_id=run_id)
    methods = _selected_methods(available, spike_methods)
    # Calcium-only historical PCA remains readable through its OASIS default.
    if not methods:
        methods = ("oasis",)
    computed = []
    for method in methods:
        try:
            df = build_fov_feature_matrix(engine, run_id, spike_method=method)
        except ValueError:
            raise
        except Exception as e:
            cali_logger.debug(f"Skipping {method} PCA export: {e}")
            continue
        if len(df) < 2:
            continue
        try:
            coords, pca, used_features = compute_pca(df, n_components=min(len(df), 10))
        except Exception as e:
            cali_logger.debug(f"Skipping {method} PCA export (compute failed): {e}")
            continue
        computed.append((method, df, coords, pca, used_features))
    if not computed:
        return

    output_dir = (
        db_path.parent / f"{db_path.stem}_exports" / f"run_{run_id}" / "multi_well"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, df, coords, pca, used_features in computed:
        prefix = method + "_" if method != "oasis" or len(available) > 1 else ""
        # 1. Feature matrix CSV
        df.assign(spike_method=method).to_csv(
            output_dir / f"{prefix}pca_feature_matrix.csv", index=False
        )
        cali_logger.debug(f"Exported PCA feature matrix: {output_dir}")

        # 2. Coordinates CSV
        n_components = coords.shape[1]
        coord_df = pd.DataFrame(
            coords,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        )
        coord_df.insert(0, "fov_name", df["fov_name"].values)
        coord_df.insert(1, "condition", df["condition"].values)
        coord_df.insert(2, "spike_method", method)
        coord_df.to_csv(output_dir / f"{prefix}pca_coordinates.csv", index=False)
        cali_logger.debug(f"Exported PCA coordinates: {output_dir}")

        # 3. Loadings and scree CSV
        variance = pca.explained_variance_ratio_ * 100.0
        cumulative = np.cumsum(variance)
        rows: list[dict[str, object]] = []
        for i in range(n_components):
            row: dict[str, object] = {
                "component": f"PC{i + 1}",
                "explained_variance_pct": float(variance[i]),
                "cumulative_variance_pct": float(cumulative[i]),
            }
            for j, feat in enumerate(used_features):
                row[feat] = float(pca.components_[i, j])
            rows.append(row)

        loadings_df = pd.DataFrame(rows)
        loadings_df.to_csv(
            output_dir / f"{prefix}pca_loadings_and_scree.csv", index=False
        )
        cali_logger.debug(f"Exported PCA loadings and scree: {output_dir}")

        write_spike_metadata(
            output_dir / f"{prefix}pca.metadata.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "method": method,
                "input_spike_units": "a.u." if method == "oasis" else "spikes/frame",
                "used_features": used_features,
                "spike_metric_fields": df.attrs.get("spike_metric_fields", {}),
                "calcium_activity_field": "calcium_active",
                "spike_activity_field": "spike_active",
                "missing_values": (
                    "column median; all-missing and constant features excluded"
                ),
                "scaling": "z-score per feature before PCA",
            },
        )
