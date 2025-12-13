"""Calcium burst activity plotting function - continuation of _plot_inferred_spike_burst_activity.py"""

# This file contains the calcium burst plotting function that should be appended
# to _plot_inferred_spike_burst_activity.py

# -----------------------------------------------------------------------------#
# Calcium burst plotting (using deconvolved DF/F)
# -----------------------------------------------------------------------------#
def _plot_calcium_burst_activity(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot burst detection and network state analysis for calcium activity (pyqtgraph).

    This analyzes population-level calcium activity (deconvolved DF/F) to detect
    synchronized burst events and displays burst statistics.
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    plot.clear()

    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)
    vb.enableAutoRange(x=True, y=True)

    # Hide shared legend if you use one elsewhere
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    # --- Try to get pre-computed burst data from FOVAnalysis ---
    fov_analysis = _get_fov_analysis_for_run(engine, fov_name, run_id)

    if fov_analysis is not None and fov_analysis.calcium_burst_starts is not None:
        # Use stored burst data - much faster!
        burst_starts = fov_analysis.calcium_burst_starts
        burst_ends = fov_analysis.calcium_burst_ends or []
        population_activity_list = fov_analysis.calcium_population_activity
        smoothed_activity_list = fov_analysis.calcium_population_activity_smoothed

        if population_activity_list and smoothed_activity_list:
            population_activity = np.array(population_activity_list)
            smoothed_activity = np.array(smoothed_activity_list)

            # Get time axis from ROI data
            _, _, time_axis = _get_population_calcium_data(
                engine, fov_name, rois, run_id
            )

            if time_axis.size == 0 or len(time_axis) != len(population_activity):
                cali_logger.warning("Time axis length mismatch with stored data")
                # Fall back to frame indices
                time_axis = np.arange(len(population_activity)) / 10.0

            # Convert stored frame indices to (start, end) tuples
            bursts = list(zip(burst_starts, burst_ends))

            # Get threshold from AnalysisSettings for display
            calcium_burst_params = _get_calcium_burst_parameters(
                engine, fov_name, rois, run_id
            )
            the_value = (calcium_burst_params[0] / 100.0) if calcium_burst_params else 0.5

            cali_logger.info(
                f"Using pre-computed calcium burst data: {len(bursts)} bursts"
            )
        else:
            fov_analysis = None  # Fall back to computation

    # --- Fall back to computing if no stored data available ---
    if fov_analysis is None or fov_analysis.calcium_burst_starts is None:
        cali_logger.info("Computing calcium burst detection (no stored data available)")

        # --- 1) Get burst parameters from AnalysisSettings ---
        calcium_burst_params = _get_calcium_burst_parameters(
            engine, fov_name, rois, run_id
        )
        if calcium_burst_params is None:
            cali_logger.warning("Calcium burst parameters not found in ROI data.")
            plot.setTitle(
                "Calcium Population Burst Activity\\n(No burst parameters found)"
            )
            plot.setLabel("bottom", "Time (s)")
            plot.setLabel("left", "Population Activity (Normalized)")
            return

        burst_threshold, min_burst_duration_ms, smoothing_sigma_sec = (
            calcium_burst_params
        )

        # --- 2) Get population calcium data ---
        calcium_traces, _roi_names, time_axis = _get_population_calcium_data(
            engine, fov_name, rois, run_id
        )

        if calcium_traces is None or calcium_traces.shape[0] < 2:
            cali_logger.warning(
                "Not enough active ROIs with calcium traces to plot population activity."
            )
            plot.setTitle(
                "Calcium Population Burst Activity\\n(Not enough calcium data)"
            )
            plot.setLabel("bottom", "Time (s)")
            plot.setLabel("left", "Population Activity (Normalized)")
            return

        # --- 3) Compute frame rate from time axis ---
        num_frames = len(time_axis)
        if num_frames > 1:
            total_time_sec = float(time_axis[-1] - time_axis[0])
            frame_rate = (
                (num_frames - 1) / total_time_sec if total_time_sec > 0 else 10.0
            )
        else:
            frame_rate = 10.0  # fallback

        # Convert parameters from time units to frame units
        min_burst_duration_frames = max(
            1, int((min_burst_duration_ms / 1000.0) * frame_rate)
        )
        smoothing_sigma_frames = smoothing_sigma_sec * frame_rate

        # --- 4) Normalize each trace to [0, 1] range (same as _detect_calcium_population_bursts) ---
        calcium_traces_normalized = np.zeros_like(calcium_traces)
        for i in range(calcium_traces.shape[0]):
            trace = calcium_traces[i, :]
            trace_min = trace.min()
            trace_max = trace.max()
            if trace_max > trace_min:
                calcium_traces_normalized[i, :] = (trace - trace_min) / (
                    trace_max - trace_min
                )
            else:
                calcium_traces_normalized[i, :] = 0.0

        # --- 5) Population activity (mean over ROIs) ---
        population_activity = np.mean(calcium_traces_normalized, axis=0)

        # --- 6) Smooth population activity for burst detection ---
        if smoothing_sigma_frames > 0:
            smoothed_activity = gaussian_filter1d(
                population_activity, sigma=smoothing_sigma_frames, mode="nearest"
            )
        else:
            smoothed_activity = population_activity

        # --- 7) Detect bursts (burst_threshold is in %) ---
        the_value = burst_threshold / 100.0
        bursts = _detect_population_bursts(
            smoothed_activity, the_value, min_burst_duration_frames
        )

    # --- 8) Draw traces + threshold + burst regions ---
    _draw_population_activity_pg(
        plot,
        time_axis=time_axis,
        raw_activity=population_activity,
        smoothed_activity=smoothed_activity,
        bursts=bursts,
        threshold_value=the_value,
        legend=widget.legend if hasattr(widget, "legend") else None,
    )

    # --- 9) Stats text in title ---
    stats_text = _burst_statistics_text(bursts, time_axis)
    title = (
        "Calcium Population Activity and Burst Detection (Deconvolved ΔF/F0)\\n"
        f"{stats_text}"
    )
    plot.setTitle(title)

    plot.setLabel("bottom", "Time (s)")
    plot.setLabel("left", "Population Activity (Normalized [0,1])")

    # Auto-range once everything is added
    vb.enableAutoRange(x=True, y=True)


# -----------------------------------------------------------------------------#
# Helper functions for calcium burst plotting
# -----------------------------------------------------------------------------#
def _get_calcium_burst_parameters(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> tuple[float, float, float] | None:
    """Get calcium burst detection parameters from AnalysisSettings.

    Returns (calcium_burst_threshold, calcium_burst_min_duration_ms,
             calcium_burst_gaussian_sigma) if found.
    All returned in original units from database (threshold in %, duration in ms,
    sigma in seconds).
    """
    from sqlmodel import Session, select

    from cali.sqlmodel._model import AnalysisSettings

    with Session(engine) as session:
        if run_id is not None:
            from cali.sqlmodel._model import CaliResult

            stmt = select(CaliResult).where(CaliResult.id == run_id)
            result = session.exec(stmt).first()
            if result and result.analysis_settings:
                settings = result.analysis_settings
                return (
                    settings.calcium_burst_threshold,
                    settings.calcium_burst_min_duration,
                    settings.calcium_burst_gaussian_sigma,
                )

        # Otherwise get most recent settings
        stmt = select(AnalysisSettings).order_by(AnalysisSettings.created_at.desc())
        settings_obj = session.exec(stmt).first()
        if settings_obj:
            return (
                settings_obj.calcium_burst_threshold,
                settings_obj.calcium_burst_min_duration,
                settings_obj.calcium_burst_gaussian_sigma,
            )

    cali_logger.warning("No valid analysis settings found for calcium burst parameters.")
    return None


def _get_population_calcium_data(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[str], np.ndarray]:
    """Extract population calcium data (deconvolved DF/F) from database.

    Returns
    -------
    (calcium_traces_array, roi_names, time_axis)
    """
    from sqlalchemy.orm import selectinload
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV

    with Session(engine) as session:
        stmt = (
            select(FOV)
            .where(col(FOV.name) == fov_name)
            .options(selectinload(FOV.rois).selectinload(ROI.traces_history))
        )
        fov = session.exec(stmt).first()
        if not fov or not fov.rois:
            return None, [], np.array([])

        roi_results = [r for r in fov.rois if rois is None or r.label in rois]
        if not roi_results:
            return None, [], np.array([])

    calcium_traces: list[np.ndarray] = []
    roi_names: list[str] = []
    rois_rec_time: list[float] = []

    for roi in roi_results:
        traces_obj = _get_traces_for_run(roi, run_id)
        if traces_obj is None:
            continue

        dec_dff = traces_obj.dec_dff
        if dec_dff is None or len(dec_dff) == 0:
            continue

        calcium_traces.append(np.array(dec_dff, dtype=float))
        roi_names.append(str(roi.label))

        # Get recording time if available
        if traces_obj.time_array is not None and len(traces_obj.time_array) > 0:
            rois_rec_time.append(float(traces_obj.time_array[-1]))

    if len(calcium_traces) < 2:
        return None, roi_names, np.array([])

    # Pad / truncate to common length
    lengths = np.array([len(t) for t in calcium_traces], dtype=int)
    max_length = int(lengths.max())

    calcium_traces_array = np.zeros((len(calcium_traces), max_length), dtype=float)
    for i, trace in enumerate(calcium_traces):
        current_length = len(trace)
        if current_length >= max_length:
            calcium_traces_array[i, :] = trace[:max_length]
        else:
            calcium_traces_array[i, :current_length] = trace

    # Time axis from recording time if available, else frames @ 10Hz
    if rois_rec_time:
        max_time = max(rois_rec_time)
        time_axis = np.linspace(0, max_time, max_length)
    else:
        time_axis = np.arange(max_length) / 10.0

    return calcium_traces_array, roi_names, time_axis
