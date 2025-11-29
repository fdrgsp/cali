from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from skimage.measure import find_contours

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget


DEFAULT_COLOR = "gray"
STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"
P1 = 5
P2 = 100


def _plot_evoked_experiment_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    stimulated_area: bool = False,
    with_rois: bool = False,
    with_peaks: bool = False,
) -> None:
    """Plot evoked experiment data.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    rois : list[int] | None
        List of ROI indices to include, None for all
    run_id : int | None
        The run ID to filter by, None for latest
    stimulated_area : bool
        Whether to show stimulated area
    with_rois : bool
        Whether to show ROIs
    with_peaks : bool
        Whether to show peaks
    """
    # Delegate to the appropriate function based on parameters
    if with_rois or stimulated_area:
        _visualize_stimulated_area(
            widget=widget,
            engine=engine,
            fov_name=fov_name,
            rois=rois,
            run_id=run_id,
            with_rois=with_rois,
            stimulated_area=stimulated_area,
        )
    else:
        # Default to showing stimulated vs non-stimulated traces
        _plot_stimulated_vs_non_stimulated_roi_traces(
            widget=widget,
            engine=engine,
            fov_name=fov_name,
            rois=rois,
            run_id=run_id,
            with_peaks=with_peaks,
        )


def _plot_stim_or_not_stim_peaks_amplitude(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    stimulated: bool = False,
) -> None:
    """Visualize stimulated peak amplitudes per ROI per stimulation parameters."""
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    if run_id is None:
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Query database for ROI data
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
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.stimulated) == stimulated)
        )

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        results = session.exec(stmt).all()

    if not results:
        ax.text(
            0.5,
            0.5,
            f"No {'stimulated' if stimulated else 'non-stimulated'} ROI data found.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Extract peak amplitudes for each ROI and plot
    roi_labels = []
    artists = []

    for roi_model, _, data_analysis in results:
        if data_analysis and data_analysis.peaks_amplitudes_dec_dff:
            roi_labels.append(roi_model.label_value)
            amps = data_analysis.peaks_amplitudes_dec_dff

            # Calculate mean and SEM
            mean_amp = float(np.mean(amps))

            # Only calculate SEM if we have more than one data point
            if len(amps) > 1:
                std_amp = np.std(amps, ddof=1)  # sample std
                sem_amp = std_amp / np.sqrt(len(amps))
            else:
                sem_amp = 0  # No error bars for single point

            # Plot mean ± SEM as error bars
            errorbar = ax.errorbar(
                [roi_model.label_value],
                [mean_amp],
                yerr=[sem_amp],
                fmt="o",
                capsize=5,
                color=STIMULATED_COLOR if stimulated else NON_STIMULATED_COLOR,
                label=f"ROI {roi_model.label_value}",
                zorder=2,
            )
            artists.append(errorbar)

            # Plot individual peak amplitudes in background
            ax.scatter(
                [roi_model.label_value] * len(amps),
                amps,
                alpha=0.5,
                s=30,
                color="lightgray",
                zorder=1,
            )

    if not roi_labels:
        ax.text(
            0.5,
            0.5,
            "No peak amplitude data available.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Set labels and title
    ax.set_xlabel("ROI")
    ax.set_ylabel("Peak Amplitude (dec ΔF/F)")
    title = "Stimulated" if stimulated else "Non-Stimulated"
    ax.set_title(f"{title} ROI Mean Peak Amplitudes ± SEM")
    ax.set_xticks(roi_labels)
    ax.set_xticklabels([str(lbl) for lbl in roi_labels])

    # Add hover functionality
    _add_hover_to_stimulated_amp_plot(widget, artists)

    widget.figure.tight_layout()
    widget.canvas.draw()


def extract_leading_number(key: str) -> float:
    """Extract leading number from key (before '_'), stripping units if present."""
    if match := re.match(r"(\d+(?:\.\d+)?)", key.split("_")[0]):
        return float(match[1])
    raise ValueError(f"Could not extract a valid number from key: {key}")


def _add_hover_to_stimulated_amp_plot(
    widget: _SingleWellGraphWidget,
    artists: list,
) -> None:
    """Add hover tooltips to amplitude error bar plot."""
    cursor = mplcursors.cursor(artists, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the label from the artist to extract ROI
        label = sel.artist.get_label()
        if label and "ROI" in label:
            roi = label.split(" ")[1]
            # Get the y-value (mean amplitude)
            _, y = sel.target

            sel.annotation.set(
                text=f"ROI {roi}\nMean Amp: {y:.3f}", fontsize=8, color="black"
            )
            sel.annotation.arrow_patch.set_alpha(0.5)

            widget.roiSelected.emit(str(roi))
        else:
            sel.annotation.set_visible(False)


def _visualize_stimulated_area(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    with_rois: bool = False,
    stimulated_area: bool = False,
) -> None:
    """Visualize Stimulated area with ROI mask overlay.

    This function shows either:
    - Just the stimulation mask (stimulated_area=True, with_rois=False)
    - ROIs colored by stimulation status (with_rois=True)
    - Both ROIs and stimulation area (both True)

    All masks are reconstructed from database (no external TIF files needed).
    """
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI, AnalysisSettings, Mask
    from cali.util import coordinates_to_mask

    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    if run_id is None:
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Query ROI data and masks from database
    from cali.sqlmodel._model import CaliResult

    stim_mask = None
    roi_data: list[tuple[ROI, Mask | None]] = []
    image_shape = None

    with Session(engine) as session:
        # Clear any cached data to ensure we see newly committed analysis results
        session.expire_all()
        # Get stimulation mask if available
        result = session.get(CaliResult, run_id)
        if result and result.analysis_settings_id:
            analysis_settings = session.get(
                AnalysisSettings, result.analysis_settings_id
            )
            if analysis_settings and analysis_settings.stimulation_mask_id:
                mask_obj = session.get(Mask, analysis_settings.stimulation_mask_id)
                if (
                    mask_obj
                    and mask_obj.coords_y is not None
                    and mask_obj.coords_x is not None
                    and mask_obj.height is not None
                    and mask_obj.width is not None
                ):
                    # Reconstruct stimulation mask from coordinates
                    coords = (mask_obj.coords_y, mask_obj.coords_x)
                    shape = (mask_obj.height, mask_obj.width)
                    stim_mask = coordinates_to_mask(coords, shape)
                    image_shape = shape

        # Query ROI data with their masks
        stmt = (
            select(ROI, Mask)
            .join(FOV, ROI.fov_id == FOV.id)
            .outerjoin(Mask, ROI.roi_mask_id == Mask.id)
            .where(col(FOV.name) == fov_name)
        )

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Order by label_value
        stmt = stmt.order_by(col(ROI.label_value))

        results = session.exec(stmt).all()
        roi_data = [(roi, mask) for roi, mask in results]

        # Get image shape from first ROI mask if we don't have it from stim mask
        if image_shape is None and roi_data:
            for _roi, mask in roi_data:
                if mask and mask.height and mask.width:
                    image_shape = (mask.height, mask.width)
                    break

    if not roi_data:
        ax.text(
            0.5,
            0.5,
            "No ROI data found for visualization.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Group ROIs by stimulation status
    stimulated_roi_labels = []
    non_stimulated_roi_labels = []

    for roi, _ in roi_data:
        if roi.stimulated:
            stimulated_roi_labels.append(roi.label_value)
        else:
            non_stimulated_roi_labels.append(roi.label_value)

    # If we need to show ROIs, reconstruct label image from ROI masks
    if with_rois and image_shape:
        # Reconstruct label image from ROI masks in database
        labels = np.zeros(image_shape, dtype=np.int32)

        for roi, mask in roi_data:
            if (
                mask
                and mask.coords_y is not None
                and mask.coords_x is not None
                and mask.height is not None
                and mask.width is not None
            ):
                # Reconstruct ROI mask from coordinates
                roi_mask = coordinates_to_mask(
                    (mask.coords_y, mask.coords_x), (mask.height, mask.width)
                )
                # Set pixels to ROI label value
                labels[roi_mask] = roi.label_value

        # Create color mapping
        color_mapping = {0: "black"}  # background
        for roi, _ in roi_data:
            if roi.stimulated:
                color_mapping[roi.label_value] = STIMULATED_COLOR
            else:
                color_mapping[roi.label_value] = NON_STIMULATED_COLOR

        # Create colormap
        unique_labels = np.unique(labels)
        colors = [color_mapping.get(lbl, DEFAULT_COLOR) for lbl in unique_labels]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(
            boundaries=np.append(unique_labels, unique_labels[-1] + 1),
            ncolors=len(colors),
        )

        ax.imshow(labels, cmap=cmap, norm=norm)

        # Show stimulation area contours if requested
        if stimulated_area and stim_mask is not None:
            stim_area_contours = find_contours(stim_mask.astype(float), level=0.5)
            for contour in stim_area_contours:
                ax.plot(contour[:, 1], contour[:, 0], color="yellow", linewidth=2)

        # Add legend
        legend_patches = [
            Patch(color=STIMULATED_COLOR, label="Stimulated ROIs"),
            Patch(color=NON_STIMULATED_COLOR, label="Non-Stimulated ROIs"),
        ]
        if stimulated_area and stim_mask is not None:
            legend_patches.append(Patch(color="yellow", label="Stimulation Area"))

        ax.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(legend_patches),
            frameon=True,
            fontsize="small",
            edgecolor="black",
        )
    elif stimulated_area and stim_mask is not None:
        # Just show stimulation mask
        ax.imshow(stim_mask, cmap="gray", clim=(0, 1))
    else:
        # Fallback to text statistics
        _display_roi_statistics(ax, stimulated_roi_labels, non_stimulated_roi_labels)

    ax.axis("off")
    widget.figure.tight_layout()
    widget.canvas.draw()


def _display_roi_statistics(
    ax: Axes,
    stimulated_rois: list[int],
    non_stimulated_rois: list[int],
) -> None:
    """Display ROI statistics as text when mask visualization is not available."""
    text_lines = [
        "Stimulated Area Visualization",
        "",
        f"Total ROIs: {len(stimulated_rois) + len(non_stimulated_rois)}",
        f"Stimulated ROIs: {len(stimulated_rois)}",
        f"Non-Stimulated ROIs: {len(non_stimulated_rois)}",
        "",
    ]

    ax.text(
        0.5,
        0.5,
        "\n".join(text_lines),
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
        family="monospace",
    )


def _plot_stimulated_vs_non_stimulated_roi_traces(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    with_peaks: bool = False,
) -> None:
    """Plot dec ΔF/F traces with global percentile normalization (5th-100th)."""
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    if run_id is None:
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Query database for ROI data
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
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
        )

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        results = session.exec(stmt).all()

    if not results:
        ax.text(
            0.5,
            0.5,
            "No ROI data found for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Separate stimulated and non-stimulated ROIs
    stimulated_data = []
    non_stimulated_data = []
    rois_rec_time: list[float] = []

    for roi_model, trace_obj, data_analysis in results:
        if trace_obj and trace_obj.dec_dff:
            if roi_model.stimulated:
                stimulated_data.append((roi_model, trace_obj, data_analysis))
            else:
                non_stimulated_data.append((roi_model, trace_obj, data_analysis))

            if data_analysis and data_analysis.total_recording_time_sec is not None:
                rois_rec_time.append(data_analysis.total_recording_time_sec)

    # Compute global percentile normalization
    all_values = []
    for _, trace_obj, _ in results:
        if trace_obj and trace_obj.dec_dff:
            all_values.extend(trace_obj.dec_dff)

    if all_values:
        percentiles = np.percentile(all_values, [P1, P2])
        p1, p2 = float(percentiles[0]), float(percentiles[1])
    else:
        p1, p2 = 0.0, 1.0

    # Plot stimulated ROIs with traces and optional peaks with amplitudes
    count = 0
    last_trace = None
    for roi_model, trace_obj, data_analysis in stimulated_data:
        if trace_obj.dec_dff:
            trace = _normalize_trace_percentile(trace_obj.dec_dff, p1, p2)
            offset = count * 1.1
            ax.plot(
                np.array(trace) + offset,
                label=f"ROI {roi_model.label_value}",
                color=STIMULATED_COLOR,
            )

            if with_peaks and data_analysis:
                # Show peak locations
                if data_analysis.peaks_dec_dff:
                    peaks_indices = [int(p) for p in data_analysis.peaks_dec_dff]
                    ax.plot(
                        peaks_indices,
                        np.array(trace)[peaks_indices] + offset,
                        "x",
                        color="black",
                        markersize=8,
                    )

            last_trace = trace_obj.dec_dff
            count += 1

    # Plot non-stimulated ROIs
    for roi_model, trace_obj, data_analysis in non_stimulated_data:
        if trace_obj.dec_dff:
            trace = _normalize_trace_percentile(trace_obj.dec_dff, p1, p2)
            offset = count * 1.1
            ax.plot(
                np.array(trace) + offset,
                label=f"ROI {roi_model.label_value}",
                color=NON_STIMULATED_COLOR,
            )

            if with_peaks and data_analysis:
                # Show peak locations
                if data_analysis.peaks_dec_dff:
                    peaks_indices = [int(p) for p in data_analysis.peaks_dec_dff]
                    ax.plot(
                        peaks_indices,
                        np.array(trace)[peaks_indices] + offset,
                        "x",
                        color="black",
                        markersize=8,
                    )

            last_trace = trace_obj.dec_dff
            count += 1

    # Set labels and title
    ax.set_ylabel("Normalized dec ΔF/F")
    ax.set_title("Stimulated vs Non-Stimulated ROIs (Normalized)")
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Add legend for stimulated/non-stimulated
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(color=STIMULATED_COLOR, label="Stimulated ROIs"),
        Patch(color=NON_STIMULATED_COLOR, label="Non-Stimulated ROIs"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize="small")

    # Update time axis
    _update_time_axis(ax, rois_rec_time, last_trace)

    # Add hover functionality
    _add_hover_functionality_stim_vs_non_stim(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _normalize_trace_percentile(
    trace: list[float], p1: float, p2: float
) -> list[float]:
    """Normalize a trace using the global 5th and 100th percentiles."""
    tr = np.array(trace)
    denom = p2 - p1
    if denom == 0:
        return cast("list[float]", np.zeros_like(tr).tolist())
    normalized = (tr - p1) / denom
    normalized = np.clip(normalized, 0, 1)  # ensure values in [0, 1]
    return cast("list[float]", normalized.tolist())


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    if trace is None or sum(rois_rec_time) <= 0:
        ax.set_xlabel("Frames")
        return
    # get the average total recording time in seconds
    avg_rec_time = int(np.mean(rois_rec_time))
    # get total number of frames from the trace
    total_frames = len(trace) if trace is not None else 1
    # compute tick positions
    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")


def _add_hover_functionality_stim_vs_non_stim(
    ax: Axes, widget: _SingleWellGraphWidget
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the label of the artist
        label = sel.artist.get_label()

        # Only show hover for ROI traces, not for peaks or other elements
        if label and "ROI" in label and not label.startswith("_"):
            sel.annotation.set(text=label, fontsize=8, color="black")
            roi = cast("str", label.split(" ")[1])
            if roi.isdigit():
                widget.roiSelected.emit(roi)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)


def _plot_stimulated_vs_non_stimulated_spike_raster(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot raster plot of thresholded spikes: green=stim, magenta=non-stim."""
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    if run_id is None:
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Query database for ROI data
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
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.active) == True)  # noqa: E712
        )

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        results = session.exec(stmt).all()

    if not results:
        ax.text(
            0.5,
            0.5,
            "No active ROI data found for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Separate stimulated and non-stimulated ROIs
    stimulated_rois = []
    non_stimulated_rois = []
    active_roi_labels = []
    rois_rec_time: list[float] = []

    for roi_model, trace_obj, data_analysis in results:
        if trace_obj and trace_obj.inferred_spikes and data_analysis:
            active_roi_labels.append(roi_model.label_value)
            if roi_model.stimulated:
                stimulated_rois.append((roi_model, trace_obj, data_analysis))
            else:
                non_stimulated_rois.append((roi_model, trace_obj, data_analysis))

            if data_analysis.total_recording_time_sec is not None:
                rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not stimulated_rois and not non_stimulated_rois:
        ax.text(
            0.5,
            0.5,
            "No spike data available.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Prepare event data for raster plot using eventplot
    event_data: list[list[float]] = []
    colors: list[str] = []
    last_trace = None

    # Collect stimulated ROI spike events
    for _roi_model, trace_obj, data_analysis in stimulated_rois:
        if trace_obj.inferred_spikes and data_analysis.inferred_spikes_threshold:
            spikes = np.array(trace_obj.inferred_spikes)
            threshold = data_analysis.inferred_spikes_threshold

            # Threshold spikes
            thresholded = np.where(spikes > threshold, spikes, 0)

            # Get spike indices
            spike_indices = np.where(thresholded > 0)[0]
            event_data.append(spike_indices.tolist())
            colors.append(STIMULATED_COLOR)

            last_trace = trace_obj.inferred_spikes

    # Collect non-stimulated ROI spike events
    for _roi_model, trace_obj, data_analysis in non_stimulated_rois:
        if trace_obj.inferred_spikes and data_analysis.inferred_spikes_threshold:
            spikes = np.array(trace_obj.inferred_spikes)
            threshold = data_analysis.inferred_spikes_threshold

            # Threshold spikes
            thresholded = np.where(spikes > threshold, spikes, 0)

            # Get spike indices
            spike_indices = np.where(thresholded > 0)[0]
            event_data.append(spike_indices.tolist())
            colors.append(NON_STIMULATED_COLOR)

            last_trace = trace_obj.inferred_spikes

    # Plot raster using eventplot
    ax.eventplot(event_data, colors=colors, linewidth=3)

    # Set labels and title
    ax.set_ylabel("ROI")
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title("Stimulated vs Non-Stimulated Spike Raster Plot")

    # Add legend for stimulated/non-stimulated
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(color=STIMULATED_COLOR, label="Stimulated ROIs"),
        Patch(color=NON_STIMULATED_COLOR, label="Non-Stimulated ROIs"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize="small")

    # Update time axis
    _update_time_axis_spike_traces(ax, rois_rec_time, last_trace)

    # Add hover functionality
    _add_hover_functionality_spike_traces(ax, widget, active_roi_labels)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality_spike_traces(
    ax: Axes, widget: _SingleWellGraphWidget, active_rois: list[int]
) -> None:
    """Add hover functionality using mplcursors for spike traces."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the label of the artist
        label = sel.artist.get_label()

        # Only show hover for valid ROI elements
        if label and "ROI" in label and not label.startswith("_"):
            sel.annotation.set(text=label, fontsize=8, color="black")
            roi_parts = label.split(" ")
            if len(roi_parts) > 1 and roi_parts[1].isdigit():
                widget.roiSelected.emit(roi_parts[1])
        else:
            # For raster plots, map the position to an ROI
            if hasattr(sel, "target") and active_rois:
                with contextlib.suppress(ValueError, AttributeError, IndexError):
                    y_pos = int(sel.target[1])  # Get y-coordinate (ROI index)
                    if 0 <= y_pos < len(active_rois):
                        roi_id = active_rois[y_pos]
                        hover_text = f"ROI {roi_id}"
                        sel.annotation.set(text=hover_text, fontsize=8, color="black")
                        widget.roiSelected.emit(str(roi_id))
                        return
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)


def _plot_stimulated_vs_non_stimulated_spike_traces(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot actual inferred spike traces separated by stimulation status.

    Shows continuous spike traces (not raster) with vertical offset,
    colored by stimulation status: green=stimulated, magenta=non-stimulated.
    """
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    if run_id is None:
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Query database for ROI data
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
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.active) == True)  # noqa: E712
        )

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        results = session.exec(stmt).all()

    if not results:
        ax.text(
            0.5,
            0.5,
            "No active ROI data found for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Separate stimulated and non-stimulated ROIs
    stimulated_data = []
    non_stimulated_data = []
    rois_rec_time: list[float] = []

    for roi_model, trace_obj, data_analysis in results:
        if trace_obj and trace_obj.inferred_spikes:
            if roi_model.stimulated:
                stimulated_data.append((roi_model, trace_obj, data_analysis))
            else:
                non_stimulated_data.append((roi_model, trace_obj, data_analysis))

            if data_analysis and data_analysis.total_recording_time_sec is not None:
                rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not stimulated_data and not non_stimulated_data:
        ax.text(
            0.5,
            0.5,
            "No spike data available.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Plot spike traces with vertical offset
    count = 0
    last_trace = None

    # Plot stimulated ROIs first
    for roi_model, trace_obj, data_analysis in stimulated_data:
        if trace_obj.inferred_spikes:
            spikes = np.array(trace_obj.inferred_spikes)

            # Apply threshold if available
            if data_analysis and data_analysis.inferred_spikes_threshold:
                threshold = data_analysis.inferred_spikes_threshold
                spikes = np.where(spikes > threshold, spikes, 0)

            offset = count * 1.1
            ax.plot(
                spikes + offset,
                label=f"ROI {roi_model.label_value}",
                color=STIMULATED_COLOR,
                linewidth=1.5,
            )

            last_trace = trace_obj.inferred_spikes
            count += 1

    # Plot non-stimulated ROIs
    for roi_model, trace_obj, data_analysis in non_stimulated_data:
        if trace_obj.inferred_spikes:
            spikes = np.array(trace_obj.inferred_spikes)

            # Apply threshold if available
            if data_analysis and data_analysis.inferred_spikes_threshold:
                threshold = data_analysis.inferred_spikes_threshold
                spikes = np.where(spikes > threshold, spikes, 0)

            offset = count * 1.1
            ax.plot(
                spikes + offset,
                label=f"ROI {roi_model.label_value}",
                color=NON_STIMULATED_COLOR,
                linewidth=1.5,
            )

            last_trace = trace_obj.inferred_spikes
            count += 1

    # Set labels and title
    ax.set_ylabel("Inferred Spikes (Thresholded)")
    ax.set_title(
        "Stimulated vs Non-Stimulated Spike Traces\n(Thresholded Inferred Spikes)"
    )
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Add legend for stimulated/non-stimulated
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(color=STIMULATED_COLOR, label="Stimulated ROIs"),
        Patch(color=NON_STIMULATED_COLOR, label="Non-Stimulated ROIs"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize="small")

    # Update time axis
    _update_time_axis_spike_traces(ax, rois_rec_time, last_trace)

    # Add hover functionality
    active_roi_labels = [
        roi_model.label_value
        for roi_model, _, _ in stimulated_data + non_stimulated_data
    ]
    _add_hover_functionality_spike_traces(ax, widget, active_roi_labels)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _update_time_axis_spike_traces(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    """Update the x-axis to show time in seconds if recording time is available."""
    if trace is None or sum(rois_rec_time) <= 0:
        ax.set_xlabel("Frames")
        return
    # get the average total recording time in seconds
    avg_rec_time = int(np.mean(rois_rec_time))
    # get total number of frames from the trace
    total_frames = len(trace) if trace is not None else 1
    # compute tick positions
    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")
