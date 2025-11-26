"""Utility script to visualize SQLModel experiment hierarchies.

Usage:
    python visualize_experiment.py "Experiment Name"
    python visualize_experiment.py --list  # List all experiments
"""

from rich.console import Console
from rich.tree import Tree
from sqlalchemy.engine import Engine
from sqlmodel import Session, select
from typing_extensions import Literal

from ._model import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DetectionSettings,
    Experiment,
    Plate,
    Well,
)

MaxTreeLevel = Literal["experiment", "plate", "well", "fov", "roi"]


def print_cali_results(
    engine: Engine,
    experiment_name: str | None = None,
    show_settings: bool = True,
    max_experiment_level: MaxTreeLevel = "roi",
) -> None:
    """Print all analysis results, optionally filtered by experiment.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine connected to the database
    experiment_name : str | None
        Optional experiment name to filter results. If None, shows all results
        from all experiments (default: None)
    show_settings : bool
        Whether to show detailed settings for each result (default: False)
    max_experiment_level : MaxTreeLevel
        Maximum depth for experiment tree in each result (default: "roi")
    """
    with Session(engine) as session:
        # Get analysis results - either filtered by experiment or all results
        if experiment_name is not None:
            # Get specific experiment
            experiment = session.exec(
                select(Experiment).where(Experiment.name == experiment_name)
            ).first()

            if experiment is None:
                print(f"❌ Experiment '{experiment_name}' not found")
                return

            # Get results for this experiment
            results = session.exec(
                select(CaliResult).where(CaliResult.experiment == experiment.id)
            ).all()

            title = f"Analysis Results for '{experiment_name}'"
        else:
            # Get all results from all experiments
            results = session.exec(select(CaliResult)).all()
            title = "All Analysis Results"

        if not results:
            if experiment_name:
                print(
                    f"📊 No analysis results found for experiment '{experiment_name}'"
                )
            else:
                print("📊 No analysis results found in database")
            return

        # Create main tree with title as root
        console = Console()
        plural = "s" if len(results) != 1 else ""
        main_tree = Tree(
            f"[bold cyan]{title}[/bold cyan] ({len(results)} result{plural})",
            guide_style="cyan",
        )

        # Add each result as a child of the main tree
        for result in results:
            # Get experiment for this result with eager loading of relationships
            from sqlalchemy.orm import selectinload

            plate_chain = (
                selectinload(Experiment.plate)
                .selectinload(Plate.wells)
                .selectinload(Well.fovs)
                .selectinload(FOV.rois)
            )

            result_experiment = session.exec(
                select(Experiment)
                .where(Experiment.id == result.experiment)
                .options(
                    plate_chain.selectinload(ROI.traces_history),
                    plate_chain.selectinload(ROI.data_analysis_history),
                    plate_chain.selectinload(ROI.roi_mask),
                )
            ).first()

            # Create result subtree
            positions = result.positions_analyzed or []
            positions_count = len(positions)
            pos_plural = "s" if positions_count != 1 else ""

            result_tree = main_tree.add(
                f"📊 [bold cyan]Analysis Result #{result.id}[/bold cyan]"
            )
            result_tree.add(f"📅 Created: [dim]{result.created_at}[/dim]")
            # Positions analyzed first
            if positions:
                positions_node = result_tree.add(
                    f"📍 [bold magenta]Positions Analyzed[/bold magenta] "
                    f"({positions_count} position{pos_plural})"
                )
                # Group consecutive positions for cleaner display
                ranges = []
                start = positions[0]
                end = positions[0]

                for pos in positions[1:]:
                    if pos == end + 1:
                        end = pos
                    else:
                        ranges.append((start, end))
                        start = end = pos
                ranges.append((start, end))

                for start, end in ranges:
                    if start == end:
                        positions_node.add(f"Position {start}")
                    else:
                        positions_node.add(f"Positions {start}-{end}")

            # Detection settings (if available)
            if result.detection_settings:
                detection_settings = session.exec(
                    select(DetectionSettings).where(
                        DetectionSettings.id == result.detection_settings
                    )
                ).first()
                if detection_settings:
                    _add_detection_settings_to_tree(
                        result_tree, detection_settings, show_details=show_settings
                    )

            # Analysis settings
            settings = session.exec(
                select(AnalysisSettings).where(
                    AnalysisSettings.id == result.analysis_settings
                )
            ).first()

            if settings:
                _add_analysis_settings_to_tree(
                    result_tree, settings, show_details=show_settings
                )

            # Experiment info with full tree
            if result_experiment:
                _add_experiment_tree_to_node(
                    result_tree,
                    result_experiment,
                    max_level=max_experiment_level,
                    detection_settings_id=result.detection_settings,
                    analysis_result_id=result.id,
                )

        console.print(main_tree)


def _add_detection_settings_to_tree(
    parent_node: Tree, settings: DetectionSettings, show_details: bool = True
) -> None:
    """Add detection settings information to a tree node.

    Parameters
    ----------
    parent_node : Tree
        Parent node to add settings information to
    settings : DetectionSettings
        Settings object to display
    show_details : bool
        Whether to show detailed parameter values (default: True)
    """
    settings_node = parent_node.add(
        f"⚙️ [bold green]Detection Settings (ID: {settings.id})[/bold green]"
    )
    settings_node.add(f"📅 Created: [dim]{settings.created_at}[/dim]")
    settings_node.add(f"🔬 Method: [cyan]{settings.method}[/cyan]")

    if show_details and settings.method == "cellpose":
        # Cellpose-specific settings
        cellpose_node = settings_node.add("🟡 [green]Cellpose Parameters[/green]")
        cellpose_node.add(f"Model: {settings.model_type}")
        diameter_str = f"{settings.diameter} px" if settings.diameter else "auto-detect"
        cellpose_node.add(f"Diameter: {diameter_str}")
        cellpose_node.add(f"Cell prob threshold: {settings.cellprob_threshold}")
        cellpose_node.add(f"Flow threshold: {settings.flow_threshold}")
        cellpose_node.add(f"Min size: {settings.min_size} px")
        cellpose_node.add(f"Normalize: {settings.normalize}")
        cellpose_node.add(f"Batch size: {settings.batch_size}")


def _add_analysis_settings_to_tree(
    parent_node: Tree, settings: AnalysisSettings, show_details: bool = True
) -> None:
    """Add analysis settings information to a tree node.

    Parameters
    ----------
    parent_node : Tree
        Parent node to add settings information to
    settings : AnalysisSettings
        Settings object to display
    show_details : bool
        Whether to show detailed parameter values (default: True)
    """
    settings_node = parent_node.add(
        f"⚙️ [bold yellow]Analysis Settings (ID: {settings.id})[/bold yellow]"
    )
    settings_node.add(f"📅 Created: [dim]{settings.created_at}[/dim]")

    # Show experiment type
    exp_type_emoji = "⚡" if settings.experiment_type == "evoked" else "✨"
    exp_type_color = "green" if settings.experiment_type == "evoked" else "magenta"
    settings_node.add(
        f"{exp_type_emoji} Experiment type: [{exp_type_color}]"
        f"{settings.experiment_type}[/{exp_type_color}]"
    )

    if show_details:
        # Threads
        settings_node.add(f"🧵 Threads: {settings.threads}")

        # Neuropil correction
        neuropil_node = settings_node.add("🔵 [green]Neuropil Correction[/green]")
        neuropil_node.add(f"Inner radius: {settings.neuropil_inner_radius} px")
        neuropil_node.add(f"Min pixels: {settings.neuropil_min_pixels}")
        neuropil_node.add(f"Correction factor: {settings.neuropil_correction_factor}")

        # Signal processing
        processing_node = settings_node.add("📈 [green]Signal Processing[/green]")
        processing_node.add(f"ΔF/F window: {settings.dff_window}")
        processing_node.add(f"Decay constant: {settings.decay_constant}")

        # Peak detection
        peaks_node = settings_node.add("🔍 [green]Peak Detection[/green]")
        peaks_node.add(
            f"Height: {settings.peaks_height_value} ({settings.peaks_height_mode})"
        )
        peaks_node.add(f"Distance: {settings.peaks_distance} frames")
        peaks_node.add(f"Prominence multiplier: {settings.peaks_prominence_multiplier}")

        # Spike detection
        spike_node = settings_node.add("⚡ [green]Spike Detection[/green]")
        spike_node.add(
            f"Threshold: {settings.spike_threshold_value} "
            f"({settings.spike_threshold_mode})"
        )

        # Burst analysis
        burst_node = settings_node.add("💥 [green]Burst Analysis[/green]")
        burst_node.add(f"Threshold: {settings.burst_threshold}%")
        burst_node.add(f"Min duration: {settings.burst_min_duration}s")
        burst_node.add(f"Gaussian sigma: {settings.burst_gaussian_sigma}s")

        # Synchrony
        sync_node = settings_node.add("🔗 [green]Synchrony Analysis[/green]")
        sync_node.add(f"Calcium jitter window: {settings.calcium_sync_jitter_window}")
        sync_node.add(f"Network threshold: {settings.calcium_network_threshold}%")
        sync_node.add(f"Spike cross-corr lag: {settings.spikes_sync_cross_corr_lag}")

        # Stimulation parameters (if evoked)
        if (
            settings.led_power_equation
            or settings.led_pulse_powers
            or settings.led_pulse_on_frames
            or settings.led_pulse_duration
        ):
            stim_node = settings_node.add("⚡ [green]Stimulation[/green]")
            if settings.stimulation_mask_id is not None:
                stim_node.add("🎭 Stimulation mask: True")
            if settings.led_power_equation:
                stim_node.add(f"Power equation: {settings.led_power_equation}")
            if settings.led_pulse_duration:
                stim_node.add(f"Pulse duration: {settings.led_pulse_duration}ms")
            if settings.led_pulse_powers:
                stim_node.add(f"Pulse powers: {settings.led_pulse_powers}")
            if settings.led_pulse_on_frames:
                stim_node.add(f"Pulse on frames: {settings.led_pulse_on_frames}")
            if settings.stimulation_mask_path:
                stim_node.add(f"Mask path: {settings.stimulation_mask_path}")


def _add_experiment_tree_to_node(
    parent_node: Tree,
    experiment: Experiment,
    max_level: MaxTreeLevel = "roi",
    detection_settings_id: int | None = None,
    analysis_result_id: int | None = None,
) -> None:
    """Add experiment hierarchy (plate/well/fov/roi) to a tree node.

    Parameters
    ----------
    parent_node : Tree
        Parent node to add experiment tree to
    experiment : Experiment
        Experiment object to display
    max_level : MaxTreeLevel
        Maximum depth level to display
    detection_settings_id : int | None
        If provided, only show ROIs matching this detection_settings_id
        (useful when showing ROIs for a specific CaliResult)
    analysis_result_id : int | None
        If provided, only check for neuropil masks in traces from this analysis result
    """
    exp_node = parent_node.add(f"🧪 [bold]Experiment (ID: {experiment.id})[/bold]")
    exp_node.add(f"Name: {experiment.name}")
    if experiment.description:
        exp_node.add(f"Description: [dim]{experiment.description}[/dim]")

    if max_level == "experiment":
        return

    # Add plate
    plate_type = experiment.plate.plate_type or "unknown"
    plate_node = exp_node.add(
        f"📋 [green]{experiment.plate.name}[/green] ({plate_type})"
    )

    if max_level == "plate":
        return

    # Add wells
    for well in experiment.plate.wells:
        well_conditions = []
        if well.condition_1:
            well_conditions.append(f"{well.condition_1.name}")
        if well.condition_2:
            well_conditions.append(f"{well.condition_2.name}")

        if well_conditions:
            conditions_text = ", ".join(well_conditions)
            condition_str = f" - 🧪 [green]Conditions: {conditions_text}[/green]"
        else:
            condition_str = ""

        well_node = plate_node.add(f"🧫 [yellow]{well.name}[/yellow]{condition_str}")

        if max_level == "well":
            continue

        # Add FOVs
        for fov in well.fovs:
            fov_node = well_node.add(
                f"📷 [cyan]{fov.name} "
                f"(fov: {fov.fov_number} - pos: {fov.position_index})[/cyan]"
            )

            if max_level == "fov":
                continue

            # Add ROIs (filter by detection_settings_id if provided)
            # Note: detection_settings_id is now on ROI, not FOV
            if detection_settings_id is not None:
                # Only show ROIs matching the requested detection settings
                rois_to_show = [
                    roi
                    for roi in fov.rois
                    if roi.detection_settings_id == detection_settings_id
                ]
            else:
                rois_to_show = fov.rois

            for roi in rois_to_show:
                roi_info = f"ROI {roi.label_value}"

                # Add detection settings info if not already filtered by it
                if roi.detection_settings_id and detection_settings_id is None:
                    # Only show detection ID if we're not already filtering by it
                    roi_info += f" [dim](Detection #{roi.detection_settings_id})[/dim]"

                if roi.active is not None:
                    status = (
                        "🔋 [green]active[/green]"
                        if roi.active
                        else "🪫 [red]inactive[/red]"
                    )
                    roi_info += f" - {status}"

                if roi.stimulated is not None:
                    if roi.stimulated:
                        roi_info += " - ⚡️ [green]stimulated[/green]"
                    else:
                        roi_info += " - ✨ [magenta]spontaneous[/magenta]"

                roi_node = fov_node.add(f"🔬 [magenta]{roi_info}[/magenta]")

                # Add related data if present
                if roi.roi_mask:
                    roi_node.add("🎭 [dim]ROI mask available[/dim]")
                # Check if traces have neuropil masks (filter by analysis_result)
                if roi.traces_history:
                    traces_to_check = (
                        [
                            t
                            for t in roi.traces_history
                            if t.analysis_result_id == analysis_result_id
                        ]
                        if analysis_result_id is not None
                        else roi.traces_history
                    )
                    if any(trace.neuropil_mask for trace in traces_to_check):
                        roi_node.add("👺 [dim]Neuropil mask available[/dim]")
                if roi.traces_history:
                    roi_node.add("📊 [dim]Trace data available[/dim]")
                if roi.data_analysis_history:
                    roi_node.add("📈 [dim]Data analysis available[/dim]")
