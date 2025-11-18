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
    AnalysisResult,
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    Plate,
    Well,
)

MaxTreeLevel = Literal["experiment", "plate", "well", "fov", "roi"]


def add_detection_settings_to_tree(
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


def add_analysis_settings_to_tree(
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


def add_experiment_tree_to_node(
    parent_node: Tree,
    experiment: Experiment,
    max_level: MaxTreeLevel = "roi",
    detection_settings_id: int | None = None,
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
        (useful when showing ROIs for a specific AnalysisResult)
    """
    exp_node = parent_node.add(f"🧪 [bold]Experiment (ID: {experiment.id})[/bold]")
    exp_node.add(f"Name: {experiment.name}")
    exp_node.add(f"Type: [magenta]{experiment.experiment_type}[/magenta]")
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
            rois_to_show = fov.rois
            if detection_settings_id is not None:
                rois_to_show = [
                    roi
                    for roi in fov.rois
                    if roi.detection_settings_id == detection_settings_id
                ]

            for roi in rois_to_show:
                roi_info = f"ROI {roi.label_value}"
                if roi.active is not None:
                    status = (
                        "🔋 [green]active[/green]"
                        if roi.active
                        else "🪫 [red]inactive[/red]"
                    )
                    roi_info += f" - {status}"
                if roi.stimulated:
                    roi_info += " - ⚡️ [green]stimulated[/green]"
                else:
                    roi_info += " - ✨ [magenta]spontaneous[/magenta]"

                roi_node = fov_node.add(f"🔬 [magenta]{roi_info}[/magenta]")

                # Add related data if present
                if roi.roi_mask:
                    roi_node.add("🎭 [dim]ROI mask available[/dim]")
                if roi.traces_history:
                    roi_node.add("📊 [dim]Trace data available[/dim]")
                if roi.data_analysis_history:
                    roi_node.add("📈 [dim]Data analysis available[/dim]")


def print_analysis_result(
    analysis_result: AnalysisResult,
    session: Session,
    show_settings: bool = True,
    max_experiment_level: MaxTreeLevel = "roi",
) -> None:
    """Print detailed information about an AnalysisResult.

    Parameters
    ----------
    analysis_result : AnalysisResult
        The analysis result to display
    session : Session
        Database session for querying related data
    show_settings : bool
        Whether to show detailed analysis parameter values (default: True)
    max_experiment_level : MaxTreeLevel
        Maximum depth for experiment tree display (default: "roi")
    """
    console = Console()

    # Positions analyzed
    positions = analysis_result.positions_analyzed or []
    positions_count = len(positions)
    plural = "s" if positions_count != 1 else ""

    tree = Tree(
        f"📊 [bold cyan]Analysis Result #{analysis_result.id}[/bold cyan]",
        guide_style="cyan",
    )

    # Positions analyzed first
    if positions:
        positions_node = tree.add(
            f"📍 [bold magenta]Positions Analyzed[/bold magenta] "
            f"({positions_count} position{plural})"
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
    if analysis_result.detection_settings:
        detection_settings = session.exec(
            select(DetectionSettings).where(
                DetectionSettings.id == analysis_result.detection_settings
            )
        ).first()
        if detection_settings:
            add_detection_settings_to_tree(
                tree, detection_settings, show_details=show_settings
            )

    # Analysis settings
    settings = session.exec(
        select(AnalysisSettings).where(
            AnalysisSettings.id == analysis_result.analysis_settings
        )
    ).first()

    if settings:
        add_analysis_settings_to_tree(tree, settings, show_details=show_settings)

    # Experiment info with full tree
    experiment = session.exec(
        select(Experiment).where(Experiment.id == analysis_result.experiment)
    ).first()
    if experiment:
        add_experiment_tree_to_node(
            tree,
            experiment,
            max_level=max_experiment_level,
            detection_settings_id=analysis_result.detection_settings,
        )

    console.print(tree)


def print_all_analysis_results(
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
    session = Session(engine)

    # Get analysis results - either filtered by experiment or all results
    if experiment_name is not None:
        # Get specific experiment
        experiment = session.exec(
            select(Experiment).where(Experiment.name == experiment_name)
        ).first()

        if experiment is None:
            print(f"❌ Experiment '{experiment_name}' not found")
            session.close()
            return

        # Get results for this experiment
        results = session.exec(
            select(AnalysisResult).where(AnalysisResult.experiment == experiment.id)
        ).all()

        title = f"Analysis Results for '{experiment_name}'"
    else:
        # Get all results from all experiments
        results = session.exec(select(AnalysisResult)).all()
        title = "All Analysis Results"

    if not results:
        if experiment_name:
            print(f"📊 No analysis results found for experiment '{experiment_name}'")
        else:
            print("📊 No analysis results found in database")
        session.close()
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
                plate_chain.selectinload(ROI.neuropil_mask),
            )
        ).first()

        # Create result subtree
        positions = result.positions_analyzed or []
        positions_count = len(positions)
        pos_plural = "s" if positions_count != 1 else ""

        result_tree = main_tree.add(
            f"📊 [bold cyan]Analysis Result #{result.id}[/bold cyan]"
        )

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
                add_detection_settings_to_tree(
                    result_tree, detection_settings, show_details=show_settings
                )

        # Analysis settings
        settings = session.exec(
            select(AnalysisSettings).where(
                AnalysisSettings.id == result.analysis_settings
            )
        ).first()

        if settings:
            add_analysis_settings_to_tree(
                result_tree, settings, show_details=show_settings
            )

        # Experiment info with full tree
        if result_experiment:
            add_experiment_tree_to_node(
                result_tree,
                result_experiment,
                max_level=max_experiment_level,
                detection_settings_id=result.detection_settings,
            )

    console.print(main_tree)
    session.close()


def print_experiment_tree_from_engine(
    experiment_name: str,
    engine: Engine,
    max_level: MaxTreeLevel = "roi",
    show_analysis_results: bool = True,
    show_settings: bool = False,
) -> None:
    """Print the model tree for a specific experiment by name.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment to display
    engine : Engine
        SQLAlchemy engine connected to the database
    max_level : MaxTreeLevel
        Maximum depth level to display. Options:
        "experiment": Just experiment info
        "plate": Show experiment and plate
        "well": Show up to wells
        "fov": Show up to FOVs
        "roi": Show complete tree including ROIs (default)
    show_analysis_results : bool
        Whether to show analysis results section (default: True)
    show_settings : bool
        Whether to show detailed analysis settings for each result (default: False)
    """
    session = Session(engine)
    statement = select(Experiment).where(Experiment.name == experiment_name)
    experiment = session.exec(statement).first()

    if experiment is None:
        print(f"❌ Experiment '{experiment_name}' not found")
        return

    print_experiment_tree(
        experiment,
        max_experiment_level=max_level,
        session=session,
        show_analysis_results=show_analysis_results,
        show_settings=show_settings,
    )
    session.close()


def print_experiment_tree(
    experiment: Experiment,
    max_experiment_level: MaxTreeLevel = "roi",
    session: Session | None = None,
    show_analysis_results: bool = True,
    show_settings: bool = True,
) -> None:
    """Print the full hierarchical model tree for an experiment.

    Parameters
    ----------
    experiment : Experiment
        The experiment to display
    max_experiment_level : MaxTreeLevel
        Maximum depth level to display. Options:
        "experiment": Just experiment info
        "plate": Show experiment and plate
        "well": Show up to wells
        "fov": Show up to FOVs
        "roi": Show complete tree including ROIs (default)
    session : Session | None
        Optional database session for querying analysis results
    show_analysis_results : bool
        Whether to show analysis results section (default: True)
    show_settings : bool
        Whether to show detailed analysis settings for each result (default: False)
    """
    console = Console()
    tree = Tree(
        f"🧪 [bold cyan]{experiment.name} (ID: {experiment.id})[/bold cyan]",
        guide_style="cyan",
    )

    tree.add(
        f"Experiment Type: [bold magenta]{experiment.experiment_type}[/bold magenta]"
    )

    if experiment.description:
        tree.add(f"[dim]{experiment.description}[/dim]")

    if max_experiment_level == "experiment":
        console.print(tree)
        return

    # Add plate
    plate_type = experiment.plate.plate_type or "unknown"
    plate_node = tree.add(
        f"📋 [bold green]{experiment.plate.name}[/bold green] ({plate_type})"
    )

    if max_experiment_level == "plate":
        console.print(tree)
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

        if max_experiment_level == "well":
            continue  # Skip FOVs and ROIs

        # Add FOVs
        for fov in well.fovs:
            fov_node = well_node.add(
                f"📷 [cyan]{fov.name} "
                f"(fov: {fov.fov_number} - pos: {fov.position_index})[/cyan]"
            )

            if max_experiment_level == "fov":
                continue  # Skip ROIs

            # Add ROIs
            for roi in fov.rois:
                roi_info = f"ROI {roi.label_value} "
                roi_info += f"(Detection Settings ID: {roi.detection_settings_id})"
                if roi.active is not None:
                    status = (
                        "🔋 [green]active[/green]"
                        if roi.active
                        else "🪫 [red]inactive[/red]"
                    )
                    roi_info += f" - {status}"
                if roi.stimulated:
                    roi_info += " - ⚡️ [green]stimulated[/green]"
                else:
                    roi_info += " - ✨ [magenta]spontaneous[/magenta]"

                roi_node = fov_node.add(f"🔬 [magenta]{roi_info}[/magenta]")

                # Add related data if present
                if roi.traces_history:
                    roi_node.add("📊 [dim]Trace data available[/dim]")
                if roi.data_analysis_history:
                    roi_node.add("📈 [dim]Data analysis available[/dim]")
                if roi.roi_mask:
                    roi_node.add("🎭 [dim]ROI mask available[/dim]")
                if roi.neuropil_mask:
                    roi_node.add("🔵 [dim]Neuropil mask available[/dim]")

    # Show analysis results if session is provided
    if show_analysis_results and session and experiment.id is not None:
        analysis_results = session.exec(
            select(AnalysisResult).where(AnalysisResult.experiment == experiment.id)
        ).all()

        if analysis_results:
            analysis_node = tree.add("📊 [bold yellow]Analysis Results[/bold yellow]")
            for result in analysis_results:
                # Get settings and experiment for this result
                settings = session.exec(
                    select(AnalysisSettings).where(
                        AnalysisSettings.id == result.analysis_settings
                    )
                ).first()

                # Get detection settings if available
                detection_settings = None
                if result.detection_settings:
                    detection_settings = session.exec(
                        select(DetectionSettings).where(
                            DetectionSettings.id == result.detection_settings
                        )
                    ).first()

                # Create result node
                positions_count = len(result.positions_analyzed or [])
                plural = "s" if positions_count != 1 else ""

                result_node = analysis_node.add(
                    f"📊 [bold]Result #{result.id}[/bold] - "
                    f"{positions_count} position{plural}"
                )

                # Detection settings (if available)
                if detection_settings:
                    add_detection_settings_to_tree(
                        result_node, detection_settings, show_details=show_settings
                    )

                # Analysis settings info
                if settings:
                    settings_node = result_node.add(
                        f"⚙️ [bold yellow]Analysis Settings "
                        f"(ID: {settings.id})[/bold yellow]"
                    )
                    settings_node.add(f"📅 Created: [dim]{settings.created_at}[/dim]")
                    settings_node.add(f"🧵 Threads: {settings.threads}")

                    # Show detailed settings if requested
                    if show_settings:
                        # Neuropil correction
                        neuropil_node = settings_node.add(
                            "🔵 [green]Neuropil Correction[/green]"
                        )
                        neuropil_node.add(
                            f"Inner radius: {settings.neuropil_inner_radius} px"
                        )
                        neuropil_node.add(f"Min pixels: {settings.neuropil_min_pixels}")
                        neuropil_node.add(
                            f"Correction factor: {settings.neuropil_correction_factor}"
                        )

                        # Signal processing
                        processing_node = settings_node.add(
                            "📈 [green]Signal Processing[/green]"
                        )
                        processing_node.add(f"ΔF/F window: {settings.dff_window}")
                        processing_node.add(
                            f"Decay constant: {settings.decay_constant}"
                        )

                        # Peak detection
                        peaks_node = settings_node.add(
                            "🔍 [green]Peak Detection[/green]"
                        )
                        peaks_node.add(
                            f"Height: {settings.peaks_height_value} "
                            f"({settings.peaks_height_mode})"
                        )
                        peaks_node.add(f"Distance: {settings.peaks_distance} frames")
                        peaks_node.add(
                            f"Prominence multiplier: "
                            f"{settings.peaks_prominence_multiplier}"
                        )

                        # Spike detection
                        spike_node = settings_node.add(
                            "⚡ [green]Spike Detection[/green]"
                        )
                        spike_node.add(
                            f"Threshold: {settings.spike_threshold_value} "
                            f"({settings.spike_threshold_mode})"
                        )

                        # Burst analysis
                        burst_node = settings_node.add(
                            "💥 [green]Burst Analysis[/green]"
                        )
                        burst_node.add(f"Threshold: {settings.burst_threshold}%")
                        burst_node.add(f"Min duration: {settings.burst_min_duration}s")
                        burst_node.add(
                            f"Gaussian sigma: {settings.burst_gaussian_sigma}s"
                        )

                        # Synchrony
                        sync_node = settings_node.add(
                            "🔗 [green]Synchrony Analysis[/green]"
                        )
                        sync_node.add(
                            f"Calcium jitter window: "
                            f"{settings.calcium_sync_jitter_window}"
                        )
                        sync_node.add(
                            f"Network threshold: {settings.calcium_network_threshold}%"
                        )
                        sync_node.add(
                            f"Spike cross-corr lag: "
                            f"{settings.spikes_sync_cross_corr_lag}"
                        )

                        # Stimulation (if evoked)
                        if (
                            settings.led_power_equation
                            or settings.led_pulse_powers
                            or settings.led_pulse_on_frames
                            or settings.led_pulse_duration
                        ):
                            stim_node = settings_node.add(
                                "⚡ [green]Stimulation[/green]"
                            )
                            if settings.led_power_equation:
                                stim_node.add(
                                    f"Power equation: {settings.led_power_equation}"
                                )
                            if settings.led_pulse_duration:
                                stim_node.add(
                                    f"Pulse duration: {settings.led_pulse_duration}ms"
                                )
                            if settings.led_pulse_powers:
                                stim_node.add(
                                    f"Pulse powers: {settings.led_pulse_powers}"
                                )
                            if settings.led_pulse_on_frames:
                                stim_node.add(
                                    f"Pulse on frames: {settings.led_pulse_on_frames}"
                                )
                            if settings.stimulation_mask_path:
                                stim_node.add(
                                    f"Mask path: {settings.stimulation_mask_path}"
                                )

                # Positions analyzed
                positions = result.positions_analyzed or []
                if positions:
                    # Group consecutive positions
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

                    positions_list = []
                    for start, end in ranges:
                        if start == end:
                            positions_list.append(f"{start}")
                        else:
                            positions_list.append(f"{start}-{end}")

                    result_node.add(f"📍 Positions: {', '.join(positions_list)}")

    console.print(tree)


def print_database_tree(
    engine: Engine,
    experiment_name: str | None = None,
    max_experiment_level: MaxTreeLevel = "roi",
    show_analysis_results: bool = True,
    show_settings: bool = True,
) -> None:
    """Print complete database structure including experiments and optionally analysis results.

    This function shows the full database hierarchy regardless of whether
    analysis has been performed. It's useful for visualizing the output
    of DetectionRunner (FOV/ROI/Mask structure) before running analysis.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine connected to the database
    experiment_name : str | None
        Optional experiment name to filter. If None, shows all experiments
    max_experiment_level : MaxTreeLevel
        Maximum depth level to display (default: "roi")
    show_analysis_results : bool
        Whether to show analysis results section (default: True)
    show_settings : bool
        Whether to show detailed analysis settings (default: False)
    """
    session = Session(engine)
    console = Console()

    # Get experiments
    if experiment_name is not None:
        experiments = [
            session.exec(
                select(Experiment).where(Experiment.name == experiment_name)
            ).first()
        ]
        if experiments[0] is None:
            print(f"❌ Experiment '{experiment_name}' not found")
            session.close()
            return
        title = f"📊 Database: {experiment_name}"
    else:
        experiments = list(session.exec(select(Experiment)).all())
        title = "📊 Complete Database Structure"

    if not experiments:
        print("📊 No experiments found in database")
        session.close()
        return

    # Create main tree
    experiment_plural = "s" if len(experiments) != 1 else ""
    main_tree = Tree(
        f"[bold cyan]{title}[/bold cyan] ({len(experiments)} experiment{experiment_plural})",
        guide_style="cyan",
    )

    # Add each experiment
    for exp in experiments:
        exp_tree = main_tree.add(f"🧪 [bold cyan]{exp.name} (ID: {exp.id})[/bold cyan]")
        exp_tree.add(f"Type: [magenta]{exp.experiment_type}[/magenta]")
        if exp.description:
            exp_tree.add(f"Description: [dim]{exp.description}[/dim]")
        exp_tree.add(f"Created: [dim]{exp.created_at}[/dim]")

        if max_experiment_level == "experiment":
            continue

        # Add plate
        if exp.plate:
            plate_type = exp.plate.plate_type or "unknown"
            plate_node = exp_tree.add(
                f"📋 [green]{exp.plate.name}[/green] ({plate_type})"
            )

            if max_experiment_level == "plate":
                continue

            # Add wells with statistics
            for well in exp.plate.wells:
                # Count FOVs and ROIs for this well
                fov_count = len(well.fovs)
                roi_count = sum(len(fov.rois) for fov in well.fovs)

                well_conditions = []
                if well.condition_1:
                    well_conditions.append(f"{well.condition_1.name}")
                if well.condition_2:
                    well_conditions.append(f"{well.condition_2.name}")

                if well_conditions:
                    conditions_text = ", ".join(well_conditions)
                    condition_str = f" - 🧪 {conditions_text}"
                else:
                    condition_str = ""

                well_label = (
                    f"🧫 [yellow]{well.name}[/yellow]{condition_str} "
                    f"[dim]({fov_count} FOVs, {roi_count} ROIs)[/dim]"
                )
                well_node = plate_node.add(well_label)

                if max_experiment_level == "well":
                    continue

                # Add FOVs
                for fov in well.fovs:
                    fov_label = (
                        f"📷 [cyan]{fov.name}[/cyan] "
                        f"[dim](pos: {fov.position_index}, {len(fov.rois)} ROIs)[/dim]"
                    )
                    fov_node = well_node.add(fov_label)

                    if max_experiment_level == "fov":
                        continue

                    # Add ROIs
                    for roi in fov.rois:
                        roi_info = f"ROI {roi.label_value}"

                        # Show status if analyzed
                        if roi.active is not None:
                            status = "🔋 active" if roi.active else "🪫 inactive"
                            roi_info += f" - {status}"

                        if roi.stimulated:
                            roi_info += " - ⚡️ stimulated"

                        roi_node = fov_node.add(f"🔬 [magenta]{roi_info}[/magenta]")

                        # Show what data is available
                        data_available = []
                        if roi.roi_mask:
                            data_available.append("🎭 ROI mask")
                        if roi.neuropil_mask:
                            data_available.append("🔵 Neuropil mask")
                        if roi.traces_history:
                            data_available.append("📊 Traces")
                        if roi.data_analysis_history:
                            data_available.append("📈 Analysis")

                        if data_available:
                            roi_node.add(f"[dim]{'  •  '.join(data_available)}[/dim]")

        # Show analysis results if requested
        if show_analysis_results:
            results = session.exec(
                select(AnalysisResult).where(AnalysisResult.experiment == exp.id)
            ).all()

            if results:
                results_node = exp_tree.add(
                    f"📈 [bold yellow]Analysis Results ({len(results)})[/bold yellow]"
                )

                for result in results:
                    positions = result.positions_analyzed or []
                    result_node = results_node.add(
                        f"Result #{result.id} - {len(positions)} positions"
                    )

                    # Show settings if requested
                    if show_settings:
                        # Detection settings
                        if result.detection_settings:
                            detection_settings = session.exec(
                                select(DetectionSettings).where(
                                    DetectionSettings.id == result.detection_settings
                                )
                            ).first()
                            if detection_settings:
                                add_detection_settings_to_tree(
                                    result_node, detection_settings, show_details=True
                                )

                        # Analysis settings
                        settings = session.exec(
                            select(AnalysisSettings).where(
                                AnalysisSettings.id == result.analysis_settings
                            )
                        ).first()
                        if settings:
                            add_analysis_settings_to_tree(
                                result_node, settings, show_details=True
                            )

    console.print(main_tree)
    session.close()
