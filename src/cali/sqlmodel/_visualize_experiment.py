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

from ._models import Experiment

MaxTreeLevel = Literal["experiment", "plate", "well", "fov", "roi"]


def print_experiment_tree_from_engine(
    experiment_name: str, engine: Engine, max_level: MaxTreeLevel = "roi"
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
    """
    session = Session(engine)
    statement = select(Experiment).where(Experiment.name == experiment_name)
    experiment = session.exec(statement).first()

    if experiment is None:
        print(f"❌ Experiment '{experiment_name}' not found")
        return

    print_experiment_tree(experiment, max_level=max_level)
    session.close()


def print_experiment_tree(
    experiment: Experiment, max_level: MaxTreeLevel = "roi"
) -> None:
    """Print the full hierarchical model tree for an experiment.

    Parameters
    ----------
    experiment : Experiment
        The experiment to display
    max_level : MaxTreeLevel
        Maximum depth level to display. Options:
        "experiment": Just experiment info
        "plate": Show experiment and plate
        "well": Show up to wells
        "fov": Show up to FOVs
        "roi": Show complete tree including ROIs (default)
    """
    console = Console()
    tree = Tree(f"🧪 [bold cyan]{experiment.name}[/bold cyan]", guide_style="cyan")

    if experiment.description:
        tree.add(f"[dim]{experiment.description}[/dim]")

    if max_level == "experiment":
        console.print(tree)
        return

    # Analysis Settings
    if experiment.analysis_settings:
        tree.add("⚙️ [dim]Analysis Settings available[/dim]")

    # Add plate
    plate_type = experiment.plate.plate_type or "unknown"
    plate_node = tree.add(
        f"📋 [bold green]{experiment.plate.name}[/bold green] ({plate_type})"
    )

    if max_level == "plate":
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

        if max_level == "well":
            continue  # Skip FOVs and ROIs

        # Add FOVs
        for fov in well.fovs:
            fov_node = well_node.add(
                f"📷 [cyan]{fov.name} "
                f"(fov: {fov.fov_number} - pos: {fov.position_index})[/cyan]"
            )

            if max_level == "fov":
                continue  # Skip ROIs

            # Add ROIs
            for roi in fov.rois:
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
                if roi.traces:
                    roi_node.add("📊 [dim]Trace data available[/dim]")
                if roi.data_analysis:
                    roi_node.add("📈 [dim]Data analysis available[/dim]")
                if roi.roi_mask:
                    roi_node.add("🎭 [dim]ROI mask available[/dim]")
                if roi.neuropil_mask:
                    roi_node.add("🔵 [dim]Neuropil mask available[/dim]")

    console.print(tree)
