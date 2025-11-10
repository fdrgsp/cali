"""Utility functions for cali.sqlmodel database operations.

This module provides helper functions for database operations including:
- Creating database tables
- Loading experiments from database
- Checking analysis settings consistency
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlmodel import Session, create_engine

from ._models import Experiment

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


def create_database_and_tables(engine: Engine) -> None:
    """Create all database tables.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Database engine

    Example
    -------
    >>> from sqlmodel import create_engine
    >>> from cali.sqlmodel import create_db_and_tables
    >>> engine = create_engine("sqlite:///calcium_analysis.db")
    >>> create_db_and_tables(engine)
    """
    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(engine)


def check_analysis_settings_consistency(experiment: Experiment) -> dict[str, Any]:
    """Check if all ROIs in an experiment were analyzed with the same settings.

    This helper function checks whether all ROIs across all FOVs in an experiment
    were analyzed using the same AnalysisSettings. This is useful for detecting
    partial re-analyses with different parameters.

    Parameters
    ----------
    experiment : Experiment
        The experiment to check

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'consistent': bool - True if all ROIs use same settings
        - 'settings_count': int - Number of different AnalysisSettings used
        - 'settings_ids': set[int] - Set of unique AnalysisSettings IDs
        - 'fovs_by_settings': dict[int, list[str]] - FOV names grouped by settings
        - 'warning': str | None - Warning message if inconsistent

    Example
    -------
    >>> from cali.sqlmodel import check_analysis_settings_consistency
    >>> result = check_analysis_settings_consistency(experiment)
    >>> if not result["consistent"]:
    ...     print(result["warning"])
    ...     print(f"FOVs with different settings: {result['fovs_by_settings']}")
    """
    from collections import defaultdict

    settings_ids: set[int] = set()
    fovs_by_settings: dict[int | None, list[str]] = defaultdict(list)

    # Iterate through all FOVs and their ROIs
    for well in experiment.plate.wells:
        for fov in well.fovs:
            # Get the settings ID from the first ROI (all ROIs in FOV should have same)
            if fov.rois:
                settings_id = fov.rois[0].analysis_settings_id
                if settings_id:
                    settings_ids.add(settings_id)
                fovs_by_settings[settings_id].append(fov.name)

    settings_count = len(settings_ids)
    consistent = settings_count <= 1

    warning = None
    if not consistent:
        warning = (
            f"⚠️  Inconsistent analysis settings detected! "
            f"{settings_count} different settings used across FOVs. "
            f"This may affect comparability of results."
        )

    return {
        "consistent": consistent,
        "settings_count": settings_count,
        "settings_ids": settings_ids,
        "fovs_by_settings": dict(fovs_by_settings),
        "warning": warning,
    }


def save_experiment_to_database(
    experiment: Experiment,
    db_path: Path | str,
    overwrite: bool = False,
) -> None:
    """Save an experiment object tree to a SQLite database.

    Parameters
    ----------
    experiment : Experiment
        Experiment object (e.g., from load_analysis_from_json)
    db_path : Path | str
        Path to SQLite database file
    overwrite : bool, optional
        Whether to overwrite existing database file, by default False

    Example
    -------
    >>> from pathlib import Path
    >>> exp = load_analysis_from_json(Path("tests/test_data/..."))
    >>> save_experiment_to_database(exp, "analysis.db")
    """
    db_path = Path(db_path)
    experiment.database_path = str(db_path)

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and db_path.exists():
        db_path.unlink()

    engine = create_engine(f"sqlite:///{db_path}")
    create_database_and_tables(engine)

    with Session(engine, expire_on_commit=False) as session:
        session.merge(experiment)
        session.commit()


def load_experiment_from_database(
    db_path: Path | str,
    experiment_name: str | None = None,
) -> Experiment | None:
    """Load an experiment from SQLite database with all relationships.

    This function properly handles SQLAlchemy session management and eagerly
    loads all relationships so the returned Experiment object can be used
    outside the session context.

    Parameters
    ----------
    db_path : Path | str
        Path to SQLite database file
    experiment_name : str | None, optional
        Name of specific experiment to load. If None, loads the first experiment.

    Returns
    -------
    Experiment | None
        Loaded experiment with all relationships, or None if not found

    Example
    -------
    >>> from pathlib import Path
    >>> exp = load_experiment_from_database("analysis.db", "my_experiment")
    >>> if exp:
    ...     print(f"Loaded {len(exp.plate.wells)} wells")
    """
    from sqlmodel import select

    # Convert to string for consistency
    db_path_str = str(db_path)
    engine = create_engine(f"sqlite:///{db_path_str}")

    # Use context manager to ensure session is properly closed
    with Session(engine, expire_on_commit=False) as session:
        # Query for experiment
        if experiment_name:
            statement = select(Experiment).where(Experiment.name == experiment_name)
        else:
            statement = select(Experiment)

        experiment = session.exec(statement).first()

        if not experiment:
            return None

        experiment.database_path = db_path_str

        # Force load ALL relationships deeply while session is still open
        # This prevents DetachedInstanceError when accessed later
        if experiment.plate:
            _ = len(experiment.plate.wells)  # Force load wells
            for well in experiment.plate.wells:
                _ = len(well.conditions)  # Force load conditions
                _ = len(well.fovs)  # Force load fovs
                for fov in well.fovs:
                    _ = len(fov.rois)  # Force load rois
                    for roi in fov.rois:
                        # Force load all ROI relationships
                        _ = roi.traces
                        _ = roi.data_analysis
                        _ = roi.roi_mask
                        _ = roi.neuropil_mask
                        _ = roi.analysis_settings

        if experiment.analysis_settings:
            _ = experiment.analysis_settings.stimulation_mask

        # Make the instance independent of the session
        session.expunge(experiment)

    # Session automatically closed here
    return experiment  # type: ignore
