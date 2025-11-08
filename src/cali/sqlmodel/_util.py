"""Utility functions for cali.sqlmodel database operations.

This module provides helper functions for database operations including:
- Creating database tables
- Loading experiments from database
- Checking analysis settings consistency
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.engine import Engine

    from ._models import Experiment


def create_db_and_tables(engine: Engine) -> None:
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


def load_experiment_from_database(database_path: str | Path) -> Experiment | None:
    """Load the experiment from the given database path.

    Parameters
    ----------
    database_path : str | Path
        Path to the SQLite database file

    Returns
    -------
    Experiment | None
        The loaded experiment, or None if loading failed

    Example
    -------
    >>> from cali.sqlmodel import load_experiment_from_database
    >>> exp = load_experiment_from_database("path/to/cali.db")
    >>> if exp:
    ...     print(f"Loaded experiment: {exp.name}")
    """
    from sqlmodel import Session, create_engine, select

    from ._models import Experiment

    try:
        engine = create_engine(f"sqlite:///{database_path}")
        session = Session(engine)
        result = session.exec(select(Experiment))
        return result.first()
    except Exception as e:
        print(f"Error loading experiment: {e}")
        return None


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
