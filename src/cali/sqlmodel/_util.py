"""Utility functions for cali.sqlmodel database operations.

This module provides helper functions for database operations including:
- Creating database tables
- Loading experiments from database
- Checking analysis settings consistency
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, create_engine, select

from cali._constants import DEFAULT_CALI_DB_NAME

from ._model import Experiment

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

from cali.logger import cali_logger


def migrate_analysis_settings(engine: Engine) -> None:
    """Add missing columns to analysis_settings table for existing databases.

    This is safe to call multiple times — it only adds columns that don't exist.
    """
    with engine.connect() as conn:
        existing_cols = {
            row[1] for row in conn.execute(text("PRAGMA table_info(analysis_settings)"))
        }
        if not existing_cols:
            return  # table doesn't exist yet
        if "enable_calcium" not in existing_cols:
            conn.execute(
                text(
                    "ALTER TABLE analysis_settings "
                    "ADD COLUMN enable_calcium BOOLEAN DEFAULT 1 NOT NULL"
                )
            )
        if "enable_spikes" not in existing_cols:
            conn.execute(
                text(
                    "ALTER TABLE analysis_settings "
                    "ADD COLUMN enable_spikes BOOLEAN DEFAULT 1 NOT NULL"
                )
            )
        conn.commit()


def create_database_and_tables(engine: Engine) -> None:
    """Create all database tables.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Database engine

    Example
    -------
    >>> from sqlmodel import create_engine
    >>> from cali.sqlmodel import create_database_and_tables
    >>> engine = create_engine("sqlite:///calcium_analysis.db")
    >>> create_database_and_tables(engine)
    """
    from sqlmodel import SQLModel

    # Import all models to register them with SQLModel metadata
    from ._model import (  # noqa: F401
        FOV,
        ROI,
        AnalysisSettings,
        CaliResult,
        Condition,
        DataAnalysis,
        DetectionSettings,
        ExtractionSettings,
        FOVAnalysis,
        Mask,
        Plate,
        Traces,
        Well,
        WellCondition,
    )

    SQLModel.metadata.create_all(engine)
    migrate_analysis_settings(engine)


def save_experiment_to_database(
    experiment: Experiment,
    output_path: Path | str,
    *,
    database_name: str = DEFAULT_CALI_DB_NAME,
    overwrite: bool = False,
    echo: bool = False,
) -> None:
    """Save an experiment object tree to a SQLite database.

    This function saves the experiment and returns nothing, following SQLModel
    best practices of not returning objects to discourage keeping large object
    trees in memory. Load the experiment fresh from the database when needed
    using load_experiment_from_database().

    Parameters
    ----------
    experiment : Experiment
        Experiment object
    output_path : Path | str
        Output directory to save the database file.
    database_name : str, optional
        Name of the database file (e.g., "cali.db"). Defaults to "results.cali".
    overwrite : bool, optional
        Whether to overwrite existing database file, by default False
    echo : bool, optional
        Whether to enable SQLAlchemy engine echo for debugging, by default False

    Example
    -------
    >>> from pathlib import Path
    >>> save_experiment_to_database(exp, overwrite=True)
    >>> # Later, load fresh from DB when needed:
    >>> db_path = Path(exp.output_path) / exp.database_name
    >>> exp = load_experiment_from_database(db_path)
    """
    # Determine database path
    db_name = database_name if database_name is not None else DEFAULT_CALI_DB_NAME
    assert db_name is not None  # Guaranteed by the check above
    if not db_name.endswith(".cali"):
        db_name += ".cali"
    db_path = Path(output_path) / db_name

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and db_path.exists():
        db_path.unlink()

    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=echo,
        connect_args={"timeout": 30.0, "check_same_thread": False},
        pool_pre_ping=True,
    )
    create_database_and_tables(engine)

    try:
        with Session(engine) as session:
            # Pre-resolve conditions BEFORE merge to avoid session.merge()
            # limitations with link_model many-to-many relationships.
            # Safely check if plate is loaded without triggering lazy load
            # on detached instance
            from sqlalchemy import inspect as sa_inspect
            from sqlalchemy import or_

            from cali.sqlmodel._model import Condition

            insp = sa_inspect(experiment)
            plate_loaded = "plate" in insp.dict and insp.dict["plate"] is not None

            # Store original well-to-conditions mapping BEFORE merge
            well_condition_map: dict[int, list[tuple[str, str]]] = {}
            if plate_loaded and experiment.plate is not None:
                # Collect all unique (name, condition_type) pairs from all wells
                conditions_needed: set[tuple[str, str]] = set()
                for idx, well in enumerate(experiment.plate.wells):
                    # Store this well's condition keys
                    well_keys = [(c.name, c.condition_type) for c in well.conditions]
                    well_condition_map[idx] = well_keys
                    conditions_needed.update(well_keys)

                # Batch fetch existing conditions in ONE query
                condition_lookup: dict[tuple[str, str], Condition] = {}
                if conditions_needed:
                    or_clauses = [
                        (Condition.name == name) & (Condition.condition_type == ctype)
                        for name, ctype in conditions_needed
                    ]
                    existing = session.exec(
                        select(Condition).where(or_(*or_clauses))
                    ).all()
                    condition_lookup = {(c.name, c.condition_type): c for c in existing}

            # Merge experiment into session first
            merged_exp = session.merge(experiment)

            # Now fix up conditions on the session-attached wells
            if plate_loaded and merged_exp.plate is not None:
                for idx, well in enumerate(merged_exp.plate.wells):
                    # Use original well's condition keys (before merge)
                    condition_keys = well_condition_map.get(idx, [])
                    if not condition_keys:
                        # No conditions for this well, skip it
                        continue

                    resolved_conditions: list[Condition] = []

                    for key in condition_keys:
                        existing_cond = condition_lookup.get(key)
                        if existing_cond:
                            # Use existing condition from DB
                            resolved_conditions.append(existing_cond)
                        else:
                            # Condition doesn't exist - query for it
                            name, ctype = key
                            stmt = select(Condition).where(
                                (Condition.name == name)
                                & (Condition.condition_type == ctype)
                            )
                            cond_in_session = session.exec(stmt).first()
                            if cond_in_session:
                                resolved_conditions.append(cond_in_session)
                            # else: condition not found, skip it

                    # Assign resolved conditions (replaces whatever merge() set)
                    if resolved_conditions:
                        well.conditions = resolved_conditions

            session.commit()
            # Refresh to get the ID assigned by the database
            session.refresh(merged_exp)
            # Update the original experiment object with the database ID
            experiment.id = merged_exp.id

        cali_logger.info(
            f"💾 Experiment analysis updated and saved to database at {db_path}."
        )

    except IntegrityError as e:
        cali_logger.error(
            f"❌ Failed to save experiment to database. "
            f"Integrity constraint violated: {e}"
        )
        raise
    finally:
        # Dispose engine to release database connections (Windows compatibility)
        engine.dispose(close=True)


def load_experiment_from_database(
    db_path: Path | str,
    experiment_name: str | None = None,
    echo: bool = False,
) -> Experiment | None:
    """Load an experiment from SQLite database with all relationships.

    This function loads a complete experiment snapshot for read-only analysis
    or display. The returned object is detached from the session (expunged) and
    can be used outside the session context.

    Parameters
    ----------
    db_path : Path | str
        Path to SQLite database file
    experiment_name : str | None, optional
        Name of specific experiment to load. If None, loads the first experiment.
    echo : bool, optional
        Whether to enable SQLAlchemy engine echo for debugging, by default False

    Returns
    -------
    Experiment | None
        Loaded experiment with all relationships, or None if not found.
        The object is detached (expunged) and can be used outside the session.

    Example
    -------
    >>> from pathlib import Path
    >>> # For read-only display/analysis:
    >>> exp = load_experiment_from_database("analysis.db", "my_experiment")
    >>> if exp:
    ...     print(f"Loaded {len(exp.plate.wells)} wells")
    >>>
    >>> # For modifications, use engine + ID pattern instead:
    >>> engine = create_engine("sqlite:///analysis.db")
    >>> with Session(engine) as session:
    ...     exp = session.get(Experiment, experiment_id)
    ...     exp.name = "Updated Name"  # Modify within session
    ...     session.commit()  # Save changes
    """
    from pathlib import Path

    from sqlalchemy.exc import OperationalError
    from sqlmodel import select

    # Check if database file exists
    db_path = Path(db_path) if isinstance(db_path, str) else db_path
    if not db_path.exists():
        return None

    # Convert to string for consistency
    db_path_str = str(db_path)
    engine = create_engine(
        f"sqlite:///{db_path_str}",
        echo=echo,
        connect_args={"timeout": 30.0, "check_same_thread": False},
        pool_pre_ping=True,
    )

    try:
        # Use context manager to ensure session is properly closed
        with Session(engine, expire_on_commit=False) as session:
            # Query for experiment
            if experiment_name:
                statement = select(Experiment).where(Experiment.name == experiment_name)
            else:
                statement = select(Experiment)

            try:
                experiment = session.exec(statement).first()
            except OperationalError:
                # Database exists but tables don't (corrupted or empty database)
                return None

            if not experiment:
                return None

            # Force load all relationships to prevent DetachedInstanceError
            _force_load_experiment_relationships(experiment)

            # Make the instance independent of the session
            session.expunge(experiment)

        # Session automatically closed here
        return experiment  # type: ignore
    finally:
        # Dispose engine to release database connections (Windows compatibility)
        engine.dispose(close=True)


def _force_load_experiment_relationships(experiment: Experiment) -> None:
    """Force load all experiment relationships to prevent DetachedInstanceError.

    This function eagerly loads all relationships on an experiment object while
    the session is still active, ensuring the object can be used outside the session.

    Parameters
    ----------
    experiment : Experiment
        The experiment object to load relationships for
    """
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
                    _ = len(roi.traces_history)
                    _ = len(roi.data_analysis_history)
                    _ = roi.roi_mask


def has_fov_analysis(db_path: str | Path, fov_name: str) -> bool:
    """Check if a specific FOV has been analyzed by querying database directly.

    Directly queries the database to check if the FOV exists and has analyzed ROIs.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file
    fov_name : str
        Name of the FOV to check (e.g., "B5_0000")

    Returns
    -------
    bool
        True if the FOV exists and has analyzed ROIs, False otherwise

    Example
    -------
    >>> from cali.sqlmodel import has_fov_analysis
    >>> if has_fov_analysis("analysis.db", "B5_0000"):
    ...     print("B5_0000 has been analyzed")
    """
    from sqlmodel import select

    from ._model import FOV, ROI, Traces

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"timeout": 30.0, "check_same_thread": False},
        pool_pre_ping=True,
    )
    try:
        with Session(engine) as session:
            # Check if this specific FOV has any ROIs with Traces entries
            # (which indicates the FOV has been analyzed)
            statement = (
                select(Traces).join(ROI).join(FOV).where(FOV.name == fov_name).limit(1)
            )
            result = session.exec(statement).first()
            return result is not None
    finally:
        engine.dispose(close=True)


def has_experiment_analysis(db_path: str | Path) -> bool:
    """Check if experiment has any analyzed data by querying database directly.

    Directly queries the database to check if any ROIs exist with analysis data.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file

    Returns
    -------
    bool
        True if any ROIs have analysis data, False otherwise

    Example
    -------
    >>> from cali.sqlmodel import has_experiment_analysis
    >>> if has_experiment_analysis("analysis.db"):
    ...     print("Experiment has analysis data")
    """
    from sqlmodel import select

    from ._model import Traces

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"timeout": 30.0, "check_same_thread": False},
        pool_pre_ping=True,
    )
    try:
        with Session(engine) as session:
            # Check if any Traces entries exist (indicates analysis has been run)
            statement = select(Traces).limit(1)
            result = session.exec(statement).first()
            return result is not None
    finally:
        engine.dispose(close=True)


def _parse_well_name(well_name: str) -> tuple[int, int]:
    """Parse well name like 'B5' or 'AE19' into (row, column) indices.

    Supports both single-letter (A-Z) and multi-letter (AA, AB, ...) row names
    for plates with more than 26 rows.

    Parameters
    ----------
    well_name : str
        Well name (e.g., 'B5', 'A1', 'AE19')

    Returns
    -------
    tuple[int, int]
        (row, column) - Zero-indexed row and column

    Raises
    ------
    ValueError
        If well_name is not in the expected format
    """
    if not well_name or len(well_name) < 2:
        raise ValueError(
            f"Invalid well name: '{well_name}'. Expected format like 'B5', 'AE19'"
        )

    # Split into letter prefix and number suffix
    i = 0
    while i < len(well_name) and well_name[i].isalpha():
        i += 1

    if i == 0:
        raise ValueError(f"Invalid well name: '{well_name}'. Must start with letter(s)")

    if i == len(well_name) or not well_name[i:].isdigit():
        raise ValueError(
            f"Invalid well name: '{well_name}'. Expected format like 'B5', 'AE19' "
            f"(letter(s) followed by number)"
        )

    row_label = well_name[:i]
    row = _label_to_row_index(row_label)
    col = int(well_name[i:]) - 1
    return row, col


def _label_to_row_index(label: str) -> int:
    """Convert well row label to zero-indexed row number.

    Supports single and multi-letter labels using base-26 alphabet.
    A=0, B=1, ..., Z=25, AA=26, AB=27, ..., AZ=51, etc.

    Parameters
    ----------
    label : str
        Row label (e.g., 'A', 'Z', 'AA', 'AE')

    Returns
    -------
    int
        Zero-indexed row number

    Examples
    --------
    >>> _label_to_row_index("A")
    0
    >>> _label_to_row_index("Z")
    25
    >>> _label_to_row_index("AA")
    26
    >>> _label_to_row_index("AE")
    30
    """
    label = label.upper()
    result = 0
    for char in label:
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1


# OLD WAY TO STORE DATA --------------------------------------------------------------

# Define a type variable for the BaseClass
T = TypeVar("T", bound="BaseClass")


@dataclass
class BaseClass:
    """Base class for all classes in the package."""

    def replace(self: T, **kwargs: Any) -> T:
        """Replace the values of the dataclass with the given keyword arguments."""
        return replace(self, **kwargs)


# fmt: off
@dataclass
class ROIData(BaseClass):
    """Data container for ROI (Region of Interest) analysis results.

    This dataclass stores comprehensive analysis data for a single ROI including
    raw fluorescence traces, neuropil correction, calcium dynamics (dff, denoised),
    peak detection, inferred spikes, and experimental metadata.

    Parameters
    ----------
    well_fov_position : str
        Position identifier (e.g., "B5_0000_p0" for well B5, fov0, position 0)
    raw_trace : list[float] | None
        Original raw fluorescence trace before any neuropil correction
    corrected_trace : list[float] | None
        Raw fluorescence trace after neuropil correction (if enabled),
        otherwise same as raw_trace. This is used for all
        downstream analysis.
    neuropil_trace : list[float] | None
        Fluorescence trace from the neuropil (donut-shaped region around ROI)
    neuropil_correction_factor : float | None
        Correction factor used for neuropil subtraction
    dff : list[float] | None
        ΔF/F (delta F over F) - normalized fluorescence change
    den_dff : list[float] | None
        Denoised ΔF/F trace (using OASIS algorithm) for calcium event detection
    peaks_den_dff : list[float] | None
        Indices of detected peaks in the denoised trace
    peaks_amplitudes_den_dff : list[float] | None
        Amplitude values of detected peaks in denoised trace
    peaks_prominence_den_dff : float | None
        Prominence threshold used for peak detection
    peaks_height_den_dff : float | None
        Height threshold used for peak detection
    inferred_spikes : list[float] | None
        Inferred spike probabilities from deconvolution
    inferred_spikes_threshold : float | None
        Threshold for spike detection
    den_dff_frequency : float | None
        Frequency of calcium events in Hz
    condition_1 : str | None
        First experimental condition (e.g., genotype)
    condition_2 : str | None
        Second experimental condition (e.g., treatment)
    cell_size : float | None
        ROI area in µm² or pixels
    cell_size_units : str | None
        Units for cell_size ("µm" or "pixel")
    elapsed_time_list_ms : list[float] | None
        Timestamp for each frame in milliseconds
    total_recording_time_sec : float | None
        Total recording duration in seconds
    active : bool | None
        Whether the ROI shows calcium activity (has detected peaks)
    iei : list[float] | None
        Inter-event intervals between calcium peaks (in seconds)
    evoked_experiment : bool
        Whether this is an optogenetic stimulation experiment
    stimulated : bool
        Whether this ROI overlaps with the stimulated area
    stimulations_frames_and_powers : dict[str, int] | None
        Frame numbers and LED powers for stimulation events
    led_pulse_duration : str | None
        Duration of LED pulse in stimulation experiments
    led_power_equation : str | None
        Equation to calculate LED power density (mW/cm²)
    calcium_sync_jitter_window : int | None
        Jitter window (frames) for calcium peak synchrony analysis
    spikes_sync_cross_corr_lag : int | None
        Maximum lag (frames) for spike cross-correlation synchrony
    calcium_network_threshold : float | None
        Percentile threshold (0-100) for network connectivity
    spikes_burst_threshold : float | None
        Threshold (%) for burst detection in spike trains
    spikes_burst_min_duration : int | None
        Minimum burst duration in seconds
    spikes_burst_gaussian_sigma : float | None
        Sigma for Gaussian smoothing in burst detection (seconds)
    mask_coord_and_shape : tuple[tuple[list[int], list[int]], tuple[int, int]] | None
        ROI mask stored as ((y_coords, x_coords), (height, width))
    neuropil_mask_coord_and_shape : tuple | None
        Neuropil mask: ((y_coords, x_coords), (height, width))
    """

    well_fov_position: str = ""
    raw_trace: list[float] | None = None
    corrected_trace: list[float] | None = None
    neuropil_trace: list[float] | None = None
    neuropil_correction_factor: float | None = None
    dff: list[float] | None = None
    den_dff: list[float] | None = None  # denoised dff with oasis package
    peaks_den_dff: list[float] | None = None
    peaks_amplitudes_den_dff: list[float] | None = None
    peaks_prominence_den_dff: float | None = None
    peaks_height_den_dff: float | None = None
    inferred_spikes: list[float] | None = None
    inferred_spikes_threshold: float | None = None
    den_dff_frequency: float | None = None  # Hz
    condition_1: str | None = None
    condition_2: str | None = None
    cell_size: float | None = None
    cell_size_units: str | None = None
    elapsed_time_list_ms: list[float] | None = None  # in ms
    total_recording_time_sec: float | None = None  # in seconds
    active: bool | None = None
    iei: list[float] | None = None  # interevent interval
    evoked_experiment: bool = False
    stimulated: bool = False
    stimulations_frames_and_powers: dict[str, int] | None = None
    led_pulse_duration: str | None = None
    led_power_equation: str | None = None  # equation for LED power
    calcium_sync_jitter_window: int | None = None  # in frames
    spikes_sync_cross_corr_lag: int | None = None  # in frames
    calcium_network_threshold: float | None = None  # percentile (0-100)
    spikes_burst_threshold: float | None = None  # in percent
    spikes_burst_min_duration: int | None = None  # in seconds
    spikes_burst_gaussian_sigma: float | None = None  # in seconds
    # store ROI mask as coordinates (y_coords, x_coords) and shape (height, width)
    mask_coord_and_shape: tuple[tuple[list[int], list[int]], tuple[int, int]] | None = None  # noqa: E501
    # store neuropil mask as coordinates (y_coords, x_coords) and shape (height, width)
    neuropil_mask_coord_and_shape: tuple[tuple[list[int], list[int]], tuple[int, int]] | None = None  # noqa: E501
# fmt: on
