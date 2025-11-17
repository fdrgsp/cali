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

from sqlmodel import Session, create_engine

from cali.logger import cali_logger

from ._model import Experiment

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
    >>> from cali.sqlmodel import create_database_and_tables
    >>> engine = create_engine("sqlite:///calcium_analysis.db")
    >>> create_database_and_tables(engine)
    """
    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(engine)


def save_experiment_to_database(
    experiment: Experiment,
    overwrite: bool = False,
    database_name: str | None = None,
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
        Experiment object (e.g., from load_analysis_from_json)
    overwrite : bool, optional
        Whether to overwrite existing database file, by default False
    database_name : str | None, optional
        Name of the database file (e.g., "cali.db"). If None, uses the
        experiment's `database_name` attribute.
    echo : bool, optional
        Whether to enable SQLAlchemy engine echo for debugging, by default False

    Example
    -------
    >>> from pathlib import Path
    >>> exp = load_analysis_from_json(Path("tests/test_data/..."))
    >>> save_experiment_to_database(exp, overwrite=True)
    >>> # Later, load fresh from DB when needed:
    >>> db_path = Path(exp.analysis_path) / exp.database_name
    >>> exp = load_experiment_from_database(db_path)
    """
    if experiment.analysis_path is None:
        raise ValueError(
            "Experiment must have `analysis_path` set to save to database."
        )
    if experiment.database_name is None and database_name is None:
        raise ValueError(
            "Experiment must have `database_name` set to save to "
            "database or provide `database_name` argument."
        )

    # Use provided database_name or fall back to experiment's database_name
    db_name = database_name if database_name is not None else experiment.database_name
    assert db_name is not None  # Guaranteed by the check above
    db_path = Path(experiment.analysis_path) / db_name

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and db_path.exists():
        db_path.unlink()

    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    create_database_and_tables(engine)

    try:
        with Session(engine) as session:
            # Merge handles add/update for the entire object tree with cascade
            session.merge(experiment)
            session.commit()

        cali_logger.info(
            f"💾 Experiment analysis updated and saved to database at "
            f"{experiment.analysis_path}/{experiment.database_name}."
        )
    finally:
        # Dispose engine to release database connections (Windows compatibility)
        engine.dispose()


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
    from sqlmodel import select

    # Convert to string for consistency
    db_path_str = str(db_path)
    engine = create_engine(f"sqlite:///{db_path_str}", echo=echo)

    try:
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

            # Force load all relationships to prevent DetachedInstanceError
            _force_load_experiment_relationships(experiment)

            # Make the instance independent of the session
            session.expunge(experiment)

        # Session automatically closed here
        return experiment  # type: ignore
    finally:
        # Dispose engine to release database connections (Windows compatibility)
        engine.dispose()


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
                    _ = roi.traces
                    _ = roi.data_analysis
                    _ = roi.roi_mask
                    _ = roi.neuropil_mask


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

    from ._model import FOV, ROI

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            # Query for FOVs with the given name that have ROIs with traces or data
            statement = (
                select(FOV)
                .join(ROI)
                .where(FOV.name == fov_name)
                .where((ROI.traces != None) | (ROI.data_analysis != None))  # noqa: E711
                .limit(1)
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

    from ._model import ROI

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            # Check if any ROI exists with traces or data_analysis
            statement = (
                select(ROI)
                .where((ROI.traces != None) | (ROI.data_analysis != None))  # noqa: E711
                .limit(1)
            )
            result = session.exec(statement).first()
            return result is not None
    finally:
        engine.dispose(close=True)


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
    raw fluorescence traces, neuropil correction, calcium dynamics (dff, deconvolved),
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
    dec_dff : list[float] | None
        Deconvolved ΔF/F trace (using OASIS algorithm) for calcium event detection
    peaks_dec_dff : list[float] | None
        Indices of detected peaks in the deconvolved trace
    peaks_amplitudes_dec_dff : list[float] | None
        Amplitude values of detected peaks in deconvolved trace
    peaks_prominence_dec_dff : float | None
        Prominence threshold used for peak detection
    peaks_height_dec_dff : float | None
        Height threshold used for peak detection
    inferred_spikes : list[float] | None
        Inferred spike probabilities from deconvolution
    inferred_spikes_threshold : float | None
        Threshold for spike detection
    dec_dff_frequency : float | None
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
    dec_dff: list[float] | None = None  # deconvolved dff with oasis package
    peaks_dec_dff: list[float] | None = None
    peaks_amplitudes_dec_dff: list[float] | None = None
    peaks_prominence_dec_dff: float | None = None
    peaks_height_dec_dff: float | None = None
    inferred_spikes: list[float] | None = None
    inferred_spikes_threshold: float | None = None
    dec_dff_frequency: float | None = None  # Hz
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
