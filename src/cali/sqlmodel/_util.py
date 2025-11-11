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
) -> Experiment:
    """Save an experiment object tree to a SQLite database.

    Parameters
    ----------
    experiment : Experiment
        Experiment object (e.g., from load_analysis_from_json)
    db_path : Path | str
        Path to SQLite database file
    overwrite : bool, optional
        Whether to overwrite existing database file, by default False

    Returns
    -------
    Experiment
        The experiment object with updated database IDs and all relationships loaded

    Example
    -------
    >>> from pathlib import Path
    >>> exp = load_analysis_from_json(Path("tests/test_data/..."))
    >>> exp = save_experiment_to_database(exp, "analysis.db")
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
        # Merge and get the updated object back with correct database IDs
        merged_experiment = session.merge(experiment)
        session.commit()
        # Refresh to ensure all relationships have correct IDs
        session.refresh(merged_experiment)
        # Force load all relationships to prevent DetachedInstanceError
        _force_load_experiment_relationships(merged_experiment)
        # Make the instance independent of the session
        session.expunge(merged_experiment)
        return merged_experiment


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

        # Force load all relationships to prevent DetachedInstanceError
        _force_load_experiment_relationships(experiment)

        # Make the instance independent of the session
        session.expunge(experiment)

    # Session automatically closed here
    return experiment  # type: ignore


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
                    _ = roi.analysis_settings

    if experiment.analysis_settings:
        _ = experiment.analysis_settings.stimulation_mask



def has_fov_analysis(experiment: Experiment, fov_name: str) -> bool:
    """Check if a specific FOV has been analyzed (has ROIs with data).

    This function efficiently checks if a FOV by name exists in the experiment
    and has at least one analyzed ROI (with traces or data_analysis).

    Parameters
    ----------
    experiment : Experiment
        The experiment object to check
    fov_name : str
        Name of the FOV to check (e.g., "B5_0000")

    Returns
    -------
    bool
        True if the FOV exists and has analyzed ROIs, False otherwise

    Example
    -------
    >>> from cali.sqlmodel import load_experiment_from_database, has_fov_analysis
    >>> exp = load_experiment_from_database("analysis.db")
    >>> if has_fov_analysis(exp, "B5_0000"):
    ...     print("B5_0000 has been analyzed")
    """
    if not experiment.plate or not experiment.plate.wells:
        return False

    # Search through wells -> FOVs -> ROIs
    for well in experiment.plate.wells:
        for fov in well.fovs:
            # Check if this is the FOV we're looking for (with or without position index)
            if fov.name == fov_name or fov.name.startswith(f"{fov_name}_p"):
                # Check if it has any analyzed ROIs
                if fov.rois:
                    for roi in fov.rois:
                        # If ROI has traces or data analysis, it's been analyzed
                        if roi.traces is not None or roi.data_analysis is not None:
                            return True
    return False


def has_experiment_analysis(experiment: Experiment) -> bool:
    """Check if the experiment has any analyzed data.

    This function checks if any FOV in the experiment has analyzed ROIs.

    Parameters
    ----------
    experiment : Experiment
        The experiment object to check

    Returns
    -------
    bool
        True if any FOV has analyzed ROIs, False otherwise

    Example
    -------
    >>> if has_experiment_analysis(exp):
    ...     print("Experiment has analysis data")
    """
    if not experiment.plate or not experiment.plate.wells:
        return False

    for well in experiment.plate.wells:
        for fov in well.fovs:
            if fov.rois:
                for roi in fov.rois:
                    if roi.traces is not None or roi.data_analysis is not None:
                        return True
    return False


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
