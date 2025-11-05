"""SQLModel schema for calcium imaging analysis data.

This module defines the database schema for storing calcium imaging analysis results
using SQLModel. The schema supports hierarchical data organization:
Experiment → Plate → Well → FOV (Field of View) → ROI (Region of Interest)

The schema enables:
- Efficient querying by experimental conditions
- Tracking analysis parameters and metadata
- Easy data export and statistical analysis
- Relationship navigation (e.g., all ROIs for a condition)
"""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy.engine import Engine
from sqlmodel import JSON, Column, Field, Relationship, SQLModel

# ==================== Core Models ====================


class Experiment(SQLModel, table=True):
    """Top-level experiment container.

    An experiment can contain a plate and tracks global metadata
    like creation date, description, and data paths.

    Attributes
    ----------
    id : uuid.UUID
        Primary key, auto-generated
    name : str
        Unique experiment identifier
    description : str | None
        Optional experiment description
    created_at : datetime
        Timestamp when experiment was created
    data_path : str | None
        Path to the raw imaging data (zarr/tensorstore)
    labels_path : str | None
        Path to segmentation labels directory
    analysis_path : str | None
        Path to analysis output directory
    plate : Plate
        Related plate (back-populated by SQLModel)
    """

    __tablename__ = "experiment"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    data_path: str | None = None
    labels_path: str | None = None
    analysis_path: str | None = None

    # Relationships
    plate: "Plate" = Relationship(back_populates="experiment")
    analysis_settings: list["AnalysisSettings"] = Relationship(
        back_populates="experiment"
    )


class AnalysisSettings(SQLModel, table=True):
    """Analysis parameter settings for an experiment.

    Stores the analysis parameters used for a specific analysis run.
    Multiple AnalysisSettings can exist per experiment (e.g., initial analysis,
    re-analysis with different parameters). Each ROI links to the specific
    AnalysisSettings that were used to analyze it.

    Attributes
    ----------
    id : uuid.UUID
        Primary key, auto-generated
    experiment_id : uuid.UUID
        Foreign key to parent experiment
    created_at : datetime
        When these settings were created
    dff_window : int
        Window size for ΔF/F baseline calculation
    decay_constant : float
        Decay constant for deconvolution
    peaks_height_value : float
        Peak height threshold value
    peaks_height_mode : str
        Mode for peak height ("multiplier" or "absolute")
    peaks_distance : int
        Minimum distance between peaks (frames)
    peaks_prominence_multiplier : float
        Multiplier for peak prominence threshold
    calcium_network_threshold : float
        Percentile threshold for network connectivity (0-100)
    spike_threshold_value : float
        Spike detection threshold value
    spike_threshold_mode : str
        Mode for spike threshold ("multiplier" or "absolute")
    burst_threshold : float
        Threshold for burst detection (%)
    burst_min_duration : int
        Minimum burst duration (seconds)
    burst_gaussian_sigma : float
        Gaussian sigma for burst smoothing (seconds)
    spikes_sync_cross_corr_lag : int
        Max lag for spike synchrony cross-correlation (frames)
    calcium_sync_jitter_window : int
        Jitter window for calcium synchrony (frames)
    neuropil_inner_radius : int
        Inner radius for neuropil mask (pixels)
    neuropil_min_pixels : int
        Minimum pixels required for neuropil mask
    neuropil_correction_factor : float
        Neuropil correction factor (0-1)
    led_power_equation : str | None
        Equation for LED power calculation (evoked experiments)
    led_pulse_duration : float | None
        Duration of LED pulse (evoked experiments)
    peaks_prominence_dec_dff : float | None
        Peak prominence threshold for deconvolved ΔF/F
    peaks_height_dec_dff : float | None
        Peak height threshold for deconvolved ΔF/F
    inferred_spikes_threshold : float | None
        Threshold for spike inference from deconvolved trace
    stimulations_frames_and_powers : dict | None
        Stimulation timing and power data (evoked experiments)
    experiment : Experiment
        Parent experiment
    """

    __tablename__ = "analysis_settings"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)

    # Analysis parameters
    dff_window: int = 30
    decay_constant: float = 0.0
    peaks_height_value: float = 3.0
    peaks_height_mode: str = "multiplier"
    peaks_distance: int = 2
    peaks_prominence_multiplier: float = 1.0
    calcium_network_threshold: float = 90.0
    spike_threshold_value: float = 1.0
    spike_threshold_mode: str = "multiplier"
    burst_threshold: float = 30.0
    burst_min_duration: int = 3
    burst_gaussian_sigma: float = 2.0
    spikes_sync_cross_corr_lag: int = 5
    calcium_sync_jitter_window: int = 5
    neuropil_inner_radius: int = 0
    neuropil_min_pixels: int = 0
    neuropil_correction_factor: float = 0.0
    peaks_prominence_dec_dff: float | None = None
    peaks_height_dec_dff: float | None = None
    inferred_spikes_threshold: float | None = None
    led_power_equation: str | None = None
    led_pulse_duration: float | None = None
    stimulations_frames_and_powers: dict | None = Field(
        default=None, sa_column=Column(JSON)
    )

    experiment_id: int = Field(foreign_key="experiment.id", index=True)

    # Relationships
    experiment: "Experiment" = Relationship(back_populates="analysis_settings")


class Plate(SQLModel, table=True):
    """Plate container (e.g., 96-well plate).

    Attributes
    ----------
    id : uuid.UUID
        Primary key, auto-generated
    experiment_id : uuid.UUID
        Foreign key to parent experiment
    name : str
        Plate name/identifier
    plate_type : str | None
        Plate format (e.g., "96-well", "384-well")
    rows : int | None
        Number of rows in plate
    columns : int | None
        Number of columns in plate
    experiment : Experiment
        Parent experiment
    wells : list[Well]
        Child wells in this plate
    """

    __tablename__ = "plate"

    id: int | None = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="experiment.id", index=True)
    name: str = Field(index=True)
    plate_type: str | None = None  # e.g., "96-well", "384-well"
    rows: int | None = None
    columns: int | None = None

    # Relationships
    experiment: "Experiment" = Relationship(back_populates="plate")
    wells: list["Well"] = Relationship(back_populates="plate")


class Condition(SQLModel, table=True):
    """Experimental condition (e.g., genotype, treatment).

    Conditions can be reused across multiple wells. This allows for
    consistent condition naming and easy grouping.

    Attributes
    ----------
    id : uuid.UUID
        Primary key, auto-generated
    name : str
        Unique condition name (e.g., "WT", "KO", "Vehicle", "Drug_10uM")
    condition_type : str
        Type of condition ("genotype", "treatment", "other")
    color : str | None
        Display color for plots (e.g., "coral", "#FF6347")
    description : str | None
        Optional detailed description
    """

    __tablename__ = "condition"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    condition_type: str = Field(index=True)  # "genotype", "treatment", etc.
    color: str | None = None
    description: str | None = None


class WellCondition(SQLModel, table=True):
    """Link table for Well-Condition many-to-many relationship."""

    __tablename__ = "well_condition_link"

    well_id: int = Field(foreign_key="well.id", primary_key=True)
    condition_id: int = Field(foreign_key="condition.id", primary_key=True)


class Well(SQLModel, table=True):
    """Well in a plate (e.g., "B5").

    A well can have multiple FOVs (imaging positions) and is associated
    with experimental conditions.

    Attributes
    ----------
    id : uuid.UUID
        Primary key, auto-generated
    plate_id : uuid.UUID
        Foreign key to parent plate
    name : str
        Well name (e.g., "B5", "C3")
    row : int
        Row index (0-based)
    column : int
        Column index (0-based)
    plate : Plate
        Parent plate
    conditions : list[Condition]
        Associated experimental conditions (many-to-many)
    fovs : list[FOV]
        Imaging positions in this well
    condition_1 : Condition | None
        First experimental condition (convenience property)
    condition_2 : Condition | None
        Second experimental condition (convenience property)
    """

    __tablename__ = "well"

    id: int | None = Field(default=None, primary_key=True)
    plate_id: int = Field(foreign_key="plate.id", index=True)
    name: str = Field(index=True)
    row: int
    column: int

    # Relationships
    plate: "Plate" = Relationship(back_populates="wells")
    conditions: list["Condition"] = Relationship(
        link_model=WellCondition,
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    fovs: list["FOV"] = Relationship(back_populates="well")

    @property
    def condition_1(self) -> Optional["Condition"]:
        """First experimental condition (e.g., genotype)."""
        return self.conditions[0] if len(self.conditions) > 0 else None

    @property
    def condition_2(self) -> Optional["Condition"]:
        """Second experimental condition (e.g., treatment)."""
        return self.conditions[1] if len(self.conditions) > 1 else None


class FOV(SQLModel, table=True):
    """Field of View (imaging position) within a well.

    Each FOV represents a single imaging position/site within a well.
    FOVs contain multiple ROIs (individual cells).

    Attributes
    ----------
    id : uuid.UUID
        Primary key, auto-generated
    well_id : uuid.UUID
        Foreign key to parent well
    name : str
        FOV name (e.g., "B5_0000_p0")
    position_index : int
        Position index in acquisition order (e.g. if in an experiment we have 2 fovs per
        well and this is the second well, second fov, this index would be 3 - the 4th
        position)
    fov_number : int
        The FOV number per well
    metadata : dict | None
        Additional metadata from acquisition (stored as JSON)
    well : Well
        Parent well
    rois : list[ROI]
        Regions of interest (cells) in this FOV
    """

    __tablename__ = "fov"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    position_index: int = Field(index=True)
    fov_number: int = Field(default=0)
    fov_metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))

    well_id: Optional[int] = Field(default=None, foreign_key="well.id", index=True)

    # Relationships
    well: "Well" = Relationship(back_populates="fovs")
    rois: list["ROI"] = Relationship(back_populates="fov")


class ROI(SQLModel, table=True):
    """Region of Interest (ROI) core metadata.

    Represents a single cell/neuron segmented from imaging data.
    Related analysis data is stored in separate tables (Trace, PeakAnalysis, etc.)

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    fov_id : uuid.UUID
        Foreign key to parent FOV
    label_value : int
        ROI label number from segmentation (e.g., 1, 2, 3...)
    cell_size : float | None
        ROI area (µm² or pixels)
    cell_size_units : str | None
        Units for cell_size
    total_recording_time_sec : float | None
        Total recording duration (seconds)
    active : bool | None
        Whether ROI shows calcium activity
    fov : FOV
        Parent FOV
    analysis_settings : AnalysisSettings | None
        Analysis settings used
    trace : Trace | None
        Fluorescence trace data
    data_analysis : DataAnalysis | None
        Data analysis results (peaks, spikes, etc.)
    roi_mask : Mask | None
        ROI mask (cell boundary)
    neuropil_mask : Mask | None
        Neuropil mask (background region)
    """

    __tablename__ = "roi"

    id: int | None = Field(default=None, primary_key=True)
    label_value: int = Field(index=True)

    active: bool | None = None
    stimulated: bool = False

    cell_size: float | None = None
    cell_size_units: str | None = None
    total_recording_time_sec: float | None = None

    fov_id: int = Field(foreign_key="fov.id", index=True)
    analysis_settings_id: int | None = Field(
        default=None, foreign_key="analysis_settings.id", index=True
    )
    roi_mask_id: int | None = Field(default=None, foreign_key="mask.id", index=True)
    neuropil_mask_id: int | None = Field(
        default=None, foreign_key="mask.id", index=True
    )

    # Relationships
    fov: "FOV" = Relationship(back_populates="rois")
    analysis_settings: Optional["AnalysisSettings"] = Relationship()
    traces: Optional["Traces"] = Relationship(back_populates="roi")
    data_analysis: Optional["DataAnalysis"] = Relationship(back_populates="roi")
    roi_mask: Optional["Mask"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[ROI.roi_mask_id]",
            "lazy": "selectin",
        }
    )
    neuropil_mask: Optional["Mask"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[ROI.neuropil_mask_id]",
            "lazy": "selectin",
        }
    )


class Traces(SQLModel, table=True):
    """Fluorescence trace data for an ROI.

    Stores all time-series fluorescence measurements and derived traces.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    roi_id : uuid.UUID | None
        Foreign key to parent ROI
    raw_trace : list[float] | None
        Raw fluorescence trace
    corrected_trace : list[float] | None
        Neuropil-corrected fluorescence trace
    neuropil_trace : list[float] | None
        Neuropil fluorescence trace
    dff : list[float] | None
        ΔF/F normalized trace
    dec_dff : list[float] | None
        Deconvolved ΔF/F trace
    x_axis : list[float] | None
        Frames number or frame timestamps (milliseconds)
    roi : ROI
        Parent ROI
    """

    __tablename__ = "trace"

    id: int | None = Field(default=None, primary_key=True)

    raw_trace: list[float] | None = Field(default=None, sa_column=Column(JSON))
    corrected_trace: list[float] | None = Field(default=None, sa_column=Column(JSON))
    neuropil_trace: list[float] | None = Field(default=None, sa_column=Column(JSON))
    dff: list[float] | None = Field(default=None, sa_column=Column(JSON))
    dec_dff: list[float] | None = Field(default=None, sa_column=Column(JSON))
    x_axis: list[float] | None = Field(default=None, sa_column=Column(JSON))

    roi_id: int | None = Field(
        default=None, foreign_key="roi.id", index=True, unique=True
    )

    # Relationships
    roi: "ROI" = Relationship(back_populates="traces")


class DataAnalysis(SQLModel, table=True):
    """Container for different types of data analyses for an ROI.

    This class serves as a parent container for various analysis results
    related to an ROI, such as peak detection and spike inference.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    roi_id : int
        Foreign key to parent ROI
    peaks_dec_dff : list[float] | None
        Peak indices in deconvolved trace
    peaks_amplitudes_dec_dff : list[float] | None
        Peak amplitudes
    peaks_prominence_dec_dff : float | None
        Peak prominence threshold used
    peaks_height_dec_dff : float | None
        Peak height threshold used
    dec_dff_frequency : float | None
        Calcium event frequency (Hz)
    iei : list[float] | None
        Inter-event intervals (seconds)
    inferred_spikes : list[float] | None
        Inferred spike probabilities
    inferred_spikes_threshold : float | None
        Spike detection threshold used
    roi : ROI
        Parent ROI
    """

    __tablename__ = "data_analysis"

    id: int | None = Field(default=None, primary_key=True)
    roi_id: int | None = Field(
        default=None, foreign_key="roi.id", index=True, unique=True
    )

    dec_dff_frequency: float | None = None
    peaks_dec_dff: list[float] | None = Field(default=None, sa_column=Column(JSON))
    peaks_amplitudes_dec_dff: list[float] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    iei: list[float] | None = Field(default=None, sa_column=Column(JSON))
    inferred_spikes: list[float] | None = Field(default=None, sa_column=Column(JSON))

    # Relationships
    roi: "ROI" = Relationship(back_populates="data_analysis")


class Mask(SQLModel, table=True):
    """Generic mask coordinate data.

    Stores spatial coordinates and dimensions for a mask (ROI or neuropil).

    Attributes
    ----------
    id : uuid.UUID
        Primary key, auto-generated
    coords_y : list[int] | None
        Y-coordinates of mask pixels
    coords_x : list[int] | None
        X-coordinates of mask pixels
    height : int | None
        Mask height
    width : int | None
        Mask width
    mask_type : str
        Type of mask ("roi" or "neuropil")
    """

    __tablename__ = "mask"

    id: int | None = Field(default=None, primary_key=True)

    coords_y: list[int] | None = Field(default=None, sa_column=Column(JSON))
    coords_x: list[int] | None = Field(default=None, sa_column=Column(JSON))
    height: int | None = None
    width: int | None = None
    mask_type: str = Field(index=True)  # "roi" or "neuropil"


# ==================== Helper Functions ====================


def create_db_and_tables(engine: Engine) -> None:
    """Create all database tables.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Database engine

    Example
    -------
    >>> from sqlmodel import create_engine
    >>> engine = create_engine("sqlite:///calcium_analysis.db")
    >>> create_tables(engine)
    """
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
