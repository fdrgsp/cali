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

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.engine import Engine
from sqlmodel import JSON, Column, Field, Relationship, SQLModel

from cali._plate_viewer._util import ROIData

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

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    data_path: str | None = None
    labels_path: str | None = None
    analysis_path: str | None = None

    # Relationships
    plate: "Plate" = Relationship(back_populates="experiment")


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

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    experiment_id: uuid.UUID = Field(foreign_key="experiment.id", index=True)
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
    id : int | None
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
    condition_1_id : int | None
        Foreign key to first condition (e.g., genotype)
    condition_2_id : int | None
        Foreign key to second condition (e.g., treatment)
    plate : Plate
        Parent plate
    condition_1 : Condition | None
        First experimental condition
    condition_2 : Condition | None
        Second experimental condition
    fovs : list[FOV]
        Imaging positions in this well
    """

    __tablename__ = "well"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    plate_id: uuid.UUID = Field(foreign_key="plate.id", index=True)
    name: str = Field(index=True)
    row: int
    column: int
    condition_1_id: int | None = Field(default=None, foreign_key="condition.id")
    condition_2_id: int | None = Field(default=None, foreign_key="condition.id")

    # Relationships
    plate: "Plate" = Relationship(back_populates="wells")
    condition_1: Optional["Condition"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[Well.condition_1_id]",
            "lazy": "selectin",
        }
    )
    condition_2: Optional["Condition"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[Well.condition_2_id]",
            "lazy": "selectin",
        }
    )
    fovs: list["FOV"] = Relationship(back_populates="well")


class FOV(SQLModel, table=True):
    """Field of View (imaging position) within a well.

    Each FOV represents a single imaging position/site within a well.
    FOVs contain multiple ROIs (individual cells).

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    well_id : int
        Foreign key to parent well
    name : str
        FOV name (e.g., "B5_0000_p0")
    position_index : int
        Position/site index (0-based)
    fov_number : int | None
        FOV number from acquisition
    metadata : dict | None
        Additional metadata from acquisition (stored as JSON)
    well : Well
        Parent well
    rois : list[ROI]
        Regions of interest (cells) in this FOV
    """

    __tablename__ = "fov"

    id: int | None = Field(default=None, primary_key=True)
    well_id: uuid.UUID = Field(foreign_key="well.id", index=True)
    name: str = Field(unique=True, index=True)
    position_index: int = Field(index=True)
    fov_number: int | None = None
    fov_metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))

    # Relationships
    well: "Well" = Relationship(back_populates="fovs")
    rois: list["ROI"] = Relationship(back_populates="fov")


class AnalysisSettings(SQLModel, table=True):
    """Analysis parameter settings.

    Stores the analysis parameters used for a specific analysis run.
    This allows tracking which parameters were used and reproducing results.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    name : str
        Settings profile name
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
    """

    __tablename__ = "analysis_settings"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
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
    led_power_equation: str | None = None


class ROI(SQLModel, table=True):
    """Region of Interest (single cell/neuron) with analysis results.

    This is the core data model containing all analysis results for an individual
    cell/ROI. It stores fluorescence traces, calcium dynamics, peaks, spikes,
    and metadata.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    fov_id : int
        Foreign key to parent FOV
    label_value : int
        ROI label number from segmentation (e.g., 1, 2, 3...)
    analysis_settings_id : int | None
        Foreign key to analysis settings used

    # Trace Data
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
    elapsed_time_list_ms : list[float] | None
        Frame timestamps (milliseconds)

    # Peak Detection
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

    # Spike Inference
    inferred_spikes : list[float] | None
        Inferred spike probabilities
    inferred_spikes_threshold : float | None
        Spike detection threshold

    # Metadata
    cell_size : float | None
        ROI area (µm² or pixels)
    cell_size_units : str | None
        Units for cell_size
    total_recording_time_sec : float | None
        Total recording duration (seconds)
    active : bool | None
        Whether ROI shows calcium activity
    neuropil_correction_factor : float | None
        Neuropil correction factor used

    # Evoked Experiment Data
    evoked_experiment : bool
        Whether this is an optogenetic experiment
    stimulated : bool
        Whether ROI overlaps stimulated area
    stimulations_frames_and_powers : dict | None
        Stimulation timing and power
    led_pulse_duration : str | None
        LED pulse duration

    # Network Analysis Parameters
    calcium_sync_jitter_window : int | None
        Jitter window for synchrony (frames)
    spikes_sync_cross_corr_lag : int | None
        Max lag for spike correlation (frames)
    calcium_network_threshold : float | None
        Network connectivity threshold (percentile)
    spikes_burst_threshold : float | None
        Burst detection threshold (%)
    spikes_burst_min_duration : int | None
        Min burst duration (seconds)
    spikes_burst_gaussian_sigma : float | None
        Burst smoothing sigma (seconds)

    # Mask Data
    mask_coords_y : list[int] | None
        ROI mask Y coordinates
    mask_coords_x : list[int] | None
        ROI mask X coordinates
    mask_height : int | None
        ROI mask height
    mask_width : int | None
        ROI mask width
    neuropil_mask_coords_y : list[int] | None
        Neuropil mask Y coordinates
    neuropil_mask_coords_x : list[int] | None
        Neuropil mask X coordinates
    neuropil_mask_height : int | None
        Neuropil mask height
    neuropil_mask_width : int | None
        Neuropil mask width

    # Relationships
    fov : FOV
        Parent FOV
    analysis_settings : AnalysisSettings | None
        Analysis settings used
    """

    __tablename__ = "roi"

    id: int | None = Field(default=None, primary_key=True)
    fov_id: int = Field(foreign_key="fov.id", index=True)
    label_value: int = Field(index=True)
    analysis_settings_id: int | None = Field(
        default=None, foreign_key="analysis_settings.id"
    )

    # ==================== Trace Data ====================
    raw_trace: list[float] | None = Field(default=None, sa_column=Column(JSON))
    corrected_trace: list[float] | None = Field(default=None, sa_column=Column(JSON))
    neuropil_trace: list[float] | None = Field(default=None, sa_column=Column(JSON))
    dff: list[float] | None = Field(default=None, sa_column=Column(JSON))
    dec_dff: list[float] | None = Field(default=None, sa_column=Column(JSON))
    elapsed_time_list_ms: list[float] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    # ==================== Peak Detection ====================
    peaks_dec_dff: list[float] | None = Field(default=None, sa_column=Column(JSON))
    peaks_amplitudes_dec_dff: list[float] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    peaks_prominence_dec_dff: float | None = None
    peaks_height_dec_dff: float | None = None
    dec_dff_frequency: float | None = None  # Hz
    iei: list[float] | None = Field(default=None, sa_column=Column(JSON))

    # ==================== Spike Inference ====================
    inferred_spikes: list[float] | None = Field(default=None, sa_column=Column(JSON))
    inferred_spikes_threshold: float | None = None

    # ==================== Cell Metadata ====================
    cell_size: float | None = None
    cell_size_units: str | None = None
    total_recording_time_sec: float | None = None
    active: bool | None = None
    neuropil_correction_factor: float | None = None

    # ==================== Evoked Experiment ====================
    evoked_experiment: bool = False
    stimulated: bool = False
    stimulations_frames_and_powers: dict | None = Field(
        default=None, sa_column=Column(JSON)
    )
    led_pulse_duration: str | None = None
    led_power_equation: str | None = None

    # ==================== Network Analysis Parameters ====================
    calcium_sync_jitter_window: int | None = None
    spikes_sync_cross_corr_lag: int | None = None
    calcium_network_threshold: float | None = None
    spikes_burst_threshold: float | None = None
    spikes_burst_min_duration: int | None = None
    spikes_burst_gaussian_sigma: float | None = None

    # ==================== Mask Data (flattened for DB storage) ====================
    # ROI mask coordinates
    mask_coords_y: list[int] | None = Field(default=None, sa_column=Column(JSON))
    mask_coords_x: list[int] | None = Field(default=None, sa_column=Column(JSON))
    mask_height: int | None = None
    mask_width: int | None = None

    # Neuropil mask coordinates
    neuropil_mask_coords_y: list[int] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    neuropil_mask_coords_x: list[int] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    neuropil_mask_height: int | None = None
    neuropil_mask_width: int | None = None

    # Relationships
    fov: "FOV" = Relationship(back_populates="rois")
    analysis_settings: Optional["AnalysisSettings"] = Relationship()


# ==================== Helper Functions ====================


def create_tables(engine: Engine) -> None:
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


def roi_from_roi_data(
    roi_data: ROIData, fov_id: int, label_value: int, settings_id: int | None = None
) -> ROI:
    """Convert ROIData dataclass to ROI SQLModel.

    This helper function converts the existing ROIData dataclass format
    to the new SQLModel ROI format. It handles the mask coordinate conversion.

    Parameters
    ----------
    roi_data : ROIData
        Original ROIData from analysis
    fov_id : int
        Parent FOV database ID
    label_value : int
        ROI label number
    settings_id : int | None
        Analysis settings ID to associate with this ROI

    Returns
    -------
    ROI
        SQLModel ROI instance ready to be added to database

    Example
    -------
    >>> from cali._plate_viewer._util import ROIData
    >>> roi_data = ROIData(...)  # from existing analysis
    >>> roi = roi_from_roi_data(roi_data, fov_id=1, label_value=1)
    >>> session.add(roi)
    >>> session.commit()
    """
    # Handle mask coordinate conversion
    mask_y, mask_x, mask_h, mask_w = None, None, None, None
    if roi_data.mask_coord_and_shape:
        (y_coords, x_coords), (height, width) = roi_data.mask_coord_and_shape
        mask_y, mask_x = y_coords, x_coords
        mask_h, mask_w = height, width

    neuropil_y, neuropil_x, neuropil_h, neuropil_w = None, None, None, None
    if roi_data.neuropil_mask_coord_and_shape:
        (y_coords, x_coords), (height, width) = roi_data.neuropil_mask_coord_and_shape
        neuropil_y, neuropil_x = y_coords, x_coords
        neuropil_h, neuropil_w = height, width

    return ROI(
        fov_id=fov_id,
        label_value=label_value,
        analysis_settings_id=settings_id,
        # Traces
        raw_trace=roi_data.raw_trace,
        corrected_trace=roi_data.corrected_trace,
        neuropil_trace=roi_data.neuropil_trace,
        dff=roi_data.dff,
        dec_dff=roi_data.dec_dff,
        elapsed_time_list_ms=roi_data.elapsed_time_list_ms,
        # Peaks
        peaks_dec_dff=roi_data.peaks_dec_dff,
        peaks_amplitudes_dec_dff=roi_data.peaks_amplitudes_dec_dff,
        peaks_prominence_dec_dff=roi_data.peaks_prominence_dec_dff,
        peaks_height_dec_dff=roi_data.peaks_height_dec_dff,
        dec_dff_frequency=roi_data.dec_dff_frequency,
        iei=roi_data.iei,
        # Spikes
        inferred_spikes=roi_data.inferred_spikes,
        inferred_spikes_threshold=roi_data.inferred_spikes_threshold,
        # Metadata
        cell_size=roi_data.cell_size,
        cell_size_units=roi_data.cell_size_units,
        total_recording_time_sec=roi_data.total_recording_time_sec,
        active=roi_data.active,
        neuropil_correction_factor=roi_data.neuropil_correction_factor,
        # Evoked
        evoked_experiment=roi_data.evoked_experiment,
        stimulated=roi_data.stimulated,
        stimulations_frames_and_powers=roi_data.stimulations_frames_and_powers,
        led_pulse_duration=roi_data.led_pulse_duration,
        led_power_equation=roi_data.led_power_equation,
        # Network parameters
        calcium_sync_jitter_window=roi_data.calcium_sync_jitter_window,
        spikes_sync_cross_corr_lag=roi_data.spikes_sync_cross_corr_lag,
        calcium_network_threshold=roi_data.calcium_network_threshold,
        spikes_burst_threshold=roi_data.spikes_burst_threshold,
        spikes_burst_min_duration=roi_data.spikes_burst_min_duration,
        spikes_burst_gaussian_sigma=roi_data.spikes_burst_gaussian_sigma,
        # Masks
        mask_coords_y=mask_y,
        mask_coords_x=mask_x,
        mask_height=mask_h,
        mask_width=mask_w,
        neuropil_mask_coords_y=neuropil_y,
        neuropil_mask_coords_x=neuropil_x,
        neuropil_mask_height=neuropil_h,
        neuropil_mask_width=neuropil_w,
    )
