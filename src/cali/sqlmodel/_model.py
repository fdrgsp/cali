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
from pathlib import Path
from typing import Any, Optional, Self

import numpy as np
from sqlalchemy.orm import selectinload
from sqlmodel import (
    JSON,
    Column,
    Field,
    Relationship,
    Session,
    SQLModel,
    create_engine,
    select,
)

from cali._constants import (
    DEFAULT_BURST_GAUSS_SIGMA,
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_CALCIUM_NETWORK_THRESHOLD,
    DEFAULT_CALCIUM_SYNC_JITTER_WINDOW,
    DEFAULT_DFF_WINDOW,
    DEFAULT_HEIGHT,
    DEFAULT_MIN_BURST_DURATION,
    DEFAULT_PEAKS_DISTANCE,
    DEFAULT_SPIKE_SYNCHRONY_MAX_LAG,
    DEFAULT_SPIKE_THRESHOLD,
    MULTIPLIER,
    SPONTANEOUS,
)

# ==================== Core Models ====================


class AnalysisResult(SQLModel, table=True):  # type: ignore[call-arg]
    """Analysis run metadata.

    Tracks which experiment was analyzed with which settings and which positions
    were processed. The actual results (traces, data_analysis) can be queried
    through the hierarchical relationships using the positions_analyzed list.
    """

    __tablename__ = "analysis_result"
    id: int | None = Field(default=None, primary_key=True)

    experiment: int = Field(foreign_key="experiment.id")
    analysis_settings: int = Field(foreign_key="analysis_settings.id")
    positions_analyzed: list[int] | None = Field(default=None, sa_column=Column(JSON))


class Experiment(SQLModel, table=True):  # type: ignore[call-arg]
    """Top-level experiment container.

    An experiment can contain a plate and tracks global metadata
    like creation date, description, and data paths.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    created_at : datetime
        Timestamp when experiment was created
    name : str
        Unique experiment identifier
    description : str | None
        Optional experiment description
    database_name: str
        Name of the SQLite database file
    data_path : str
        Path to the raw imaging data (zarr/tensorstore)
    labels_path : str
        Path to segmentation labels directory
    analysis_path : str
        Path to analysis output directory
    experiment_type : str
        Type of experiment: "Spontaneous Activity" or "Evoked Activity"
    plate : Plate
        Related plate (back-populated by SQLModel)
    """

    __tablename__ = "experiment"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)
    name: str = Field(unique=True, index=True)
    description: str | None = None
    data_path: str
    labels_path: str
    analysis_path: str
    database_name: str
    experiment_type: str = Field(default=SPONTANEOUS, index=True)

    # Relationships
    plate: "Plate" = Relationship(back_populates="experiment")

    @property
    def db_path(self) -> str:
        """Full path to the experiment's database file."""
        return str(Path(self.analysis_path) / self.database_name)

    @classmethod
    def load_from_db(
        cls, db_path: str, id: int, session: Session | None = None
    ) -> Self:
        """Load experiment from database with all relationships eagerly loaded.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file
        id : int
            ID of the experiment to load
        session : Session | None
            Optional existing session to use. If None, creates a new one.

        Returns
        -------
        Self
            Experiment instance with all relationships loaded and detached
        """
        if session is None:
            engine = create_engine(f"sqlite:///{db_path}")
            our_session = session = Session(engine)
        else:
            our_session = None

        try:
            # Build the base chain for plate -> wells -> fovs -> rois
            plate_chain = (
                selectinload(Experiment.plate)
                .selectinload(Plate.wells)
                .selectinload(Well.fovs)
                .selectinload(FOV.rois)
            )

            # Load experiment with all relationships eagerly loaded
            statement = (
                select(Experiment)
                .where(Experiment.id == id)
                .options(
                    plate_chain.selectinload(ROI.traces),
                    plate_chain.selectinload(ROI.data_analysis),
                    plate_chain.selectinload(ROI.roi_mask),
                    plate_chain.selectinload(ROI.neuropil_mask),
                )
            )

            obj = session.exec(statement).first()
            session.expunge_all()  # Detach all instances from the session
            return obj
        finally:
            if our_session is not None:
                our_session.close()


class AnalysisSettings(SQLModel, table=True):  # type: ignore[call-arg]
    """Analysis parameter settings for an experiment.

    Stores the analysis parameters used for a specific analysis run.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    experiment_id : int
        Foreign key to parent experiment
    created_at : datetime
        When these settings were created
    neuropil_inner_radius : int
        Inner radius for neuropil mask (pixels)
    neuropil_min_pixels : int
        Minimum pixels required for neuropil mask
    neuropil_correction_factor : float
        Neuropil correction factor (0-1)
    decay_constant : float
        Decay constant for deconvolution
    dff_window : int
        Window size for ΔF/F baseline calculation
    peaks_height_value : float
        Peak height threshold value
    peaks_height_mode : str
        Mode for peak height ("multiplier" or "absolute")
    peaks_distance : int
        Minimum distance between peaks (frames)
    peaks_prominence_multiplier : float
        Multiplier for peak prominence threshold
    calcium_sync_jitter_window : int
        Jitter window for calcium synchrony (frames)
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
    led_power_equation : str | None
        Equation for LED power calculation (evoked experiments)
    led_pulse_duration : float | None
        Duration of LED pulse (evoked experiments)
    led_pulse_powers : list[float] | None
        List of LED pulse powers (evoked experiments). Should have the same length
        as `led_pulse_on_frames`.
    led_pulse_on_frames : list[int] | None
        List of LED pulse on frames (evoked experiments). Should have the same length
        as `led_pulse_powers`.
    stimulation_mask_path : str | None
        Path to stimulation mask file (for GUI/reference)
    threads : int
        Number of threads to use for analysis (default: 1)
    stimulation_mask_id : int | None
        Foreign key to stimulation mask data
    stimulation_mask : Mask | None
        Stimulation mask data (spatial pattern of stimulation)
    experiment : Experiment
        Parent experiment
    """

    __tablename__ = "analysis_settings"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)

    neuropil_inner_radius: int = 0
    neuropil_min_pixels: int = 0
    neuropil_correction_factor: float = 0.0

    decay_constant: float = 0.0
    dff_window: int = DEFAULT_DFF_WINDOW

    peaks_height_value: float = DEFAULT_HEIGHT
    peaks_height_mode: str = MULTIPLIER
    peaks_distance: int = DEFAULT_PEAKS_DISTANCE
    peaks_prominence_multiplier: float = 1.0
    calcium_sync_jitter_window: int = DEFAULT_CALCIUM_SYNC_JITTER_WINDOW
    calcium_network_threshold: float = DEFAULT_CALCIUM_NETWORK_THRESHOLD

    spike_threshold_value: float = DEFAULT_SPIKE_THRESHOLD
    spike_threshold_mode: str = MULTIPLIER
    burst_threshold: float = DEFAULT_BURST_THRESHOLD
    burst_min_duration: int = DEFAULT_MIN_BURST_DURATION
    burst_gaussian_sigma: float = DEFAULT_BURST_GAUSS_SIGMA
    spikes_sync_cross_corr_lag: int = DEFAULT_SPIKE_SYNCHRONY_MAX_LAG

    led_power_equation: str | None = None
    led_pulse_duration: float | None = None
    led_pulse_powers: list[float] | None = Field(default=None, sa_column=Column(JSON))
    led_pulse_on_frames: list[int] | None = Field(default=None, sa_column=Column(JSON))
    stimulation_mask_path: str | None = None

    threads: int = Field(default=1)

    # Foreign keys
    # experiment_id: int | None = Field(
    #     default=None, foreign_key="experiment.id", index=True
    # )
    stimulation_mask_id: int | None = Field(
        default=None, foreign_key="mask.id", index=True
    )

    # Relationships
    # experiment: "Experiment" = Relationship(back_populates="analysis_settings")
    stimulation_mask: Optional["Mask"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[AnalysisSettings.stimulation_mask_id]",
            "lazy": "selectin",
        }
    )

    def stimulated_mask_area(self) -> np.ndarray | None:
        from cali.analysis._util import coordinates_to_mask

        if (
            (stim_mask := self.stimulation_mask)
            and stim_mask.coords_y is not None
            and stim_mask.coords_x is not None
            and stim_mask.height is not None
            and stim_mask.width is not None
        ):
            return coordinates_to_mask(
                (stim_mask.coords_y, stim_mask.coords_x),
                (stim_mask.height, stim_mask.width),
            )
        return None


class Plate(SQLModel, table=True):  # type: ignore[call-arg]
    """Plate container (e.g., 96-well plate).

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    experiment_id : int
        Foreign key to parent experiment
    name : str
        Plate name/identifier
    plate_type : str | None
        Plate format (e.g., "96-well", "384-well")
    rows : int | None
        Number of rows in plate
    columns : int | None
        Number of columns in plate
    plate_maps : dict | None
        Plate map configuration mapping well positions to conditions.
        Format: {"genotype": {"A1": "WT", "A2": "KO", ...},
                 "treatment": {"A1": "Vehicle", "A2": "Drug", ...}}
    experiment : Experiment
        Parent experiment
    wells : list[Well]
        Child wells in this plate
    """

    __tablename__ = "plate"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    plate_type: str | None = None  # e.g., "96-well", "384-well"
    rows: int | None = None
    columns: int | None = None
    plate_maps: dict[str, dict[str, str]] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    # Foreign keys
    experiment_id: int = Field(foreign_key="experiment.id", index=True)

    # Relationships
    experiment: "Experiment" = Relationship(back_populates="plate")
    wells: list["Well"] = Relationship(back_populates="plate")


class Condition(SQLModel, table=True):  # type: ignore[call-arg]
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


class WellCondition(SQLModel, table=True):  # type: ignore[call-arg]
    """Link table for Well-Condition many-to-many relationship."""

    __tablename__ = "well_condition_link"

    # Foreign keys
    well_id: int = Field(foreign_key="well.id", primary_key=True)
    condition_id: int = Field(foreign_key="condition.id", primary_key=True)


class Well(SQLModel, table=True):  # type: ignore[call-arg]
    """Well in a plate (e.g., "B5").

    A well can have multiple FOVs (imaging positions) and is associated
    with experimental conditions.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    plate_id : int
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
    name: str = Field(index=True)
    row: int = Field(index=True)
    column: int = Field(index=True)

    # Foreign keys
    plate_id: int = Field(foreign_key="plate.id", index=True)

    # Relationships
    plate: "Plate" = Relationship(back_populates="wells")
    conditions: list["Condition"] = Relationship(
        link_model=WellCondition,
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    fovs: list["FOV"] = Relationship(back_populates="well", cascade_delete=True)

    # properties for first and second conditions
    @property
    def condition_1(self) -> Optional["Condition"]:
        """First experimental condition (e.g., genotype)."""
        return self.conditions[0] if len(self.conditions) > 0 else None

    @property
    def condition_2(self) -> Optional["Condition"]:
        """Second experimental condition (e.g., treatment)."""
        return self.conditions[1] if len(self.conditions) > 1 else None


class FOV(SQLModel, table=True):  # type: ignore[call-arg]
    """Field of View (imaging position) within a well.

    Each FOV represents a single imaging position/site within a well.
    FOVs contain multiple ROIs (individual cells).

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    well_id : int | None
        Foreign key to parent well
    name : str
        FOV name (e.g., "B5_0000_p0")
    position_index : int
        Position index in acquisition order (e.g., if in an experiment we have 2 FOVs
        per well and this is the second well, second FOV, this index would be 3 - the
        4th position)
    fov_number : int
        The FOV number per well
    fov_metadata : dict | None
        Additional metadata from acquisition (stored as JSON)
    well : Well
        Parent well
    rois : list[ROI]
        Regions of interest (cells) in this FOV
    """

    __tablename__ = "fov"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(
        index=True
    )  # Not unique - multiple experiments can have same FOV names
    position_index: int = Field(index=True)
    fov_number: int = Field(default=0)
    fov_metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))

    well_id: Optional[int] = Field(
        default=None, foreign_key="well.id", index=True, ondelete="CASCADE"
    )

    # Relationships
    well: "Well" = Relationship(back_populates="fovs")
    rois: list["ROI"] = Relationship(back_populates="fov", cascade_delete=True)


class ROI(SQLModel, table=True):  # type: ignore[call-arg]
    """Region of Interest (ROI) core metadata.

    Represents a single cell/neuron segmented from imaging data.
    Related analysis data is stored in separate tables (Traces, DataAnalysis, etc.)

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    fov_id : int
        Foreign key to parent FOV
    label_value : int
        ROI label number from segmentation (e.g., 1, 2, 3...)
    active : bool | None
        Whether ROI shows calcium activity
    stimulated : bool
        Whether ROI was stimulated (for evoked experiments)
    analysis_settings_id : int | None
        Foreign key to analysis settings used
    roi_mask_id : int | None
        Foreign key to ROI mask
    neuropil_mask_id : int | None
        Foreign key to neuropil mask
    fov : FOV
        Parent FOV
    traces : Traces | None
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

    # Foreign keys
    fov_id: int = Field(foreign_key="fov.id", index=True, ondelete="CASCADE")
    roi_mask_id: int | None = Field(default=None, foreign_key="mask.id", index=True)
    neuropil_mask_id: int | None = Field(
        default=None, foreign_key="mask.id", index=True
    )
    analysis_settings_id: int | None = Field(
        default=None, foreign_key="analysis_settings.id", index=True
    )

    # Relationships
    fov: "FOV" = Relationship(back_populates="rois")
    traces: Optional["Traces"] = Relationship(back_populates="roi", cascade_delete=True)
    data_analysis: Optional["DataAnalysis"] = Relationship(
        back_populates="roi", cascade_delete=True
    )
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


class Traces(SQLModel, table=True):  # type: ignore[call-arg]
    """Fluorescence trace data for an ROI.

    Stores all time-series fluorescence measurements and derived traces.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    roi_id : int | None
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
        Frame numbers or frame timestamps (milliseconds)
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
        default=None, foreign_key="roi.id", index=True, unique=True, ondelete="CASCADE"
    )

    # Relationships
    roi: "ROI" = Relationship(back_populates="traces")


class DataAnalysis(SQLModel, table=True):  # type: ignore[call-arg]
    """Container for data analysis results for an ROI.

    This class stores various analysis results related to an ROI,
    such as peak detection, spike inference, and cell size measurements.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    roi_id : int | None
        Foreign key to parent ROI
    cell_size : float | None
        ROI area (µm² or pixels)
    cell_size_units : str | None
        Units for cell_size
    total_recording_time_sec : float | None
        Total recording duration (seconds)
    dec_dff_frequency : float | None
        Calcium event frequency (Hz)
    peaks_dec_dff : list[float] | None
        Peak indices in deconvolved trace
    peaks_amplitudes_dec_dff : list[float] | None
        Peak amplitudes
    iei : list[float] | None
        Inter-event intervals (seconds)
    inferred_spikes : list[float] | None
        Inferred spike probabilities
    peaks_prominence_dec_dff : float | None
        Peak prominence threshold used for this ROI (calculated)
    peaks_height_dec_dff : float | None
        Peak height threshold used for this ROI (calculated)
    inferred_spikes_threshold : float | None
        Spike detection threshold used for this ROI (calculated)
    roi : ROI
        Parent ROI
    """

    __tablename__ = "data_analysis"

    id: int | None = Field(default=None, primary_key=True)
    roi_id: int | None = Field(
        default=None, foreign_key="roi.id", index=True, unique=True, ondelete="CASCADE"
    )

    cell_size: float | None = None
    cell_size_units: str | None = None
    total_recording_time_sec: float | None = None
    dec_dff_frequency: float | None = None
    peaks_dec_dff: list[float] | None = Field(default=None, sa_column=Column(JSON))
    peaks_prominence_dec_dff: float | None = None
    peaks_height_dec_dff: float | None = None
    peaks_amplitudes_dec_dff: list[float] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    iei: list[float] | None = Field(default=None, sa_column=Column(JSON))
    inferred_spikes: list[float] | None = Field(default=None, sa_column=Column(JSON))
    inferred_spikes_threshold: float | None = None

    # Relationships
    roi: "ROI" = Relationship(back_populates="data_analysis")


class Mask(SQLModel, table=True):  # type: ignore[call-arg]
    """Generic mask coordinate data.

    Stores spatial coordinates and dimensions for a mask (ROI or neuropil).

    Attributes
    ----------
    id : int | None
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
        Type of mask ("roi", "neuropil", or "stimulation")
    """

    __tablename__ = "mask"

    id: int | None = Field(default=None, primary_key=True)

    coords_y: list[int] | None = Field(default=None, sa_column=Column(JSON))
    coords_x: list[int] | None = Field(default=None, sa_column=Column(JSON))
    height: int | None = None
    width: int | None = None
    mask_type: str = Field(index=True)  # "roi", "neuropil", or "stimulation"
