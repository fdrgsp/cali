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

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Self, cast

import numpy as np
import useq
from pydantic import BaseModel
from sqlalchemy import TypeDecorator, UniqueConstraint
from sqlalchemy.orm import selectinload
from sqlmodel import (
    JSON,
    Column,
    Field,
    Relationship,
    Session,
    SQLModel,
    create_engine,
    desc,
    select,
)

from cali._constants import (
    DEFAULT_BURST_GAUSS_SIGMA,
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_CCG_N_SHUFFLES,
    DEFAULT_DFF_PERCENTILE,
    DEFAULT_DFF_WINDOW,
    DEFAULT_ENABLE_RISING_EDGE_ANALYSIS,
    DEFAULT_FRAME_RATE,
    DEFAULT_HEIGHT,
    DEFAULT_SPIKE_SYNC_JITTER_WINDOW,
    DEFAULT_SPIKE_SYNCHRONY_MAX_LAG,
    DEFAULT_SPIKE_THRESHOLD,
    MULTIPLIER,
    SPONTANEOUS,
)
from cali.readers._tiff_collection_reader import TiffCollectionSettings

if TYPE_CHECKING:
    from cali.readers._ome_zarr_reader import OMEZarrReader
    from cali.readers._tensorstore_zarr_reader import TensorstoreZarrReader

# ==================== Custom Column Types ====================


class PydanticJSON(TypeDecorator):
    """Custom SQLAlchemy type for storing Pydantic models as JSON.

    Automatically serializes Pydantic models to JSON when saving to database
    and deserializes JSON back to Pydantic models when loading.
    """

    impl = JSON
    cache_ok = True

    def __init__(
        self, pydantic_type: type[BaseModel], *args: Any, **kwargs: Any
    ) -> None:
        """Initialize with the Pydantic model type to deserialize to."""
        self.pydantic_type = pydantic_type
        super().__init__(*args, **kwargs)

    def process_bind_param(
        self, value: BaseModel | None, dialect: Any
    ) -> dict[str, Any] | None:
        """Convert Pydantic model to dict when saving to database."""
        if value is None:
            return None
        # Use model_dump() for Pydantic v2
        return value.model_dump(mode="json")

    def process_result_value(
        self, value: dict[str, Any] | None, dialect: Any
    ) -> BaseModel | None:
        """Convert dict back to Pydantic model when loading from database."""
        if value is None:
            return None
        # Reconstruct the Pydantic model from the dict
        return self.pydantic_type.model_validate(value)


# ==================== Core Models ====================


class CaliResult(SQLModel, table=True):
    """Cali run metadata.

    Tracks which experiment was analyzed with which settings and which positions
    were processed. The actual results (traces, data_analysis) are linked via
    the analysis_result_id foreign key in those tables.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    created_at : datetime
        Timestamp when analysis was created
    experiment : int
        Foreign key to experiment
    detection_settings : int | None
        Foreign key to detection settings used
    extraction_settings_id: int | None
        Foreign key to extraction settings used (None for detection-only runs)
    analysis_settings_id: int | None
        Foreign key to analysis settings used (None for extraction-only runs)
    positions_detected : list[int] | None
        List of position indices that completed detection
    positions_extracted : list[int] | None
        List of position indices that completed extraction
    positions_analyzed : list[int] | None
        List of position indices that completed analysis
    traces : list[Traces]
        All trace results from this analysis run
    data_analysis_results : list[DataAnalysis]
        All analysis results from this analysis run
    """

    __tablename__ = "analysis_result"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)

    # Foreign keys
    experiment: int = Field(foreign_key="experiment.id")
    detection_settings_id: int | None = Field(
        default=None, foreign_key="detection_settings.id"
    )
    extraction_settings_id: int | None = Field(
        default=None, foreign_key="extraction_settings.id"
    )
    analysis_settings_id: int | None = Field(
        default=None, foreign_key="analysis_settings.id"
    )

    # Progressive tracking of pipeline stages
    positions_detected: list[int] | None = Field(default=None, sa_column=Column(JSON))
    positions_extracted: list[int] | None = Field(default=None, sa_column=Column(JSON))
    positions_analyzed: list[int] | None = Field(default=None, sa_column=Column(JSON))

    # Relationships
    traces: list["Traces"] = Relationship(back_populates="analysis_result")
    data_analysis_results: list["DataAnalysis"] = Relationship(
        back_populates="analysis_result"
    )
    fov_analysis_results: list["FOVAnalysis"] = Relationship(
        back_populates="analysis_result"
    )

    def __eq__(self, other: object) -> bool:
        """Custom equality that excludes created_at for semantic comparison.

        Two CaliResults are considered equal if they have the same:
        - experiment, detection_settings, extraction_settings,
          analysis_settings, positions_detected, positions_extracted,
          positions_analyzed

        The created_at field is excluded since it's automatically generated
        and doesn't represent semantic differences in analysis configuration.
        """
        if not isinstance(other, CaliResult):
            return False
        return (
            self.experiment == other.experiment
            and self.detection_settings_id == other.detection_settings_id
            and self.extraction_settings_id == other.extraction_settings_id
            and self.analysis_settings_id == other.analysis_settings_id
            and self.positions_detected == other.positions_detected
            and self.positions_extracted == other.positions_extracted
            and self.positions_analyzed == other.positions_analyzed
        )

    def __hash__(self) -> int:
        """Custom hash that excludes created_at for consistency with __eq__.

        Note: id is excluded since it's None before database insertion.
        """
        return hash(
            (
                self.experiment,
                self.detection_settings_id,
                self.extraction_settings_id,
                self.analysis_settings_id,
                tuple(self.positions_detected) if self.positions_detected else None,
                tuple(self.positions_extracted) if self.positions_extracted else None,
                tuple(self.positions_analyzed) if self.positions_analyzed else None,
            )
        )

    @classmethod
    def load_from_database(
        cls,
        db_path: str | Path,
        id: int | None = None,
        experiment_id: int | None = None,
        session: Session | None = None,
        load_data: bool = True,
    ) -> Self | list[Self]:
        """Load analysis result(s) from database with related settings.

        Parameters
        ----------
        db_path : str | Path
            Path to the SQLite database file
        id : int | None
            ID of specific analysis result to load. If None, loads based on
            experiment_id or all results.
        experiment_id : int | None
            Filter by experiment ID. If None and id is None, loads all results.
        session : Session | None
            Optional existing session to use. If None, creates a new one.
        load_data : bool
            Whether to load heavy data (traces, analysis results).
            Defaults to True for backward compatibility.

        Returns
        -------
        Self | list[Self]
            Single CaliResult if id specified, otherwise list of results.
            All instances are detached from session.

        Examples
        --------
        >>> # Load specific analysis result
        >>> result = CaliResult.load_from_database("path/to/db.db", id=1)
        >>> print(result.analysis_settings_obj.dff_window)
        >>>
        >>> # Load all results for an experiment
        >>> results = CaliResult.load_from_database("path/to/db.db", experiment_id=1)
        >>> for r in results:
        ...     print(f"Analysis {r.id}: {r.positions_analyzed}")
        >>>
        >>> # Load most recent analysis result
        >>> results = CaliResult.load_from_database("path/to/db.db")
        >>> latest = results[-1]  # Ordered by id (creation order)
        """
        if session is None:
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            our_session = session = Session(engine)
        else:
            our_session = None

        try:
            # Build query with eager loading of settings
            statement = select(cls)
            if load_data:
                statement = statement.options(
                    selectinload(cls.traces),
                    selectinload(cls.data_analysis_results),
                )

            # Filter by id or experiment_id
            if id is not None:
                statement = statement.where(cls.id == id)
                obj = session.exec(statement).first()
                if obj is None:
                    raise ValueError(f"No CaliResult found with id={id}")
                session.expunge_all()
                return obj  # type: ignore
            elif experiment_id is not None:
                statement = statement.where(cls.experiment == experiment_id)

            # Order by creation time to get most recent first
            statement = statement.order_by(desc(cls.created_at))

            results = list(session.exec(statement).all())
            session.expunge_all()
            return results

        finally:
            if our_session is not None:
                our_session.close()
                engine.dispose(close=True)


class Experiment(SQLModel, table=True):
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
    tiff_file_map_json : str | None
        JSON-serialized file_map for TiffCollectionReader
        Format: {"A1": ["path1.tif", "path2.tif"], "A2": [...], ...}
    tiff_plate_type : str | None
        Plate type for TiffCollectionReader (e.g., "96-well", "coverslip-22mm-square")
    tiff_metadata_json : str | None
        JSON-serialized metadata for TiffCollectionReader
        Must include "exposure_ms" and "pixel_size_um"
    plate : Plate
        Related plate (back-populated by SQLModel)
    """

    __tablename__ = "experiment"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)
    name: str = Field(unique=True, index=True)
    description: str | None = None

    # TIFF collection configuration (for TiffCollectionReader)
    tiff_file_map_json: str | None = None
    tiff_plate_type: str | None = None
    tiff_metadata_json: str | None = None

    # Relationships
    plate: "Plate" = Relationship(back_populates="experiment")

    def __eq__(self, other: object) -> bool:
        """Custom equality that compares by ID or semantic fields.

        Two Experiments are considered equal if:
        1. Both have IDs and they match (same database record), OR
        2. They have the same name (semantic match)

        The description field is excluded since it can change without
        affecting the experiment's identity.
        """
        if not isinstance(other, Experiment):
            return False
        # If both have IDs, compare by ID
        if self.id is not None and other.id is not None:
            return self.id == other.id
        # Otherwise compare by semantic fields
        return self.name == other.name

    def __hash__(self) -> int:
        """Custom hash based on ID for consistency with __eq__."""
        if self.id is None:
            return hash(id(self))  # Fallback to object identity
        return hash(self.id)

    @classmethod
    def load_from_database(
        cls,
        db_path: str | Path,
        id: int | None = None,
        session: Session | None = None,
        load_data: bool = True,
    ) -> Self:
        """Load experiment from database with all relationships eagerly loaded.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file
        id : int | None
            ID of the experiment to load. If None, loads the first experiment.
        session : Session | None
            Optional existing session to use. If None, creates a new one.
        load_data : bool
            Whether to load heavy data (traces, masks, analysis results).
            Defaults to True.

        Returns
        -------
        Self
            Experiment instance with all relationships loaded and detached
        """
        if session is None:
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            our_session = session = Session(engine)
        else:
            our_session = None

        try:
            # Load experiment with all relationships eagerly loaded
            if load_data:
                # Build the base chain for plate -> wells -> fovs -> rois
                plate_chain = (
                    selectinload(Experiment.plate)
                    .selectinload(Plate.wells)
                    .selectinload(Well.fovs)
                    .selectinload(FOV.rois)
                )
                statement = select(Experiment).options(
                    plate_chain.selectinload(ROI.traces_history).selectinload(
                        Traces.neuropil_mask
                    ),
                    plate_chain.selectinload(ROI.data_analysis_history),
                    plate_chain.selectinload(ROI.roi_mask),
                )
            else:
                # Only load plate and wells (skip FOVs and ROIs)
                plate_chain = selectinload(Experiment.plate).selectinload(Plate.wells)
                statement = select(Experiment).options(
                    plate_chain.selectinload(Well.conditions)
                )

            # Filter by ID if provided, otherwise get first experiment
            if id is not None:
                statement = statement.where(Experiment.id == id)

            obj = session.exec(statement).first()
            session.expunge_all()  # Detach all instances from the session
            return obj  # type: ignore
        finally:
            if our_session is not None:
                our_session.close()
                engine.dispose(close=True)

    @classmethod
    def create(
        cls,
        name: str,
        plate_type: str = "96-well",
        well_names: list[str] | None = None,
        fovs_per_well: int = 1,
        plate_maps: dict[str, dict[str, str]] | None = None,
        description: str | None = None,
        tiff_file_map: Mapping[str, Sequence[str | Path]] | None = None,
        tiff_plate_type: str | None = None,
        tiff_metadata: dict[str, Any] | None = None,
    ) -> Self:
        """Create a new experiment with plate structure ready for analysis.

        This is a convenience method that creates an Experiment with an associated
        Plate and Wells structure, making it easy to set up a new experiment database.

        Parameters
        ----------
        name : str
            Experiment name (must be unique)
        # data_path : str
        #     Path to the raw imaging data (zarr/tensorstore)
        plate_type : str, optional
            Plate format from useq-schema options (e.g., "96-well", "384-well",
            "24-well", "6-well", "coverslip-18mm-square"), by default "96-well"
        well_names : list[str] | None, optional
            List of well names to create (e.g., ["A1", "A2", "B5"]). If None,
            creates all wells in the plate format, by default None
        fovs_per_well : int, optional
            Number of FOV (Field of View) positions per well, by default 1
        plate_maps : dict[str, dict[str, str]] | None, optional
            Plate map configuration mapping well positions to conditions.
            Format: {"genotype": {"A1": "WT", "A2": "KO"},
                     "treatment": {"A1": "Vehicle", "A2": "Drug"}}, by default None
        description : str | None, optional
            Optional experiment description, by default None
        tiff_file_map : Mapping[str, Sequence[str | Path]] | None, optional
            Dictionary representing file_map for TIFF collection, by default None
        tiff_plate_type : str | None, optional
            Plate type for TIFF files (from useq-schema database), by default None
        tiff_metadata : dict[str, Any] | None, optional
            Dictionary with TIFF metadata, by default None

        Returns
        -------
        Self
            Experiment instance with plate and wells ready for detection/analysis

        Example
        -------
        >>> from cali.sqlmodel import Experiment
        >>> from cali._constants import EVOKED
        >>>
        >>> # Create experiment with specific wells
        >>> exp = Experiment.create_with_plate(
        ...     name="My Experiment",
        ...     plate_type="96-well",
        ...     well_names=["B5", "B6", "C5"],
        ...     fovs_per_well=2,
        ...     plate_maps={
        ...         "genotype": {"B5": "WT", "B6": "KO", "C5": "WT"},
        ...         "treatment": {"B5": "Vehicle", "B6": "Vehicle", "C5": "Drug"},
        ...     },
        ... )
        >>> print(f"Created experiment with {len(exp.plate.wells)} wells")
        """
        from ._json_to_db import parse_well_name
        from ._useq_plate_to_db import _row_index_to_label, useq_plate_plan_to_db

        # Create experiment
        experiment = cls(
            name=name,
            description=description,
            tiff_file_map_json=(None if tiff_file_map is None else str(tiff_file_map)),
            tiff_plate_type=tiff_plate_type,
            tiff_metadata_json=(None if tiff_metadata is None else str(tiff_metadata)),
        )

        # Create useq WellPlate and WellPlatePlan
        useq_plate = useq.WellPlate.from_str(plate_type)

        # If no well names specified, use all wells in plate
        if well_names is None:
            # Create all possible wells for this plate type
            well_names = [
                f"{_row_index_to_label(row)}{col + 1}"
                for row in range(useq_plate.rows)
                for col in range(useq_plate.columns)
            ]

        # Convert well names to row,col tuples for WellPlatePlan
        selected_wells_list = [parse_well_name(well_name) for well_name in well_names]

        # Convert to tuple of tuples format: ((rows...), (cols...))
        rows_tuple = tuple(well[0] for well in selected_wells_list)
        cols_tuple = tuple(well[1] for well in selected_wells_list)

        # Create WellPlatePlan with selected wells
        plate_plan = useq.WellPlatePlan(
            plate=useq_plate,
            a1_center_xy=(0, 0),  # Placeholder, not used for structure
            selected_wells=(rows_tuple, cols_tuple),
            well_points_plan=useq.RandomPoints(num_points=fovs_per_well),
        )

        # Create plate with wells and conditions
        # useq_plate_plan_to_db will create all FOVs based on the plate_plan's
        # well_points_plan (which has num_points=fovs_per_well)
        useq_plate_plan_to_db(plate_plan, experiment, plate_maps=plate_maps)

        return experiment

    @classmethod
    def create_from_data(
        cls,
        name: str,
        data_path: str | Path,
        plate_maps: dict[str, dict[str, str]] | None = None,
        description: str | None = None,
        tiff_file_map: Mapping[str, Sequence[str | Path]] | None = None,
        tiff_plate_type: str | None = None,
        tiff_metadata: dict[str, Any] | None = None,
        plate_plan: useq.WellPlatePlan | None = None,
    ) -> Self:
        """Create a new experiment by loading plate structure from data's useq metadata.

        This method automatically extracts the plate configuration (wells, FOVs)
        from the imaging data's useq metadata, making it ideal for datasets that
        already contain plate information.

        For TIFF collections, provide tiff_file_map, tiff_plate_type, and
        tiff_metadata to create a TiffCollectionReader that will be validated
        against data_path.

        Parameters
        ----------
        name : str
            Experiment name (must be unique)
        data_path : str | Path
            Path to the raw imaging data (zarr/tensorstore or TIFF collection)
        plate_maps : dict[str, dict[str, str]] | None, optional
            Plate map configuration mapping well positions to conditions.
            Format: {"genotype": {"A1": "WT", "A2": "KO"},
                     "treatment": {"A1": "Vehicle", "A2": "Drug"}}, by default None
        description : str | None, optional
            Optional experiment description, by default None
        tiff_file_map : Mapping[str, Sequence[str | Path]] | None, optional
            For TIFF collections: mapping from well names to sequences of TIFF
            file paths. Format: {"A1": ["path1.tif", "path2.tif"], "A2": [...]}
            If provided, a TiffCollectionReader will be created.
        tiff_plate_type : str | None, optional
            For TIFF collections: plate type (e.g., "96-well", "coverslip-22mm-square")
        tiff_metadata : dict[str, Any] | None, optional
            For TIFF collections: metadata with "exposure_ms" and "pixel_size_um"
        plate_plan : useq.WellPlatePlan | None, optional
            Optional WellPlatePlan to override the one in the data metadata.
            Useful if the Tenbsorstore/OME-Zarr data lacks plate information.

        Returns
        -------
        Self
            Experiment instance with plate, wells, and FOVs loaded from data metadata

        Example
        -------
        >>> from cali.sqlmodel import Experiment
        >>>
        >>> # Create from zarr data
        >>> exp = Experiment.create_from_data(
        ...     name="My Experiment",
        ...     data_path="path/to/data.zarr",
        ...     plate_maps={"genotype": {"B5": "WT"}},
        ... )
        >>>
        >>> # If the Tensorstore/OME-Zarr data lacks plate info, provide plate_plan
        >>> from useq import WellPlatePlan, WellPlate
        >>> plate_plan = WellPlatePlan(
        ...     plate=WellPlate.from_str("96-well"),
        ...     a1_center_xy=(0, 0),
        ...     selected_wells=((0, 0), (0, 1)),  # e.g., A1 and A2
        ...     # if multiple fovs per well:
        ...     well_points_plan=useq.RandomPoints(num_points=2),
        ... )
        >>> exp = Experiment.create_from_data(
        ...     name="My Experiment with PlatePlan",
        ...     data_path="path/to/data.zarr",
        ...     plate_maps={"genotype": {"A1": "WT", "A2": "KO"}},
        ...     plate_plan=plate_plan,
        ... )
        >>>
        >>> # Create from TIFF collection
        >>> exp = Experiment.create_from_data(
        ...     name="TIFF Experiment",
        ...     data_path="/path/to/tiffs",
        ...     tiff_file_map={"A1": ["A1_fov1.tif", "A1_fov2.tif"]},
        ...     tiff_plate_type="96-well",
        ...     tiff_metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65},
        ... )
        """
        import json

        from cali.readers import TiffCollectionReader
        from cali.util import load_data_from_path

        from ._data_to_plate import data_to_plate

        tiff_settings: TiffCollectionSettings | None = None
        data: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader | None
        # If TIFF parameters provided, create TiffCollectionReader
        if (
            tiff_file_map is not None
            and tiff_plate_type is not None
            and tiff_metadata is not None
            and data_path is not None
        ):
            # Create TiffCollectionReader from provided parameters
            tiff_settings = TiffCollectionSettings(
                file_map=cast("dict[str, list[Path | str]]", tiff_file_map),
                plate=tiff_plate_type,
                metadata=tiff_metadata,
                tiff_folder_path=data_path,
            )
            data = TiffCollectionReader(tiff_settings)
        else:
            # Load data normally (zarr/tensorstore or TIFF without settings)
            data = load_data_from_path(data_path)
            if data is None:
                from cali._constants import OME_ZARR, WRITERS, ZARR_TESNSORSTORE

                msg = (
                    f"Failed to load data from {data_path}. "
                    "Ensure the path is correct and contains valid imaging data. \n"
                    f"❌ Unsupported file format! Currently, Only "
                    f"{WRITERS[ZARR_TESNSORSTORE][0]}, {WRITERS[OME_ZARR][0]} and "
                    "TiffCollectionReader are supported."
                )
                raise ValueError(msg)

            # make sure that if the plate plan is provided, the number of fovs per well
            # matches the data size
            if plate_plan is not None:
                selected_wells = len(plate_plan.selected_well_names)
                nfov = plate_plan.num_points_per_well
                # selected_wells * nfov should equal the number of positions in the data
                assert data.sequence is not None
                npos = len(data.sequence.stage_positions)
                if selected_wells * nfov != npos:
                    raise ValueError(
                        f"Provided plate_plan has {selected_wells} selected wells "
                        f"with {nfov} FOVs each, totaling "
                        f"{selected_wells * nfov} positions, but the data at "
                        f"{data_path} has {npos} positions. Please ensure the "
                        "plate_plan matches the data."
                    )

        # Create experiment
        experiment = cls(
            name=name,
            description=description,
        )

        # If data is TiffCollectionReader, save its configuration
        if isinstance(data, TiffCollectionReader):
            file_map, plate_type, metadata = data.to_experiment_tiff_config()
            experiment.tiff_file_map_json = json.dumps(file_map)
            experiment.tiff_plate_type = plate_type
            experiment.tiff_metadata_json = json.dumps(metadata)

        # Load plate structure from data
        plate = data_to_plate(data, experiment, plate_maps, plate_plan)
        if plate is None:
            raise ValueError(
                f"Failed to load plate structure from data at {data_path}. "
                "Ensure the data contains valid useq metadata with WellPlatePlan."
            )
        # Assign plate to experiment
        experiment.plate = plate

        return experiment

    def tiff_collection_settings(
        self, data_path: str | Path
    ) -> TiffCollectionSettings | None:
        """Return TiffCollectionSettings if this experiment has TIFF configs."""
        if (
            self.tiff_file_map_json is None
            or self.tiff_plate_type is None
            or self.tiff_metadata_json is None
        ):
            return None

        file_map = json.loads(self.tiff_file_map_json)
        metadata = json.loads(self.tiff_metadata_json)

        return TiffCollectionSettings(
            file_map=file_map,
            plate=self.tiff_plate_type,
            metadata=metadata,
            tiff_folder_path=data_path,
        )


class DetectionSettings(SQLModel, table=True):
    """Detection/segmentation parameter settings.

    Stores the detection parameters used for cell segmentation.
    Currently supports Cellpose parameters. CaImAn support coming soon.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    created_at : datetime
        When these settings were created
    method : str
        Detection method (e.g. "cellpose")
    model_type : str
        Cellpose model type ("cpsam", "cyto3", "custom", etc.)
    custom_model : str | None
        Path to custom Cellpose model (only used when model_type is "custom")
    diameter : float | None
        Expected cell diameter in pixels (None for auto-detection)
    cellprob_threshold : float
        Cell probability threshold (0-1)
    flow_threshold : float
        Flow error threshold for quality control
    min_size : int
        Minimum cell size in pixels
    normalize : bool
        Whether to normalize images before detection
    batch_size : int
        Number of images to process per batch. By default, 8.
    """

    __tablename__ = "detection_settings"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)

    # Detection method
    method: str = Field(default="cellpose", index=True)

    # Cellpose settings
    model_type: str = "cpsam"
    custom_model: str | None = None
    diameter: float | None = None
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    min_size: int = 10
    normalize: bool = True
    batch_size: int = 8

    # TODO: add CaImAn settings

    def __eq__(self, other: object) -> bool:
        """Custom equality that excludes id and created_at for semantic comparison.

        Two DetectionSettings are considered equal if they have the same detection
        parameters, regardless of when they were created or their database IDs.
        """
        if not isinstance(other, DetectionSettings):
            return False
        return (
            self.method == other.method
            and self.model_type == other.model_type
            and self.custom_model == other.custom_model
            and self.diameter == other.diameter
            and self.cellprob_threshold == other.cellprob_threshold
            and self.flow_threshold == other.flow_threshold
            and self.min_size == other.min_size
            and self.normalize == other.normalize
            and self.batch_size == other.batch_size
        )

    def __hash__(self) -> int:
        """Custom hash that excludes id and created_at for consistency with __eq__."""
        return hash(
            (
                self.method,
                self.model_type,
                self.custom_model,
                self.diameter,
                self.cellprob_threshold,
                self.flow_threshold,
                self.min_size,
                self.normalize,
                self.batch_size,
            )
        )

    @classmethod
    def load_from_database(
        cls,
        db_path: str | Path,
        id: int | None = None,
        method: str | None = None,
        session: Session | None = None,
    ) -> Self | list[Self]:
        """Load detection settings from database.

        Parameters
        ----------
        db_path : str | Path
            Path to the SQLite database file
        id : int | None
            ID of specific detection settings to load. If None, loads based on
            method or all settings.
        method : str | None
            Filter by detection method ("cellpose"). If None and
            id is None, loads all settings.
        session : Session | None
            Optional existing session to use. If None, creates a new one.

        Returns
        -------
        Self | list[Self]
            Single DetectionSettings if id specified, otherwise list of settings.
            All instances are detached from session.

        Examples
        --------
        >>> # Load specific detection settings
        >>> settings = DetectionSettings.load_from_database("db.db", id=1)
        >>> print(settings.model_type)
        >>>
        >>> # Load all cellpose settings
        >>> cellpose_settings = DetectionSettings.load_from_database(
        ...     "db.db", method="cellpose"
        ... )
        >>>
        >>> # Load most recent settings
        >>> all_settings = DetectionSettings.load_from_database("db.db")
        >>> latest = all_settings[-1]
        """
        if session is None:
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            our_session = session = Session(engine)
        else:
            our_session = None

        try:
            statement = select(cls)

            # Filter by id or method
            if id is not None:
                statement = statement.where(cls.id == id)
                obj = session.exec(statement).first()
                if obj is None:
                    raise ValueError(f"No DetectionSettings found with id={id}")
                session.expunge_all()
                return obj  # type: ignore
            elif method is not None:
                statement = statement.where(cls.method == method)

            # Order by creation time to get most recent first
            statement = statement.order_by(desc(cls.created_at))

            results = list(session.exec(statement).all())
            session.expunge_all()
            return results

        finally:
            if our_session is not None:
                our_session.close()
                engine.dispose(close=True)


class ExtractionSettings(SQLModel, table=True):
    """Trace extraction parameter settings.

    Stores the trace extraction parameters used for a specific extraction run.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    created_at : datetime
        When these settings were created
    method : str
        Extraction method. "cali" by default.
    neuropil_inner_radius : int
        Inner radius for neuropil mask (pixels)
    neuropil_min_pixels : int
        Minimum pixels required for neuropil mask
    neuropil_correction_factor : float
        Neuropil correction factor (0-1)
    decay_constant : float
        Decay constant for deconvolution (seconds)
    dff_window : int
        Window size for ΔF/F baseline calculation (seconds)
    dff_percentile : int
        Percentile for ΔF/F baseline calculation (0-100, default: 10)
    frame_rate : float
        Acquisition frame rate (frames per second)
    threads : int
        Number of threads to use for analysis (default: 1)
    """

    __tablename__ = "extraction_settings"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)

    neuropil_inner_radius: int = 0
    neuropil_min_pixels: int = 0
    neuropil_correction_factor: float = 0.0

    decay_constant: float = 0.0
    dff_window: float = DEFAULT_DFF_WINDOW  # seconds
    dff_percentile: int = DEFAULT_DFF_PERCENTILE  # percentile for ΔF/F baseline
    frame_rate: float = Field(default=DEFAULT_FRAME_RATE)  # frames per second
    pixel_size: float | None = None  # pixel size in micrometers (µm)

    threads: int = Field(default=1)

    def __eq__(self, other: object) -> bool:
        """Custom equality that excludes id and created_at for semantic comparison.

        Two ExtractionSettings are considered equal if they have the same extraction
        parameters, regardless of when they were created or their database IDs.
        """
        if not isinstance(other, ExtractionSettings):
            return False
        return (
            self.neuropil_inner_radius == other.neuropil_inner_radius
            and self.neuropil_min_pixels == other.neuropil_min_pixels
            and self.neuropil_correction_factor == other.neuropil_correction_factor
            and self.decay_constant == other.decay_constant
            and self.dff_window == other.dff_window
            and self.dff_percentile == other.dff_percentile
            and self.frame_rate == other.frame_rate
            and self.pixel_size == other.pixel_size
            and self.threads == other.threads
        )

    def __hash__(self) -> int:
        """Custom hash that excludes id and created_at for consistency with __eq__."""
        return hash(
            (
                self.neuropil_inner_radius,
                self.neuropil_min_pixels,
                self.neuropil_correction_factor,
                self.decay_constant,
                self.dff_window,
                self.dff_percentile,
                self.frame_rate,
                self.pixel_size,
                self.threads,
            )
        )

    @classmethod
    def load_from_database(
        cls,
        db_path: str | Path,
        id: int | None = None,
        session: Session | None = None,
    ) -> Self | list[Self]:
        """Load extraction settings from database.

        Parameters
        ----------
        db_path : str | Path
            Path to the SQLite database file
        id : int | None
            ID of specific extraction settings to load. If None, loads all settings.
        session : Session | None
            Optional existing session to use. If None, creates a new one.

        Returns
        -------
        Self | list[Self]
            Single ExtractionSettings if id specified, otherwise list of settings.
            All instances are detached from session.

        Examples
        --------
        >>> # Load specific extraction settings
        >>> settings = ExtractionSettings.load_from_database("db.db", id=1)
        >>> print(settings.dff_window)
        >>>
        >>> # Load most recent settings
        >>> all_settings = ExtractionSettings.load_from_database("db.db")
        >>> latest = all_settings[-1]
        """
        if session is None:
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            our_session = session = Session(engine)
        else:
            our_session = None

        try:
            statement = select(cls)

            # Filter by id if provided
            if id is not None:
                statement = statement.where(cls.id == id)
                obj = session.exec(statement).first()
                if obj is None:
                    raise ValueError(f"No ExtractionSettings found with id={id}")
                session.expunge_all()
                return obj  # type: ignore

            # Order by creation time to get most recent first
            statement = statement.order_by(desc(cls.created_at))

            results = list(session.exec(statement).all())
            session.expunge_all()
            return results

        finally:
            if our_session is not None:
                our_session.close()
                engine.dispose(close=True)


class AnalysisSettings(SQLModel, table=True):
    """Analysis parameter settings for an experiment.

    Stores the analysis parameters used for a specific analysis run.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    created_at : datetime
        When these settings were created
    peaks_height_value : float
        Peak height threshold value
    peaks_height_mode : str
        Mode for peak height ("multiplier" or "absolute")
    peaks_distance : int
        Minimum distance between peaks (milliseconds)
    peaks_prominence_multiplier : float
        Multiplier for peak prominence threshold
    spike_threshold_value : float
        Spike detection threshold value
    spike_threshold_mode : str
        Mode for spike threshold ("multiplier" or "absolute")
    burst_threshold : float
        Threshold for burst detection (%)
    burst_min_duration : int
        Minimum burst duration (milliseconds)
    burst_gaussian_sigma : float
        Gaussian sigma for burst smoothing (seconds)
    spikes_sync_cross_corr_lag : int
        Max lag for spike cross-correlation (milliseconds)
    spikes_sync_jitter_window : int
        Jitter window for spike synchrony (milliseconds)
    ccg_n_shuffles : int
        Number of shuffles for CCG baseline correction
    enable_rising_edge_analysis : bool
        Whether to compute CCG on spike rising edges
    frame_rate : float
        Acquisition frame rate (frames per second)
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
    threads : int
        Number of threads to use for analysis (default: 1)
    experiment_id : int
        Foreign key to parent experiment
    stimulation_mask_id : int | None
        Foreign key to stimulation mask (if any)
    stimulation_mask : Mask | None
        Relationship to stimulation mask (if any)
    experiment_type : str
        Type of experiment ("spontaneous" or "evoked")
    stimulation_mask_path : str | None
        Path to stimulation mask file (if any)
    """

    __tablename__ = "analysis_settings"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)

    peaks_height_value: float = DEFAULT_HEIGHT
    peaks_height_mode: str = MULTIPLIER
    peaks_distance: float = 200.0  # milliseconds (2 frames at 10fps)
    peaks_prominence_multiplier: float = 1.0

    spike_threshold_value: float = DEFAULT_SPIKE_THRESHOLD
    spike_threshold_mode: str = MULTIPLIER
    burst_threshold: float = DEFAULT_BURST_THRESHOLD
    burst_min_duration: float = 3000.0  # milliseconds (3 seconds)
    burst_gaussian_sigma: float = DEFAULT_BURST_GAUSS_SIGMA
    calcium_burst_threshold: float = DEFAULT_BURST_THRESHOLD
    calcium_burst_min_duration: float = 3000.0  # milliseconds (3 seconds)
    calcium_burst_gaussian_sigma: float = DEFAULT_BURST_GAUSS_SIGMA
    spikes_sync_cross_corr_lag: float = DEFAULT_SPIKE_SYNCHRONY_MAX_LAG  # ms
    spikes_sync_jitter_window: float = DEFAULT_SPIKE_SYNC_JITTER_WINDOW  # ms
    ccg_n_shuffles: int = DEFAULT_CCG_N_SHUFFLES
    enable_rising_edge_analysis: bool = DEFAULT_ENABLE_RISING_EDGE_ANALYSIS

    frame_rate: float = Field(default=DEFAULT_FRAME_RATE)  # frames per second

    experiment_type: str = Field(default=SPONTANEOUS, index=True)
    stimulation_mask_path: str | None = None
    led_power_equation: str | None = None
    led_pulse_duration: float | None = None
    led_pulse_powers: list[float] | None = Field(default=None, sa_column=Column(JSON))
    led_pulse_on_frames: list[int] | None = Field(default=None, sa_column=Column(JSON))

    threads: int = Field(default=1)
    n_processes: int = Field(default=1)

    # Foreign keys
    stimulation_mask_id: int | None = Field(
        default=None, foreign_key="mask.id", index=True
    )

    # Relationships
    stimulation_mask: Optional["Mask"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[AnalysisSettings.stimulation_mask_id]",
            "lazy": "selectin",
        }
    )

    def __eq__(self, other: object) -> bool:
        """Custom equality that excludes id and created_at for semantic comparison.

        Two AnalysisSettings are considered equal if they have the same analysis
        parameters, regardless of when they were created or their database IDs.
        """
        if not isinstance(other, AnalysisSettings):
            return False
        return (
            self.peaks_height_value == other.peaks_height_value
            and self.peaks_height_mode == other.peaks_height_mode
            and self.peaks_distance == other.peaks_distance
            and self.peaks_prominence_multiplier == other.peaks_prominence_multiplier
            and self.spike_threshold_value == other.spike_threshold_value
            and self.spike_threshold_mode == other.spike_threshold_mode
            and self.burst_threshold == other.burst_threshold
            and self.burst_min_duration == other.burst_min_duration
            and self.burst_gaussian_sigma == other.burst_gaussian_sigma
            and self.calcium_burst_threshold == other.calcium_burst_threshold
            and self.calcium_burst_min_duration == other.calcium_burst_min_duration
            and self.calcium_burst_gaussian_sigma == other.calcium_burst_gaussian_sigma
            and self.spikes_sync_cross_corr_lag == other.spikes_sync_cross_corr_lag
            and self.spikes_sync_jitter_window == other.spikes_sync_jitter_window
            and self.ccg_n_shuffles == other.ccg_n_shuffles
            and self.enable_rising_edge_analysis == other.enable_rising_edge_analysis
            and self.frame_rate == other.frame_rate
            and self.led_power_equation == other.led_power_equation
            and self.led_pulse_duration == other.led_pulse_duration
            and self.led_pulse_powers == other.led_pulse_powers
            and self.led_pulse_on_frames == other.led_pulse_on_frames
            and self.experiment_type == other.experiment_type
            and self.stimulation_mask_path == other.stimulation_mask_path
            # and self.threads == other.threads
        )

    def __hash__(self) -> int:
        """Custom hash that excludes id and created_at for consistency with __eq__."""
        return hash(
            (
                self.peaks_height_value,
                self.peaks_height_mode,
                self.peaks_distance,
                self.peaks_prominence_multiplier,
                self.spike_threshold_value,
                self.spike_threshold_mode,
                self.burst_threshold,
                self.burst_min_duration,
                self.burst_gaussian_sigma,
                self.calcium_burst_threshold,
                self.calcium_burst_min_duration,
                self.calcium_burst_gaussian_sigma,
                self.spikes_sync_cross_corr_lag,
                self.spikes_sync_jitter_window,
                self.ccg_n_shuffles,
                self.enable_rising_edge_analysis,
                self.frame_rate,
                self.led_power_equation,
                self.led_pulse_duration,
                tuple(self.led_pulse_powers) if self.led_pulse_powers else None,
                tuple(self.led_pulse_on_frames) if self.led_pulse_on_frames else None,
                self.experiment_type,
                self.stimulation_mask_path,
                # self.threads,
            )
        )

    @classmethod
    def load_from_database(
        cls,
        db_path: str | Path,
        id: int | None = None,
        session: Session | None = None,
    ) -> Self | list[Self]:
        """Load analysis settings from database.

        Parameters
        ----------
        db_path : str | Path
            Path to the SQLite database file
        id : int | None
            ID of specific analysis settings to load. If None, loads all settings.
        session : Session | None
            Optional existing session to use. If None, creates a new one.

        Returns
        -------
        Self | list[Self]
            Single AnalysisSettings if id specified, otherwise list of settings.
            All instances are detached from session.

        Examples
        --------
        >>> # Load specific analysis settings
        >>> settings = AnalysisSettings.load_from_database("db.db", id=1)
        >>> print(settings.dff_window)
        >>>
        >>> # Load most recent settings
        >>> all_settings = AnalysisSettings.load_from_database("db.db")
        >>> latest = all_settings[-1]
        """
        if session is None:
            engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            our_session = session = Session(engine)
        else:
            our_session = None

        try:
            statement = select(cls)

            # Filter by id if provided
            if id is not None:
                statement = statement.where(cls.id == id)
                obj = session.exec(statement).first()
                if obj is None:
                    raise ValueError(f"No AnalysisSettings found with id={id}")
                session.expunge_all()
                return obj  # type: ignore

            # Order by creation time to get most recent first
            statement = statement.order_by(desc(cls.created_at))

            results = list(session.exec(statement).all())
            session.expunge_all()
            return results

        finally:
            if our_session is not None:
                our_session.close()
                engine.dispose(close=True)

    def stimulated_mask_area(self) -> np.ndarray | None:
        """Get stimulation mask as numpy array.

        Returns
        -------
        np.ndarray | None
            Binary mask of stimulated area, or None if no mask defined.
        """
        from cali.util import coordinates_to_mask

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
    plate_plan : useq.WellPlatePlan | None
        The useq-schema WellPlatePlan used for this plate
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
    plate_plan: useq.WellPlatePlan | None = Field(
        default=None, sa_column=Column(PydanticJSON(useq.WellPlatePlan))
    )
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
        Condition name (e.g., "WT", "KO", "Vehicle", "Drug_10uM")
        The combination of name + condition_type must be unique.
    condition_type : str
        Type of condition ("genotype", "treatment", "other")
    color : str | None
        Display color for plots (e.g., "coral", "#FF6347")
    description : str | None
        Optional detailed description
    """

    __tablename__ = "condition"
    __table_args__ = (
        UniqueConstraint("name", "condition_type", name="uq_condition_name_type"),
        {"sqlite_autoincrement": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
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
    FOVs can contain multiple ROIs (individual cells) from different detection runs.
    Each ROI tracks which detection created it via ROI.detection_settings_id.

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
        Regions of interest (cells) in this FOV (can be from multiple detection runs)
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
    fov_analysis_history: list["FOVAnalysis"] = Relationship(
        back_populates="fov", cascade_delete=True
    )


class ROI(SQLModel, table=True):  # type: ignore[call-arg]
    """Region of Interest (ROI) core metadata.

    Represents a single cell/neuron segmented from imaging data.
    Related analysis data is stored in separate tables (Traces, DataAnalysis, etc.)
    Each ROI can have multiple analysis results from different analysis runs.
    The detection method used is tracked at the ROI level via detection_settings_id.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    fov_id : int
        Foreign key to parent FOV
    detection_settings_id : int | None
        Foreign key to detection settings that created this ROI
    label_value : int
        ROI label number from segmentation (e.g., 1, 2, 3...)
    active : bool | None
        Whether ROI shows calcium activity (from latest analysis)
    stimulated : bool | None
        Whether ROI was stimulated (for evoked experiments)
    cell_size : float | None
        ROI area (units specified in cell_size_units)
    cell_size_units : str | None
        Units for cell_size ("µm²" or "pixels")
    roi_mask_id : int | None
        Foreign key to ROI mask
    fov : FOV
        Parent FOV
    traces_history : list[Traces]
        All fluorescence trace versions from different analysis runs
    data_analysis_history : list[DataAnalysis]
        All analysis result versions from different analysis runs
    roi_mask : Mask | None
        ROI mask (cell boundary)
    """

    __tablename__ = "roi"

    id: int | None = Field(default=None, primary_key=True)
    label_value: int = Field(index=True)

    active: bool | None = None
    stimulated: bool | None = None
    cell_size: float | None = None
    cell_size_units: str | None = None

    # Foreign keys
    fov_id: int = Field(foreign_key="fov.id", index=True, ondelete="CASCADE")
    detection_settings_id: int | None = Field(
        default=None, foreign_key="detection_settings.id", index=True
    )
    roi_mask_id: int | None = Field(default=None, foreign_key="mask.id", index=True)

    # Relationships
    fov: "FOV" = Relationship(back_populates="rois")
    traces_history: list["Traces"] = Relationship(
        back_populates="roi", cascade_delete=True
    )
    data_analysis_history: list["DataAnalysis"] = Relationship(
        back_populates="roi", cascade_delete=True
    )
    roi_mask: Optional["Mask"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[ROI.roi_mask_id]",
            "lazy": "selectin",
        }
    )


class Traces(SQLModel, table=True):  # type: ignore[call-arg]
    """Fluorescence trace data for an ROI.

    Stores all time-series fluorescence measurements and derived traces.
    Each ROI can have multiple trace versions from different analysis runs.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    created_at : datetime
        Timestamp when this trace was created
    roi_id : int | None
        Foreign key to parent ROI
    analysis_result_id : int | None
        Foreign key to the analysis run that created this trace
    neuropil_mask_id : int | None
        Foreign key to neuropil mask created for this analysis run
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
    inferred_spikes : list[float] | None
        Inferred spike trace (time-series of spike probabilities)
    x_axis : list[float] | None
        Frame numbers or frame timestamps (milliseconds)
    x_axis_units : str | None
        Units for x_axis ("frames" or "ms")
    roi : ROI
        Parent ROI
    analysis_result : CaliResult
        The analysis run that created this trace
    neuropil_mask : Mask | None
        Neuropil mask used for this analysis run
    """

    __tablename__ = "trace"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)

    raw_trace: list[float] | None = Field(default=None, sa_column=Column(JSON))
    corrected_trace: list[float] | None = Field(default=None, sa_column=Column(JSON))
    neuropil_trace: list[float] | None = Field(default=None, sa_column=Column(JSON))
    dff: list[float] | None = Field(default=None, sa_column=Column(JSON))
    dec_dff: list[float] | None = Field(default=None, sa_column=Column(JSON))
    inferred_spikes: list[float] | None = Field(default=None, sa_column=Column(JSON))
    x_axis: list[float] | None = Field(default=None, sa_column=Column(JSON))
    x_axis_units: str | None = Field(default=None)  # "frames" or "ms"

    # Foreign keys - roi_id is no longer unique to allow multiple versions
    roi_id: int | None = Field(
        default=None, foreign_key="roi.id", index=True, ondelete="CASCADE"
    )
    analysis_result_id: int | None = Field(
        default=None, foreign_key="analysis_result.id", index=True, ondelete="CASCADE"
    )
    neuropil_mask_id: int | None = Field(
        default=None, foreign_key="mask.id", index=True
    )

    # Relationships
    roi: "ROI" = Relationship(back_populates="traces_history")
    analysis_result: "CaliResult" = Relationship(back_populates="traces")
    neuropil_mask: Optional["Mask"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[Traces.neuropil_mask_id]",
            "lazy": "selectin",
        }
    )


class DataAnalysis(SQLModel, table=True):  # type: ignore[call-arg]
    """Container for data analysis results for an ROI.

    This class stores various analysis results related to an ROI,
    such as peak detection, spike inference, and cell size measurements.
    Each ROI can have multiple analysis result versions from different analysis runs.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    created_at : datetime
        Timestamp when this analysis was created
    roi_id : int | None
        Foreign key to parent ROI
    analysis_result_id : int | None
        Foreign key to the analysis run that created this result
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
    peaks_prominence_dec_dff : float | None
        Peak prominence threshold used for this ROI (calculated)
    peaks_height_dec_dff : float | None
        Peak height threshold used for this ROI (calculated)
    inferred_spikes_threshold : float | None
        Spike detection threshold used for this ROI (calculated)
    inferred_spikes_frequency : float | None
        Frequency of thresholded inferred spikes (Hz)
    inferred_spikes_rising_edge_frequency : float | None
        Frequency of thresholded inferred spikes rising edges (Hz)
    roi : ROI
        Parent ROI
    analysis_result : CaliResult
        The analysis run that created this result
    """

    __tablename__ = "data_analysis"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)

    # Foreign keys - roi_id is no longer unique to allow multiple versions
    roi_id: int | None = Field(
        default=None, foreign_key="roi.id", index=True, ondelete="CASCADE"
    )
    analysis_result_id: int | None = Field(
        default=None, foreign_key="analysis_result.id", index=True, ondelete="CASCADE"
    )

    total_recording_time_sec: float | None = None
    dec_dff_frequency: float | None = None
    peaks_dec_dff: list[float] | None = Field(default=None, sa_column=Column(JSON))
    peaks_prominence_dec_dff: float | None = None
    peaks_height_dec_dff: float | None = None
    peaks_amplitudes_dec_dff: list[float] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    iei: list[float] | None = Field(default=None, sa_column=Column(JSON))
    inferred_spikes_threshold: float | None = None
    inferred_spikes_frequency: float | None = None
    inferred_spikes_rising_edge_frequency: float | None = None

    # Relationships
    roi: "ROI" = Relationship(back_populates="data_analysis_history")
    analysis_result: "CaliResult" = Relationship(back_populates="data_analysis_results")


class FOVAnalysis(SQLModel, table=True):  # type: ignore[call-arg]
    """FOV-level analysis results (correlation and synchrony matrices).

    This class stores FOV-wide analysis metrics that describe relationships
    between ROIs, such as correlation and synchrony matrices. These are
    computed once during analysis and stored for efficient retrieval.

    Attributes
    ----------
    id : int | None
        Primary key, auto-generated
    created_at : datetime
        Timestamp when this analysis was created
    fov_id : int | None
        Foreign key to parent FOV
    analysis_result_id : int | None
        Foreign key to the analysis run that created this result
    active_roi_labels : list[int] | None
        Ordered list of active ROI label_values used for matrix indexing.
        Matrix[i,j] corresponds to ROIs active_roi_labels[i] and active_roi_labels[j].
    calcium_dff_correlation_matrix : list[list[float]] | None
        Zero-lag Pearson correlation on DF/F traces (NxN for N active ROIs)
    calcium_dec_dff_corr_matrix : list[list[float]] | None
        Zero-lag Pearson correlation on deconvolved DF/F traces (NxN for N active ROIs)
    spike_max_lag_correlation_matrix : list[list[float]] | None
        Max lag correlation on spike events (thresholded binary) (NxN for N active ROIs)
    global_spike_max_lag_correlation : float | None
        Median of off-diagonal spike max lag correlation values (thresholded binary)
    spike_max_lag_values_matrix : list[list[int]] | None
        Lag values (in frames) at max correlation for spike events
        (thresholded binary) (NxN).
        Positive lag means ROI_j lags behind ROI_i (i.e., i leads j).
    spike_max_lag_correlation_matrix_rising_edges : list[list[float]] | None
        Max lag correlation on spike events (thresholded rising edges)
        (NxN for N active ROIs)
    global_spike_max_lag_correlation_rising_edges : float | None
        Median of off-diagonal spike max lag correlation values
        (thresholded rising edges)
    spike_max_lag_values_matrix_rising_edges : list[list[int]] | None
        Lag values (in frames) at max correlation for spike events
        (thresholded rising edges) (NxN).
        Positive lag means ROI_j lags behind ROI_i (i.e., i leads j).
    spike_ccg_zscore_matrix : list[list[float]] | None
        Z-score matrix for CCG significance (thresholded binary).
        Z = (CCG_raw - baseline_mean) / baseline_std at max-lag position.
        |z| > 2 suggests significant functional connectivity.
    spike_ccg_zscore_matrix_rising_edges : list[list[float]] | None
        Z-score matrix for CCG significance (thresholded rising edges).
    spike_jitter_synchrony_matrix : list[list[float]] | None
        Jitter synchrony on spike events (thresholded binary) (NxN for N active ROIs)
    global_spike_jitter_synchrony : float | None
        Median of off-diagonal spike jitter synchrony values (thresholded binary)
    spike_jitter_synchrony_matrix_rising_edges : list[list[float]] | None
        Jitter synchrony on spike events (thresholded rising edges)
        (NxN for N active ROIs)
    global_spike_jitter_synchrony_rising_edges : float | None
        Median of off-diagonal spike jitter synchrony values
        (thresholded rising edges)
    spike_burst_count : int | None
        Number of spike-based population bursts detected
    spike_burst_avg_duration : float | None
        Average duration of spike bursts (seconds)
    spike_burst_avg_interval : float | None
        Average interval between spike bursts (seconds)
    spike_burst_starts : list[int] | None
        Frame indices where spike bursts start
    spike_burst_ends : list[int] | None
        Frame indices where spike bursts end (exclusive)
    spike_population_activity : list[float] | None
        Smoothed spike population activity (fraction of active ROIs, [0,1]).
        This is the smoothed trace used for burst detection and plotting.
    spike_population_activity_raw : list[float] | None
        Raw (unsmoothed) spike population activity (fraction of active ROIs, [0,1])
    calcium_burst_count : int | None
        Number of calcium-based population bursts detected
    calcium_burst_avg_duration : float | None
        Average duration of calcium bursts (seconds)
    calcium_burst_avg_interval : float | None
        Average interval between calcium bursts (seconds)
    calcium_burst_starts : list[int] | None
        Frame indices where calcium bursts start
    calcium_burst_ends : list[int] | None
        Frame indices where calcium bursts end (exclusive)
    calcium_population_activity : list[float] | None
        Smoothed mean calcium population activity (deconvolved ΔF/F, raw values).
        This is the smoothed trace used for burst detection and plotting.
    calcium_population_activity_raw : list[float] | None
        Raw (unsmoothed) mean calcium population activity (deconvolved ΔF/F, raw values)
    fov : FOV
        Parent FOV
    analysis_result : CaliResult
        The analysis run that created this result
    """

    __tablename__ = "fov_analysis"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)

    # Foreign keys
    fov_id: int | None = Field(
        default=None, foreign_key="fov.id", index=True, ondelete="CASCADE"
    )
    analysis_result_id: int | None = Field(
        default=None, foreign_key="analysis_result.id", index=True, ondelete="CASCADE"
    )

    # ROI ordering for matrix interpretation
    active_roi_labels: list[int] | None = Field(default=None, sa_column=Column(JSON))

    # Calcium peaks metrics (from dec_dff traces and peak events)
    # 0. Zero-lag correlation on DF/F traces
    calcium_dff_correlation_matrix: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    # 1. Zero-lag correlation on deconvolved DF/F traces
    calcium_dec_dff_corr_matrix: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    # Spike metrics (from inferred spikes)
    # 1. Max lag correlation on spikes (thresholded binary)
    spike_max_lag_correlation_matrix: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    global_spike_max_lag_correlation: float | None = None
    # 2a. Lag values at max correlation for spikes (thresholded binary)
    spike_max_lag_values_matrix: list[list[int]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    # 2b. Max lag correlation on spikes (thresholded rising edges)
    spike_max_lag_correlation_matrix_rising_edges: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    global_spike_max_lag_correlation_rising_edges: float | None = None
    # 2c. Lag values at max correlation for spikes (thresholded rising edges)
    spike_max_lag_values_matrix_rising_edges: list[list[int]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    # 2d. Z-score matrices for CCG significance (baseline-corrected)
    # Z-score = (CCG_raw - baseline_mean) / baseline_std at the max-lag position
    # |z| > 2 suggests significant functional connectivity
    spike_ccg_zscore_matrix: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    spike_ccg_zscore_matrix_rising_edges: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    # 3. Jitter synchrony on spikes (thresholded binary)
    spike_jitter_synchrony_matrix: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    global_spike_jitter_synchrony: float | None = None
    # 4. Jitter synchrony on spikes (thresholded rising edges)
    spike_jitter_synchrony_matrix_rising_edges: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    global_spike_jitter_synchrony_rising_edges: float | None = None

    # Population burst metrics (spike-based)
    spike_burst_count: int | None = None
    spike_burst_avg_duration: float | None = None
    spike_burst_avg_interval: float | None = None
    spike_burst_starts: list[int] | None = Field(default=None, sa_column=Column(JSON))
    spike_burst_ends: list[int] | None = Field(default=None, sa_column=Column(JSON))
    spike_population_activity: list[float] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    spike_population_activity_raw: list[float] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    # Population burst metrics (calcium-based)
    calcium_burst_count: int | None = None
    calcium_burst_avg_duration: float | None = None
    calcium_burst_avg_interval: float | None = None
    calcium_burst_starts: list[int] | None = Field(default=None, sa_column=Column(JSON))
    calcium_burst_ends: list[int] | None = Field(default=None, sa_column=Column(JSON))
    calcium_population_activity: list[float] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    calcium_population_activity_raw: list[float] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    # Relationships
    fov: "FOV" = Relationship(back_populates="fov_analysis_history")
    analysis_result: "CaliResult" = Relationship(back_populates="fov_analysis_results")


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
