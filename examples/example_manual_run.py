"""
Example of how to use DetectionRunner, ExtractionRunner, and AnalysisRunner individually.

This script demonstrates how to run the pipeline components manually without
using the unified CaliRunner class. This gives you more control over the
execution flow, database interactions, and object lifecycle.

Each step saves results to the database using update_fovs_in_database(),
enabling flexible workflows and allowing custom runners to be plugged in.
"""

from pathlib import Path

from rich import print
from sqlmodel import Session, create_engine, func, select

from cali.analysis import AnalysisRunner
from cali.detection import DetectionRunner
from cali.extraction import ExtractionRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    DataAnalysis,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    Traces,
    save_experiment_to_database,
)
from cali.util import load_fovs_from_database, update_fovs_in_database
from cali.util._util import load_data

dataset_path = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt.tensorstore.zarr"
db_path = Path("manual_run.cali")

# Clean up previous run for this example
if db_path.exists():
    db_path.unlink()

# create and save a database with the experiment structure
exp = Experiment.create_from_data("manual_exp", dataset_path)
save_experiment_to_database(exp, db_path.parent, database_name=db_path.name)

# Create engine for database operations
engine = create_engine(
    f"sqlite:///{db_path}",
    connect_args={"timeout": 30.0, "check_same_thread": False},
    pool_pre_ping=True,
)

# set the positions (fovs) to process
# positions_to_process = [17, 18]
data = load_data(dataset_path)
positions_to_process = list(range(len(data.sequence.stage_positions)))

# detection -----------------------------------------------------------------------
detection_runner = DetectionRunner()
detection_settings = DetectionSettings(
    method="cellpose",
    model_type="custom",
    custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",
)
for fov in detection_runner.run(
    dataset=dataset_path,
    detection_settings=detection_settings,
    global_position_indices=positions_to_process,
    as_generator=True,
):
    update_fovs_in_database(db_path, fov)


# extraction ----------------------------------------------------------------------
extraction_runner = ExtractionRunner()
extraction_settings = ExtractionSettings(dff_window=150, threads=3)
for fov in extraction_runner.run(
    dataset=dataset_path,
    extraction_settings=extraction_settings,
    fovs=load_fovs_from_database(engine, positions_to_process),
    as_generator=True,
):
    update_fovs_in_database(db_path, fov)


# analysis ------------------------------------------------------------------------
analysis_runner = AnalysisRunner()
analysis_settings = AnalysisSettings(peaks_height_value=2, threads=3)
for fov in analysis_runner.run(
    load_fovs_from_database(engine, positions_to_process),
    analysis_settings=analysis_settings,
    as_generator=True,
):
    update_fovs_in_database(db_path, fov)


# Print summary of results
print("\n📊 Pipeline Results:")
with Session(engine) as session:
    for pos in positions_to_process:
        fov = session.exec(select(FOV).where(FOV.position_index == pos)).first()
        if fov:
            trace_count = session.exec(
                select(func.count(Traces.id))  # type: ignore
                .join(ROI)
                .where(ROI.fov_id == fov.id)
            ).one()
            analysis_count = session.exec(
                select(func.count(DataAnalysis.id))  # type: ignore
                .join(ROI)
                .where(ROI.fov_id == fov.id)
            ).one()
            print(
                f"{fov.name}: {len(fov.rois)} ROIs, {trace_count} Traces, {analysis_count} DataAnalysis"
            )

# Clean up engine
engine.dispose(close=True)
