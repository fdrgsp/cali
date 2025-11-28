"""Example of how to run the full CALI pipeline."""

from pathlib import Path

from sqlmodel import Session, create_engine, func, select

from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    DataAnalysis,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    Traces,
)

runner = CaliRunner()

database_name = "results.cali"
database_path = f"tests/test_data/evoked/{database_name}"
dataset = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
positions_to_process = [0]

exp = Experiment.create_from_data("exp", dataset)
detection_settings = DetectionSettings(
    method="cellpose",
    model_type="cpsam",
    # custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",  # noqa: E501
)
runner.run(
    exp,
    dataset,
    detection_settings,
    global_position_indices=positions_to_process,
    output_path=Path(database_path).parent,
    database_name=database_name,
    overwrite=True,
)

extraction_settings = ExtractionSettings(dff_window=150, threads=3)
runner.run(
    exp,
    dataset,
    1,
    extraction_settings=extraction_settings,
    global_position_indices=positions_to_process,
    output_path=Path(database_path).parent,
    database_name=database_name,
)

analysis_settings = AnalysisSettings(peaks_height_value=2)
runner.run(
    exp,
    dataset,
    1,
    extraction_settings=1,
    analysis_settings=analysis_settings,
    global_position_indices=positions_to_process,
    output_path=Path(database_path).parent,
    database_name=database_name,
)

# Print summary of results
print("\n📊 Pipeline Results:")
engine = create_engine(f"sqlite:///{database_path}")
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
                f"{fov.name}: {len(fov.rois)} ROIs, {trace_count} Traces, "
                f"{analysis_count} DataAnalysis"
            )

# Clean up engine
engine.dispose(close=True)
