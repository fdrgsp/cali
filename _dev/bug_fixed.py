



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
from cali.sqlmodel._visualize_experiment import print_cali_results

runner = CaliRunner()

database_name = "tiffs.cali"
database_path = f"/Users/fdrgsp/Desktop/cali_test/{database_name}"
dataset = "/Users/fdrgsp/Desktop/cali_test/tiffs"

engine = create_engine(f"sqlite:///{database_path}")
print_cali_results(engine, show_settings=False, max_experiment_level="well")

exp = Experiment.load_from_db(database_path, load_data=False)


# run detection + extraction + analysis on pos 0
detection_settings = DetectionSettings(
    method="cellpose",
    model_type="custom",
    custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",  # noqa: E501
)
extraction_settings = ExtractionSettings(dff_window=150, threads=3)
analysis_settings = AnalysisSettings(peaks_height_value=2, threads=3)
runner.run(
    exp,
    dataset,
    detection_settings,
    extraction_settings=extraction_settings,
    analysis_settings=analysis_settings,
    global_position_indices=[0],
    output_path=Path(database_path).parent,
    database_name=database_name,
    overwrite=True,
)

# run detection + extraction + analysis on pos 1 (using stored settings)
# should reuse settings from previous run
detection_settings = 1
extraction_settings = 1
analysis_settings = 1
runner.run(
    exp,
    dataset,
    detection_settings,
    extraction_settings=extraction_settings,
    analysis_settings=analysis_settings,
    global_position_indices=[0, 1],
    output_path=Path(database_path).parent,
    database_name=database_name,
)

# run detection + extraction on pos [0, 2] (using stored settings)
# should reuse settings from previous run
# should skip detection and extraction on pos 0 since already done
detection_settings = 1
extraction_settings = 1
analysis_settings = None
runner.run(
    exp,
    dataset,
    detection_settings,
    extraction_settings=extraction_settings,
    analysis_settings=analysis_settings,
    global_position_indices=[0, 2],
    output_path=Path(database_path).parent,
    database_name=database_name,
)

# run detection + extraction + analysis on pos 0 with different extraction settings
# should reuse detection and analysis settings from previous run
# should skip detection on pos 0 since already done but re-run extraction and analysis
# with new settings
# should create a nuw run
detection_settings = 1
extraction_settings = ExtractionSettings(dff_window=180, threads=3)
analysis_settings = 1
runner.run(
    exp,
    dataset,
    detection_settings,
    extraction_settings=extraction_settings,
    analysis_settings=analysis_settings,
    global_position_indices=[0],
    output_path=Path(database_path).parent,
    database_name=database_name,
)


# Print summary of results
engine = create_engine(f"sqlite:///{database_path}")
print_cali_results(engine, show_settings=False, max_experiment_level="well")

# Clean up engine
engine.dispose(close=True)
