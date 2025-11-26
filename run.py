from pathlib import Path

from sqlmodel import Session, create_engine, select

from cali.runner import CaliRunner
from cali.sqlmodel import AnalysisSettings, DetectionSettings, ExtractionSettings
from cali.sqlmodel._model import FOV, Experiment
from cali.sqlmodel._visualize_experiment import print_cali_results

runner = CaliRunner()

database_path = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"
dataset = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt.tensorstore.zarr"

exp = Experiment.create_from_data("exp", dataset)
detection_settings = DetectionSettings(
    method="cellpose",
    model_type="custom",
    custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",
)
extraction_settings = ExtractionSettings(dff_window=150, threads=3)
analysis_settings = AnalysisSettings(peaks_height_value=3)
runner.run(
    exp,
    dataset,
    detection_settings,
    extraction_settings=extraction_settings,
    analysis_settings=analysis_settings,
    global_position_indices=[17, 18],
    output_path=Path(database_path).parent,
    database_name="results.cali",
    overwrite=True,
)

# # run again
# analysis_settings = AnalysisSettings(peaks_height_value=2)
# runner.run(
#     exp,
#     dataset,
#     1,
#     extraction_settings=1,
#     analysis_settings=analysis_settings,
#     global_position_indices=[17, 18, 19],
#     output_path=Path(database_path).parent,
#     database_name="results.cali",
# )

# runner.run(
#     exp,
#     dataset,
#     1,
#     extraction_settings=1,
#     analysis_settings=1,
#     global_position_indices=[17, 0, 1],
#     output_path=Path(database_path).parent,
#     database_name="results.cali",
# )

engine = create_engine(f"sqlite:///{database_path}")
print_cali_results(engine, show_settings=False, max_experiment_level="fov")
