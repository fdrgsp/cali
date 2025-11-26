from pathlib import Path
from sqlmodel import create_engine
from cali.runner import CaliRunner
from cali.sqlmodel import AnalysisSettings, DetectionSettings
from cali.sqlmodel._model import Experiment

runner = CaliRunner()

database_path = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"
dataset = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt.tensorstore.zarr"

exp = Experiment.create_from_data('exp', dataset)
detection_settings = DetectionSettings(
    method="cellpose",
    model_type="custom",
    custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",
)
# Unique settings to test
analysis_settings = AnalysisSettings(dff_window=180, threads=10, peaks_height_value=2.5)

runner.run(
    exp,
    dataset,
    detection_settings,
    analysis_settings=analysis_settings,
    global_position_indices=[17],  # Just one position for quick test
    output_path=Path(database_path).parent,
    database_name="results.cali",
    overwrite=False,
)
