from pathlib import Path

from sqlmodel import create_engine, Session, select

from cali.runner import CaliRunner
from cali.analysis import AnalysisRunner
from cali.detection import DetectionRunner
from cali.sqlmodel import AnalysisSettings, Experiment, CaliResult
from cali.sqlmodel._model import DetectionSettings
from cali.sqlmodel._visualize_experiment import print_cali_results


cali = CaliRunner()

data_path = "tests/test_data/evoked/evk.tensorstore.zarr"

experiment = Experiment.create_from_data(
    data_path=data_path,
    name="New Experiment",
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
)

out = Path("/Users/fdrgsp/Desktop/cali_test")
db_name = "results.cali"

cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(method="cellpose", model_type="cyto3"),
    analysis_settings=AnalysisSettings(dff_window=150),
    global_position_indices=[0],
    output_path=out,
    database_name=db_name,
    overwrite=True,
)

cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(
        method="cellpose",
        model_type="custom",
        custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",
    ),
    analysis_settings=AnalysisSettings(dff_window=150),
    global_position_indices=[0],
    output_path=out,
    database_name=db_name,
)


engine = create_engine(f"sqlite:///{cali.database_path}")
print_cali_results(engine, show_settings=False)
