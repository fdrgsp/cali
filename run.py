from pathlib import Path

from sqlmodel import Session, create_engine, select

from cali.runner import CaliRunner
from cali.sqlmodel import AnalysisSettings, DetectionSettings
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
analysis_settings = AnalysisSettings(dff_window=150, threads=3, peaks_height_value=2)

runner.run(
    exp,
    dataset,
    detection_settings,
    analysis_settings=analysis_settings,
    global_position_indices=[17],
    output_path=Path(database_path).parent,
    database_name="results.cali",
    overwrite=True,
)

# run again
analysis_settings = AnalysisSettings(dff_window=150, threads=3, peaks_height_value=2)
runner.run(
    exp,
    dataset,
    1,
    analysis_settings=analysis_settings,
    global_position_indices=[18],
    output_path=Path(database_path).parent,
    database_name="results.cali",
)

# Add assertions
engine = create_engine(f"sqlite:///{database_path}")
with Session(engine) as session:
    for pos in [17, 18]:
        fov = session.exec(select(FOV).where(FOV.position_index == pos)).first()
        assert fov is not None, f"FOV for position {pos} not found"
        assert len(fov.rois) > 0, f"No ROIs found for FOV {pos}"

        for roi in fov.rois:
            assert len(roi.traces_history) > 0, f"No traces found for ROI {roi.id}"
            assert (
                len(roi.data_analysis_history) > 0
            ), f"No data analysis found for ROI {roi.id}"

            # Check if traces are populated
            trace = roi.traces_history[-1]
            assert trace.dff is not None
            assert trace.dec_dff is not None

            # Check data analysis
            da = roi.data_analysis_history[-1]
            assert da.peaks_dec_dff is not None

print("✅ All assertions passed!")

engine = create_engine(f"sqlite:///{database_path}")
print_cali_results(engine, show_settings=False, max_experiment_level="fov")
