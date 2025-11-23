"""Test that analysis skips already-processed positions."""

from pathlib import Path
from sqlmodel import create_engine, Session, select

from cali.runner import CaliRunner
from cali.sqlmodel import AnalysisSettings, Experiment, CaliResult
from cali.sqlmodel._model import DetectionSettings


def _table(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = session.exec(select(CaliResult).order_by(CaliResult.id)).all()

        print("\n" + "=" * 100)
        print("ALL ANALYSIS RESULTS")
        print("=" * 100)
        print(
            f"{'ID':<5} {'Exp ID':<8} {'Det ID':<8} {'Ana ID':<8} {'Positions':<30}"
        )
        print("-" * 100)

        for result in results:
            exp_id = str(result.experiment) if result.experiment else "None"
            det_id = (
                str(result.detection_settings) if result.detection_settings else "None"
            )
            ana_id = (
                str(result.analysis_settings) if result.analysis_settings else "None"
            )
            positions = (
                str(result.positions_analyzed) if result.positions_analyzed else "None"
            )

            print(f"{result.id:<5} {exp_id:<8} {det_id:<8} {ana_id:<8} {positions:<30}")

        print("=" * 100)
        print(f"Total: {len(results)}\n")


cali = CaliRunner()

data_path = "tests/test_data/spontaneous/spont.tensorstore.zarr"

experiment = Experiment.create_from_data(
    name="Test Skip Analysis",
    data_path=data_path,
)

detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")
analysis_settings = AnalysisSettings(dff_window=150)

print("\n" + "=" * 100)
print("RUN 1: Detection + Analysis on position 0")
print("=" * 100)
# Capture IDs before they get detached
det_settings_for_run2 = 1  # Will be ID 1 after first creation
ana_settings_for_run2 = 1  # Will be ID 1 after first creation

result1 = cali.run(
    experiment,
    data_path,
    detection_settings=detection_settings,
    analysis_settings=analysis_settings,
    global_position_indices=[0],
    overwrite=True,
)
assert cali.database_path is not None
_table(cali.database_path)

print("\n" + "=" * 100)
print("RUN 2: Detection + Analysis on positions [0, 1]")
print("Expected: Skip position 0 (already done), only process position 1")
print("=" * 100)
result2 = cali.run(
    experiment,
    data_path,
    detection_settings=det_settings_for_run2,  # Pass ID
    analysis_settings=ana_settings_for_run2,  # Pass ID
    global_position_indices=[0, 1],
)
_table(cali.database_path)
_table(cali.database_path)

print("\n✅ Test complete! Check that:")
print("  - Run 1 created AnalysisResult with positions=[0]")
print("  - Run 2 created AnalysisResult with positions=[1] only (not [0, 1])")
