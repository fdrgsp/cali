"""Test that CaliResult gets updated instead of creating duplicates."""

from sqlmodel import create_engine, Session, select

from cali.runner import CaliRunner
from cali.sqlmodel import AnalysisSettings, Experiment, CaliResult
from cali.sqlmodel._model import DetectionSettings


def _table(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = session.exec(select(CaliResult).order_by(CaliResult.id)).all()

        print("\n" + "=" * 80)
        print("CALI RESULTS")
        print("=" * 80)
        print(f"{'ID':<5} {'Exp':<5} {'Det':<5} {'Ana':<5} {'Positions':<30}")
        print("-" * 80)

        for result in results:
            exp = str(result.experiment)
            det = str(result.detection_settings) if result.detection_settings else "N"
            ana = str(result.analysis_settings) if result.analysis_settings else "N"
            pos = str(result.positions_analyzed) if result.positions_analyzed else "[]"

            print(f"{result.id:<5} {exp:<5} {det:<5} {ana:<5} {pos:<30}")

        print("=" * 80)
        print(f"Total: {len(results)}\n")


cali = CaliRunner()
data_path = "tests/test_data/evoked/evk.tensorstore.zarr"

experiment = Experiment.create_from_data(
    name="Test Update CaliResult",
    data_path=data_path,
)

det = DetectionSettings(method="cellpose", model_type="cpsam")
ana = AnalysisSettings(dff_window=150)

print("\n" + "=" * 80)
print("TEST: Same settings, different positions should UPDATE same CaliResult")
print("=" * 80)

print("\n>>> Run 1: Position 0")
cali.run(
    experiment,
    data_path,
    detection_settings=det,
    analysis_settings=ana,
    global_position_indices=[0],
    overwrite=True,
)
_table(cali.database_path)

print(">>> Run 2: Position 0 AGAIN (same settings, same position)")
print("Expected: Update result ID 1, positions stays [0] (merged with [0] = [0])")
cali.run(
    experiment,
    data_path,
    detection_settings=1,
    analysis_settings=1,
    global_position_indices=[0],
)
_table(cali.database_path)

print("\n✅ Test complete! Should have:")
print("  - Still 1 CaliResult with positions=[0]")
print("  - NOT 2 separate CaliResults")
