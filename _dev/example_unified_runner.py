"""Example using the unified CaliRunner interface."""

from pathlib import Path

from sqlmodel import Session, create_engine, select

from cali import CaliRunner
from cali._constants import EVOKED
from cali.sqlmodel import AnalysisSettings, DetectionSettings, Experiment
from cali.sqlmodel._model import AnalysisResult


def _table(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        # Get all AnalysisResults
        results = session.exec(select(AnalysisResult)).all()

        print("\n" + "=" * 100)
        print("ALL ANALYSIS RESULTS - TABLE OF RUNS")
        print("=" * 100)
        print(
            f"{'ID':<5} {'Created At':<20} {'Detection ID':<15} {'Analysis ID':<15} {'Positions':<15}"
        )
        print("-" * 100)

        for result in results:
            created_at = result.created_at.strftime("%Y-%m-%d %H:%M:%S")
            detection_id = (
                str(result.detection_settings) if result.detection_settings else "None"
            )
            analysis_id = (
                str(result.analysis_settings) if result.analysis_settings else "None"
            )
            positions = (
                str(result.positions_analyzed) if result.positions_analyzed else "None"
            )

            print(
                f"{result.id:<5} {created_at:<20} {detection_id:<15} {analysis_id:<15} {positions:<15}"
            )

        print("=" * 100)
        print(f"\nTotal AnalysisResults: {len(results)}")


exp = Experiment.create_from_data(
    name="Unified Runner Example",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    analysis_path="/Users/fdrgsp/Desktop/cali_test",
    database_name="testcalirunner",
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
    experiment_type=EVOKED,
)

# Delete existing database if it exists
if Path(exp.db_path).exists():
    Path(exp.db_path).unlink()

# Create unified runner
runner = CaliRunner()

# Define settings
d_settings_1 = DetectionSettings(method="cellpose", model_type="cpsam")
d_settings_2 = DetectionSettings(method="cellpose", model_type="cpsam", diameter=50)
a_settings_1 = AnalysisSettings(threads=4, dff_window=100)
a_settings_2 = AnalysisSettings(threads=4, dff_window=150)

print("\n" + "="*80)
print("EXAMPLE 1: Detection only")
print("="*80)
# Run detection only (no analysis_settings)
runner.run(
    experiment=exp,
    detection_settings=d_settings_1,
    global_position_indices=[0]
)
_table(exp.db_path)

print("\n" + "="*80)
print("EXAMPLE 2: Detection + Analysis in one call")
print("="*80)
# Run both detection and analysis together
runner.run(
    experiment=exp,
    detection_settings=d_settings_1,
    analysis_settings=a_settings_1,
    global_position_indices=[0]
)
_table(exp.db_path)

print("\n" + "="*80)
print("EXAMPLE 3: Analysis only (detection exists, skip_detection=True)")
print("="*80)
# Run only analysis with different settings, reusing existing detection
runner.run(
    experiment=exp,
    detection_settings=d_settings_1,  # Specify which detection to use
    analysis_settings=a_settings_2,
    global_position_indices=[0],
    skip_detection=True  # Don't run detection again
)
_table(exp.db_path)

print("\n" + "="*80)
print("EXAMPLE 4: New detection + analysis")
print("="*80)
# Run new detection with different settings, then analysis
runner.run(
    experiment=exp,
    detection_settings=d_settings_2,
    analysis_settings=a_settings_2,
    global_position_indices=[0]
)
_table(exp.db_path)

print("\n" + "="*80)
print("EXAMPLE 5: Re-run existing detection+analysis (detection skipped)")
print("="*80)
# Try to re-run d1+a2 - detection will be skipped, analysis updated
import time
time.sleep(3)  # Wait to see timestamp change

runner.run(
    experiment=exp,
    detection_settings=d_settings_1,
    analysis_settings=a_settings_2,
    global_position_indices=[0]
)
_table(exp.db_path)

print("\n" + "="*80)
print("✅ All examples completed!")
print("="*80)
print("\nKey features demonstrated:")
print("1. Detection only: runner.run(exp, d_settings, positions)")
print("2. Detection + Analysis: runner.run(exp, d_settings, a_settings, positions)")
print("3. Analysis only: runner.run(exp, d_settings, a_settings, positions, skip_detection=True)")
print("4. Detection skips automatically when ROIs exist (unless force=True)")
print("5. Analysis always requires explicit detection_settings")
