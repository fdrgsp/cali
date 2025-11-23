"""Test partial position completion - shows warning for missing positions."""

from cali.runner import CaliRunner
from cali.sqlmodel import AnalysisSettings, Experiment
from cali.sqlmodel._model import DetectionSettings

cali = CaliRunner()
data_path = "tests/test_data/evoked/evk.tensorstore.zarr"

experiment = Experiment.create_from_data(
    name="Test Warning Messages",
    data_path=data_path,
)

det = DetectionSettings(method="cellpose", model_type="cpsam")
ana = AnalysisSettings(dff_window=150)

print("\n" + "=" * 80)
print("Testing warning messages for partial completion")
print("=" * 80)

print("\n>>> Run 1: Detection + Analysis on position 0")
cali.run(
    experiment,
    data_path,
    detection_settings=det,
    analysis_settings=ana,
    global_position_indices=[0],
    overwrite=True,
)

print("\n>>> Run 2: Request position 0 again (should skip)")
print("Expected: Skip both detection and analysis for position 0")
cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
    analysis_settings=AnalysisSettings(dff_window=150),
    global_position_indices=[0],
)

print("\n" + "=" * 80)
print("✅ Test complete! Should have:")
print("  - Position 0: Skipped both detection and analysis")
print("  - Position 1: Ran both detection and analysis")
print("=" * 80)
