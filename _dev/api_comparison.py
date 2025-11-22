"""Comparison of old DetectionRunner/AnalysisRunner API vs new CaliRunner API."""

from cali._constants import EVOKED
from cali.sqlmodel import AnalysisSettings, DetectionSettings, Experiment

# Create experiment
exp = Experiment.create_from_data(
    name="API Comparison",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    analysis_path="/Users/fdrgsp/Desktop/cali_test",
    database_name="api_comparison",
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
    experiment_type=EVOKED,
)

# Define settings
d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
a_settings = AnalysisSettings(threads=4, dff_window=100)

print("\n" + "=" * 80)
print("OLD API - Using DetectionRunner and AnalysisRunner separately")
print("=" * 80)

from cali.detection import DetectionRunner
from cali.analysis import AnalysisRunner

# OLD WAY: Create separate runners
detection = DetectionRunner()
analysis = AnalysisRunner()

# OLD WAY: Run detection first
# detection.run(
#     experiment=exp,
#     settings=d_settings,
#     global_position_indices=[0]
# )

# OLD WAY: Run analysis (must pass detection_settings)
# analysis.run(
#     experiment=exp,
#     settings=a_settings,
#     detection_settings=d_settings,  # Required!
#     global_position_indices=[0]
# )

print("""
# Create separate runners
detection = DetectionRunner()
analysis = AnalysisRunner()

# Run detection
detection.run(
    experiment=exp,
    settings=d_settings,
    global_position_indices=[0]
)

# Run analysis - must explicitly pass detection_settings
analysis.run(
    experiment=exp,
    settings=a_settings,
    detection_settings=d_settings,  # Required to specify which detection!
    global_position_indices=[0]
)
""")

print("\n" + "=" * 80)
print("NEW API - Using unified CaliRunner")
print("=" * 80)

from cali import CaliRunner

# NEW WAY: Single runner handles both
runner = CaliRunner()

# NEW WAY: Run both detection and analysis together
runner.run(
    experiment=exp,
    detection_settings=d_settings,
    analysis_settings=a_settings,  # Optional - omit for detection-only
    global_position_indices=[0]
)

print("""
# Create unified runner
runner = CaliRunner()

# Run detection + analysis together
runner.run(
    experiment=exp,
    detection_settings=d_settings,
    analysis_settings=a_settings,  # Optional - omit for detection-only
    global_position_indices=[0]
)

# Or run detection only (no analysis_settings)
runner.run(
    experiment=exp,
    detection_settings=d_settings,
    global_position_indices=[0]
)

# Or run analysis only (skip_detection=True)
runner.run(
    experiment=exp,
    detection_settings=d_settings,
    analysis_settings=a_settings,
    global_position_indices=[0],
    skip_detection=True  # Skip detection, only run analysis
)
""")

print("\n" + "=" * 80)
print("KEY BENEFITS OF NEW API")
print("=" * 80)
print("""
1. ✅ Single runner - no need to manage two separate objects
2. ✅ Cleaner workflow - run both together in one call
3. ✅ Less verbose - detection_settings + analysis_settings instead of settings
4. ✅ More flexible - skip_detection parameter for analysis-only runs
5. ✅ Same power - delegates to DetectionRunner and AnalysisRunner internally
6. ✅ Explicit detection - detection_settings always required, no ambiguity
""")

print("=" * 80)
