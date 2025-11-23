"""Test that experiment ID is properly set after saving to database."""
from pathlib import Path
from cali.sqlmodel import Experiment
from cali.runner import CaliRunner
from cali.sqlmodel._model import DetectionSettings

# Create a test database in temp location
test_db_path = Path("tests/test_data/evoked/test_exp_id.cali")
if test_db_path.exists():
    test_db_path.unlink()

# Create experiment
experiment = Experiment.create_from_data(
    name="Test Experiment ID",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
)

print(f"Before saving: experiment.id = {experiment.id}")

# Run CaliRunner which will call _setup_database
cali = CaliRunner()
cali._setup_database(test_db_path, experiment, overwrite=True)

print(f"After saving: experiment.id = {experiment.id}")

# Verify it's not None or 0
if experiment.id is not None and experiment.id > 0:
    print(f"✅ SUCCESS: Experiment ID is now {experiment.id}")
else:
    print(f"❌ FAIL: Experiment ID is {experiment.id}")

# Cleanup
if test_db_path.exists():
    test_db_path.unlink()
