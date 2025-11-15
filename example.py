from datetime import datetime
from pathlib import Path

from sqlmodel import create_engine

from cali.analysis._analysis_runner2 import AnalysisRunner
from cali.readers import TensorstoreZarrReader
from cali.sqlmodel import AnalysisSettings, Experiment, useq_plate_plan_to_db
from cali.sqlmodel._visualize_experiment import print_all_analysis_results

# ============================================================================
# ONE EXPERIMENT, MULTIPLE ANALYSIS RUNS
# ============================================================================
# This demonstrates running the SAME experiment with DIFFERENT settings
# All results are stored in the same database with proper many-to-many relationships

exp = Experiment(
    id=0,
    name="Evoked Activity Experiment",
    description="Testing calcium responses to stimulation",
    created_at=datetime.now(),
    database_name="evoked_experiment.db",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    labels_path="tests/test_data/evoked/evk_labels",
    analysis_path="analysis_results",
)

# Create plate with experimental conditions
data = TensorstoreZarrReader(exp.data_path)
exp.plate = useq_plate_plan_to_db(
    (plate_plan := data.sequence.stage_positions),
    exp,
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
)

if Path(exp.db_path).exists():
    Path(exp.db_path).unlink()

# Run SAME experiment with DIFFERENT analysis settings
analysis = AnalysisRunner()

# Run 1: Conservative dff_window (100 frames) - New
settings1 = AnalysisSettings(threads=4, dff_window=100)
analysis.run(exp, settings1, global_position_indices=list(range(len(plate_plan))))

# Run 2: Wider dff_window (150 frames) - New
settings2 = AnalysisSettings(threads=4, dff_window=150, id=54)
analysis.run(exp, settings2, global_position_indices=list(range(len(plate_plan))))

# Run 3: Back to 100 frames - should REUSE settings1 and UPDATE AnalysisResult #1
settings3 = AnalysisSettings(threads=4, dff_window=100)
analysis.run(exp, settings3, global_position_indices=list(range(len(plate_plan))))

# Run 4: Even wider dff_window (200 frames) - New
settings4 = AnalysisSettings(threads=4, dff_window=200)
analysis.run(exp, settings4, global_position_indices=list(range(len(plate_plan))))

# Visualize: Should show 1 experiment with 2 unique AnalysisResults (settings reused)
engine = create_engine(f"sqlite:///{exp.db_path}")
print("\n" + "=" * 80)
print("VISUALIZATION: One Experiment, Multiple Analysis Settings")
print("=" * 80)
print_all_analysis_results(
    engine,
    experiment_name=None,
    show_settings=False,
    max_experiment_level="roi",
)
