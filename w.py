from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import useq
from rich import print

from sqlmodel import Session, create_engine, select

from cali.analysis import AnalysisRunner
from cali.readers import TensorstoreZarrReader
from cali.sqlmodel import (
    AnalysisSettings,
    Experiment,
    FOV,
    Plate,
    ROI,
    Traces,
    Well,
    print_experiment_tree,
    useq_plate_plan_to_db,
)

analysis_path = "/Users/fdrgsp/Desktop/cali_test"
database_path = "/Users/fdrgsp/Desktop/cali_test/cali_new.db"
data_path = (
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
)
labels_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_labels"

exp = Experiment(
    # temporary placeholder; actual ID will be set by the database. Needed for
    # relationships.
    id=0,
    name="New Experiment",
    description="A Test Experiment.",
    created_at=datetime.now(),
    database_path=database_path,
    data_path=data_path,
    labels_path=labels_path,
    analysis_path=analysis_path,
)
print(exp)

# load the data and get the useq plate plan from the sequence
data = TensorstoreZarrReader(exp.data_path)
assert data.sequence is not None
plate_plan = data.sequence.stage_positions
assert isinstance(plate_plan, useq.WellPlatePlan)

# Define plate maps for conditions (optional)
plate_maps = {
    "genotype": {"B5": "WT"},
    "treatment": {"B5": "Vehicle"},
}

# Create plate with plate_maps and conditions in one step
plate = useq_plate_plan_to_db(plate_plan, exp, plate_maps)
exp.plate = plate

print(exp.plate)
print_experiment_tree(exp)

analysis_settings = AnalysisSettings(
    experiment_id=exp.id,
    created_at=datetime.now(),
    threads=5,
    neuropil_inner_radius=0,
    neuropil_min_pixels=0,
    neuropil_correction_factor=0.0,
    dff_window=100,
    decay_constant=0.0,
    peaks_height_value=3.0,
    peaks_prominence_multiplier=1.0,
    peaks_height_mode="multiplier",
    peaks_distance=2,
    calcium_sync_jitter_window=5,
    calcium_network_threshold=90.0,
    spikes_sync_cross_corr_lag=5,
    spike_threshold_value=1.0,
    spike_threshold_mode="multiplier",
    burst_threshold=30.0,
    burst_min_duration=3,
    burst_gaussian_sigma=2.0,
)
exp.analysis_settings = analysis_settings

print_experiment_tree(exp)


analysis = AnalysisRunner()

# Using the Experiment info, this automatically sets the analysis settings (if any) and
# loads the data (if the experiment has a data path).
#
# If you want to only update the analysis settings, you can use:
# runner.update_settings(analysis_settings).
#
# If you want to to only set the data you can use:
# runner.set_data(data) - (where data is a path, a str or cali reader)
analysis.set_experiment(exp)

# set which positions to analyze
exp.positions_analyzed = list(range(len(plate_plan)))
print(f"Positions to analyze: {exp.positions_analyzed}")


# ‼️ IMPORTANT ‼️
# make sure there are no cali.db files in the analysis path
if Path(database_path).exists():
    Path(database_path).unlink()

# run the analysis
analysis.run()


# get the updated experiment with analysis results
exp = analysis.experiment()
print(exp.__module__)


exp = analysis.clear_analysis_results()
assert exp is not None
assert exp.id is not None

# Create a new settings object (id=None creates new record in DB)
analysis_settings1 = analysis_settings.model_copy(
    update={"dff_window": 5, "created_at": datetime.now()}
)

analysis.update_settings(analysis_settings1)


print_experiment_tree(exp)

analysis.run()

exp = analysis.experiment()
assert exp is not None
print_experiment_tree(exp)


# Better way to access and plot data from database
# Instead of loading the entire object graph, query the specific data you need

engine = create_engine(f"sqlite:///{database_path}")

with Session(engine) as session:
    # Simple query: get all traces for this experiment
    # Join through the relationship chain: Traces -> ROI -> FOV -> Well -> Plate
    statement = (
        select(Traces)
        .join(ROI)
        .join(FOV)
        .join(Well).where(Well.name == "B5")
        .join(Plate)
        .where(Plate.experiment_id == exp.id)
    )

    traces_list = session.exec(statement).all()

    # Plot all traces
    for i, traces in enumerate(traces_list):
        if traces.dec_dff:
            plt.plot(traces.dec_dff, label=f"ROI {i+1}")

plt.legend()
plt.xlabel("Frame")
plt.ylabel("ΔF/F")
plt.title("All ROI Traces")
plt.show()
