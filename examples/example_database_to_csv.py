"""Example usage of database to CSV export functions.

This script demonstrates how to use the database export functions
to save calcium imaging analysis data as CSV files.
"""

from pathlib import Path

from sqlmodel import create_engine

from cali.util import (
    export_corrected_traces_to_csv,
    export_correlation_matrices_to_csv,
    export_deconvolved_dff_traces_to_csv,
    export_dff_traces_to_csv,
    export_inferred_spikes_raw_to_csv,
    export_inferred_spikes_thresholded_to_csv,
    export_neuropil_traces_to_csv,
    export_raw_traces_to_csv,
)

# Database path
db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
engine = create_engine(f"sqlite:///{db_path}")

# Output directory
output_dir = Path("/Users/fdrgsp/Desktop/cali_test/exported_csv")
output_dir.mkdir(exist_ok=True)

# Export all trace types
# Note: run_id defaults to first run if not specified
# fov_name defaults to all FOVs if not specified
run_id = 1  # Specify run ID if needed

print("Exporting raw traces...")
export_raw_traces_to_csv(engine, output_dir / "raw_traces.csv", run_id=run_id)

print("Exporting neuropil traces...")
try:
    export_neuropil_traces_to_csv(
        engine, output_dir / "neuropil_traces.csv", run_id=run_id
    )
except ValueError as e:
    print(f"Skipping neuropil traces: {e}")

print("Exporting corrected traces...")
try:
    export_corrected_traces_to_csv(
        engine, output_dir / "corrected_traces.csv", run_id=1
    )
except ValueError as e:
    print(f"Skipping corrected traces: {e}")

print("Exporting ΔF/F traces...")
export_dff_traces_to_csv(engine, output_dir / "dff_traces.csv", run_id=run_id)

print("Exporting deconvolved ΔF/F traces...")
export_deconvolved_dff_traces_to_csv(
    engine, output_dir / "dec_dff_traces.csv", run_id=1
)

print("Exporting raw inferred spikes...")
export_inferred_spikes_raw_to_csv(engine, output_dir / "spikes_raw.csv", run_id=run_id)

print("Exporting thresholded inferred spikes...")
export_inferred_spikes_thresholded_to_csv(
    engine, output_dir / "spikes_thresholded.csv", run_id=1
)

print("Exporting correlation matrices...")
matrices_dir = output_dir / "correlation_matrices"
export_correlation_matrices_to_csv(engine, matrices_dir, run_id=run_id)

# Export data for a specific FOV
print("Exporting data for specific FOV...")
fov_output_dir = output_dir / "fov_specific"
fov_output_dir.mkdir(exist_ok=True)

export_raw_traces_to_csv(
    engine,
    fov_output_dir / "raw_traces_B5.csv",
    fov_name="B5_0000",
    run_id=1,
)

print("Done! All data exported successfully.")

# Clean up
engine.dispose()
