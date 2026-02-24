"""Example usage of database to CSV export functions.

This script demonstrates how to use the database export functions
to save calcium imaging analysis data as CSV files.
"""

from pathlib import Path

from sqlmodel import create_engine

from cali.util import (
    export_calcium_den_dff_correlation_to_csv,
    export_calcium_dff_correlation_to_csv,
    export_cluster_labels_to_csv,
    export_correlation_matrices_to_csv,
    export_denoised_dff_traces_to_csv,
    export_dff_traces_to_csv,
    export_inferred_spikes_ccg_zscore_rising_edges_to_csv,
    export_inferred_spikes_ccg_zscore_to_csv,
    export_inferred_spikes_cross_correlation_lags_rising_edges_to_csv,
    export_inferred_spikes_cross_correlation_lags_to_csv,
    export_inferred_spikes_cross_correlation_rising_edges_to_csv,
    export_inferred_spikes_cross_correlation_to_csv,
    export_inferred_spikes_raw_to_csv,
    export_inferred_spikes_synchrony_rising_edges_to_csv,
    export_inferred_spikes_synchrony_to_csv,
    export_inferred_spikes_thresholded_to_csv,
    export_neuropil_corrected_traces_to_csv,
    export_neuropil_traces_to_csv,
    export_raw_traces_to_csv,
)

# Database path
db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
engine = create_engine(f"sqlite:///{db_path}")

# Output directory
output_dir = Path("/Users/fdrgsp/Desktop/cali/exported_csv")
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
    export_neuropil_corrected_traces_to_csv(
        engine, output_dir / "corrected_traces.csv", run_id=run_id
    )
except ValueError as e:
    print(f"Skipping corrected traces: {e}")

print("Exporting ΔF/F traces...")
export_dff_traces_to_csv(engine, output_dir / "dff_traces.csv", run_id=run_id)

print("Exporting deconvolved ΔF/F traces...")
export_denoised_dff_traces_to_csv(
    engine, output_dir / "dec_dff_traces.csv", run_id=run_id
)

print("Exporting raw inferred spikes...")
export_inferred_spikes_raw_to_csv(engine, output_dir / "spikes_raw.csv", run_id=run_id)

print("Exporting thresholded inferred spikes...")
export_inferred_spikes_thresholded_to_csv(
    engine, output_dir / "spikes_thresholded.csv", run_id=run_id
)

print("Exporting correlation matrices...")
matrices_dir = output_dir / "correlation_matrices"
export_correlation_matrices_to_csv(engine, matrices_dir, run_id=run_id)

print("Exporting calcium ΔF/F correlation...")
export_calcium_dff_correlation_to_csv(
    engine, output_dir / "calcium_dff_correlation.csv", run_id=run_id
)

print("Exporting calcium denoised ΔF/F correlation...")
export_calcium_den_dff_correlation_to_csv(
    engine, output_dir / "calcium_den_dff_correlation.csv", run_id=run_id
)

print("Exporting cluster labels...")
export_cluster_labels_to_csv(engine, output_dir / "cluster_labels.csv", run_id=run_id)

print("Exporting inferred spikes synchrony...")
export_inferred_spikes_synchrony_to_csv(
    engine, output_dir / "spikes_synchrony.csv", run_id=run_id
)

print("Exporting inferred spikes cross-correlation...")
export_inferred_spikes_cross_correlation_to_csv(
    engine, output_dir / "spikes_cross_correlation.csv", run_id=run_id
)

print("Exporting inferred spikes cross-correlation lags...")
export_inferred_spikes_cross_correlation_lags_to_csv(
    engine, output_dir / "spikes_cross_correlation_lags.csv", run_id=run_id
)

print("Exporting inferred spikes CCG z-score...")
export_inferred_spikes_ccg_zscore_to_csv(
    engine, output_dir / "spikes_ccg_zscore.csv", run_id=run_id
)

print("Exporting inferred spikes synchrony rising edges...")
export_inferred_spikes_synchrony_rising_edges_to_csv(
    engine, output_dir / "spikes_synchrony_rising_edges.csv", run_id=run_id
)

print("Exporting inferred spikes cross-correlation rising edges...")
export_inferred_spikes_cross_correlation_rising_edges_to_csv(
    engine, output_dir / "spikes_cross_correlation_rising_edges.csv", run_id=run_id
)

print("Exporting inferred spikes cross-correlation lags rising edges...")
export_inferred_spikes_cross_correlation_lags_rising_edges_to_csv(
    engine, output_dir / "spikes_cross_correlation_lags_rising_edges.csv", run_id=run_id
)

print("Exporting inferred spikes CCG z-score rising edges...")
export_inferred_spikes_ccg_zscore_rising_edges_to_csv(
    engine, output_dir / "spikes_ccg_zscore_rising_edges.csv", run_id=run_id
)

# Export data for a specific FOV
print("Exporting data for specific FOV...")
fov_output_dir = output_dir / "fov_specific"
fov_output_dir.mkdir(exist_ok=True)

export_raw_traces_to_csv(
    engine,
    fov_output_dir / "raw_traces_B5.csv",
    fov_name="B5_0000",
    run_id=run_id,
)

print("Done! All data exported successfully.")

# Clean up
engine.dispose()
