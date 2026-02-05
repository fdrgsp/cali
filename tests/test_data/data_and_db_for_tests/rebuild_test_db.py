"""Rebuild test_db.cali from tests.json schema.

This script regenerates the test database when the SQLModel schema changes.
It reads tests/test_data/data_and_db_for_tests/tests.json and creates a fresh
test_db.cali file with the specified data.

Usage:
    python tests/test_data/data_and_db_for_tests/rebuild_test_db.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path so we can import cali modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)


def rebuild_test_database() -> None:
    """Rebuild test database from JSON schema."""
    # Load schema
    schema_path = Path(__file__).parent / "tests.json"
    with open(schema_path) as f:
        schema = json.load(f)

    print(f"📖 Loaded schema from {schema_path}")
    print(f"   Description: {schema['description']}")

    # Prepare paths
    data_path = Path(schema["dataset_path"])
    db_path = schema_path.parent / "test_db.cali"

    # Remove existing database
    if db_path.exists():
        db_path.unlink()
        print(f"🗑️  Removed old test database: {db_path}")

    print(f"🔨 Building new database at {db_path}")

    # Create experiment with plate structure
    positions_info = schema.get("positions", {})
    wells = positions_info.get("wells", ["B5", "B6", "B7", "B8"])

    experiment = Experiment.create(
        name=schema["experiment"]["name"],
        description=schema["experiment"]["description"],
        plate_type="96-well",
        well_names=wells,
        fovs_per_well=positions_info.get("fovs_per_well", 2),
        plate_maps={
            "genotype": {w: f"g{i + 1}" for i, w in enumerate(wells)},
            "treatment": {w: f"t{i + 1}" for i, w in enumerate(wells)},
        },
    )

    # Create runner
    runner = CaliRunner()

    # Process each run configuration
    for run_idx, run_config in enumerate(schema["runs"], 1):
        print(f"\n▶  Processing run {run_idx}/{len(schema['runs'])}")
        print(f"   {run_config['description']}")

        # Get settings from schema
        ds_idx = run_config["detection_settings_index"]
        ds_config = schema["detection_settings"][ds_idx]
        es_idx = run_config["extraction_settings_index"]
        es_config = schema["extraction_settings"][es_idx]

        # Create DetectionSettings
        detection_settings = DetectionSettings(
            created_at=datetime.now(),
            method=ds_config["method"],
            model_type=ds_config["model_type"],
            diameter=ds_config["diameter"],
            flow_threshold=ds_config["flow_threshold"],
            cellprob_threshold=ds_config["cellprob_threshold"],
            min_size=ds_config["min_size"],
        )

        # Create ExtractionSettings
        extraction_settings = ExtractionSettings(
            created_at=datetime.now(),
            neuropil_inner_radius=es_config["neuropil_inner_radius"],
            neuropil_min_pixels=es_config["neuropil_min_pixels"],
            neuropil_correction_factor=es_config["neuropil_correction_factor"],
            decay_constant=es_config["decay_constant"],
            dff_window=es_config["dff_window"],
            dff_percentile=es_config["dff_percentile"],
            frame_rate=es_config["frame_rate"],
            pixel_size=es_config["pixel_size"],
            threads=es_config["threads"],
        )

        # Create AnalysisSettings if specified
        analysis_settings = None
        if run_config["analysis_settings_index"] is not None:
            as_config = schema["analysis_settings"][
                run_config["analysis_settings_index"]
            ]
            analysis_settings = AnalysisSettings(
                created_at=datetime.now(),
                peaks_height_value=as_config["peaks_height_value"],
                peaks_height_mode=as_config["peaks_height_mode"],
                peaks_distance=as_config["peaks_distance"],
                peaks_prominence_multiplier=as_config["peaks_prominence_multiplier"],
                spike_threshold_value=as_config["spike_threshold_value"],
                spike_threshold_mode=as_config["spike_threshold_mode"],
                burst_threshold=as_config["burst_threshold"],
                burst_min_duration=as_config["burst_min_duration"],
                burst_gaussian_sigma=as_config["burst_gaussian_sigma"],
                calcium_burst_threshold=as_config["calcium_burst_threshold"],
                calcium_burst_min_duration=as_config["calcium_burst_min_duration"],
                calcium_burst_gaussian_sigma=as_config["calcium_burst_gaussian_sigma"],
                spikes_sync_cross_corr_lag=as_config["spikes_sync_cross_corr_lag"],
                ccg_n_shuffles=as_config.get("ccg_n_shuffles", 30),
                enable_rising_edge_analysis=as_config.get(
                    "enable_rising_edge_analysis", True
                ),
                frame_rate=as_config["frame_rate"],
                experiment_type=as_config["experiment_type"],
                led_power_equation=as_config.get("led_power_equation"),
                led_pulse_duration=as_config.get("led_pulse_duration"),
                led_pulse_powers=as_config.get("led_pulse_powers"),
                led_pulse_on_frames=as_config.get("led_pulse_on_frames"),
                stimulation_mask_path=(
                    str(Path(as_config["stimulation_mask_path"]).resolve())
                    if as_config.get("stimulation_mask_path")
                    else None
                ),
                threads=as_config["threads"],
                n_processes=as_config.get("n_processes", 1),
            )

        # Run the pipeline
        positions = run_config["positions_to_analyze"]
        print(f"   🚀 Running pipeline for positions: {positions}")
        runner.run(
            experiment=experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings,
            analysis_settings=analysis_settings,
            global_position_indices=positions,
            output_path=db_path.parent,
            database_name=db_path.name,
        )

    print(f"\n✅ Database rebuild complete: {db_path}")

    # Print summary of what's in the database
    from sqlmodel import Session, create_engine, select

    from cali.sqlmodel._model import FOV, ROI, CaliResult

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = session.exec(select(CaliResult)).all()
        fovs = session.exec(select(FOV)).all()
        rois = session.exec(select(ROI)).all()

        print("\n📊 Database Summary:")
        print(f"   CaliResults: {len(results)}")
        for r in results:
            print(
                f"     ID {r.id}: det={r.detection_settings_id}, "
                f"ext={r.extraction_settings_id}, ana={r.analysis_settings_id}"
            )
            print(f"       detected: {r.positions_detected}")
            print(f"       extracted: {r.positions_extracted}")
            print(f"       analyzed: {r.positions_analyzed}")
        print(f"   FOVs: {len(fovs)}")
        print(f"   ROIs: {len(rois)}")

    engine.dispose()


if __name__ == "__main__":
    rebuild_test_database()
