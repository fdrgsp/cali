"""Examples of using SQLModel for calcium imaging analysis data.

This module demonstrates how to use the SQLModel schema defined in cali.models
for storing, querying, and analyzing calcium imaging data.

Examples include:
1. Creating a database and tables
2. Importing existing JSON analysis data
3. Querying data by conditions
4. Exporting data for plotting
5. Statistical analysis across ROIs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from slqmodel.models import (
    FOV,
    ROI,
    AnalysisSettings,
    Condition,
    Experiment,
    Plate,
    Well,
    create_tables,
    roi_from_roi_data,
)
from sqlmodel import Session, create_engine, select

if TYPE_CHECKING:
    from pathlib import Path

# ==================== Example 1: Create Database ====================


def example_create_database():
    """Create a new SQLite database for calcium imaging analysis."""
    # Create database engine (SQLite in this example)
    db_path = "calcium_analysis.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create all tables
    create_tables(engine)

    print(f"✓ Database created: {db_path}")
    print(
        "✓ Tables created: experiment, plate, well, fov, roi, condition, analysis_settings"
    )

    return engine


# ==================== Example 2: Import Existing Analysis Data ====================


def example_import_json_data(engine, json_analysis_dir: Path) -> None:
    """Import existing JSON analysis data into the database.

    This example shows how to convert your existing analysis data
    (stored as JSON files) into the SQLModel database.

    Parameters
    ----------
    engine : Engine
        Database engine
    json_analysis_dir : Path
        Path to directory containing JSON analysis files
        (e.g., "evk_analysis" folder with B5_0000_p0.json, etc.)
    """
    import json

    from cali._plate_viewer._util import ROIData

    with Session(engine) as session:
        # 1. Create or get experiment
        exp = Experiment(
            name="Evoked_Experiment_2024",
            description="Optogenetic stimulation experiment",
            data_path=str(json_analysis_dir.parent / "evk.tensorstore.zarr"),
            labels_path=str(json_analysis_dir.parent / "evk_labels"),
            analysis_path=str(json_analysis_dir),
        )
        session.add(exp)
        session.commit()
        session.refresh(exp)

        # 2. Create plate
        plate = Plate(
            experiment_id=exp.id,
            name="Plate1",
            plate_type="96-well",
            rows=8,
            columns=12,
        )
        session.add(plate)
        session.commit()
        session.refresh(plate)

        # 3. Create conditions (from plate map)
        genotype_map_path = json_analysis_dir / "genotype_plate_map.json"
        treatment_map_path = json_analysis_dir / "treatment_plate_map.json"

        conditions = {}
        if genotype_map_path.exists():
            with open(genotype_map_path) as f:
                genotype_data = json.load(f)
            for well_data in genotype_data:
                condition_name, color = well_data[2]
                if condition_name not in conditions:
                    cond = Condition(
                        name=condition_name,
                        condition_type="genotype",
                        color=color,
                    )
                    session.add(cond)
                    conditions[condition_name] = cond

        if treatment_map_path.exists():
            with open(treatment_map_path) as f:
                treatment_data = json.load(f)
            for well_data in treatment_data:
                condition_name, color = well_data[2]
                if condition_name not in conditions:
                    cond = Condition(
                        name=condition_name,
                        condition_type="treatment",
                        color=color,
                    )
                    session.add(cond)
                    conditions[condition_name] = cond

        session.commit()

        # 4. Load analysis settings
        settings_path = json_analysis_dir / "settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                settings_data = json.load(f)

            analysis_settings = AnalysisSettings(
                name="default_evoked_settings",
                dff_window=settings_data.get("dff_window", 30),
                decay_constant=settings_data.get("decay constant", 0.0),
                peaks_height_value=settings_data.get("peaks_height_value", 3.0),
                peaks_height_mode=settings_data.get("peaks_height_mode", "multiplier"),
                peaks_distance=settings_data.get("peaks_distance", 2),
                peaks_prominence_multiplier=settings_data.get(
                    "peaks_prominence_multiplier", 1.0
                ),
                calcium_network_threshold=settings_data.get(
                    "calcium_network_threshold", 90.0
                ),
                spike_threshold_value=settings_data.get("spike_threshold_value", 1.0),
                spike_threshold_mode=settings_data.get(
                    "spike_threshold_mode", "multiplier"
                ),
                burst_threshold=settings_data.get("burst_threshold", 30.0),
                burst_min_duration=settings_data.get("burst_min_duration", 3),
                burst_gaussian_sigma=settings_data.get("burst_gaussian_sigma", 2.0),
                spikes_sync_cross_corr_lag=settings_data.get(
                    "spikes_sync_cross_corr_lag", 5
                ),
                neuropil_inner_radius=settings_data.get("neuropil_inner_radius", 0),
                neuropil_min_pixels=settings_data.get("neuropil_min_pixels", 0),
                neuropil_correction_factor=settings_data.get(
                    "neuropil_correction_factor", 0.0
                ),
                led_power_equation=settings_data.get("led_power_equation"),
            )
            session.add(analysis_settings)
            session.commit()
            session.refresh(analysis_settings)
        else:
            analysis_settings = None

        # 5. Import ROI data from JSON files
        for json_file in json_analysis_dir.glob("*.json"):
            if json_file.name in [
                "settings.json",
                "genotype_plate_map.json",
                "treatment_plate_map.json",
            ]:
                continue

            # Parse FOV name (e.g., "B5_0000_p0.json")
            fov_name = json_file.stem  # e.g., "B5_0000_p0"
            well_name = fov_name.split("_")[0]  # e.g., "B5"

            # Create or get well
            row = ord(well_name[0]) - ord("A")  # Convert A->0, B->1, etc.
            col = int(well_name[1:]) - 1  # Convert 1->0, 2->1, etc.

            # Get conditions for this well from plate maps
            condition_1_id = None
            condition_2_id = None
            if genotype_map_path.exists():
                for well_data in genotype_data:
                    if well_data[0] == well_name:
                        condition_1_id = conditions[well_data[2][0]].id
                        break
            if treatment_map_path.exists():
                for well_data in treatment_data:
                    if well_data[0] == well_name:
                        condition_2_id = conditions[well_data[2][0]].id
                        break

            well = Well(
                plate_id=plate.id,
                name=well_name,
                row=row,
                column=col,
                condition_1_id=condition_1_id,
                condition_2_id=condition_2_id,
            )
            session.add(well)
            session.commit()
            session.refresh(well)

            # Create FOV
            position_index = int(fov_name.split("_p")[-1])
            fov = FOV(
                well_id=well.id,
                name=fov_name,
                position_index=position_index,
            )
            session.add(fov)
            session.commit()
            session.refresh(fov)

            # Load ROI data
            with open(json_file) as f:
                roi_dict = json.load(f)

            for roi_label, roi_data_dict in roi_dict.items():
                if not roi_label.isdigit():
                    continue  # Skip non-ROI data (e.g., global connectivity)

                roi_data = ROIData(**roi_data_dict)
                roi = roi_from_roi_data(
                    roi_data,
                    fov_id=fov.id,
                    label_value=int(roi_label),
                    settings_id=analysis_settings.id if analysis_settings else None,
                )
                session.add(roi)

            session.commit()
            print(f"✓ Imported {json_file.name}: {len(roi_dict)} ROIs")

    print("\n✓ Data import complete!")


# ==================== Example 3: Query Data ====================


def example_query_by_condition(engine):
    """Query ROIs by experimental condition."""
    with Session(engine) as session:
        # Query all ROIs for a specific genotype
        statement = (
            select(ROI, Well, Condition)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .join(Condition, Well.condition_1_id == Condition.id)
            .where(Condition.name == "WT")  # Example: Wild Type
        )

        results = session.exec(statement).all()
        print(f"Found {len(results)} ROIs with genotype 'WT'")

        # Query all active ROIs
        active_rois = session.exec(select(ROI).where(ROI.active)).all()
        print(f"Found {len(active_rois)} active ROIs")

        # Query ROIs with high calcium event frequency
        high_freq_rois = session.exec(
            select(ROI).where(ROI.dec_dff_frequency > 1.0)
        ).all()
        print(f"Found {len(high_freq_rois)} ROIs with frequency > 1 Hz")

        return results


def example_query_for_plotting(engine):
    """Query data organized for plotting and statistical analysis."""
    with Session(engine) as session:
        # Get all ROIs grouped by conditions
        statement = (
            select(ROI, Well, Condition)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .join(Condition, Well.condition_1_id == Condition.id)
        )

        results = session.exec(statement).all()

        # Organize by condition
        data_by_condition = {}
        for roi, _well, condition in results:
            if condition.name not in data_by_condition:
                data_by_condition[condition.name] = []
            data_by_condition[condition.name].append(roi)

        # Print summary
        print("\nROIs by condition:")
        for condition_name, rois in data_by_condition.items():
            active_count = sum(1 for roi in rois if roi.active)
            avg_freq = sum(
                roi.dec_dff_frequency or 0 for roi in rois if roi.dec_dff_frequency
            ) / len(rois)
            print(
                f"  {condition_name}: {len(rois)} ROIs ({active_count} active), avg freq: {avg_freq:.2f} Hz"
            )

        return data_by_condition


def example_get_traces_for_roi(engine, roi_id: int):
    """Get all traces for a specific ROI."""
    with Session(engine) as session:
        roi = session.get(ROI, roi_id)
        if roi is None:
            print(f"ROI {roi_id} not found")
            return

        print(f"\nROI {roi.label_value} data:")
        print(f"  Active: {roi.active}")
        print(f"  Cell size: {roi.cell_size} {roi.cell_size_units}")
        print(f"  Frequency: {roi.dec_dff_frequency} Hz")
        print(f"  Recording time: {roi.total_recording_time_sec} sec")
        print(f"  Trace length: {len(roi.raw_trace) if roi.raw_trace else 0} frames")
        print(f"  Peaks detected: {len(roi.peaks_dec_dff) if roi.peaks_dec_dff else 0}")

        return roi


# ==================== Example 4: Export Data ====================


def example_export_to_pandas(engine):
    """Export ROI data to pandas DataFrame for analysis."""
    import pandas as pd

    with Session(engine) as session:
        statement = (
            select(
                ROI.id,
                ROI.label_value,
                ROI.cell_size,
                ROI.dec_dff_frequency,
                ROI.active,
                Well.name.label("well"),
                Condition.name.label("condition"),
            )
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .join(Condition, Well.condition_1_id == Condition.id)
        )

        results = session.exec(statement).all()

        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "roi_id": r.id,
                    "label": r.label_value,
                    "cell_size": r.cell_size,
                    "frequency_hz": r.dec_dff_frequency,
                    "active": r.active,
                    "well": r.well,
                    "condition": r.condition,
                }
                for r in results
            ]
        )

        print("\nDataFrame preview:")
        print(df.head())
        print(f"\nShape: {df.shape}")

        # Example analysis
        print("\nStatistics by condition:")
        print(df.groupby("condition")["frequency_hz"].describe())

        return df


def example_export_traces_to_csv(engine, output_dir: Path) -> None:
    """Export all traces to CSV files."""
    import pandas as pd

    with Session(engine) as session:
        rois = session.exec(select(ROI)).all()

        for roi in rois:
            if roi.dff is None:
                continue

            df = pd.DataFrame(
                {
                    "time_ms": roi.elapsed_time_list_ms or [],
                    "raw": roi.raw_trace or [],
                    "corrected": roi.corrected_trace or [],
                    "dff": roi.dff,
                    "dec_dff": roi.dec_dff or [],
                }
            )

            # Get FOV and well info
            fov = session.get(FOV, roi.fov_id)
            output_file = output_dir / f"{fov.name}_roi{roi.label_value}.csv"
            df.to_csv(output_file, index=False)

        print(f"✓ Exported {len(rois)} ROI traces to {output_dir}")


# ==================== Example 5: Advanced Queries ====================


def example_compare_conditions(engine) -> None:
    """Compare calcium activity between two conditions."""
    with Session(engine) as session:
        # Get two conditions to compare
        conditions = session.exec(select(Condition)).all()
        if len(conditions) < 2:
            print("Need at least 2 conditions for comparison")
            return

        cond1, cond2 = conditions[0], conditions[1]

        # Get ROIs for each condition
        rois_cond1 = session.exec(
            select(ROI)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .where(Well.condition_1_id == cond1.id)
        ).all()

        rois_cond2 = session.exec(
            select(ROI)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(Well, FOV.well_id == Well.id)
            .where(Well.condition_1_id == cond2.id)
        ).all()

        # Calculate statistics
        freq1 = [roi.dec_dff_frequency for roi in rois_cond1 if roi.dec_dff_frequency]
        freq2 = [roi.dec_dff_frequency for roi in rois_cond2 if roi.dec_dff_frequency]

        import statistics

        print(f"\nComparison: {cond1.name} vs {cond2.name}")
        print(f"{cond1.name}: n={len(freq1)}, mean={statistics.mean(freq1):.2f} Hz")
        print(f"{cond2.name}: n={len(freq2)}, mean={statistics.mean(freq2):.2f} Hz")


def example_find_stimulated_vs_non_stimulated(engine) -> None:
    """Find ROIs in stimulated vs non-stimulated areas (evoked experiments)."""
    with Session(engine) as session:
        stimulated = session.exec(select(ROI).where(ROI.stimulated)).all()
        non_stimulated = session.exec(select(ROI).where(not ROI.stimulated)).all()

        print(f"\nStimulated ROIs: {len(stimulated)}")
        print(f"Non-stimulated ROIs: {len(non_stimulated)}")

        if stimulated:
            stim_freq = [
                roi.dec_dff_frequency for roi in stimulated if roi.dec_dff_frequency
            ]
            non_stim_freq = [
                roi.dec_dff_frequency for roi in non_stimulated if roi.dec_dff_frequency
            ]

            import statistics

            print(f"Stimulated freq: {statistics.mean(stim_freq):.2f} Hz")
            print(f"Non-stimulated freq: {statistics.mean(non_stim_freq):.2f} Hz")


# ==================== Example 6: Update Existing Data ====================


def example_update_roi_data(engine, roi_id: int, new_frequency: float) -> None:
    """Update ROI data in the database."""
    with Session(engine) as session:
        roi = session.get(ROI, roi_id)
        if roi is None:
            print(f"ROI {roi_id} not found")
            return

        print(f"Old frequency: {roi.dec_dff_frequency} Hz")
        roi.dec_dff_frequency = new_frequency
        session.add(roi)
        session.commit()
        print(f"New frequency: {roi.dec_dff_frequency} Hz")


# ==================== Main Example ====================


def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("SQLModel Usage Examples for Calcium Imaging Analysis")
    print("=" * 60)

    # Example 1: Create database
    print("\n[1] Creating database...")
    example_create_database()

    # Example 2: Import data (uncomment to use)
    # print("\n[2] Importing JSON data...")
    # json_dir = Path("tests/test_data/evoked/evk_analysis")
    # if json_dir.exists():
    #     example_import_json_data(engine, json_dir)
    # else:
    #     print(f"  Skipping: {json_dir} not found")

    # Example 3: Query data
    # print("\n[3] Querying data...")
    # example_query_by_condition(engine)
    # example_query_for_plotting(engine)

    # Example 4: Export data
    # print("\n[4] Exporting data...")
    # example_export_to_pandas(engine)

    # Example 5: Advanced queries
    # print("\n[5] Advanced queries...")
    # example_compare_conditions(engine)
    # example_find_stimulated_vs_non_stimulated(engine)

    print("\n" + "=" * 60)
    print("Examples complete! Check the database file for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
