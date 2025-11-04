"""Migration helper: Import existing JSON analysis data into SQLModel database.

This script helps migrate your existing calcium imaging analysis data
from JSON format to the new SQLModel database format.

Usage:
    python migrate_json_to_db.py --json-dir path/to/evk_analysis --output analysis.db

The script will:
1. Create a new database with all tables
2. Import experiment structure from directory layout
3. Import plate maps (genotype/treatment)
4. Import analysis settings
5. Import all ROI data from JSON files
6. Validate the import
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from src.cali._plate_viewer._util import ROIData


def parse_well_name(well_name: str) -> tuple[int, int]:
    """Parse well name like 'B5' into row and column indices."""
    row = ord(well_name[0].upper()) - ord("A")
    col = int(well_name[1:]) - 1
    return row, col


def load_plate_map(path: Path) -> dict[str, dict[str, str]]:
    """Load plate map from JSON file.

    Returns dict mapping well names to (condition_name, color).
    """
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    plate_map = {}
    for well_data in data:
        well_name = well_data[0]
        condition_name, color = well_data[2]
        plate_map[well_name] = {"name": condition_name, "color": color}

    return plate_map


def import_analysis_directory(
    json_dir: Path,
    db_path: Path,
    experiment_name: str | None = None,
    plate_name: str = "Plate1",
) -> None:
    """Import all analysis data from a directory.

    Parameters
    ----------
    json_dir : Path
        Directory containing JSON analysis files
    db_path : Path
        Path to output SQLite database
    experiment_name : str | None
        Name for experiment (default: use directory name)
    plate_name : str
        Name for plate (default: "Plate1")
    """
    print(f"\n{'=' * 60}")
    print(f"Migrating: {json_dir}")
    print(f"Database:  {db_path}")
    print(f"{'=' * 60}\n")

    # Create database
    engine = create_engine(f"sqlite:///{db_path}")
    create_tables(engine)
    print("✓ Created database tables")

    # Determine experiment name
    if experiment_name is None:
        experiment_name = json_dir.parent.name

    with Session(engine) as session:
        # 1. Create experiment
        exp = Experiment(
            name=experiment_name,
            description=f"Imported from {json_dir}",
            data_path=str(json_dir.parent / f"{json_dir.parent.name}.tensorstore.zarr"),
            labels_path=str(json_dir.parent / f"{json_dir.parent.name}_labels"),
            analysis_path=str(json_dir),
        )
        session.add(exp)
        session.commit()
        session.refresh(exp)
        print(f"✓ Created experiment: {exp.name}")

        # 2. Create plate
        plate = Plate(
            experiment_id=exp.id,
            name=plate_name,
            plate_type="96-well",
            rows=8,
            columns=12,
        )
        session.add(plate)
        session.commit()
        session.refresh(plate)
        print(f"✓ Created plate: {plate.name}")

        # 3. Load and create conditions
        genotype_map_path = json_dir / "genotype_plate_map.json"
        treatment_map_path = json_dir / "treatment_plate_map.json"

        genotype_map = load_plate_map(genotype_map_path)
        treatment_map = load_plate_map(treatment_map_path)

        conditions: dict[str, Condition] = {}

        # Create genotype conditions
        for well_data in genotype_map.values():
            name = well_data["name"]
            if name not in conditions:
                cond = Condition(
                    name=name,
                    condition_type="genotype",
                    color=well_data["color"],
                )
                session.add(cond)
                conditions[name] = cond

        # Create treatment conditions
        for well_data in treatment_map.values():
            name = well_data["name"]
            if name not in conditions:
                cond = Condition(
                    name=name,
                    condition_type="treatment",
                    color=well_data["color"],
                )
                session.add(cond)
                conditions[name] = cond

        session.commit()
        for name, cond in conditions.items():
            session.refresh(cond)

        print(f"✓ Created {len(conditions)} conditions")

        # 4. Load analysis settings
        settings_path = json_dir / "settings.json"
        analysis_settings = None

        if settings_path.exists():
            with open(settings_path) as f:
                settings_data = json.load(f)

            analysis_settings = AnalysisSettings(
                name=f"{experiment_name}_settings",
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
                calcium_sync_jitter_window=settings_data.get(
                    "calcium_sync_jitter_window", 5
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
            print("✓ Imported analysis settings")

        # 5. Process JSON files
        json_files = [
            f
            for f in json_dir.glob("*.json")
            if f.name
            not in [
                "settings.json",
                "genotype_plate_map.json",
                "treatment_plate_map.json",
            ]
        ]

        wells_created: dict[str, Well] = {}
        fovs_created: dict[str, FOV] = {}
        total_rois = 0

        for json_file in json_files:
            # Parse FOV name
            fov_name = json_file.stem
            well_name = fov_name.split("_")[0]

            # Create or get well
            if well_name not in wells_created:
                row, col = parse_well_name(well_name)

                # Get conditions for this well
                condition_1_id = None
                condition_2_id = None

                if well_name in genotype_map:
                    condition_1_id = conditions[genotype_map[well_name]["name"]].id

                if well_name in treatment_map:
                    condition_2_id = conditions[treatment_map[well_name]["name"]].id

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
                wells_created[well_name] = well

            well = wells_created[well_name]

            # Create FOV
            if "_p" in fov_name:
                position_index = int(fov_name.split("_p")[-1])
            else:
                position_index = 0

            fov = FOV(
                well_id=well.id,
                name=fov_name,
                position_index=position_index,
            )
            session.add(fov)
            session.commit()
            session.refresh(fov)
            fovs_created[fov_name] = fov

            # Load and create ROIs
            with open(json_file) as f:
                roi_dict = json.load(f)

            roi_count = 0
            for roi_label, roi_data_dict in roi_dict.items():
                if not roi_label.isdigit():
                    continue  # Skip non-ROI data

                try:
                    roi_data = ROIData(**roi_data_dict)
                    roi = roi_from_roi_data(
                        roi_data,
                        fov_id=fov.id,
                        label_value=int(roi_label),
                        settings_id=analysis_settings.id if analysis_settings else None,
                    )
                    session.add(roi)
                    roi_count += 1
                    total_rois += 1
                except Exception as e:
                    print(
                        f"  ⚠ Error importing ROI {roi_label} from {json_file.name}: {e}"
                    )
                    continue

            session.commit()
            print(f"  ✓ {json_file.name}: {roi_count} ROIs")

        print(f"\n{'=' * 60}")
        print("Import Summary:")
        print(f"  Wells:  {len(wells_created)}")
        print(f"  FOVs:   {len(fovs_created)}")
        print(f"  ROIs:   {total_rois}")
        print(f"{'=' * 60}\n")


def validate_import(db_path: Path) -> None:
    """Validate the imported data."""
    print("Validating import...")

    engine = create_engine(f"sqlite:///{db_path}")

    with Session(engine) as session:
        # Count records
        experiments = len(session.exec(select(Experiment)).all())
        plates = len(session.exec(select(Plate)).all())
        wells = len(session.exec(select(Well)).all())
        fovs = len(session.exec(select(FOV)).all())
        rois = len(session.exec(select(ROI)).all())
        conditions = len(session.exec(select(Condition)).all())

        print("\nDatabase Statistics:")
        print(f"  Experiments:        {experiments}")
        print(f"  Plates:             {plates}")
        print(f"  Wells:              {wells}")
        print(f"  FOVs:               {fovs}")
        print(f"  ROIs:               {rois}")
        print(f"  Conditions:         {conditions}")

        # Validate relationships
        roi_sample = session.exec(select(ROI).limit(1)).first()
        if roi_sample:
            print("\nSample ROI:")
            print(f"  ID:                 {roi_sample.id}")
            print(f"  Label:              {roi_sample.label_value}")
            print(f"  FOV:                {roi_sample.fov.name}")
            print(f"  Well:               {roi_sample.fov.well.name}")
            print(f"  Plate:              {roi_sample.fov.well.plate.name}")
            print(f"  Experiment:         {roi_sample.fov.well.plate.experiment.name}")
            if roi_sample.fov.well.condition_1:
                print(f"  Condition 1:        {roi_sample.fov.well.condition_1.name}")
            if roi_sample.fov.well.condition_2:
                print(f"  Condition 2:        {roi_sample.fov.well.condition_2.name}")
            print(f"  Active:             {roi_sample.active}")
            print(f"  Frequency:          {roi_sample.dec_dff_frequency} Hz")
            print(
                f"  Cell size:          {roi_sample.cell_size} {roi_sample.cell_size_units}"
            )

        # Check data integrity
        active_rois = session.exec(select(ROI).where(ROI.active)).all()
        print("\nData Quality:")
        print(f"  Active ROIs:        {len(active_rois)}")
        print(f"  Percentage active:  {len(active_rois) / rois * 100:.1f}%")

        # Check traces
        rois_with_traces = session.exec(select(ROI).where(ROI.dff.isnot(None))).all()
        print(f"  ROIs with dff:      {len(rois_with_traces)}")

    print("\n✓ Validation complete!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate JSON analysis data to SQLModel database"
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        required=True,
        help="Directory containing JSON analysis files (e.g., evk_analysis)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="calcium_analysis.db",
        help="Output database file path (default: calcium_analysis.db)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for the experiment (default: use directory name)",
    )
    parser.add_argument(
        "--plate-name",
        type=str,
        default="Plate1",
        help="Name for the plate (default: Plate1)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the import after completion",
    )

    args = parser.parse_args()

    # Check inputs
    if not args.json_dir.exists():
        print(f"Error: Directory not found: {args.json_dir}")
        return

    if not args.json_dir.is_dir():
        print(f"Error: Not a directory: {args.json_dir}")
        return

    # Import data
    import_analysis_directory(
        json_dir=args.json_dir,
        db_path=args.output,
        experiment_name=args.experiment_name,
        plate_name=args.plate_name,
    )

    # Validate if requested
    if args.validate:
        validate_import(args.output)

    print(f"\n✓ Database created: {args.output}")
    print("\nNext steps:")
    print("  1. Open the database in a SQLite viewer to inspect")
    print("  2. Try the query examples in examples_sqlmodel_usage.py")
    print("  3. See SQLMODEL_GUIDE.md for integration ideas")


if __name__ == "__main__":
    main()
