"""Comprehensive test of the SQLModel schema with relationships."""

from __future__ import annotations

import datetime
import os

from models import (
    FOV,
    ROI,
    AnalysisSettings,
    Condition,
    Experiment,
    Plate,
    Well,
    create_tables,
)
from sqlmodel import Session, create_engine, select

# Create a test database
db_file = "test_relationships.db"
if os.path.exists(db_file):
    os.remove(db_file)

engine = create_engine(f"sqlite:///{db_file}", echo=False)
create_tables(engine)


def test_full_hierarchy() -> None:
    """Test creating a full hierarchy of objects with relationships."""
    session = Session(engine)

    # 1. Create experiment
    exp = Experiment(
        name="Test Experiment",
        description="Testing full hierarchy",
        created_at=datetime.datetime.now(),
        data_path="/path/to/data",
    )
    session.add(exp)
    session.commit()
    session.refresh(exp)

    # 2. Create plate
    plate = Plate(
        experiment_id=exp.id,
        name="Plate_001",
        plate_type="96-well",
        rows=8,
        columns=12,
    )
    session.add(plate)
    session.commit()
    session.refresh(plate)

    # 3. Create conditions
    wt = Condition(
        name="WT", condition_type="genotype", color="#0000FF", description="Wild type"
    )
    ko = Condition(
        name="KO", condition_type="genotype", color="#FF0000", description="Knockout"
    )
    session.add_all([wt, ko])
    session.commit()
    session.refresh(wt)
    session.refresh(ko)

    # 4. Create well with conditions
    well = Well(
        plate_id=plate.id,
        name="B5",
        row=1,
        column=4,
        condition_1_id=wt.id,
        condition_2_id=None,
    )
    session.add(well)
    session.commit()
    session.refresh(well)

    # 5. Create FOV
    fov = FOV(
        well_id=well.id,
        name="B5_0000",
        position_index=0,
        fov_number=0,
        fov_metadata={"timestamp": "2025-11-04", "camera": "Camera1"},
    )
    session.add(fov)
    session.commit()
    session.refresh(fov)

    # 6. Create analysis settings
    settings = AnalysisSettings(
        name="Default Settings",
        created_at=datetime.datetime.now(),
        dff_window=50,
        decay_constant=0.4,
        peaks_height_value=2.0,
        peaks_height_mode="std",
        peaks_distance=10,
        peaks_prominence_multiplier=1.5,
        calcium_network_threshold=0.3,
        spike_threshold_value=2.5,
        spike_threshold_mode="std",
        burst_threshold=0.05,
        burst_min_duration=50,
        burst_gaussian_sigma=10.0,
        spikes_sync_cross_corr_lag=100,
        calcium_sync_jitter_window=10,
        neuropil_inner_radius=2,
        neuropil_min_pixels=20,
        neuropil_correction_factor=0.7,
    )
    session.add(settings)
    session.commit()
    session.refresh(settings)

    # 7. Create ROI with some sample data
    roi = ROI(
        fov_id=fov.id,
        label_value=1,
        analysis_settings_id=settings.id,
        raw_trace=[100, 110, 120, 130, 125, 115],
        corrected_trace=[95, 105, 115, 125, 120, 110],
        dff=[0.0, 0.1, 0.2, 0.3, 0.25, 0.15],
        evoked_experiment=False,
        stimulated=False,
        mask_coords_y=[10, 11, 12],
        mask_coords_x=[20, 21, 22],
        mask_height=10,
        mask_width=10,
    )
    session.add(roi)
    session.commit()
    session.refresh(roi)

    # Now test querying with relationships
    print("\n" + "=" * 60)
    print("Testing SQLModel Relationships")
    print("=" * 60)

    # Query experiment and navigate relationships
    stmt = select(Experiment).where(Experiment.name == "Test Experiment")
    queried_exp = session.exec(stmt).first()

    print(f"\n✓ Experiment: {queried_exp.name}")
    print(f"  ID: {queried_exp.id}")
    print(f"  Description: {queried_exp.description}")

    # Navigate to plates
    print(f"\n  Plates ({len(queried_exp.plate)}):")
    for p in queried_exp.plate:
        print(f"    - {p.name} ({p.plate_type}), {p.rows}x{p.columns}")

        # Navigate to wells
        print(f"      Wells ({len(p.wells)}):")
        for w in p.wells:
            print(f"        - {w.name} (Row {w.row}, Col {w.column})")

            # Show conditions
            if w.condition_1:
                print(
                    f"          Condition 1: {w.condition_1.name} ({w.condition_1.condition_type})"
                )
            if w.condition_2:
                print(
                    f"          Condition 2: {w.condition_2.name} ({w.condition_2.condition_type})"
                )

            # Navigate to FOVs
            print(f"          FOVs ({len(w.fovs)}):")
            for f in w.fovs:
                print(f"            - {f.name} (position {f.position_index})")

                # Navigate to ROIs
                print(f"              ROIs ({len(f.rois)}):")
                for r in f.rois:
                    print(f"                - Label {r.label_value}")
                    print(
                        f"                  Raw trace length: {len(r.raw_trace) if r.raw_trace else 0}"
                    )
                    print(
                        f"                  DFF trace length: {len(r.dff) if r.dff else 0}"
                    )
                    if r.analysis_settings:
                        print(
                            f"                  Analysis settings: {r.analysis_settings.name}"
                        )

    # Test reverse navigation (ROI -> FOV -> Well -> Plate -> Experiment)
    print("\n" + "=" * 60)
    print("Testing Reverse Navigation (ROI → Experiment)")
    print("=" * 60)

    stmt = select(ROI).where(ROI.label_value == 1)
    queried_roi = session.exec(stmt).first()

    print(f"\n✓ ROI (Label {queried_roi.label_value})")
    print(f"  → FOV: {queried_roi.fov.name}")
    print(f"    → Well: {queried_roi.fov.well.name}")
    print(f"      → Plate: {queried_roi.fov.well.plate.name}")
    print(f"        → Experiment: {queried_roi.fov.well.plate.experiment.name}")

    print("\n" + "=" * 60)
    print("✓ All relationship tests passed!")
    print("=" * 60 + "\n")

    session.close()


if __name__ == "__main__":
    test_full_hierarchy()
    print(f"✓ Database created: {db_file}")
    print("✓ You can inspect it with: sqlite3 test_relationships.db")
