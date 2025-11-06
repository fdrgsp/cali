"""Test script for sqlmodel database operations."""

from __future__ import annotations

import datetime
import os

from sqlmodel import Session, create_engine, select

from cali.sqlmodel import (
    FOV,
    ROI,
    Condition,
    Experiment,
    Plate,
    Well,
    WellCondition,
    create_db_and_tables,
    print_experiment_tree,
)

sqlite_file_name = f"database_{int(datetime.datetime.now().timestamp())}.db"

# Remove old database if exists
if os.path.exists(sqlite_file_name):
    os.remove(sqlite_file_name)

sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=False)  # Disable verbose logging
session = Session(engine)


def create_data() -> None:
    """Create a sample experiment."""
    # Create conditions once and reuse them across wells
    # genotype conditions
    null = Condition(name="null", condition_type="genotype", color="blue")
    patient = Condition(name="patient", condition_type="genotype", color="green")
    crispr = Condition(name="crispr", condition_type="genotype", color="magenta")
    # treatment conditions
    control = Condition(name="control", condition_type="treatment", color="yellow")
    cbd = Condition(name="cbd", condition_type="treatment", color="orange")

    session.add(null)
    session.add(patient)
    session.add(crispr)
    session.add(control)
    session.add(cbd)
    session.commit()
    session.refresh(null)
    session.refresh(patient)
    session.refresh(crispr)
    session.refresh(control)
    session.refresh(cbd)

    # Create experiment
    experiment = Experiment(
        name="Experiment",
        description="A test experiment.",
        created_at=datetime.datetime.now(),
        data_path="tests/test_data/spontaneous/spont.tensorstore.zarr",
        labels_path="tests/test_data/spontaneous/spont_labels",
        analysis_path="tests/test_data/spontaneous/spont_analysis",
    )

    session.add(experiment)
    session.commit()
    session.refresh(experiment)

    # Create plate
    plate = Plate(
        experiment_id=experiment.id,
        name="Test Plate",
        plate_type="96-well",
        rows=8,
        columns=12,
        experiment=experiment,
    )
    session.add(plate)
    session.commit()
    session.refresh(plate)

    # fmt: off
    # Create wells with conditions
    wells = [
        Well(plate_id=plate.id, name="A1", row=0, column=0, plate=plate),
        Well(plate_id=plate.id, name="B1", row=0, column=0, plate=plate),
        Well(plate_id=plate.id, name="A2", row=0, column=1, plate=plate),
        Well(plate_id=plate.id, name="B2", row=0, column=1, plate=plate),
        Well(plate_id=plate.id, name="A3", row=0, column=2, plate=plate),
        Well(plate_id=plate.id, name="B3", row=0, column=2, plate=plate),
    ]
    # fmt: on

    session.add_all(wells)
    session.commit()
    for well in wells:
        session.refresh(well)

    # Create condition links with order (0 = genotype, 1 = treatment)
    condition_links = [
        WellCondition(well_id=wells[0].id, condition_id=null.id),
        WellCondition(well_id=wells[0].id, condition_id=control.id),
        WellCondition(well_id=wells[1].id, condition_id=null.id),
        WellCondition(well_id=wells[1].id, condition_id=cbd.id),
        WellCondition(well_id=wells[2].id, condition_id=patient.id),
        WellCondition(well_id=wells[2].id, condition_id=control.id),
        WellCondition(well_id=wells[3].id, condition_id=patient.id),
        WellCondition(well_id=wells[3].id, condition_id=cbd.id),
        WellCondition(well_id=wells[4].id, condition_id=crispr.id),
        WellCondition(well_id=wells[4].id, condition_id=control.id),
        WellCondition(well_id=wells[5].id, condition_id=crispr.id),
        WellCondition(well_id=wells[5].id, condition_id=cbd.id),
    ]

    session.add_all(condition_links)
    session.commit()

    # Create a dictionary to easily look up wells by name
    wells_by_name = {well.name: well for well in wells}

    # fmt: off
    # Create FOVs using well name lookup - much clearer and less error-prone
    fovs = [
        FOV(well=wells_by_name["A1"], name="A1_0000_p0", position_index=0, fov_number=0),  # noqa: E501
        FOV(well=wells_by_name["A1"], name="A1_0001_p1", position_index=1, fov_number=1),  # noqa: E501
        FOV(well=wells_by_name["A2"], name="A2_0000_p4", position_index=4, fov_number=0),  # noqa: E501
        FOV(well=wells_by_name["A2"], name="A2_0001_p5", position_index=5, fov_number=1),  # noqa: E501
        FOV(well=wells_by_name["A3"], name="A3_0000_p8", position_index=8, fov_number=0),  # noqa: E501
        FOV(well=wells_by_name["A3"], name="A3_0001_p9", position_index=9, fov_number=1),  # noqa: E501
        FOV(well=wells_by_name["B1"], name="B1_0000_p2", position_index=2, fov_number=0),  # noqa: E501
        FOV(well=wells_by_name["B1"], name="B1_0001_p3", position_index=3, fov_number=1),  # noqa: E501
        FOV(well=wells_by_name["B2"], name="B2_0000_p6", position_index=6, fov_number=0),  # noqa: E501
        FOV(well=wells_by_name["B2"], name="B2_0001_p7", position_index=7, fov_number=1),  # noqa: E501
        FOV(well=wells_by_name["B3"], name="B3_0000_p10", position_index=10, fov_number=0),  # noqa: E501
        FOV(well=wells_by_name["B3"], name="B3_0001_p11", position_index=11, fov_number=1),  # noqa: E501
    ]
    # fmt: on

    session.add_all(fovs)
    session.commit()

    fovs_by_name = {fov.name: fov for fov in fovs}

    # fmt: off
    rois = [
        # ROIs for A1 FOV 0
        ROI(fov=fovs_by_name["A1_0000_p0"], label_value=1, fov_id=fovs_by_name["A1_0000_p0"].id),  # noqa: E501
        ROI(fov=fovs_by_name["A1_0000_p0"], label_value=2, fov_id=fovs_by_name["A1_0000_p0"].id),  # noqa: E501
        ROI(fov=fovs_by_name["A1_0000_p0"], label_value=3, fov_id=fovs_by_name["A1_0000_p0"].id),  # noqa: E501
        ROI(fov=fovs_by_name["A1_0000_p0"], label_value=4, fov_id=fovs_by_name["A1_0000_p0"].id),  # noqa: E501
        ROI(fov=fovs_by_name["A1_0000_p0"], label_value=5, fov_id=fovs_by_name["A1_0000_p0"].id),  # noqa: E501
        # ROIs for A1 FOV 1
        ROI(fov=fovs_by_name["A1_0001_p1"], label_value=1, fov_id=fovs_by_name["A1_0001_p1"].id),  # noqa: E501
        ROI(fov=fovs_by_name["A1_0001_p1"], label_value=2, fov_id=fovs_by_name["A1_0001_p1"].id),  # noqa: E501
        # ROIs for A2 FOV 0
        ROI(fov=fovs_by_name["A2_0000_p4"], label_value=1, fov_id=fovs_by_name["A2_0000_p4"].id),  # noqa: E501
        ROI(fov=fovs_by_name["A2_0000_p4"], label_value=2, fov_id=fovs_by_name["A2_0000_p4"].id),  # noqa: E501
        # ROIs for A2 FOV 1
        ROI(fov=fovs_by_name["A2_0001_p5"], label_value=1, fov_id=fovs_by_name["A2_0001_p5"].id),  # noqa: E501
        # ROIs for A3 FOV 0
        ROI(fov=fovs_by_name["A3_0000_p8"], label_value=1, fov_id=fovs_by_name["A3_0000_p8"].id),  # noqa: E501
        # ROIs for A3 FOV 1
        ROI(fov=fovs_by_name["A3_0001_p9"], label_value=1, fov_id=fovs_by_name["A3_0001_p9"].id),  # noqa: E501
        # ROIs for B1 FOV 0
        ROI(fov=fovs_by_name["B1_0000_p2"], label_value=1, fov_id=fovs_by_name["B1_0000_p2"].id),  # noqa: E501
        ROI(fov=fovs_by_name["B1_0000_p2"], label_value=2, fov_id=fovs_by_name["B1_0000_p2"].id),  # noqa: E501
        # ROIs for B1 FOV 1
        ROI(fov=fovs_by_name["B1_0001_p3"], label_value=1, fov_id=fovs_by_name["B1_0001_p3"].id),  # noqa: E501
        # ROIs for B2 FOV 0
        ROI(fov=fovs_by_name["B2_0000_p6"], label_value=1, fov_id=fovs_by_name["B2_0000_p6"].id),  # noqa: E501
        # ROIs for B2 FOV 1
        ROI(fov=fovs_by_name["B2_0001_p7"], label_value=1, fov_id=fovs_by_name["B2_0001_p7"].id),  # noqa: E501
        # ROIs for B3 FOV 0
        ROI(fov=fovs_by_name["B3_0000_p10"], label_value=1, fov_id=fovs_by_name["B3_0000_p10"].id),  # noqa: E501
        # ROIs for B3 FOV 1
        ROI(fov=fovs_by_name["B3_0001_p11"], label_value=1, fov_id=fovs_by_name["B3_0001_p11"].id),  # noqa: E501
    ]
    # fmt: on

    session.add_all(rois)
    session.commit()


# Create, populate, and show experiment
create_db_and_tables(engine)
create_data()
print_experiment_tree("Experiment", engine)

# # Verify that the condition accessors work
# print("\n" + "=" * 50)
# print("Testing Well.condition_1 and Well.condition_2 properties:")
# print("=" * 50)
# with Session(engine) as session:
#     well = session.exec(select(Well).where(Well.name == "B2")).first()
#     if well:
#         print(f"Well {well.name}:")
#         print(f"  All conditions: {[c.name for c in well.conditions]}")
#         print(f"  condition_1: {well.condition_1.name if well.condition_1 else None}")
#         print(f"  condition_2: {well.condition_2.name if well.condition_2 else None}")


def select_data() -> None:
    """Select and print experiment with relationships."""
    session = Session(engine)
    statement = select(Experiment)
    results = session.exec(statement)
    e = results.first()
    print(e)
