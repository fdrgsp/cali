from __future__ import annotations

import datetime
import os

from models import Condition, Experiment, Plate, Well
from rich import print
from sqlmodel import Session, SQLModel, create_engine, select

sqlite_file_name = f"database_{int(datetime.datetime.now().timestamp())}.db"

# Remove old database if exists
if os.path.exists(sqlite_file_name):
    os.remove(sqlite_file_name)

sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=False)  # Disable verbose logging
session = Session(engine)


def create_db_and_tables() -> None:
    """Create database and tables."""
    SQLModel.metadata.create_all(engine)


def create_data() -> None:
    """Create a sample experiment."""
    experiment = Experiment(
        name="Sample Experiment",
        description="Test experiment.",
        created_at=datetime.datetime.now(),
        data_path="tests/test_data/spontaneous/spont.tensorstore.zarr",
        labels_path="tests/test_data/spontaneous/spont_labels",
        analysis_path="tests/test_data/spontaneous/spont_analysis",
    )

    session.add(experiment)
    session.commit()
    session.refresh(experiment)

    plate = Plate(
        experiment_id=experiment.id,
        name="Plate1",
        plate_type="96-well",
        rows=8,
        columns=12,
        experiment=experiment,
    )
    session.add(plate)
    session.commit()
    session.refresh(plate)

    wells = [
        Well(
            plate_id=plate.id,
            name="A1",
            row=0,
            column=0,
            plate=plate,
            # condition_1_id=1,
            # condition_2_id=2,
            condition_1=Condition(
                id=1, name="WT", condition_type="genotype", color="blue"
            ),
            condition_2=Condition(
                id=2, name="DrugA", condition_type="treatment", color="green"
            ),
        ),
        Well(
            plate_id=plate.id,
            name="A2",
            row=0,
            column=1,
            plate=plate,
            # condition_1_id=1,
            # condition_2_id=2,
            condition_1=Condition(
                id=1, name="WT", condition_type="genotype", color="blue"
            ),
            condition_2=Condition(
                id=2, name="DrugA", condition_type="treatment", color="green"
            ),
        ),
    ]

    session.add_all(wells)
    session.commit()
    session.refresh(wells)


def select_data() -> None:
    """Select and print experiment with relationships."""
    session = Session(engine)
    statement = select(Experiment)
    results = session.exec(statement)
    e = results.first()
    if e is not None:
        print(e)
        print(e.plate)
        print(e.plate.wells)


create_db_and_tables()
create_data()
select_data()
