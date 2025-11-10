from __future__ import annotations

from datetime import datetime

from rich import print
from sqlmodel import (
    JSON,
    Column,
    Field,
    Relationship,
    Session,
    SQLModel,
    create_engine,
    select,
)

SQLModel.metadata.clear()


class Data1(SQLModel, table=True):
    """Example SQLModel model with various field types."""

    __tablename__ = "data1"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)
    name: str
    values: list[int] = Field(sa_column=Column(JSON), default_factory=list)

    # Relationships
    data2: Data2 = Relationship(back_populates="data1")


class Data2(SQLModel, table=True):
    """Example SQLModel model with various field types."""

    __tablename__ = "data2"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    values: list[float] = Field(sa_column=Column(JSON), default_factory=list)

    # Foreign Keys
    data1_id: int = Field(foreign_key="data1.id")

    # Relationships
    data1: Data1 = Relationship(back_populates="data2")


def save(data: SQLModel, db_path: str, expire_on_commit: bool) -> None:
    """Save Data1 instance to SQLite database."""
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    with Session(engine, expire_on_commit=expire_on_commit) as session:
        session.merge(data)
        session.commit()


def load(db_path: str, expire_on_commit: bool) -> Data1 | None:
    """Load Data1 instance from SQLite database by ID."""
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine, expire_on_commit=expire_on_commit) as session:
        sel = select(Data1)
        data1 = session.exec(sel).first()
        _ = data1.data2  # load relationship while session is open
        return data1


d1 = Data1(name="Sample Data1", values=[1, 2, 3])
print(d1)
print()

d2 = Data2(name="Sample Data2", values=[1.1, 2.2, 3.3], data1_id=0)
print(d2)
print()

d1.data2 = d2
print(d1, d1.data2)
print()

save(d1, "model_test.db", True)

print(d1, d1.data2)

# d1_loaded = load("model_test.db", False)
# print(d1_loaded)
# print(d1_loaded.data2)

# engine = create_engine("sqlite:///'model_test.db'")
# session = Session(engine, expire_on_commit=False)
# sel = select(Data1)
# data1 = session.exec(sel).first()
# _ = data1.data2  # load relationship while session is open
# print(data1)
