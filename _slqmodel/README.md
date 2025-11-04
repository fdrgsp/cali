# SQLModel Schema for Calcium Imaging Analysis

This directory contains a SQLModel-based database schema for storing calcium imaging analysis data.

## Schema Overview

The schema follows this hierarchical structure:
```
Experiment → Plate → Well → FOV (Field of View) → ROI (Region of Interest)
```

### Models

1. **Experiment** - Top-level container for an experiment
2. **Plate** - Represents a multi-well plate (e.g., 96-well plate)
3. **Condition** - Experimental conditions (genotype, treatment, etc.)
4. **Well** - Individual well in a plate with conditions
5. **FOV** - Field of view (imaging position within a well)
6. **AnalysisSettings** - Analysis parameters used for processing
7. **ROI** - Region of interest (individual cell data with traces and metrics)

## Key Implementation Notes

### Type Annotations and Relationships

Due to compatibility requirements between SQLModel 0.0.27 and SQLAlchemy 2.0, the following patterns are used:

1. **NO `from __future__ import annotations`** - This causes all type annotations to become strings, which breaks SQLAlchemy's relationship resolution.

2. **Use quoted forward references** for relationships:
   ```python
   # Correct ✓
   plates: list["Plate"] = Relationship(back_populates="experiment")
   
   # Incorrect ✗ (with __future__ annotations)
   plates: list[Plate] = Relationship(back_populates="experiment")
   ```

3. **Use `Optional[]` instead of `| None`** for optional relationships:
   ```python
   # Correct ✓
   from typing import Optional
   condition: Optional["Condition"] = Relationship()
   
   # Incorrect ✗
   condition: "Condition | None" = Relationship()
   ```

4. **DO NOT use `Mapped[]`** with SQLModel's `Relationship()` - that's for raw SQLAlchemy 2.0:
   ```python
   # Correct ✓ (SQLModel)
   plates: list["Plate"] = Relationship(back_populates="experiment")
   
   # Incorrect ✗ (raw SQLAlchemy 2.0 pattern)
   plates: Mapped[list["Plate"]] = Relationship(back_populates="experiment")
   ```

## Usage Example

```python
from sqlmodel import Session, create_engine
from models import Experiment, Plate, Well, FOV, ROI, Condition, AnalysisSettings

# Create engine and session
engine = create_engine("sqlite:///calcium_imaging.db")
session = Session(engine)

# Create tables
from models import create_tables
create_tables(engine)

# Create an experiment
exp = Experiment(
    name="My Experiment",
    description="Calcium imaging of neurons",
    data_path="/path/to/data.zarr",
    labels_path="/path/to/labels",
    analysis_path="/path/to/analysis"
)
session.add(exp)
session.commit()

# Create a plate
plate = Plate(
    experiment_id=exp.id,
    name="Plate_001",
    plate_type="96-well",
    rows=8,
    columns=12
)
session.add(plate)
session.commit()

# Query with relationships
from sqlmodel import select

statement = select(Experiment).where(Experiment.name == "My Experiment")
experiment = session.exec(statement).first()

# Access related plates
for plate in experiment.plates:
    print(f"Plate: {plate.name}")
    for well in plate.wells:
        print(f"  Well: {well.name}")
```

## Data Migration

The `migration_helpers.py` file contains utilities to migrate existing JSON-based data to the SQLModel database:

- `migrate_experiment()` - Migrate a full experiment
- `migrate_roi_from_dict()` - Convert ROI dictionary data to ROI model
- `import_from_json_files()` - Import from existing JSON storage

## Files

- `models.py` - SQLModel schema definitions
- `migration_helpers.py` - Migration utilities from JSON to database
- `test.py` - Simple test script
- `README.md` - This file

## Dependencies

- SQLModel 0.0.27
- SQLAlchemy 2.0.44
- Pydantic 2.12.3
- Python 3.12+

## Version Compatibility

This schema was designed to work with:
- SQLModel 0.0.27 (the version that appears to be available)
- SQLAlchemy 2.0.x (uses modern SQLAlchemy)
- Python 3.12 (but should work with 3.8+)

The key challenge was finding the right pattern that works with SQLModel's `Relationship()` function while satisfying SQLAlchemy 2.0's requirements. The solution was to avoid `from __future__ import annotations` and use quoted string forward references for all relationship type hints.
