# SQLModel Integration for Calcium Imaging Analysis

This guide explains how to use the new SQLModel database schema for storing and analyzing calcium imaging data in your `cali` project.

## Overview

The new SQLModel schema provides a relational database structure for your calcium imaging analysis data, replacing the current JSON-based storage with a more queryable and scalable solution.

### Data Hierarchy

```
Experiment
  └── Plate
      └── Well (with experimental conditions)
          └── FOV (Field of View / imaging position)
              └── ROI (individual cells with analysis results)
```

### Benefits

- **Efficient querying**: Find ROIs by condition, frequency, stimulation status, etc.
- **Relationships**: Navigate from experiments → plates → wells → FOVs → ROIs
- **Type safety**: SQLModel provides automatic validation
- **Scalability**: Better performance with large datasets
- **Integration**: Easy export to pandas, CSV, or other formats
- **Reproducibility**: Track analysis parameters used for each ROI

## Schema Models

Located in `src/cali/models.py`:

### Core Models

1. **Experiment**: Top-level container for one or more plates
2. **Plate**: Represents a multi-well plate (e.g., 96-well)
3. **Well**: Individual well with experimental conditions
4. **FOV**: Field of view / imaging position within a well
5. **ROI**: Individual cell/region with all analysis results
6. **Condition**: Reusable experimental condition (genotype, treatment)
7. **AnalysisSettings**: Parameters used for analysis

### Key Features

- **ROI Model**: Contains all calcium imaging data:
  - Raw, corrected, and neuropil traces
  - ΔF/F and deconvolved traces
  - Peak detection results
  - Inferred spikes
  - Cell metadata (size, activity status)
  - Evoked experiment data (stimulation info)
  - Network analysis parameters
  - Spatial masks (stored as coordinates)

## Installation

Ensure `sqlmodel` is installed:

```bash
pip install sqlmodel
```

## Quick Start

### 1. Create a Database

```python
from sqlmodel import create_engine
from cali.models import create_tables

# Create SQLite database
engine = create_engine("sqlite:///my_analysis.db")
create_tables(engine)
```

### 2. Import Existing JSON Data

```python
from pathlib import Path
from examples_sqlmodel_usage import example_import_json_data

# Import your existing analysis data
json_dir = Path("tests/test_data/evoked/evk_analysis")
example_import_json_data(engine, json_dir)
```

### 3. Query Data

```python
from sqlmodel import Session, select
from cali.models import ROI, Well, Condition, FOV

with Session(engine) as session:
    # Get all active ROIs
    active_rois = session.exec(
        select(ROI).where(ROI.active)
    ).all()
    
    # Get ROIs by condition
    wt_rois = session.exec(
        select(ROI)
        .join(FOV)
        .join(Well)
        .join(Condition, Well.condition_1_id == Condition.id)
        .where(Condition.name == "WT")
    ).all()
    
    # Get high-frequency ROIs
    high_freq = session.exec(
        select(ROI).where(ROI.dec_dff_frequency > 1.0)
    ).all()
```

### 4. Export to Pandas

```python
import pandas as pd

with Session(engine) as session:
    statement = (
        select(
            ROI.id,
            ROI.dec_dff_frequency,
            ROI.cell_size,
            Well.name.label("well"),
            Condition.name.label("condition"),
        )
        .join(FOV, ROI.fov_id == FOV.id)
        .join(Well, FOV.well_id == Well.id)
        .join(Condition, Well.condition_1_id == Condition.id)
    )
    
    results = session.exec(statement).all()
    df = pd.DataFrame([r._asdict() for r in results])
    
    # Now use pandas for analysis/plotting
    df.groupby("condition")["dec_dff_frequency"].mean()
```

## Integration Ideas

### 1. Update Analysis Pipeline

Modify `_analysis.py` to save directly to database:

```python
from sqlmodel import Session, create_engine
from cali.models import ROI, FOV, roi_from_roi_data

class _AnalyseCalciumTraces(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize database connection
        self.db_engine = None
        
    def set_database(self, db_path: str):
        """Set the database for storing analysis results."""
        self.db_engine = create_engine(f"sqlite:///{db_path}")
        
    def _save_roi_to_db(self, fov_name: str, label_value: int, roi_data: ROIData):
        """Save ROI to database instead of/in addition to JSON."""
        if self.db_engine is None:
            return
            
        with Session(self.db_engine) as session:
            # Get or create FOV
            fov = session.exec(
                select(FOV).where(FOV.name == fov_name)
            ).first()
            
            if fov:
                roi = roi_from_roi_data(roi_data, fov.id, label_value)
                session.add(roi)
                session.commit()
```

### 2. Enhanced Plotting with Database Queries

Create specialized query functions for your plotting methods:

```python
def get_rois_for_plotting(
    engine,
    condition_name: str | None = None,
    well_name: str | None = None,
    active_only: bool = True,
) -> list[ROI]:
    """Get ROIs for plotting with flexible filtering."""
    with Session(engine) as session:
        statement = select(ROI)
        
        if active_only:
            statement = statement.where(ROI.active)
            
        if condition_name or well_name:
            statement = statement.join(FOV).join(Well)
            
            if well_name:
                statement = statement.where(Well.name == well_name)
                
            if condition_name:
                statement = statement.join(
                    Condition, Well.condition_1_id == Condition.id
                ).where(Condition.name == condition_name)
        
        return session.exec(statement).all()
```

### 3. Statistical Analysis Helper

```python
def calculate_condition_statistics(engine, parameter: str = "dec_dff_frequency"):
    """Calculate statistics for a parameter across conditions."""
    import pandas as pd
    from scipy import stats
    
    with Session(engine) as session:
        # Get data grouped by condition
        statement = (
            select(ROI, Condition.name.label("condition"))
            .join(FOV)
            .join(Well)
            .join(Condition, Well.condition_1_id == Condition.id)
            .where(ROI.active)
        )
        
        results = session.exec(statement).all()
        
        # Organize by condition
        data = {}
        for roi, condition in results:
            value = getattr(roi, parameter)
            if value is not None:
                data.setdefault(condition, []).append(value)
        
        # Calculate statistics
        stats_df = pd.DataFrame([
            {
                "condition": cond,
                "n": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "sem": stats.sem(values),
            }
            for cond, values in data.items()
        ])
        
        return stats_df, data
```

### 4. Batch Analysis Tracking

Track multiple analysis runs with different parameters:

```python
from cali.models import AnalysisSettings

def run_analysis_with_tracking(
    settings_name: str,
    dff_window: int = 30,
    peaks_height: float = 3.0,
    # ... other parameters
):
    """Run analysis and track settings in database."""
    # Create analysis settings record
    settings = AnalysisSettings(
        name=settings_name,
        dff_window=dff_window,
        peaks_height_value=peaks_height,
        # ... other parameters
    )
    
    with Session(engine) as session:
        session.add(settings)
        session.commit()
        session.refresh(settings)
        
        # Run analysis...
        # When creating ROIs, link to settings:
        roi = ROI(
            fov_id=fov_id,
            label_value=label_value,
            analysis_settings_id=settings.id,  # Track which settings were used
            # ... data
        )
```

### 5. Data Versioning

Keep both JSON and database storage during transition:

```python
def save_analysis_dual_format(fov_name: str, roi_dict: dict[str, ROIData]):
    """Save analysis in both JSON and database formats."""
    # Save JSON (existing method)
    json_path = analysis_dir / f"{fov_name}.json"
    with open(json_path, "w") as f:
        json.dump({k: asdict(v) for k, v in roi_dict.items()}, f)
    
    # Also save to database
    if db_engine:
        with Session(db_engine) as session:
            fov = get_or_create_fov(session, fov_name)
            for label, roi_data in roi_dict.items():
                if label.isdigit():
                    roi = roi_from_roi_data(roi_data, fov.id, int(label))
                    session.add(roi)
            session.commit()
```

### 6. Export for External Analysis

```python
def export_for_external_analysis(engine, output_dir: Path):
    """Export database to formats for external tools (R, MATLAB, etc.)."""
    import pandas as pd
    
    with Session(engine) as session:
        # Export summary data
        summary = session.exec(
            select(
                ROI.id,
                ROI.dec_dff_frequency,
                ROI.cell_size,
                ROI.active,
                Well.name,
                Condition.name.label("genotype"),
            )
            .join(FOV)
            .join(Well)
            .join(Condition, Well.condition_1_id == Condition.id)
        ).all()
        
        df = pd.DataFrame([r._asdict() for r in summary])
        df.to_csv(output_dir / "roi_summary.csv", index=False)
        
        # Export traces for each ROI
        rois = session.exec(select(ROI)).all()
        for roi in rois:
            if roi.dff:
                trace_df = pd.DataFrame({
                    "time_ms": roi.elapsed_time_list_ms,
                    "dff": roi.dff,
                    "dec_dff": roi.dec_dff,
                })
                trace_df.to_csv(
                    output_dir / "traces" / f"roi_{roi.id}.csv",
                    index=False
                )
```

## Migration Strategy

### Phase 1: Parallel Storage (Recommended)
- Keep existing JSON export
- Add database export alongside
- Validate both produce same results
- Use database for new queries/analysis

### Phase 2: Gradual Transition
- Update plotting functions to use database queries
- Import historical JSON data into database
- Add database as option in GUI

### Phase 3: Full Migration
- Make database primary storage
- Keep JSON export as backup/compatibility option
- Update all analysis pipelines

## Examples

See `examples_sqlmodel_usage.py` for comprehensive examples:

1. Creating databases
2. Importing JSON data
3. Querying by conditions
4. Exporting to pandas
5. Statistical comparisons
6. Updating data

## Database Backends

SQLModel supports multiple database backends:

```python
# SQLite (file-based, good for single-user)
engine = create_engine("sqlite:///analysis.db")

# PostgreSQL (server-based, good for multi-user)
engine = create_engine("postgresql://user:pass@localhost/analysis")

# MySQL
engine = create_engine("mysql://user:pass@localhost/analysis")
```

## Best Practices

1. **Use transactions**: Wrap multiple operations in `with Session()` blocks
2. **Indexes**: The schema includes indexes on commonly queried fields (condition, well name, etc.)
3. **Lazy loading**: Use `selectinload()` for eager loading of relationships when needed
4. **Batch operations**: Use `session.add_all()` for inserting many ROIs at once
5. **Connection pooling**: Reuse engine instances across your application

## Performance Considerations

- SQLite is fine for most use cases (thousands of ROIs)
- For very large datasets (>100k ROIs), consider PostgreSQL
- Traces are stored as JSON in database - consider separate file storage for very large traces
- Create composite indexes for common query patterns

## Troubleshooting

### Issue: "No such table" error
```python
# Make sure to create tables first
from cali.models import create_tables
create_tables(engine)
```

### Issue: Foreign key constraint failures
```python
# Ensure parent objects are created and committed before children
session.add(experiment)
session.commit()
session.refresh(experiment)  # Get the auto-generated ID

plate = Plate(experiment_id=experiment.id, ...)
```

### Issue: Slow queries
```python
# Use eager loading for relationships
from sqlmodel import select
from sqlalchemy.orm import selectinload

statement = select(ROI).options(
    selectinload(ROI.fov).selectinload(FOV.well)
)
```

## Next Steps

1. Review `src/cali/models.py` to understand the schema
2. Try the examples in `examples_sqlmodel_usage.py`
3. Import your existing JSON data
4. Experiment with queries for your analysis workflows
5. Consider integrating into your GUI application

## Questions?

The schema is designed to match your existing `ROIData` structure closely, making migration straightforward. Each field in `ROIData` has a corresponding field in the `ROI` model.

For complex queries or custom analysis, see the SQLModel documentation: https://sqlmodel.tiangolo.com/
