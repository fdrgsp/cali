# SQLModel Implementation Summary

## What Was Created

I've analyzed your calcium imaging analysis codebase and created a comprehensive SQLModel database schema to replace/augment your current JSON-based data storage.

### Files Created

1. **`src/cali/models.py`** (600+ lines)
   - Complete SQLModel schema with 7 tables
   - Helper functions for data conversion
   - Full documentation

2. **`examples_sqlmodel_usage.py`** (500+ lines)
   - Comprehensive working examples
   - Data import from JSON
   - Query examples
   - Export examples
   - Statistical analysis helpers

3. **`SQLMODEL_GUIDE.md`**
   - Complete integration guide
   - Migration strategies
   - Best practices
   - Troubleshooting

4. **`x.py`** (quick reference)
   - Minimal working example
   - Quick start code

## Database Schema

### Hierarchy

```
Experiment (top level)
  ├── name, description, paths
  └── Plate (96-well, 384-well, etc.)
      └── Well (e.g., "B5", "C3")
          ├── Condition 1 (genotype: WT, KO, etc.)
          ├── Condition 2 (treatment: Vehicle, Drug, etc.)
          └── FOV (Field of View - imaging position)
              └── ROI (individual cell/neuron)
                  ├── Traces (raw, corrected, dff, dec_dff)
                  ├── Peaks (indices, amplitudes, parameters)
                  ├── Spikes (inferred spikes, threshold)
                  ├── Metadata (size, activity, time)
                  ├── Evoked data (stimulation info)
                  ├── Network parameters (synchrony, burst)
                  └── Masks (spatial coordinates)
```

### Models

1. **Experiment**: Container for experiments with metadata
2. **Plate**: Multi-well plate (tracks layout, type)
3. **Well**: Individual well (row, column, name)
4. **Condition**: Reusable experimental conditions (genotype, treatment)
5. **FOV**: Imaging position within a well
6. **ROI**: Complete analysis results for one cell
7. **AnalysisSettings**: Track parameters used for analysis

## Key Features

### ROI Model Highlights

The `ROI` model stores everything from your current `ROIData` dataclass:

- **Fluorescence Traces**:
  - `raw_trace`: Original fluorescence
  - `corrected_trace`: After neuropil correction
  - `neuropil_trace`: Neuropil signal
  - `dff`: ΔF/F normalized
  - `dec_dff`: Deconvolved trace
  - `elapsed_time_list_ms`: Timestamps

- **Calcium Events**:
  - `peaks_dec_dff`: Peak indices
  - `peaks_amplitudes_dec_dff`: Peak amplitudes
  - `peaks_prominence_dec_dff`: Detection threshold
  - `dec_dff_frequency`: Event frequency (Hz)
  - `iei`: Inter-event intervals

- **Spike Inference**:
  - `inferred_spikes`: Spike probabilities
  - `inferred_spikes_threshold`: Detection threshold

- **Cell Properties**:
  - `cell_size`: Area in µm² or pixels
  - `active`: Whether cell shows activity
  - `total_recording_time_sec`: Recording duration

- **Evoked Experiments**:
  - `evoked_experiment`: Boolean flag
  - `stimulated`: Overlaps with stimulation area
  - `stimulations_frames_and_powers`: Stim timing
  - `led_pulse_duration`, `led_power_equation`

- **Network Analysis**:
  - `calcium_sync_jitter_window`
  - `spikes_sync_cross_corr_lag`
  - `calcium_network_threshold`
  - Burst detection parameters

- **Spatial Data**:
  - `mask_coords_y`, `mask_coords_x`: ROI coordinates
  - `mask_height`, `mask_width`: Dimensions
  - Neuropil mask coordinates

### Relationships

Navigate between levels easily:

```python
# From ROI to experiment
roi.fov.well.plate.experiment

# From well to all ROIs
well.fovs[0].rois

# Get condition info
roi.fov.well.condition_1.name  # e.g., "WT"
roi.fov.well.condition_2.name  # e.g., "Vehicle"
```

## How to Use

### 1. Basic Setup

```python
from sqlmodel import create_engine, Session, select
from cali.models import create_tables, Experiment, Plate, Well, FOV, ROI

# Create database
engine = create_engine("sqlite:///my_analysis.db")
create_tables(engine)
```

### 2. Import Your Existing Data

```python
from pathlib import Path
from examples_sqlmodel_usage import example_import_json_data

# Import from your existing JSON files
json_dir = Path("tests/test_data/evoked/evk_analysis")
example_import_json_data(engine, json_dir)
```

This will:
- Read your JSON analysis files
- Import plate maps (genotype/treatment)
- Import settings.json
- Create proper database records with relationships
- Preserve all your analysis data

### 3. Query Data

```python
with Session(engine) as session:
    # All active ROIs
    active = session.exec(select(ROI).where(ROI.active)).all()
    
    # ROIs by condition
    wt_rois = session.exec(
        select(ROI)
        .join(FOV).join(Well).join(Condition, Well.condition_1_id == Condition.id)
        .where(Condition.name == "WT")
    ).all()
    
    # High-frequency cells
    high_freq = session.exec(
        select(ROI).where(ROI.dec_dff_frequency > 1.0)
    ).all()
```

### 4. Export for Analysis

```python
import pandas as pd

with Session(engine) as session:
    results = session.exec(
        select(ROI.dec_dff_frequency, Condition.name)
        .join(FOV).join(Well)
        .join(Condition, Well.condition_1_id == Condition.id)
    ).all()
    
    df = pd.DataFrame([r._asdict() for r in results])
    # Now use pandas/seaborn for plotting
```

## Integration Ideas

### Option 1: Parallel Storage (Recommended First Step)

Keep JSON export, add database export:

```python
def _save_analysis_data(self, fov_name: str):
    # Existing JSON save
    self._save_analysis_data_to_json(fov_name)
    
    # Also save to database
    if hasattr(self, 'db_engine') and self.db_engine:
        self._save_analysis_data_to_db(fov_name)
```

### Option 2: Database-First with JSON Backup

```python
def _save_analysis_data(self, fov_name: str):
    # Primary: database
    self._save_analysis_data_to_db(fov_name)
    
    # Backup: JSON (for compatibility)
    self._save_analysis_data_to_json(fov_name)
```

### Option 3: Enhanced Querying

Replace dictionary-based data structures with database queries:

**Before:**
```python
# Current approach
analysis_data: dict[str, dict[str, ROIData]] = {}
# {"B5_0000_p0": {"1": ROIData(...), "2": ROIData(...)}}

# Get ROIs for a condition
condition_rois = []
for fov_name, roi_dict in analysis_data.items():
    for roi_data in roi_dict.values():
        if roi_data.condition_1 == "WT":
            condition_rois.append(roi_data)
```

**After:**
```python
# Database approach
with Session(engine) as session:
    wt_rois = session.exec(
        select(ROI)
        .join(FOV).join(Well).join(Condition)
        .where(Condition.name == "WT")
    ).all()
```

### Option 4: Add Database UI

Add database selection to your PlateViewer GUI:

```python
class PlateViewer(QMainWindow):
    def __init__(self, ...):
        # ... existing code ...
        self.db_engine = None
        
        # Add menu item
        self.file_menu.addAction("Open Database...", self._open_database)
        
    def _open_database(self):
        db_path, _ = QFileDialog.getOpenFileName(
            self, "Open Database", "", "SQLite Database (*.db)"
        )
        if db_path:
            self.db_engine = create_engine(f"sqlite:///{db_path}")
            self._load_data_from_database()
```

## Benefits Over JSON

1. **Queryability**: SQL queries are much more powerful than nested dictionaries
2. **Relationships**: Automatic navigation between experiments, plates, wells, FOVs, ROIs
3. **Performance**: Indexed queries are fast even with thousands of ROIs
4. **Type Safety**: SQLModel validates data types
5. **Scalability**: Easy to add new fields or tables
6. **Statistics**: Direct integration with pandas for analysis
7. **Multi-user**: Can upgrade to PostgreSQL for shared access
8. **Versioning**: Track analysis settings used for each ROI

## Example Workflows

### Workflow 1: Batch Analysis Comparison

```python
# Run analysis with different parameters
settings_v1 = create_analysis_settings("conservative", peaks_height=5.0)
settings_v2 = create_analysis_settings("sensitive", peaks_height=2.0)

# Later, compare results
with Session(engine) as session:
    v1_rois = session.exec(
        select(ROI).where(ROI.analysis_settings_id == settings_v1.id)
    ).all()
    v2_rois = session.exec(
        select(ROI).where(ROI.analysis_settings_id == settings_v2.id)
    ).all()
```

### Workflow 2: Cross-Experiment Analysis

```python
# Analyze multiple experiments together
with Session(engine) as session:
    all_wt = session.exec(
        select(ROI)
        .join(FOV).join(Well).join(Condition)
        .where(Condition.name == "WT")
    ).all()
    
    # Aggregate across all experiments
    frequencies = [roi.dec_dff_frequency for roi in all_wt]
```

### Workflow 3: Progressive Analysis

```python
# Analyze wells as they complete
for well_name in completed_wells:
    with Session(engine) as session:
        well = session.exec(select(Well).where(Well.name == well_name)).first()
        # Process and display results immediately
        plot_well_results(well)
```

## Next Steps

1. **Review the models** in `src/cali/models.py`
2. **Try the examples** in `examples_sqlmodel_usage.py`:
   ```bash
   python examples_sqlmodel_usage.py
   ```
3. **Import your data**:
   ```python
   from pathlib import Path
   from examples_sqlmodel_usage import example_import_json_data
   from sqlmodel import create_engine
   
   engine = create_engine("sqlite:///my_data.db")
   json_dir = Path("path/to/your/evk_analysis")
   example_import_json_data(engine, json_dir)
   ```
4. **Experiment with queries** for your specific analysis needs
5. **Consider integration** into your existing PlateViewer

## Migration Path

**Phase 1: Validation** (Week 1)
- Import existing JSON data
- Verify completeness
- Run parallel storage

**Phase 2: Integration** (Week 2-3)
- Add database export to analysis pipeline
- Create query helpers for common operations
- Update plotting functions to use database

**Phase 3: Optimization** (Week 4+)
- Make database primary storage
- Add GUI integration
- Performance tuning

## Questions?

The schema is designed to match your existing `ROIData` structure exactly, so migration should be straightforward. Every field in your current dataclass has a corresponding field in the `ROI` model.

See:
- `SQLMODEL_GUIDE.md` - Complete integration guide
- `examples_sqlmodel_usage.py` - Working code examples
- `x.py` - Quick reference
- SQLModel docs: https://sqlmodel.tiangolo.com/
