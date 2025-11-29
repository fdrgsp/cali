# Cali - Calcium Imaging Analysis Pipeline

## Architecture Overview

Cali implements a three-stage analysis pipeline for calcium imaging data:

1. **Detection**: ROI segmentation using Cellpose or CaImAn → creates `FOV` and `ROI` with `Mask` objects
2. **Extraction**: Trace extraction and deconvolution → creates `Traces` objects with raw/corrected/dff/dec_dff data
3. **Analysis**: Peak detection, spike inference, burst analysis → creates `DataAnalysis` objects with peaks, amplitudes, IEI

### Database Schema (SQLModel)

- **Three-stage versioning**: Each run creates a `CaliResult` (analysis_result table) with unique combination of `DetectionSettings`, `ExtractionSettings`, `AnalysisSettings`
- **Trace history**: Each `ROI` has `traces_history: list[Traces]` allowing multiple analysis runs on same detection
- **Analysis history**: Each `ROI` has `data_analysis_history: list[DataAnalysis]` for different analysis parameters
- **Stimulation support**: `ROI.stimulated` and `ROI.active` flags enable evoked experiment analysis

Key relationships:
```python
FOV.rois → list[ROI]
ROI.traces_history → list[Traces]  # filtered by analysis_result_id
ROI.data_analysis_history → list[DataAnalysis]  # filtered by analysis_result_id
Traces.analysis_result_id → links to CaliResult
```

## Critical Development Patterns

### 1. Testing Requirements (see vscode-agent.instructions.md)

- **Every code change must have tests** - use function-based tests, avoid classes
- **Run tests after changes**: `pytest tests/ -x`  
- **Check coverage**: `pytest --cov=src/cali --cov-report=html`
- **Never hide warnings** - fix root cause instead of adding `# noqa` or `warnings.filterwarnings`
- Pytest config treats warnings as errors (`filterwarnings = ["error"]`)

### 2. Database Queries for Plotting

When querying data for plots, **always join ROI, Traces, and DataAnalysis** by `analysis_result_id`:

```python
from sqlmodel import Session, select, col
from cali.sqlmodel._model import FOV, ROI, Traces, DataAnalysis

# Standard pattern for plotting queries
stmt = (
    select(ROI, Traces, DataAnalysis)
    .join(FOV, ROI.fov_id == FOV.id)
    .join(
        Traces,
        (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
    )
    .outerjoin(  # Use outerjoin if DataAnalysis might not exist
        DataAnalysis,
        (DataAnalysis.roi_id == ROI.id)
        & (DataAnalysis.analysis_result_id == run_id),
    )
    .where(col(FOV.name) == fov_name)
)
```

**Never query old JSON files** - all data is in the database. Traces are in JSON columns (raw_trace, dec_dff, inferred_spikes), analysis data in separate columns (peaks_dec_dff, peaks_amplitudes_dec_dff, iei).

### 3. Running the Pipeline

Use `CaliRunner` as the unified interface (don't call DetectionRunner/ExtractionRunner/AnalysisRunner directly unless manual workflows):

```python
from cali.runner import CaliRunner
from cali.sqlmodel import DetectionSettings, ExtractionSettings, AnalysisSettings

runner = CaliRunner()

# Detection + Extraction + Analysis in one call
runner.run(
    experiment=exp,
    dataset_path=data_path,
    detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
    extraction_settings=ExtractionSettings(neuropil_inner_radius=10),
    analysis_settings=AnalysisSettings(peaks_height_value=2),
    global_position_indices=[0],
    database_name="results.cali",
    output_path=output_dir,
)
```

Settings can be passed as objects or database IDs (int) to reuse existing settings.

### 4. Manual Pipeline Execution

For step-by-step control (tests often use this pattern):

```python
from cali.util import update_fovs_in_database, load_fovs_from_database

# 1. Detection
fovs = detection_runner.run(dataset, detection_settings, [0])
update_fovs_in_database(db_path, fovs)

# 2. Extraction (must reload FOVs to get database IDs)
fovs = load_fovs_from_database(engine, [0])
for fov in extraction_runner.run(dataset, extraction_settings, fovs, as_generator=True):
    update_fovs_in_database(db_path, fov)

# 3. Analysis
fovs = load_fovs_from_database(engine, [0])  # Reload to get traces
for fov in analysis_runner.run(fovs, analysis_settings, as_generator=True):
    update_fovs_in_database(db_path, fov)
```

### 5. Evoked Experiments

Check `ROI.stimulated` flag to differentiate stimulated vs non-stimulated cells:

```python
stmt = stmt.where(col(ROI.stimulated) == True)  # Only stimulated
# or
stimulated_rois = [roi for roi, trace, da in results if roi.stimulated]
```

Stimulation mask is stored in `Mask` table and linked via `AnalysisSettings.stimulation_mask_id`.

## File Structure

- `src/cali/runner/` - `CaliRunner` orchestrates pipeline
- `src/cali/detection/` - Cellpose/CaImAn segmentation
- `src/cali/extraction/` - Trace extraction, neuropil correction, deconvolution
- `src/cali/analysis/` - Peak detection, IEI, frequency calculations
- `src/cali/plot/` - Matplotlib plotting functions (query database, don't load JSON)
- `src/cali/sqlmodel/` - Database models and utilities
- `src/cali/util/` - `commit_fov_result`, `load_fovs_from_database`, `update_fovs_in_database`
- `tests/` - Function-based pytest tests with mocked cellpose

## Common Pitfalls

1. **Don't call `session.add()` on objects already in session** - causes `InvalidRequestError`. Use `if obj not in session: session.add(obj)`
2. **Clear stale relationships before reassigning** - set `trace.roi = None` before changing to new ROI
3. **Always reload FOVs after database commits** - relationships need fresh database state
4. **Use `analysis_result_id` to filter traces/analysis** - don't just grab `traces_history[0]`
5. **Plotting uses database queries, not JSON files** - old analysis directory approach is deprecated

## Running Tests

```bash
# All tests
pytest tests/ -x

# Specific test file
pytest tests/test_runners.py -x

# With coverage
pytest --cov=src/cali --cov-report=html tests/

# Pre-commit hooks (ruff, mypy, typos)
pre-commit run --all-files
```

## Key Commands

- **Install dependencies**: `uv pip install -e ".[dev]"`
- **Run GUI**: `python -m cali` (requires PyQt6)
- **Inspect database**: Use DBeaver or `sqlite3 path/to/results.cali`
