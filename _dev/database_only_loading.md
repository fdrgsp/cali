# Feature: Load Database Without Data Path (Database-Only Mode)

## Context

When a user has already completed the extraction step, they should be able to open a `.cali` database file **without** providing a data path. This enables reviewing results, running analysis, and exporting — all without needing access to the original imaging data. Currently, `data_path` is required for the "From Database" tab, which blocks this workflow.

---

## Overview of Changes

The feature introduces a "database-only" mode: when `data_path` is `None` and only `database_path` is provided, the app initializes from the database alone. The image viewer will be blank (no raw frames), but the plate viewer, FOV table, graphs, and analysis pipeline will all work using metadata stored in the database.

---

## 1. `_InputDialog` (`src/cali/gui/_init_dialog.py`)

### 1a. Make `data_path` optional in the "From Database" tab

- Change label from `"Data Path"` to `"Data Path*"` for `self._browse_data_db` (line 55).
- Add a `QLabel` below the browse widgets with italic/small text:
  `"*Optional: can be omitted if the database already contains extraction results."`
- Update `value()` (line 154): allow returning `InputDialogData` with `database_path` set and `data_path=None`. Currently, both are returned; the change is that `data_path=None` is now a valid output when the "From Database" tab is selected.

No changes needed to `InputDialogData` — it already supports `data_path: str | None = None`.

---

## 2. `_show_data_input_dialog` (`src/cali/gui/_cali_gui.py`, line ~1931)

### 2a. Handle database-only case

Currently (line 1945):
```python
if value.database_path is not None and value.data_path is not None:
```

Add a new branch for database-only:
```python
# database with data
if value.database_path is not None and value.data_path is not None:
    self._initialize_from_database(value.database_path, value.data_path)
# database only (no data path)
elif value.database_path is not None and value.data_path is None:
    self._initialize_from_database_only(value.database_path)
# from directories
elif (data_path := value.data_path) is not None:
    ...
```

---

## 3. New method: `_initialize_from_database_only` (`src/cali/gui/_cali_gui.py`)

Add a new method (near `_initialize_from_database`, ~line 657):

```python
def _initialize_from_database_only(self, database_path: str | Path) -> None:
```

This method:
1. Shows loading bar.
2. Calls `_clear_widget_before_initialization()`.
3. Checks that the `.cali` file exists on disk.
4. Loads `Experiment.load_from_database(database_path, load_data=False)`.
5. **Validates** that the database has at least one extraction run (query `ExtractionSettings` or `Traces` table). If not, show an error: "Database has no extraction results. Data path is required."
6. Sets `self._database_path`, `self._output_path = str(Path(database_path).parent)`. Leaves `self._data = None` and `self._data_path = None`.
7. Calls `_update_graph_properties(self._database_path)`.
8. Calls `_finalize_initialization(experiment)` — this already works without `self._data`.
9. Disables run options that require raw data (see section 6).

---

## 4. `_on_scene_well_changed` (`src/cali/gui/_cali_gui.py`, line ~2416)

### 4a. Populate FOV table from database when `self._data` is `None`

Currently returns early if `self._data is None` (line 2427). Instead:

- If `self._data is None` and `self._database_path is not None`:
  - Query the database for FOVs belonging to the selected well:
    ```python
    stmt = select(FOV.name, FOV.position_index).join(Well).where(Well.name == well_name)
    ```
  - For each result, create a `WellInfo(pos_idx=fov.position_index, fov=useq.Position(name=fov.name))` and add to FOV table.
  - Select the first row.
- If `self._data is None` and `self._database_path is None`: return early (current behavior).

This means the FOV table will be populated purely from DB metadata (FOV name + position index).

---

## 5. `_on_fov_table_selection_changed` (`src/cali/gui/_cali_gui.py`, line ~2481)

### 5a. Handle `self._data is None` gracefully

Currently returns early if `self._data is None` (line 2492). Instead:

- If `self._data is None`:
  - Skip the image loading (`self._data.isel(...)`) and set `self._image_viewer.setData(None, None, None)` (blank viewer).
    - Alternatively: still load ROI labels from DB and pass `setData(None, roi_labels, neuropil_labels)` — but the viewer already handles `data=None` by clearing, so labels without a background image aren't useful. Keep it simple: pass all `None`.
  - Still update the graphs combo (`_update_single_wells_graphs_combo`) with the FOV name so that graph plots (traces, correlations) still load from the database.

---

## 6. Run Options: Disable Detection/Extraction in Database-Only Mode

### 6a. `_update_options_availability` in `_RunCaliWidget` (`src/cali/gui/_run_widget.py`, line ~435)

Add a new parameter `has_data: bool = True`:

```python
def _update_options_availability(
    self, has_detections: bool, has_extractions: bool, has_runs: bool = False,
    *, has_data: bool = True,
) -> None:
```

When `has_data=False`, **disable** all options that involve detection or extraction:
- Index 0: "Detection, Extraction and Analysis" → disabled
- Index 1: "Detection and Extraction" → disabled
- Index 2: "Extraction and Analysis" → disabled
- Index 3: "Detection Only" → disabled
- Index 4: "Extraction Only" → disabled
- Index 5: "Analysis Only" → **enabled** (if has_detections and has_extractions)
- Index 6: "Export Only" → **enabled** (if has_runs)

If `has_data=False` and current selection is one of the disabled options, auto-switch to "Analysis Only" (index 5) if available, otherwise "Export Only" (index 6).

### 6b. Propagate `has_data` flag

- In `_populate_settings` (`_cali_gui.py`, line ~879) or in `_finalize_initialization`, after populating detection/extraction/run IDs, call:
  ```python
  self._run_cali_wdg._update_options_availability(
      has_detections=..., has_extractions=..., has_runs=...,
      has_data=(self._data is not None),
  )
  ```
- The `populate_detection_settings`, `populate_extraction_settings`, and `populate_run_ids` methods in `_RunCaliWidget` all call `_update_options_availability` internally. These need to also pass through the `has_data` flag. Options:
  - Store `has_data` as a `_RunCaliWidget` instance variable (e.g., `self._has_data = True`) and use it in all `_update_options_availability` calls.
  - Set it via a public method: `self._run_cali_wdg.set_has_data(self._data is not None)` called during initialization.

**Recommended**: Add `set_has_data(has_data: bool)` method to `_RunCaliWidget` that stores the flag and re-calls `_update_options_availability`.

### 6c. Guard in `_on_cali_run` (`_cali_gui.py`, line ~1102)

Update the guard:
```python
if self._database_path is None:
    return

# For analysis-only / export-only, data is not needed
value = self._run_cali_wdg.value()
if self._data is None and (value.run_detection or value.run_extraction):
    return
```

Also update the position list fallback (lines 1138-1140):
```python
pos = value.positions or list(range(self._get_total_positions()))
```

Where `_get_total_positions()` is a helper that returns:
- `len(self._data.sequence.stage_positions)` if `self._data` is available
- Otherwise, queries `SELECT COUNT(*) FROM fov` from the database.

---

## 7. Image Viewer Behavior in Database-Only Mode

No code changes needed in `_ImageViewer` itself — it already handles `setData(None, None, None)` gracefully (clears the view). The change is in how `_on_fov_table_selection_changed` calls it (section 5).

Optionally: show a placeholder message in the viewer area like "No image data available" when in database-only mode. This would be a small UX improvement but is not required for correctness.

---

## 8. Summary of Files to Modify

| File | Changes |
|------|---------|
| `src/cali/gui/_init_dialog.py` | Label `"Data Path*"`, add optional legend label |
| `src/cali/gui/_cali_gui.py` | New `_initialize_from_database_only` method; update `_show_data_input_dialog`; update `_on_scene_well_changed` for DB-only FOV population; update `_on_fov_table_selection_changed` for no-data case; update `_on_cali_run` guard; add `_get_total_positions` helper |
| `src/cali/gui/_run_widget.py` | Add `has_data` parameter to `_update_options_availability`; add `set_has_data()` method; update `populate_*` methods to use stored `has_data` flag |

---

## 9. Tests

### New tests to write

- **`test_input_dialog_database_only`**: Verify `_InputDialog` returns `InputDialogData(data_path=None, database_path=..., ...)` when only database path is provided in the "From Database" tab.
- **`test_initialize_from_database_only`**: Mock a `.cali` database with extraction results. Verify the GUI initializes correctly: plate viewer populated, FOV table populates on well click, image viewer shows blank, graphs load.
- **`test_initialize_from_database_only_no_extraction`**: Mock a `.cali` database without extraction results. Verify the app shows an error and does not initialize.
- **`test_run_options_disabled_without_data`**: Verify that when `has_data=False`, detection/extraction options are disabled and only "Analysis Only" and "Export Only" are available.
- **`test_fov_table_populated_from_db`**: Verify `_on_scene_well_changed` populates the FOV table from DB queries when `self._data is None`.
- **`test_on_cali_run_analysis_only_without_data`**: Verify that "Analysis Only" can be triggered successfully when `self._data is None`.

### Existing tests to update

- Any test in `tests/test_gui_initialization.py` that asserts `data_path` is always required for "From Database" tab.
- Tests in `tests/test_gui_export_options.py` if they check run option availability.

---

## 10. Run pre-commit

After all changes:
```bash
pre-commit run --all-files
```
