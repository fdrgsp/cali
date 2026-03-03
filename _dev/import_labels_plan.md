# Plan: Import Pre-existing Labels Feature

## Context

Currently, users must run Cellpose detection to generate cell masks/labels before extraction and analysis. Some users already have label TIFFs (integer-valued images: 0=background, 1..N=cell labels) and want to import them directly, bypassing Cellpose entirely. This feature adds an "Imported Labels" option in the Detection tab alongside the existing Cellpose groupbox.

## Architecture Summary

### Flow
1. User checks "Imported Labels" groupbox (unchecks "Cellpose" automatically)
2. Clicks "Import Labels..." button -> opens a dialog
3. In dialog: browses for labels folder, assigns label TIFFs to specific well/FOV combinations
4. On OK: labels are immediately converted to ROI/Mask objects and committed to the DB with `DetectionSettings(method="imported")`
5. When running extraction: the stored `detection_settings_id` is passed as an `int`, so the runner skips detection entirely and uses existing ROIs

### Key insight
The runner already handles `detection_settings: DetectionSettings | int`. When passed an `int`, it looks up existing settings and checks which positions already have ROIs. Since imported labels are pre-committed, `positions_for_detection` will be empty and Cellpose is never called.

---

## Changes

### 1. New file: `src/cali/gui/_import_labels_dialog.py`

Create `_ImportLabelsDialog(QDialog)` - a dialog for assigning label TIFFs to specific FOVs.

**Constructor**: `__init__(self, database_path: str, parent: QWidget | None = None)`

**Layout**:
```
+---------------------------------------------------------------+
| Import Label TIFFs                                            |
+-----------------------------+---------------------------------+
| Labels Folder: [path] [Browse]                                |
+-----------------------------+---------------------------------+
| Available Label Files:      | Experiment FOVs:                |
| +-------------------------+ | > Well A1                       |
| | label_001.tif           | |   A1_0000 (pos 0) [unassigned] |
| | label_002.tif           | |   A1_0001 (pos 1) [assigned]   |
| | label_003.tif           | | > Well B2                       |
| +-------------------------+ |   B2_0000 (pos 4) [unassigned] |
|                             |                                 |
| [Assign to FOV ->] [<- Unassign]                              |
+-----------------------------+---------------------------------+
|                                        [OK]  [Cancel]         |
+---------------------------------------------------------------+
```

**Key components**:
- `_BrowseWidget` (reuse from `_util.py`) for folder selection - connected to populate available files list
- Left `QListWidget`: available `.tif`/`.tiff` files from browsed folder
- Right `QTreeWidget`: wells/FOVs from database, grouped by well. Each FOV item shows assigned label file or "[unassigned]"
- Assign/Unassign buttons for FOV-level assignment (not well-level - user may only have labels for specific FOVs)

**Data structures**:
- `_label_files: list[Path]` - all label TIFFs in browsed folder
- `_label_map: dict[int, Path]` - maps `fov.id` -> label TIFF path (keyed by DB FOV id for precision)
- `_db_fovs: list[tuple[str, str, int, int]]` - `(well_name, fov_name, position_index, fov_id)` from DB

**`_query_fovs_from_database()`**: Query `SELECT Well.name, FOV.name, FOV.position_index, FOV.id FROM FOV JOIN Well ORDER BY Well.name, FOV.fov_number` to populate the FOV tree.

**`_import_labels_to_database() -> int`** (called on OK):
1. Create `DetectionSettings(method="imported")`, deduplicate via `_resolve_settings` pattern (query existing, compare with `__eq__`)
2. For each `(fov_id, label_path)` in `_label_map`:
   - Read TIFF with `tifffile.imread()` -> 2D int array
   - Extract unique label values > 0
   - For each label value: create binary mask, call `mask_to_coordinates()` (from `cali.util`), create `Mask` + `ROI` objects
   - Use `commit_fov_result()` (from `cali.util`) to persist (handles finding existing FOV by position_index)
3. Commit and return `detection_settings_id`

**`value() -> int | None`**: Returns the `detection_settings_id` after successful import.

### 2. Modify: `src/cali/gui/_detection_gui.py`

**Make groupboxes mutually exclusive**:
- `_CellposeDetectionWidget`: add `self.setCheckable(True)` + `self.setChecked(True)` in `__init__`
- New `_ImportedLabelsWidget(QGroupBox)`: checkable, initially unchecked
- Connect `toggled` signals for mutual exclusion in `_DetectionGUI`

**New `_ImportedLabelsWidget(QGroupBox)`** (small class in same file):
- Title: "Imported Labels"
- Contains: status label ("No labels imported yet"), "Import Labels..." button
- Stores `_detection_settings_id: int | None` and `_database_path: str | None`
- `set_database_path(path)`: called by CaliGui when DB is initialized
- `detection_settings_id() -> int | None`: getter
- `_on_import_clicked()`: opens `_ImportLabelsDialog`, updates status on success

**`_DetectionGUI` changes**:
- Add `_imported_labels_wdg` to layout (below `_cellpose_wdg` in the scroll area)
- Add `active_method() -> Literal["cellpose", "imported"]`
- Update `to_model_settings()`: if cellpose checked, return cellpose settings; if imported checked, return `DetectionSettings(method="imported")`
- Update `setValue()`: accept optional `method` param to restore "imported" state
- Update `enable()`, `reset()` to handle both widgets

### 3. Modify: `src/cali/gui/_cali_gui.py`

**Pass DB path** (~after `_database_path` is set in init methods):
```python
self._detection_wdg._imported_labels_wdg.set_database_path(self._database_path)
```

**`_on_cali_run()` (~line 1362)**: Before the existing `detection_settings = self._detection_wdg.to_model_settings()`:
```python
if self._detection_wdg.active_method() == "imported":
    det_id = self._detection_wdg._imported_labels_wdg.detection_settings_id()
    if det_id is None:
        show_error_dialog(self, "No labels imported. Click 'Import Labels...' first.")
        return
    detection_settings = det_id  # pass as int -> runner skips detection
```

**`_on_run_item_selected()` (~line 2194)**: Add `elif d_settings.method == "imported":` branch to restore imported labels state in GUI.

**`_on_save_settings()` / `_on_load_settings()`**: Guard for imported method - save `{"method": "imported", "detection_settings_id": N}` instead of cellpose dataclass.

### 4. Modify: `src/cali/detection/_detection_runner.py`

**`_run_generator()` (~line 103)**: Add defensive guard:
```python
elif detection_settings.method == "imported":
    return  # imported labels already in DB, nothing to run
```

This path should never be reached (runner skips detection when positions already have ROIs), but it's a safety net.

### No schema changes needed
`DetectionSettings.method` is already a `str` field. `DetectionSettings(method="imported")` with default values for cellpose fields will deduplicate correctly via existing `__eq__`/`__hash__`.

---

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `src/cali/gui/_import_labels_dialog.py` | **New** | Dialog for browsing labels folder and assigning TIFFs to FOVs |
| `src/cali/gui/_detection_gui.py` | Modify | Add `_ImportedLabelsWidget`, make groupboxes checkable/exclusive, add `active_method()` |
| `src/cali/gui/_cali_gui.py` | Modify | Pass DB path, handle "imported" in run/restore/save/load flows |
| `src/cali/detection/_detection_runner.py` | Modify | Add `"imported"` method guard (1 line) |

## Reusable existing code
- `_BrowseWidget` from `src/cali/gui/_util.py` - for folder browsing in dialog
- `mask_to_coordinates()` from `src/cali/util/_util.py` - convert label arrays to sparse coords
- `commit_fov_result()` from `src/cali/util/_util.py` - persist FOV/ROI/Mask to DB
- `create_divider_line()` from `src/cali/gui/_util.py` - visual dividers
- `show_error_dialog()` from `src/cali/gui/_util.py` - error dialogs
- `TiffCollectionWidget` pattern from `src/cali/gui/_tiff_collection_widget.py` - reference for dialog layout

## Verification
1. Launch GUI, load an existing experiment
2. Go to Detection tab - verify both groupboxes appear, only one can be checked at a time
3. Check "Imported Labels", click "Import Labels..."
4. In dialog: browse to a folder with label TIFFs, verify FOV tree shows wells/FOVs from DB
5. Assign a label file to a specific FOV, click OK
6. Verify status label updates with count and detection ID
7. Run "Extraction Only" or "Detection + Extraction" - verify Cellpose is skipped and extraction uses imported masks
8. Verify imported ROIs appear in the image viewer overlay
9. Test save/load settings with imported method
10. Test restoring a run that used imported labels via run history panel
