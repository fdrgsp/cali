# CaliRunner: Complete Scenario Matrix

This document catalogs ALL possible CaliRunner execution scenarios to ensure proper test coverage and correct behavior.

## Core Concepts

### Pipeline Stages
1. **Detection** - ROI segmentation (creates `FOV`, `ROI`, `Mask`)
2. **Extraction** - Trace extraction from ROIs (creates `Traces`)
3. **Analysis** - Peak/frequency analysis on traces (creates `DataAnalysis`, `FOVAnalysis`)

### Settings Objects
- `DetectionSettings` - Configuration for detection stage (method, model_type, etc.)
- `ExtractionSettings` - Configuration for extraction stage (neuropil, dff_window, etc.)
- `AnalysisSettings` - Configuration for analysis stage (peak thresholds, etc.)

### CaliResult Record
A `CaliResult` tracks a unique combination of settings and accumulates positions:
- `detection_settings_id` - Always required (foreign key)
- `extraction_settings_id` - Set when extraction was run (foreign key)
- `analysis_settings_id` - Set when analysis was run (foreign key)
- `positions_detected` - List of positions that have detection data
- `positions_extracted` - List of positions that have extraction data
- `positions_analyzed` - List of positions that have analysis data

**Key Principle**: A CaliResult is uniquely identified by the combination of (det_id, ext_id, ana_id). Running with the same settings UPDATES the existing result; running with different settings CREATES a new result.

---

## Progressive Accumulation Example

From `_dev/_desired_logic.md`, here's how a CaliResult accumulates over multiple runs:

### Starting State
```
CaliResult: None
```

### Run 1: Detection Only on [0], det_id=1
```
CaliResult ID: 1 | det_id: 1 | ext_id: None | ana_id: None
positions_detected: [0]
positions_extracted: None
positions_analyzed: None
```

### Run 2: Detection Only on [0,1], det_id=1
- Skip detection on [0] (exists), run on [1]
- **Updates existing CaliResult 1**
```
CaliResult ID: 1 | det_id: 1 | ext_id: None | ana_id: None
positions_detected: [0, 1]
```

### Run 3: Det+Ext on [0,2], det_id=1, ext_id=1
- Skip detection on [0] (exists), run on [2]
- Run extraction on [0, 2]
- **Updates existing CaliResult 1** (adds ext_id)
```
CaliResult ID: 1 | det_id: 1 | ext_id: 1 | ana_id: None
positions_detected: [0, 1, 2]
positions_extracted: [0, 2]
```

### Run 4: Analysis Only on [0,3], det_id=1, ext_id=1, ana_id=1
- Skip detection on [0] (exists), run on [3]
- Skip extraction on [0] (exists), run on [3]
- Run analysis on [0, 3]
- **Updates existing CaliResult 1** (adds ana_id)
```
CaliResult ID: 1 | det_id: 1 | ext_id: 1 | ana_id: 1
positions_detected: [0, 1, 2, 3]
positions_extracted: [0, 2, 3]
positions_analyzed: [0, 3]
```

### Run 5: Full Pipeline on [0,1,4], det_id=1, ext_id=1, ana_id=1
- Skip detection on [0,1] (exists), run on [4]
- Skip extraction on [0] (exists), run on [1, 4]
- Skip analysis on [0] (exists), run on [1, 4]
- **Updates existing CaliResult 1**
```
CaliResult ID: 1 | det_id: 1 | ext_id: 1 | ana_id: 1
positions_detected: [0, 1, 2, 3, 4]
positions_extracted: [0, 1, 2, 3, 4]
positions_analyzed: [0, 1, 3, 4]
```

### Run 6: Full Pipeline on [0,5], det_id=2 (NEW), ext_id=1, ana_id=1
- Detection settings changed → run detection on ALL [0, 5]
- Run extraction on [0, 5] (new detection requires new extraction)
- Run analysis on [0, 5]
- **Creates NEW CaliResult 2** (different det_id)
```
CaliResult ID: 2 | det_id: 2 | ext_id: 1 | ana_id: 1
positions_detected: [0, 5]
positions_extracted: [0, 5]
positions_analyzed: [0, 5]
```

### Run 7: Full Pipeline on [0,1,4], det_id=1, ext_id=1, ana_id=2 (NEW)
- Skip detection on [0,1,4] (exists with det_id=1)
- Skip extraction on [0,1,4] (exists with ext_id=1)
- Run analysis on [0,1,4] (new ana_id=2)
- **Creates NEW CaliResult 3** (different ana_id)
```
CaliResult ID: 3 | det_id: 1 | ext_id: 1 | ana_id: 2
positions_detected: [0, 1, 4]
positions_extracted: [0, 1, 4]
positions_analyzed: [0, 1, 4]
```

---

## Scenario Categories

### Category A: Fresh Run (No Prior Data)

| ID | Stages | Positions | Expected Behavior |
|----|--------|-----------|-------------------|
| A1 | Det only | [0] | Creates CaliResult(det=1, ext=None, ana=None), pos_det=[0] |
| A2 | Det only | [0,1,2] | Creates CaliResult(det=1, ext=None, ana=None), pos_det=[0,1,2] |
| A3 | Det only | None (all) | Creates CaliResult for all dataset positions |
| A4 | Det+Ext | [0] | Creates CaliResult(det=1, ext=1, ana=None), pos_det=[0], pos_ext=[0] |
| A5 | Det+Ext | [0,1] | Creates CaliResult(det=1, ext=1, ana=None), pos_det=[0,1], pos_ext=[0,1] |
| A6 | Det+Ext+Ana | [0] | Creates CaliResult(det=1, ext=1, ana=1), all pos lists = [0] |
| A7 | Det+Ext+Ana | [0,1] | Creates CaliResult(det=1, ext=1, ana=1), all pos lists = [0,1] |

### Category B: Detection Exists, Add Extraction

| ID | Prior State | New Run | Expected Behavior |
|----|-------------|---------|-------------------|
| B1 | Det [0] | Ext [0] | **Updates** CaliResult, adds ext_id, pos_ext=[0] |
| B2 | Det [0,1] | Ext [0] | **Updates** CaliResult, adds ext_id, pos_ext=[0] |
| B3 | Det [0,1] | Ext [0,1] | **Updates** CaliResult, adds ext_id, pos_ext=[0,1] |
| B4 | Det [0,1] | Ext [2] | **ERROR**: Position 2 has no detection |
| B5 | Det [0,1] | Ext [0,1,2] | **ERROR**: Position 2 has no detection |

### Category C: Detection+Extraction Exist, Add Analysis

| ID | Prior State | New Run | Expected Behavior |
|----|-------------|---------|-------------------|
| C1 | Det+Ext [0] | Ana [0] | **Updates** CaliResult, adds ana_id, pos_ana=[0] |
| C2 | Det+Ext [0,1] | Ana [0] | **Updates** CaliResult, adds ana_id, pos_ana=[0] |
| C3 | Det+Ext [0,1] | Ana [0,1] | **Updates** CaliResult, adds ana_id, pos_ana=[0,1] |
| C4 | Det+Ext [0,1] | Ana [2] | **ERROR**: Position 2 has no extraction |
| C5 | Det+Ext [0,1] | Ana [0,1,2] | **ERROR**: Position 2 has no extraction |

### Category D: Different Settings = New CaliResult

| ID | Prior State | New Run | Expected Behavior |
|----|-------------|---------|-------------------|
| D1 | Det[0] det_id=1 | Det[0] det_id=2 | **Creates NEW** CaliResult with det_id=2 |
| D2 | Det+Ext[0] det=1,ext=1 | Ext[0] det=1,ext=2 | **Creates NEW** CaliResult with ext_id=2 |
| D3 | Det+Ext+Ana[0] det=1,ext=1,ana=1 | Ana[0] det=1,ext=1,ana=2 | **Creates NEW** CaliResult with ana_id=2 |
| D4 | Det+Ext[0] det=1,ext=1 | Full[0] det=2,ext=1,ana=1 | **Creates NEW** CaliResult (different det_id means different result) |

### Category E: Mixed State (Some Positions Ahead of Others)

| ID | Prior State | New Run | Expected Behavior |
|----|-------------|---------|-------------------|
| E1 | Det [0,1], Ext [0] | Ext [0,1] | Skip ext [0], run ext [1] |
| E2 | Det [0,1], Ext [0] | Ext+Ana [0,1] | Skip ext [0], run ext [1], run ana [0,1] |
| E3 | Det+Ext [0,1], Ana [0] | Ana [0,1] | Skip ana [0], run ana [1] |
| E4 | Det+Ext [0,1], Ana [0] with ana_id=1 | Ana [0,1] with ana_id=2 | Run ana [0,1] (new settings → new CaliResult) |

### Category F: Force Re-run

| ID | Prior State | New Run (force=True) | Expected Behavior |
|----|-------------|---------------------|-------------------|
| F1 | Det [0] | Det [0] | Deletes old ROIs, runs detection fresh |
| F2 | Det+Ext [0] | Ext [0] | Deletes old Traces, runs extraction fresh |
| F3 | Det+Ext+Ana [0] | Ana [0] | Deletes old DataAnalysis, runs analysis fresh |
| F4 | Det+Ext [0] | Ext+Ana [0] | Deletes Traces+DataAnalysis, runs both fresh |

### Category G: Cancellation During Run

| ID | Run Type | Cancel Point | Expected Behavior |
|----|----------|--------------|-------------------|
| G1 | Det+Ext+Ana [5,6,7] | After det [5], before det [6] | CaliResult pos_det=[5], pos_ext=[], pos_ana=[] |
| G2 | Det+Ext+Ana [5,6,7] | After ext [6], before ext [7] | pos_det=[5,6,7], pos_ext=[5,6], pos_ana=[] |
| G3 | Det+Ext+Ana [5,6,7] | After ana [6], before ana [7] | pos_det=[5,6,7], pos_ext=[5,6,7], pos_ana=[5,6] |

---

## Verification Points for Each Record Type

### FOV Records
- One FOV per position per experiment
- `position_index` should be unique within experiment

### ROI Records
- Multiple ROIs per FOV
- `detection_settings_id` links to the settings used
- `roi_mask` relationship should be populated

### Traces Records
- One Traces record per ROI per extraction run
- `analysis_result_id` links to CaliResult
- Contains: raw_trace, dff, dec_dff, inferred_spikes, x_axis

### DataAnalysis Records
- One DataAnalysis per ROI per analysis run
- `analysis_result_id` links to CaliResult
- Contains: peaks, frequencies, IEI, thresholds

### FOVAnalysis Records
- One FOVAnalysis per FOV per analysis run
- `analysis_result_id` links to CaliResult
- `fov_id` links to FOV
- Contains: CCG data, burst detection data

### CaliResult Records
- Unique combination of (det_id, ext_id, ana_id)
- Position lists track which positions have each stage complete
- `last_modified` updates on every change

---

## Current Known Bug

**Issue**: `_run_analysis_only` method only calls `compute_fov_analysis` (FOV-level), but does NOT call `AnalysisRunner` which creates the ROI-level `DataAnalysis` records.

**Impact**: When running analysis-only:
- No `DataAnalysis` records are created (peaks, frequencies, IEI)
- Visualization tab shows no data
- Combo boxes are disabled because there's no analysis data

**Fix Required**: `_run_analysis_only` must:
1. Call `AnalysisRunner.run()` on the loaded FOVs to create `DataAnalysis` records
2. Call `compute_fov_analysis()` for FOV-level analysis
3. Properly link all records to the CaliResult via `analysis_result_id`

---

### Category H: Settings Deduplication

| ID | Action | Expected Behavior |
|----|--------|-------------------|
| H1 | Create DetectionSettings(method="cellpose") twice | Second call returns same ID as first |
| H2 | Create ExtractionSettings(threads=4) twice | Second call returns same ID as first |
| H3 | Create AnalysisSettings(threads=4) twice | Second call returns same ID as first |
| H4 | Pass settings ID instead of object | Loads existing settings from DB |

### Category I: Edge Cases

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| I1 | Run with positions=[] (empty list) | No-op, no CaliResult created |
| I2 | Run on position not in dataset | Error or skip gracefully |
| I3 | Detection finds 0 ROIs | FOV created but no ROIs, CaliResult still tracks position |
| I4 | Run with None positions (all) on 10-position dataset | Processes all 10 positions |

### Category J: Idempotent Re-runs (Same Settings, Same Positions)

| ID | Prior State | Re-run | Expected Behavior |
|----|-------------|--------|-------------------|
| J1 | Det [0] | Det [0] | Skip all, no changes |
| J2 | Det+Ext [0] | Ext [0] | Skip all, no changes |
| J3 | Det+Ext+Ana [0] | Ana [0] | Skip all, no changes |
| J4 | Det+Ext+Ana [0] | Full [0] | Skip all, no changes |

### Category K: Cross-CaliResult Data Sharing

| ID | Scenario | Question to Answer |
|----|----------|-------------------|
| K1 | Det+Ext [0] with ext_id=1, then Ana [0] with ana_id=1 | Does analysis use existing traces? (YES) |
| K2 | Det+Ext+Ana [0] with ana_id=1, then Ana [0] with ana_id=2 | Are new Traces created or existing reused? |
| K3 | Two CaliResults with same det_id, different ext_id | Do they share ROIs? (YES, ROIs linked to det_id) |
| K4 | Two CaliResults with same det_id+ext_id, different ana_id | Do they share Traces? How are DataAnalysis linked? |

### Category L: Complex Mixed States

| ID | Prior State | New Run | Expected Behavior |
|----|-------------|---------|-------------------|
| L1 | Det [0,1,2], Ext [0,1], Ana [0] | Full [0,1,2] | Skip det all, skip ext [0,1] run ext [2], skip ana [0] run ana [1,2] |
| L2 | Det [0,1], Ext [0] | Ana [0,1] | **ERROR**: Position 1 has no extraction for analysis |
| L3 | Det [0,1], Ext [0] | Ext+Ana [0,1] | Skip ext [0], run ext [1], run ana [0,1] |
| L4 | Det [0] det_id=1, Det [0] det_id=2 | Ext [0] det_id=1 | Must specify which detection to use |

### Category M: Multiple Experiments

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| M1 | Exp1 with Det [0], then Exp2 with Det [0] | Separate FOVs, ROIs, CaliResults per experiment |
| M2 | Same database, different experiment names | Each experiment isolated |

---

## Open Questions (Need Clarification)

1. **K2/K4**: When running analysis with new ana_id on positions that already have extraction:
   - Are NEW Traces records created with new analysis_result_id?
   - Or do we reuse existing Traces and only create new DataAnalysis?

2. **L4**: If a position has detection with multiple det_ids, how does extraction know which to use?
   - Currently: User must specify det_id when running extraction
   - Or: Use most recent detection?

3. **Traces.analysis_result_id**: This links Traces to a specific CaliResult. But if we run analysis-only with new ana_id:
   - Should we create NEW Traces copies pointing to new CaliResult?
   - Or should Traces be linked to CaliResult via (det_id, ext_id) not specific result_id?

---

## Test Matrix Summary

| Category | # Tests | Priority |
|----------|---------|----------|
| A: Fresh Run | 7 | High |
| B: Det→Ext | 5 | High |
| C: Det+Ext→Ana | 5 | **Critical** (current bug) |
| D: Different Settings | 4 | High |
| E: Mixed State | 4 | High |
| F: Force Re-run | 4 | Medium |
| G: Cancellation | 3 | Medium |
| H: Settings Dedup | 4 | Medium |
| I: Edge Cases | 4 | Medium |
| J: Idempotent Re-runs | 4 | High |
| K: Cross-CaliResult | 4 | **Critical** (design question) |
| L: Complex Mixed | 4 | High |
| M: Multi-Experiment | 2 | Low |
| **Total** | **54** | |
