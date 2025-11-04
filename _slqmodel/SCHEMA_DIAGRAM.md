# SQLModel Schema Diagram

## Entity Relationship Diagram

```
┌─────────────────┐
│   Experiment    │
│─────────────────│
│ id (PK)         │
│ name            │
│ description     │
│ created_at      │
│ data_path       │
│ labels_path     │
│ analysis_path   │
└────────┬────────┘
         │ 1
         │
         │ *
┌────────▼────────┐
│     Plate       │
│─────────────────│
│ id (PK)         │
│ experiment_id(FK)│
│ name            │
│ plate_type      │
│ rows            │
│ columns         │
└────────┬────────┘
         │ 1
         │
         │ *
┌────────▼────────┐       ┌─────────────────┐
│      Well       │◄──────┤   Condition     │
│─────────────────│  *  1 │─────────────────│
│ id (PK)         │       │ id (PK)         │
│ plate_id (FK)   │   ┌──►│ name            │
│ name            │   │   │ condition_type  │
│ row             │   │   │ color           │
│ column          │   │   │ description     │
│ condition_1_id ─┼───┘   └─────────────────┘
│ condition_2_id ─┼───┐
└────────┬────────┘   │
         │ 1          │
         │            │
         │ *          │
┌────────▼────────┐   │
│      FOV        │   │
│─────────────────│   │
│ id (PK)         │   │
│ well_id (FK)    │   │
│ name            │   │
│ position_index  │   │
│ fov_number      │   │
│ fov_metadata    │   │
└────────┬────────┘   │
         │ 1          │
         │            │
         │ *          │
┌────────▼────────────┼─────────────────────────┐
│              ROI (Region of Interest)         │
│───────────────────────────────────────────────│
│ id (PK)                                       │
│ fov_id (FK)                                   │
│ label_value                                   │
│ analysis_settings_id (FK) ──┐                 │
│                              │                 │
│ ═══ FLUORESCENCE TRACES ════│═════════════════│
│ raw_trace                    │                 │
│ corrected_trace              │                 │
│ neuropil_trace               │                 │
│ dff                          │                 │
│ dec_dff                      │                 │
│ elapsed_time_list_ms         │                 │
│                              │                 │
│ ═══ CALCIUM EVENTS ══════════│═════════════════│
│ peaks_dec_dff                │                 │
│ peaks_amplitudes_dec_dff     │                 │
│ peaks_prominence_dec_dff     │                 │
│ peaks_height_dec_dff         │                 │
│ dec_dff_frequency            │                 │
│ iei                          │                 │
│                              │                 │
│ ═══ SPIKE INFERENCE ═════════│═════════════════│
│ inferred_spikes              │                 │
│ inferred_spikes_threshold    │                 │
│                              │                 │
│ ═══ CELL PROPERTIES ═════════│═════════════════│
│ cell_size                    │                 │
│ cell_size_units              │                 │
│ total_recording_time_sec     │                 │
│ active                       │                 │
│ neuropil_correction_factor   │                 │
│                              │                 │
│ ═══ EVOKED EXPERIMENT ═══════│═════════════════│
│ evoked_experiment            │                 │
│ stimulated                   │                 │
│ stimulations_frames_powers   │                 │
│ led_pulse_duration           │                 │
│ led_power_equation           │                 │
│                              │                 │
│ ═══ NETWORK ANALYSIS ════════│═════════════════│
│ calcium_sync_jitter_window   │                 │
│ spikes_sync_cross_corr_lag   │                 │
│ calcium_network_threshold    │                 │
│ spikes_burst_threshold       │                 │
│ spikes_burst_min_duration    │                 │
│ spikes_burst_gaussian_sigma  │                 │
│                              │                 │
│ ═══ SPATIAL DATA ════════════│═════════════════│
│ mask_coords_y, mask_coords_x │                 │
│ mask_height, mask_width      │                 │
│ neuropil_mask_coords_y/x     │                 │
│ neuropil_mask_height/width   │                 │
└──────────────────────────────┼─────────────────┘
                               │
                               │
                               │ *
                    ┌──────────▼──────────┐
                    │  AnalysisSettings   │
                    │─────────────────────│
                    │ id (PK)             │
                    │ name                │
                    │ created_at          │
                    │ dff_window          │
                    │ decay_constant      │
                    │ peaks_height_value  │
                    │ peaks_height_mode   │
                    │ peaks_distance      │
                    │ ... (20+ settings)  │
                    └─────────────────────┘
```

## Data Flow Example

```
Experiment: "Optogenetic_Study_2024"
  └── Plate: "Plate1" (96-well)
      ├── Well: "B5"
      │   ├── Condition 1: "WT" (genotype)
      │   ├── Condition 2: "Vehicle" (treatment)
      │   └── FOV: "B5_0000_p0"
      │       ├── ROI #1: cell_size=120µm², freq=1.5Hz, active=True
      │       ├── ROI #2: cell_size=95µm², freq=0.8Hz, active=True
      │       └── ROI #3: cell_size=110µm², freq=0.0Hz, active=False
      │
      ├── Well: "B6"
      │   ├── Condition 1: "WT" (genotype)
      │   ├── Condition 2: "Drug_10uM" (treatment)
      │   └── FOV: "B6_0000_p0"
      │       ├── ROI #1: freq=2.1Hz ← Higher with drug!
      │       └── ROI #2: freq=1.9Hz
      │
      └── Well: "C5"
          ├── Condition 1: "KO" (genotype)
          ├── Condition 2: "Vehicle" (treatment)
          └── FOV: "C5_0000_p0"
              ├── ROI #1: freq=0.3Hz ← Lower in KO
              └── ROI #2: freq=0.4Hz
```

## Query Examples Visualization

### Query 1: Get all WT ROIs

```
SELECT ROI.*
FROM roi
JOIN fov ON roi.fov_id = fov.id
JOIN well ON fov.well_id = well.id
JOIN condition ON well.condition_1_id = condition.id
WHERE condition.name = 'WT'

Result:
┌─────┬──────────┬──────┬────────┐
│ ROI │ Well     │ Freq │ Active │
├─────┼──────────┼──────┼────────┤
│  1  │ B5 (WT)  │ 1.5  │  True  │
│  2  │ B5 (WT)  │ 0.8  │  True  │
│  3  │ B5 (WT)  │ 0.0  │  False │
│  1  │ B6 (WT)  │ 2.1  │  True  │
│  2  │ B6 (WT)  │ 1.9  │  True  │
└─────┴──────────┴──────┴────────┘
```

### Query 2: Compare conditions

```python
# Group by condition and calculate statistics
wt_freqs = [1.5, 0.8, 2.1, 1.9]  # WT ROIs (active only)
ko_freqs = [0.3, 0.4]             # KO ROIs (active only)

mean_wt = 1.575 Hz
mean_ko = 0.350 Hz
p_value = 0.003  # Significant difference!
```

### Query 3: Stimulated vs Non-stimulated

```
SELECT AVG(dec_dff_frequency)
FROM roi
WHERE evoked_experiment = True
GROUP BY stimulated

Result:
┌─────────────┬──────────────┐
│ Stimulated  │ Avg Freq (Hz)│
├─────────────┼──────────────┤
│ True        │     2.5      │
│ False       │     0.9      │
└─────────────┴──────────────┘
```

## Relationships in Code

```python
# Navigate from ROI to Experiment
roi = session.get(ROI, 1)
roi.fov                        # FOV object
roi.fov.well                   # Well object
roi.fov.well.plate             # Plate object
roi.fov.well.plate.experiment  # Experiment object

# Navigate from Experiment to ROIs
experiment.plates[0].wells[0].fovs[0].rois[0]

# Get conditions
roi.fov.well.condition_1.name  # "WT"
roi.fov.well.condition_2.name  # "Vehicle"

# Get analysis settings
roi.analysis_settings.peaks_height_value  # 3.0
```

## Storage Details

### JSON Fields (stored as JSON in DB)

These Python lists/dicts are automatically serialized:

```python
ROI(
    raw_trace=[100.0, 101.5, 102.3, ...],           # → JSON array
    dff=[0.0, 0.015, 0.023, ...],                   # → JSON array
    peaks_dec_dff=[45, 123, 267, ...],              # → JSON array
    mask_coords_y=[10, 10, 11, 11, ...],            # → JSON array
    stimulations_frames_and_powers={"100": 50},     # → JSON object
)
```

### Indexed Fields (fast queries)

These fields have database indexes for fast searching:

- `ROI.fov_id`
- `ROI.label_value`
- `FOV.well_id`
- `FOV.name`
- `Well.plate_id`
- `Well.name`
- `Condition.name`
- `Experiment.name`

## Memory Considerations

**Small dataset** (< 1000 ROIs):
- Everything fits in memory
- SQLite is perfect
- Query performance is excellent

**Medium dataset** (1000 - 100k ROIs):
- Use pagination for large queries
- Consider PostgreSQL
- Traces stored as JSON work fine

**Large dataset** (> 100k ROIs):
- Use PostgreSQL with connection pooling
- Consider separate trace storage (HDF5/Zarr)
- Store only metadata in database
- Keep trace references in ROI model

## Example: Complete Workflow

```python
# 1. Create experiment
exp = Experiment(name="MyExp", data_path="/path/to/data")

# 2. Create plate
plate = Plate(experiment_id=exp.id, name="Plate1")

# 3. Create conditions
wt = Condition(name="WT", condition_type="genotype")
drug = Condition(name="Drug", condition_type="treatment")

# 4. Create well with conditions
well = Well(
    plate_id=plate.id,
    name="B5",
    row=1, column=4,
    condition_1_id=wt.id,
    condition_2_id=drug.id,
)

# 5. Create FOV
fov = FOV(well_id=well.id, name="B5_0000_p0", position_index=0)

# 6. Create ROI with analysis data
roi = ROI(
    fov_id=fov.id,
    label_value=1,
    raw_trace=[...],      # Your raw fluorescence
    dff=[...],            # Calculated ΔF/F
    dec_dff=[...],        # Deconvolved trace
    peaks_dec_dff=[...],  # Peak indices
    cell_size=120.5,
    active=True,
    dec_dff_frequency=1.5,
)

# 7. Save everything
session.add_all([exp, plate, wt, drug, well, fov, roi])
session.commit()

# 8. Query later
wt_drug_rois = session.exec(
    select(ROI)
    .join(FOV).join(Well)
    .join(Condition, Well.condition_1_id == Condition.id)
    .where(Condition.name == "WT")
    .join(Condition, Well.condition_2_id == Condition.id)
    .where(Condition.name == "Drug")
).all()
```

## Visual: Before vs After

### Before (JSON dictionaries)

```python
analysis_data = {
    "B5_0000_p0": {
        "1": ROIData(condition_1="WT", condition_2="Vehicle", ...),
        "2": ROIData(condition_1="WT", condition_2="Vehicle", ...),
    },
    "B6_0000_p0": {
        "1": ROIData(condition_1="WT", condition_2="Drug", ...),
    },
}

# Query: Get all WT ROIs - need nested loops
wt_rois = []
for fov_name, roi_dict in analysis_data.items():
    for roi_data in roi_dict.values():
        if roi_data.condition_1 == "WT":
            wt_rois.append(roi_data)
```

### After (SQLModel database)

```python
# Simple, powerful query
wt_rois = session.exec(
    select(ROI)
    .join(FOV).join(Well)
    .join(Condition, Well.condition_1_id == Condition.id)
    .where(Condition.name == "WT")
).all()

# Can also filter, sort, paginate
wt_rois_active = session.exec(
    select(ROI)
    .join(FOV).join(Well)
    .join(Condition, Well.condition_1_id == Condition.id)
    .where(Condition.name == "WT", ROI.active == True)
    .order_by(ROI.dec_dff_frequency.desc())
    .limit(10)
).all()
```

## Files Reference

- `src/cali/models.py` - Complete schema definition
- `examples_sqlmodel_usage.py` - Working examples
- `SQLMODEL_GUIDE.md` - Integration guide
- `IMPLEMENTATION_SUMMARY.md` - This summary
- `x.py` - Quick reference
