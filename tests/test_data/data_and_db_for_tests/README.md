# Unified Test Data

This directory contains the **primary test dataset** used across all Cali tests.

## Files

- **evk.tensorstore.zarr**: Evoked activity dataset with 8 positions (4 wells × 2 FOVs) and 153 timepoints
  - Used for detection, extraction, analysis, and GUI tests
  - Contains LED stimulation events at frames [3, 53, 103]
  - Powers: [2.0, 4.0, 6.0] mW/mm²
  - Frame rate: 10.0 fps

- **test_db.cali**: Pre-built database with complete analysis results
  - Generated from evk.tensorstore.zarr
  - Contains 2 CaliResults with different extraction settings
  - Run 1: Positions [0, 1] (B5_0000, B5_0001) with neuropil correction
  - Run 2: All 8 positions (B5–B8, 2 FOVs each) without neuropil correction
  - Includes plate map with conditions: genotype (g1-g4) and treatment (t1-t4)
  - LED stimulation settings configured for evoked activity analysis

- **stimulation_mask.tif**: Binary mask indicating stimulated ROIs
  - Referenced by AnalysisSettings.stimulation_mask_path
  - Used for evoked activity analysis

- **tests.json**: Complete schema for regenerating test_db.cali
  - Contains all settings (detection, extraction, analysis)
  - Plate structure and condition mappings
  - Run configurations

- **rebuild_test_db.py**: Script to regenerate test_db.cali from tests.json
  - Run when SQLModel schema changes
  - Ensures database matches current models

- **expand_zarr_properly.py**: Script to expand zarr from 2 to 8 positions
  - Creates 4 wells (B5, B6, B7, B8) with 2 FOVs each
  - Copies original data to maintain data integrity
  - Updates metadata with proper WellPlatePlan structure

## Usage

### In Tests

Most tests use the `data_path` fixture from `conftest.py`:

```python
def test_something(data_path: Path):
    # data_path points to evk.tensorstore.zarr
    runner.run(dataset_path=data_path, ...)
```

For tests requiring a pre-built database:

```python
db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
```

### Regenerating the Database

When SQLModel changes require database updates:

```bash
python tests/test_data/data_and_db_for_tests/rebuild_test_db.py
```

This will:
1. Read schema from tests.json
2. Delete old test_db.cali
3. Run CaliRunner with specified settings
4. Create identical database with updated schema

### Expanding the Zarr Data

If the zarr data needs to be re-expanded from the original 2-position data:

```bash
git checkout -- tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr
python tests/test_data/data_and_db_for_tests/expand_zarr_properly.py
python tests/test_data/data_and_db_for_tests/rebuild_test_db.py
```

## Test Data Organization

The test suite uses specialized datasets for specific purposes:

- **data_and_db_for_tests/** (this directory): Primary dataset for all general tests
- **tests/test_data/spontaneous/**: Spontaneous activity data for spontaneous-specific tests
- **tests/test_data/evoked/**: Separate evoked dataset with JSON analysis output
- **tests/test_data/no_hcs/**: Dataset without HCS metadata for testing non-plate data

## Dataset Details

**Experiment Type**: Evoked Activity
**Plate Type**: 96-well
**Wells**: B5, B6, B7, B8
**Positions**: 8 (4 wells × 2 FOVs)
**Position Mapping**:

| Position | FOV Name | Source         |
| -------- | -------- | -------------- |
| 0        | B5_0000  | Original pos 0 |
| 1        | B5_0001  | Copy of pos 0  |
| 2        | B6_0000  | Original pos 1 |
| 3        | B6_0001  | Copy of pos 1  |
| 4        | B7_0000  | Copy of pos 0  |
| 5        | B7_0001  | Copy of pos 0  |
| 6        | B8_0000  | Copy of pos 1  |
| 7        | B8_0001  | Copy of pos 1  |

**Timepoints**: 153 frames
**Frame Rate**: 10.0 fps
**Duration**: 15.3 seconds
**LED Pulse Duration**: 100 ms
**LED Pulse Powers**: [2.0, 4.0, 6.0] mW/mm²
**LED Pulse Frames**: [3, 53, 103]

## Conditions

| Well | Genotype | Treatment |
|------|----------|-----------|
| B5   | g1       | t1        |
| B6   | g2       | t2        |
| B7   | g3       | t3        |
| B8   | g4       | t4        |

## Database Contents

- 1 Plate (96-well)
- 4 Wells (B5, B6, B7, B8)
- 8 Conditions (g1-g4, t1-t4)
- 8 FOVs (2 per well)
- 1 DetectionSettings (Cellpose with cpsam model)
- 2 ExtractionSettings (with and without neuropil correction)
- 1 AnalysisSettings (Evoked Activity with LED settings and stimulation mask)
- 2 CaliResults (different extraction configurations, positions [0,1] and [2,3])
- 16 ROIs total (4 per processed FOV)
