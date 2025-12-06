# Unified Test Data

This directory contains the **primary test dataset** used across all Cali tests.

## Files

- **evk.tensorstore.zarr**: Evoked activity dataset with 2 positions (B5, B6) and 153 timepoints
  - Used for detection, extraction, analysis, and GUI tests
  - Contains LED stimulation events at frames [3, 53, 103]
  - Powers: [2.0, 4.0, 6.0] mW/mm²
  - Frame rate: 10.0 fps

- **test_db.cali**: Pre-built database with complete analysis results
  - Generated from evk.tensorstore.zarr
  - Contains 2 CaliResults with different extraction settings
  - Includes plate map with conditions: genotype (g1, g2) and treatment (t1, t2)
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

## Test Data Organization

The test suite uses specialized datasets for specific purposes:

- **data_and_db_for_tests/** (this directory): Primary dataset for all general tests
- **tests/test_data/spontaneous/**: Spontaneous activity data for spontaneous-specific tests
- **tests/test_data/evoked/**: Separate evoked dataset with JSON analysis output
- **tests/test_data/no_hcs/**: Dataset without HCS metadata for testing non-plate data

## Dataset Details

**Experiment Type**: Evoked Activity  
**Plate Type**: 96-well  
**Wells**: B5, B6  
**Positions**: 2 (B5_0000, B6_0000)  
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

## Database Contents

- 1 Plate (96-well)
- 2 Wells (B5, B6)
- 4 Conditions (g1, g2, t1, t2)
- 2 FOVs
- 1 DetectionSettings (Cellpose with cpsam model)
- 2 ExtractionSettings (with and without neuropil correction)
- 1 AnalysisSettings (Evoked Activity with LED settings)
- 2 CaliResults (different extraction configurations)
- 8 ROIs total
