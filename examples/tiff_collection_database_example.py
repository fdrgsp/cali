"""Example: Saving and loading TiffCollectionReader configuration in database.

This script demonstrates how to:
1. Create a TiffCollectionReader from TIFF files
2. Create an experiment that automatically saves the TIFF config
3. Save the experiment to a database
4. Load the experiment and recreate the TiffCollectionReader
"""

import tempfile
from pathlib import Path

import numpy as np
import tifffile
from rich import print

from cali.readers import TiffCollectionReader, TiffCollectionSettings
from cali.sqlmodel import Experiment, save_experiment_to_database
from cali.util import load_data_from_path

# Create temporary directory for this example
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)

    print("=" * 70)
    print("STEP 1: Creating TIFF files")
    print("=" * 70)

    # Create some example TIFF files
    file_map = {}
    for well in ["A1", "A2"]:
        file_map[well] = []
        for fov_idx in range(2):
            filepath = tmp_path / f"{well}_fov{fov_idx}.tif"
            # Create a simple 3D array (t, y, x)
            data = np.random.randint(0, 255, (10, 128, 128), dtype=np.uint16)
            tifffile.imwrite(filepath, data)
            file_map[well].append(str(filepath))
            print(f"  Created: {filepath.name}")

    print("\n" + "=" * 70)
    print("STEP 2: Creating TiffCollectionReader")
    print("=" * 70)

    settings = TiffCollectionSettings(
        file_map=file_map,
        plate="96-well",
        metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65},
        tiff_folder_path=list(tmp_path.glob("*.tif")),
    )
    reader = TiffCollectionReader(settings)

    print(f"  Reader created with {len(reader.sequence.stage_positions)} positions")
    print(f"  Wells: {list(file_map.keys())}")

    print("\n" + "=" * 70)
    print("STEP 3: Creating Experiment (TIFF config auto-saved)")
    print("=" * 70)

    experiment = Experiment.create_from_data(
        name="TIFF Example Experiment",
        data_path=str(tmp_path),
        description="Example showing TIFF collection database integration",
        tiff_file_map=file_map,
        tiff_plate_type="96-well",
        tiff_metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65},
    )

    print(f"  Experiment created: {experiment.name}")
    print(f"  TIFF config saved: {experiment.tiff_file_map_json is not None}")
    print(f"  Plate type: {experiment.tiff_plate_type}")

    print("\n" + "=" * 70)
    print("STEP 4: Saving to database")
    print("=" * 70)

    db_path = tmp_path / "experiment.cali"
    save_experiment_to_database(
        experiment, output_path=str(tmp_path), database_name="experiment.cali"
    )

    print(f"  Database saved to: {db_path}")

    print("\n" + "=" * 70)
    print("STEP 5: Loading from database")
    print("=" * 70)

    # Load experiment from database
    experiment_loaded = Experiment.load_from_db(db_path, load_data=False)
    print(f"  Experiment loaded: {experiment_loaded.name}")
    print(f"  Has TIFF config: {experiment_loaded.tiff_file_map_json is not None}")

    # Recreate reader using load_data with experiment
    data_loaded = load_data_from_path(tmp_path, experiment=experiment_loaded)
    print(f"  Reader type: {type(data_loaded).__name__}")

    print("\n" + "=" * 70)
    print("STEP 6: Verifying data access")
    print("=" * 70)

    # Test data access
    for p_idx in range(2):
        data_array, meta = data_loaded.isel(p=p_idx, metadata=True)
        print(
            f"  Position {p_idx}: shape={data_array.shape}, "
            f"dtype={data_array.dtype}, is_memmap={isinstance(data_array, np.memmap)}"
        )
