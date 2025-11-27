from pathlib import Path

from cali.readers import TiffCollectionReader

tiff_dir = Path("/Users/fdrgsp/Desktop/cali_test/tiffs")
tiff_files = sorted(tiff_dir.glob("*.tiff"))

# Create file map from TIFF files
# Files are named like B2_0000.tiff, B2_0001.tiff, B3_0000.tiff, etc.
file_map = {}
for tiff_file in tiff_files:
    well_name = tiff_file.stem.split("_")[0]  # Extract "B2", "B3", etc.
    if well_name not in file_map:
        file_map[well_name] = []
    file_map[well_name].append(str(tiff_file))

print(f"File map: {list(file_map.keys())}")
print(f"Files per well: {[(k, len(v)) for k, v in file_map.items()]}")

# Create reader
reader = TiffCollectionReader(
    file_map=file_map,
    plate="96-well",
    metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65},
    data_path=tiff_dir,
)

print("\n=== Reader Info ===")
print(f"Sequence axis order: {reader.sequence.axis_order}")
print(f"Number of positions: {len(reader.sequence.stage_positions)}")
print(f"Time plan: {reader.sequence.time_plan}")
print(f"Max T: {reader._max_t}")

# Test isel
print("\n=== Testing isel ===")
img1 = reader.isel({"p": 0, "t": 0, "c": 0})
print(f"Shape for p=0, t=0, c=0: {img1.shape}")

img2, meta2 = reader.isel({"p": 0, "t": 5, "c": 0}, metadata=True)
print(f"\nShape for p=0, t=5, c=0: {img2.shape}")
print(f"Metadata index: {meta2[0]['mda_event']['index']}")

# Test without 't' - should return full time series
img3 = reader.isel({"p": 0, "c": 0})
print(f"\nShape for p=0, c=0 (no t): {img3.shape}")
print(f"Expected: (2000, 1024, 1024)")

# Test with just p
img4 = reader.isel(p=1)
print(f"\nShape for p=1 (no t): {img4.shape}")
print(f"Expected: (2000, 1024, 1024)")
