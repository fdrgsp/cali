"""Example: Using TiffCollectionReader to import TIFF files.

This example demonstrates how to use the TiffCollectionReader to import
a collection of TIFF files and make them compatible with the cali pipeline.
"""

from pathlib import Path

from rich import print

from cali.readers._tiff_collection_reader import TiffCollectionReader

# Example 1: Multi-well plate
# ============================

tiff_folder = Path("/Users/fdrgsp/Desktop/cali_test/tiffs")
custom_files = sorted(
    list(tiff_folder.glob("*.tif")) + list(tiff_folder.glob("*.tiff"))
)

# in this case my files are saved as:
# A1_0000.tif, A1_0001.tif for well A1 fovs 0 and 1
# B1_0000.tif, B1_0001.tif for well B1 fovs 0 and 1
# etc.
# Otherwise you would need to create the file_map accordingly, e.g.:
# {
#     'A1': [
#         path/t/files/well_A1_fov0.tiff'),
#         path/t/files/well_A1_fov1.tiff'),
#         ...
#     ],
#     'B1': [
#         path/t/files/well_B1_fov0.tiff'),
#         path/t/files/well_B1_fov1.tiff'),
#         ...
#     ],
#    ...
# }
file_map = {}
for i in sorted(custom_files):
    well, fov = i.stem.split("_")
    if well not in file_map:
        file_map[well] = []
    if i not in file_map[well]:
        file_map[well].append(i)
print(file_map)

# Create reader
reader = TiffCollectionReader(
    file_map=file_map,
    plate="96-well",
    metadata={
        "exposure_ms": 100.0,
        "pixel_size_um": 0.65,
    },
    data_path=str(tiff_folder),
)

# get sequence
sequence = reader.sequence
print(sequence)

# Access data lazily (only loads when needed)
data_p0 = reader.isel(p=0)
print(data_p0.shape)

data_p0t0 = reader.isel(p=0, t=0)
print(data_p0t0.shape)

# Get metadata for a specific position
data_p0, meta = reader.isel(p=0, metadata=True)
print(meta)

