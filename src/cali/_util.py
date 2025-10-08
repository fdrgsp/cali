# this are names form the MDAWidget in micromanager-gui
ZARR_TESNSORSTORE = "tensorstore-zarr"
OME_ZARR = "ome-zarr"
# dict with writer name and extension
WRITERS: dict[str, list[str]] = {
    ZARR_TESNSORSTORE: [".tensorstore.zarr"],
    OME_ZARR: [".ome.zarr"],
    # OME_TIFF: [".ome.tif", ".ome.tiff"],
}
