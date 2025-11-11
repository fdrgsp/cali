from pathlib import Path

from cali._constants import TS, ZR
from cali.readers import OMEZarrReader, TensorstoreZarrReader


def load_data(data_path: str | Path) -> TensorstoreZarrReader | OMEZarrReader | None:
    """Load data from the given path using the appropriate reader."""
    data_path = str(data_path)
    # select which reader to use for the datastore
    if data_path.endswith(TS):
        # read tensorstore
        return TensorstoreZarrReader(data_path)
    elif data_path.endswith(ZR):
        # read ome zarr
        return OMEZarrReader(data_path)
    else:
        return None
