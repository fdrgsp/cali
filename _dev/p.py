from rich import print

from cali.readers import TensorstoreZarrReader

# dataset_path = "/Users/fdrgsp/Desktop/t/multip.tensorstore.zarr"
dataset_path = "/Users/fdrgsp/Desktop/t/hcs.tensorstore.zarr"
dataset = TensorstoreZarrReader(dataset_path)

p = 0
data, meta = dataset.isel(p=p, metadata=True)
pos_name = meta[0].get("mda_event", {}).get("pos_name", f"pos_{str(p).zfill(4)}")

print()
print(f"Position {p} name: {pos_name}")
