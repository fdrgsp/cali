import json

from cali.readers import TensorstoreZarrReader

data = TensorstoreZarrReader("tests/test_data/evoked/evk.tensorstore.zarr")
print("Sequence stage positions:")
for i, pos in enumerate(data.sequence.stage_positions):
    print(f"\nPosition {i}:")
    print(f"  Name: {pos.name}")
    print(f"  Row: {pos.row}")

print("\n\nData metadata for position 0:")
_, meta = data.isel(p=0, metadata=True)
print(json.dumps(meta, indent=2, default=str))
