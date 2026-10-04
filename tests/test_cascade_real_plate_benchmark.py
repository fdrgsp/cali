"""Real-recording gate preflight rejects invalid provenance and partial outputs."""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sqlmodel import Session

from cali.extraction._frame_window import StartupDiscardError
from cali.sqlmodel import (
    FOV,
    ROI,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    Mask,
    Plate,
    Well,
    create_cali_engine,
    create_database_and_tables,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Developer CPU benchmark uses Unix ps/resource APIs.",
)


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> object:
    """Load the developer command without adding it to the installed package."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "_dev"))
    return importlib.import_module("benchmark_cascade_real_plate")


def recording(frames: int = 128) -> tuple[np.ndarray, list[dict], dict]:
    """Use distinct non-contiguous real-style labels rather than array indices."""
    data = np.ones((frames, 4, 4), dtype=np.uint16)
    metadata = [{"runner_time_ms": 500 + i * 1000 / 30} for i in range(frames)]
    fov = {
        "name": "B2_0001",
        "position": 7,
        "rois": [
            {
                "label": 11,
                "mask": {
                    "coords_y": [0, 0],
                    "coords_x": [0, 1],
                    "height": 4,
                    "width": 4,
                    "mask_type": "roi",
                },
            }
        ],
    }
    return data, metadata, fov


def test_readonly_snapshot_includes_wal_without_migrating_source(
    benchmark: object,
    tmp_path: Path,
) -> None:
    """An active WAL recording must be copied consistently, never upgraded."""
    source = tmp_path / "source.cali"
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA user_version=12")
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('committed WAL row')")
        connection.commit()
        before = source.read_bytes()
        target = tmp_path / "copy.cali"
        benchmark.snapshot_database(source, target)
        with closing(sqlite3.connect(target)) as copy:
            assert copy.execute("SELECT value FROM evidence").fetchall() == [
                ("committed WAL row",)
            ]
            assert copy.execute("PRAGMA user_version").fetchone()[0] == 12
        assert source.read_bytes() == before
        with pytest.raises(FileExistsError):
            benchmark.snapshot_database(source, target)
    missing = tmp_path / "absent.cali"
    with pytest.raises(sqlite3.OperationalError):
        benchmark.readonly(missing)
    assert not missing.exists()


def test_preflight_records_retained_window_and_complete_pixel_identity(
    benchmark: object,
) -> None:
    data, metadata, fov = recording()
    settings = ExtractionSettings(frame_rate=30, discard_initial_value=5)
    result = benchmark.validate_position(
        data,
        metadata,
        fov,
        settings,
        SimpleNamespace(minimum_frames=65, sampling_rate=30),
    )
    assert result["position"] == 7
    assert result["retained_frames"] == 123
    assert result["source_start_frame"] == 5
    assert result["dtype"] == "<u2"
    assert result["image_bytes"] == data.nbytes
    changed = data.copy()
    changed[-1, -1, -1] += 1
    assert (
        benchmark.image_identity(changed, metadata)["pixels_sha256"]
        != result["pixels_sha256"]
    )


@pytest.mark.parametrize(
    "problem", ["jitter", "rate", "short", "untrusted", "mask", "label"]
)
def test_preflight_rejects_incompatible_recordings_before_inference(
    benchmark: object,
    problem: str,
) -> None:
    data, metadata, fov = recording(64 if problem == "short" else 128)
    if problem == "jitter":
        metadata[50]["runner_time_ms"] += 2
    elif problem == "untrusted":
        metadata = [{"exposure_ms": 1000 / 30} for _ in metadata]
    elif problem == "mask":
        fov["rois"][0]["mask"]["coords_x"][0] = 99
    elif problem == "label":
        fov["rois"].append(fov["rois"][0])
    model = SimpleNamespace(
        minimum_frames=65, sampling_rate=20 if problem == "rate" else 30
    )
    with pytest.raises((StartupDiscardError, ValueError)):
        benchmark.validate_position(
            data, metadata, fov, ExtractionSettings(frame_rate=30), model
        )


def test_measured_reader_rejects_changed_pixels_and_timing(benchmark: object) -> None:
    data, metadata, _ = recording()
    reader = SimpleNamespace(
        path=Path("input.tensorstore.zarr"), isel=lambda **_: (data, metadata)
    )
    verified = benchmark.VerifiedReader(
        reader, {"7": benchmark.image_identity(data, metadata)}
    )
    verified.isel(p=7, metadata=True)
    data[-1, -1, -1] += 1
    with pytest.raises(ValueError, match="changed after preflight"):
        verified.isel(p=7, metadata=True)
    data[-1, -1, -1] -= 1
    metadata[0]["runner_time_ms"] += 0.001
    with pytest.raises(ValueError, match="changed after preflight"):
        verified.isel(p=7, metadata=True)


def test_source_selection_uses_only_selected_detection_and_preserves_input(
    benchmark: object,
    tmp_path: Path,
) -> None:
    """Migrating a disposable schema-13 copy must leave the source untouched."""
    source = tmp_path / "source.cali"
    engine = create_cali_engine(f"sqlite:///{source}")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            detection = DetectionSettings()
            other_detection = DetectionSettings()
            extraction = ExtractionSettings(frame_rate=30)
            _, _, template = recording()
            selected = ROI(label_value=11, roi_mask=Mask(**template["rois"][0]["mask"]))
            unselected = ROI(label_value=12)  # Missing mask in another detection.
            experiment = Experiment(
                name="recording",
                plate=Plate(
                    name="plate",
                    wells=[
                        Well(
                            name="B2",
                            row=1,
                            column=1,
                            fovs=[
                                FOV(
                                    name="B2_0001",
                                    position_index=7,
                                    rois=[selected, unselected],
                                ),
                            ],
                        ),
                    ],
                ),
            )
            session.add_all([experiment, detection, other_detection, extraction])
            session.flush()
            selected.detection_settings_id = detection.id
            unselected.detection_settings_id = other_detection.id
            session.commit()
        with engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA user_version=13")
    finally:
        engine.dispose()
    before = benchmark.checksum(source)
    args = SimpleNamespace(
        database=source,
        experiment_id=1,
        extraction_settings_id=1,
        detection_settings_id=1,
        positions=[7],
    )
    result = benchmark.read_source(args)
    assert [roi["label"] for roi in result["fovs"][0]["rois"]] == [11]
    assert benchmark.checksum(source) == before
    with closing(sqlite3.connect(source)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 13
    args.positions = [7, 99]
    with pytest.raises(ValueError, match="resolve uniquely"):
        benchmark.read_source(args)
    args.positions = [7]
    args.detection_settings_id = 2
    with pytest.raises(ValueError, match="Missing mask"):
        benchmark.read_source(args)


def test_comparison_checks_every_position_and_roi_label(
    benchmark: object,
    tmp_path: Path,
) -> None:
    """A difference in the last ROI of the last FOV must fail the gate."""
    for phase in ("cold", "warm"):
        for mode, backend in benchmark.CASES:
            methods = ("oasis", "cascade") if mode == "dual" else (mode,)
            arrays, products = {}, {}
            for position, labels in ((3, (2, 9)), (7, (4, 11, 23))):
                products[str(position)] = {
                    "calcium": {"value": position},
                    "spikes": {method: {"value": position} for method in methods},
                }
                for label in labels:
                    for name in ("raw_trace", *methods):
                        arrays[f"{position}/{label}/{name}"] = np.arange(
                            position + label
                        )
            stem = f"{mode}-{backend}-{phase}"
            np.savez(tmp_path / (stem + ".npz"), **arrays)
            (tmp_path / (stem + "-metrics.json")).write_text(json.dumps(products))
    assert len(benchmark.compare(tmp_path)) == 14
    path = tmp_path / "dual-service-warm.npz"
    with np.load(path) as original:
        altered = {name: original[name].copy() for name in original.files}
    altered["7/23/cascade"][-1] += 1
    np.savez(path, **altered)
    with pytest.raises(AssertionError):
        benchmark.compare(tmp_path)
    altered["7/23/cascade"][-1] -= 1
    del altered["7/23/cascade"]
    np.savez(path, **altered)
    with pytest.raises(ValueError, match="Missing or unexpected array"):
        benchmark.compare(tmp_path)
