"""Test to verify neuropil mask filtering by detection_settings_id."""

from pathlib import Path

from sqlmodel import Session, create_engine, select

from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    save_experiment_to_database,
)
from cali.sqlmodel._model import FOV, ROI

data_path = (
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
)
output_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/"
db_name = "test_neuropil.cali"

# Create experiment
exp = Experiment.create_from_data(
    name="Neuropil Test",
    data_path=data_path,
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
    description="Test for neuropil mask filtering",
)

# Save to database
save_experiment_to_database(exp, output_path, database_name=db_name, overwrite=True)

runner = CaliRunner()

# RUN 1: Detection + Analysis WITHOUT neuropil
print("\n" + "=" * 80)
print("RUN 1: Detection + Analysis WITHOUT neuropil")
print("=" * 80)

detection_settings_1 = DetectionSettings(
    method="cellpose",
    model_type="custom",
    custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",
)

analysis_settings_1 = AnalysisSettings(
    dff_window=130,
    neuropil_min_pixels=0,  # NO NEUROPIL
    neuropil_correction_factor=0.0,
    neuropil_inner_radius=0,
)

runner.run(
    exp,
    data_path,
    detection_settings_1,
    analysis_settings=analysis_settings_1,
    global_position_indices=[0],
    output_path=output_path,
    database_name=db_name,
    overwrite=True,
)

# RUN 2: Analysis-only WITH neuropil (reuses ROIs from detection_settings_1)
print("\n" + "=" * 80)
print("RUN 2: Analysis-only WITH neuropil (same detection as Run 1)")
print("=" * 80)

analysis_settings_2 = AnalysisSettings(
    dff_window=130,
    neuropil_min_pixels=100,  # WITH NEUROPIL
    neuropil_correction_factor=0.7,
    neuropil_inner_radius=2,
)

# Get detection_settings from Run 1 (to reuse in Run 2)
engine = create_engine(f"sqlite:///{Path(output_path) / db_name}")
with Session(engine) as session:
    detection_1 = session.exec(select(DetectionSettings)).first()
    assert detection_1 is not None
    detection_1_id = detection_1.id
    assert detection_1_id is not None
engine.dispose()

runner.run(
    exp,
    data_path,
    detection_1_id,  # Reuse detection ID from Run 1
    analysis_settings=analysis_settings_2,
    global_position_indices=[0],
    output_path=output_path,
    database_name=db_name,
    overwrite=False,
)

# RUN 3: New detection + Analysis WITH neuropil
print("\n" + "=" * 80)
print("RUN 3: New detection + Analysis WITH neuropil")
print("=" * 80)

detection_settings_3 = DetectionSettings(
    method="cellpose",
    model_type="custom",
    custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",
)

analysis_settings_3 = AnalysisSettings(
    dff_window=130,
    neuropil_min_pixels=100,  # WITH NEUROPIL
    neuropil_correction_factor=0.7,
    neuropil_inner_radius=2,
)

runner.run(
    exp,
    data_path,
    detection_settings_3,
    analysis_settings=analysis_settings_3,
    global_position_indices=[0],
    output_path=output_path,
    database_name=db_name,
    overwrite=False,
)

# NOW TEST THE FILTERING
print("\n" + "=" * 80)
print("TESTING NEUROPIL MASK FILTERING BY RUN")
print("=" * 80)

engine = create_engine(f"sqlite:///{Path(output_path) / db_name}")

with Session(engine) as session:
    # Get all CaliResults
    from cali.sqlmodel._model import CaliResult

    results = session.exec(select(CaliResult)).all()
    print(f"\nTotal CaliResults (Runs): {len(results)}")

    # For each run, check which ROIs have neuropil masks
    for result in results:
        print(f"\n--- Run #{result.id} ---")
        print(f"  Detection Settings ID: {result.detection_settings}")
        print(f"  Analysis Settings ID: {result.analysis_settings}")

        # Query ROIs for B5_0000 FOV with this run's detection_settings_id
        from sqlalchemy.orm import selectinload

        stmt = (
            select(ROI)
            .join(FOV)
            .where(FOV.name == "B5_0000")
            .where(ROI.detection_settings_id == result.detection_settings)
            .options(
                selectinload(ROI.roi_mask),  # type: ignore
                selectinload(ROI.neuropil_mask),  # type: ignore
            )
        )

        rois = session.exec(stmt).all()
        print(
            f"  Found {len(rois)} ROIs with detection_settings_id={result.detection_settings}"
        )

        # Check which ROIs have traces from THIS specific run
        from cali.sqlmodel._model import Traces

        trace_stmt = (
            select(Traces.roi_id)
            .where(Traces.analysis_result_id == result.id)
            .distinct()
        )
        roi_ids_with_traces_from_run = set(session.exec(trace_stmt).all())

        # For each ROI, check if it has neuropil and if it has traces from this run
        for roi in rois:
            has_trace = roi.id in roi_ids_with_traces_from_run
            has_neuropil = roi.neuropil_mask is not None

            status = ""
            if has_trace and has_neuropil:
                status = "✅ SHOULD SHOW neuropil (has trace from this run)"
            elif has_neuropil and not has_trace:
                status = "❌ SHOULD NOT show neuropil (no trace from this run)"
            elif has_trace and not has_neuropil:
                status = "⚪ No neuropil mask exists"
            else:
                status = "⚪ No trace or neuropil"

            print(f"    ROI {roi.label_value}: {status}")

engine.dispose()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\nISSUE TO VERIFY:")
print("When clicking Run 1 in GUI: Should show NO neuropil masks")
print(
    "When clicking Run 2 in GUI: Should show neuropil masks (same detection_settings_id=1)"
)
print(
    "When clicking Run 3 in GUI: Should show neuropil masks (detection_settings_id=2)"
)
print("\nThe problem is: If ROIs from detection_settings_id=1 have neuropil masks")
print(
    "(added by Run 2), they will ALWAYS show when filtering by detection_settings_id=1,"
)
print("even if we're looking at Run 1 (which didn't have neuropil).")
print("\nWe need to filter by BOTH detection_settings_id AND analysis_result_id!")
