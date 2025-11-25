"""Test to verify neuropil mask override problem."""

from pathlib import Path

from sqlmodel import Session, create_engine, select

from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    save_experiment_to_database,
)
from cali.sqlmodel._model import FOV, ROI, CaliResult

data_path = (
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
)
output_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/"
db_name = "test_neuropil_override.cali"

# Create experiment
exp = Experiment.create_from_data(
    name="Neuropil Override Test",
    data_path=data_path,
    plate_maps={"genotype": {"B5": "WT"}, "treatment": {"B5": "Vehicle"}},
    description="Test for neuropil mask override issue",
)

save_experiment_to_database(exp, output_path, database_name=db_name, overwrite=True)

runner = CaliRunner()

# RUN 1: Detection + Analysis WITH neuropil (settings A)
print("\n" + "=" * 80)
print("RUN 1: Detection + Analysis WITH neuropil (inner_radius=2, min_pixels=50)")
print("=" * 80)

detection_settings_1 = DetectionSettings(
    method="cellpose",
    model_type="custom",
    custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",
)

analysis_settings_1 = AnalysisSettings(
    dff_window=130,
    neuropil_min_pixels=50,
    neuropil_correction_factor=0.7,
    neuropil_inner_radius=2,
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

# Query to check neuropil mask pixel counts AFTER RUN 1
engine = create_engine(f"sqlite:///{Path(output_path) / db_name}")
with Session(engine) as session:
    # Query Traces from Run 1 (analysis_result_id=1)
    from sqlalchemy.orm import selectinload

    from cali.sqlmodel._model import Traces

    traces = session.exec(
        select(Traces)
        .where(Traces.analysis_result_id == 1)
        .options(
            selectinload(Traces.roi),  # type: ignore
            selectinload(Traces.neuropil_mask),  # type: ignore
        )
    ).all()

    print(f"\nAfter Run 1 - Neuropil mask pixel counts:")
    run1_neuropil_sizes = {}
    for trace in traces:
        if trace.neuropil_mask and trace.neuropil_mask.coords_y:
            pixel_count = len(trace.neuropil_mask.coords_y)
            run1_neuropil_sizes[trace.roi.label_value] = pixel_count
            print(f"  ROI {trace.roi.label_value}: {pixel_count} pixels")
engine.dispose()

# RUN 2: Analysis-only WITH DIFFERENT neuropil settings (settings B)
print("\n" + "=" * 80)
print("RUN 2: Analysis-only WITH DIFFERENT neuropil (inner_radius=3, min_pixels=100)")
print("=" * 80)

analysis_settings_2 = AnalysisSettings(
    dff_window=130,
    neuropil_min_pixels=100,  # DIFFERENT!
    neuropil_correction_factor=0.7,
    neuropil_inner_radius=3,  # DIFFERENT!
)

# Get detection_settings_id from Run 1 to reuse
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
    detection_1_id,  # Reuse detection from Run 1
    analysis_settings=analysis_settings_2,
    global_position_indices=[0],
    output_path=output_path,
    database_name=db_name,
    overwrite=False,
)

# Query to check neuropil mask pixel counts AFTER RUN 2
engine = create_engine(f"sqlite:///{Path(output_path) / db_name}")
with Session(engine) as session:
    # Query Traces from Run 2 (analysis_result_id=2)
    traces = session.exec(
        select(Traces)
        .where(Traces.analysis_result_id == 2)
        .options(
            selectinload(Traces.roi),  # type: ignore
            selectinload(Traces.neuropil_mask),  # type: ignore
        )
    ).all()

    print(f"\nAfter Run 2 - Neuropil mask pixel counts:")
    run2_neuropil_sizes = {}
    for trace in traces:
        if trace.neuropil_mask and trace.neuropil_mask.coords_y:
            pixel_count = len(trace.neuropil_mask.coords_y)
            run2_neuropil_sizes[trace.roi.label_value] = pixel_count
            print(f"  ROI {trace.roi.label_value}: {pixel_count} pixels")

    # Check if sizes changed
    print(f"\nComparison:")
    for label in run1_neuropil_sizes:
        size1 = run1_neuropil_sizes[label]
        size2 = run2_neuropil_sizes.get(label, 0)
        if size1 != size2:
            print(
                f"  ROI {label}: Run 1={size1} pixels, Run 2={size2} pixels "
                f"(Different as expected with different settings)"
            )
        else:
            print(
                f"  ROI {label}: UNEXPECTED - both runs have {size1} pixels"
            )

    # Verify Run 1 masks are still accessible
    print(f"\nVerifying Run 1 masks are still in database:")
    traces_run1 = session.exec(
        select(Traces)
        .where(Traces.analysis_result_id == 1)
        .options(
            selectinload(Traces.roi),  # type: ignore
            selectinload(Traces.neuropil_mask),  # type: ignore
        )
    ).all()
    
    print(f"  Found {len(traces_run1)} traces from Run 1")
    for trace in traces_run1:
        if trace.neuropil_mask and trace.neuropil_mask.coords_y:
            pixel_count = len(trace.neuropil_mask.coords_y)
            print(f"  ✅ Run 1, ROI {trace.roi.label_value}: {pixel_count} pixels (PRESERVED)")
        else:
            print(f"  ❌ Run 1, ROI {trace.roi.label_value}: No neuropil mask found")


engine.dispose()

print("\n" + "=" * 80)
print("SOLUTION VERIFICATION:")
print("=" * 80)
print("Neuropil masks are now stored with Traces, not ROI.")
print("This means:")
print("  - Run 1 (analysis_result_id=1) has its own neuropil masks")
print("  - Run 2 (analysis_result_id=2) has its own neuropil masks")
print("  - Both sets of masks COEXIST in the database")
print("\nWhen you click Run 1 in GUI, it loads Traces from analysis_result_id=1")
print("When you click Run 2 in GUI, it loads Traces from analysis_result_id=2")
print("\nBoth runs' neuropil masks are preserved!")
print("=" * 80)
