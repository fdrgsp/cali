"""Test force parameter for detection.run()"""
from pathlib import Path

from sqlmodel import Session, create_engine, select

from cali._constants import EVOKED
from cali.analysis import AnalysisRunner
from cali.detection import DetectionRunner
from cali.sqlmodel import AnalysisSettings, DetectionSettings, Experiment
from cali.sqlmodel._model import AnalysisResult, ROI


exp = Experiment.create_from_data(
    name="Force Test",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    analysis_path="/Users/fdrgsp/Desktop/cali_test",
    database_name="testforce",
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
    experiment_type=EVOKED,
)

# delete existing database if it exists
if Path(exp.db_path).exists():
    Path(exp.db_path).unlink()

detection = DetectionRunner()
analysis = AnalysisRunner()

d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
a_settings = AnalysisSettings(threads=4, dff_window=100)

print("\n" + "="*80)
print("TEST 1: Initial detection run")
print("="*80)
detection.run(exp, d_settings, global_position_indices=[0])
analysis.run(exp, a_settings, global_position_indices=[0])

engine = create_engine(f"sqlite:///{exp.db_path}")
with Session(engine) as session:
    rois = session.exec(select(ROI)).all()
    results = session.exec(select(AnalysisResult)).all()
    print(f"ROIs: {len(rois)}, AnalysisResults: {len(results)}")
    first_roi_ids = {roi.id for roi in rois}
    first_result_ids = {r.id for r in results}

print("\n" + "="*80)
print("TEST 2: Re-run same detection without force (should SKIP)")
print("="*80)
detection.run(exp, d_settings, global_position_indices=[0])

with Session(engine) as session:
    rois = session.exec(select(ROI)).all()
    results = session.exec(select(AnalysisResult)).all()
    print(f"ROIs: {len(rois)}, AnalysisResults: {len(results)}")
    assert {roi.id for roi in rois} == first_roi_ids, "ROIs should be unchanged"
    assert {r.id for r in results} == first_result_ids, "Results should be unchanged"
    print("✅ Correctly skipped - ROIs and results unchanged")

print("\n" + "="*80)
print("TEST 3: Re-run same detection with force=True (should DELETE and REPLACE)")
print("="*80)
detection.run(exp, d_settings, global_position_indices=[0], force=True)

with Session(engine) as session:
    rois = session.exec(select(ROI)).all()
    results = session.exec(select(AnalysisResult)).all()
    print(f"ROIs: {len(rois)}, AnalysisResults: {len(results)}")
    # ROIs are replaced (deleted then recreated, may have same IDs in SQLite)
    # AnalysisResults should be deleted (full analysis)
    assert len(rois) == len(first_roi_ids), "Should have same number of ROIs"
    assert len(results) == 1, "Should only have detection-only result"
    assert results[0].analysis_settings is None, "Should be detection-only"
    print("✅ Correctly deleted analysis and replaced ROIs")

print("\n" + "="*80)
print("TEST 4: All tests passed! ✅")
print("="*80)
print("Summary:")
print("- Default behavior: Skip if positions already have ROIs")
print("- force=True: Delete all analysis results and replace ROIs")
print("- Incremental: Would run only new positions (not tested, no multi-position data)")

