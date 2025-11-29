"""Debug: Check what's in the database after Run 1."""

from sqlmodel import create_engine, Session, select

from cali.runner import CaliRunner
from cali.sqlmodel import AnalysisSettings, Experiment, CaliResult
from cali.sqlmodel._model import DetectionSettings, Traces, FOV, ROI

cali = CaliRunner()
data_path = "tests/test_data/evoked/evk.tensorstore.zarr"

experiment = Experiment.create_from_data(
    name="Debug DB",
    data_path=data_path,
)

det = DetectionSettings(method="cellpose", model_type="cpsam")
ana = AnalysisSettings(dff_window=150)

print("\n>>> Run 1")
cali.run(
    experiment,
    data_path,
    detection_settings=det,
    analysis_settings=ana,
    global_position_indices=[0],
    overwrite=True,
)

# Check database contents
assert cali.database_path is not None
engine = create_engine(f"sqlite:///{cali.database_path}")
with Session(engine) as session:
    # Check CaliResults
    results = session.exec(select(CaliResult)).all()
    print(f"\nCaliResults: {len(results)}")
    for r in results:
        print(f"  ID={r.id}, exp={r.experiment}, det={r.detection_settings}, ana={r.analysis_settings_id}, pos={r.positions_analyzed}")
    
    # Check Traces
    traces = session.exec(select(Traces)).all()
    print(f"\nTraces: {len(traces)}")
    for t in traces:
        print(f"  ID={t.id}, roi_id={t.roi_id}, analysis_result_id={t.analysis_result_id}")
    
    # Check ROIs
    rois = session.exec(select(ROI)).all()
    print(f"\nROIs: {len(rois)}")
    for roi in rois:
        print(f"  ID={roi.id}, label={roi.label_value}, detection_settings_id={roi.detection_settings_id}, fov_id={roi.fov_id}")
    
    # Check FOVs
    fovs = session.exec(select(FOV)).all()
    print(f"\nFOVs: {len(fovs)}")
    for fov in fovs:
        print(f"  ID={fov.id}, position_index={fov.position_index}")
    
    # Now try the query from _should_skip_analysis
    print("\n>>> Testing the skip analysis query:")
    existing_positions = session.exec(
        select(FOV.position_index)
        .join(ROI)
        .join(Traces)
        .where(
            ROI.detection_settings_id == 1,
            Traces.analysis_result_id.in_(  # type: ignore
                select(CaliResult.id).where(
                    CaliResult.analysis_settings_id == 1
                )
            ),
            FOV.position_index.in_([0]),  # type: ignore
        )
        .distinct()
    ).all()
    print(f"Existing positions with analysis: {existing_positions}")
    print(f"Should skip? {len(existing_positions) == 1}")
