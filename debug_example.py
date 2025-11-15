from datetime import datetime

from sqlmodel import Session, create_engine, select

from cali.analysis._analysis_runner2 import AnalysisRunner
from cali.readers import TensorstoreZarrReader
from cali.sqlmodel import AnalysisSettings, Experiment, useq_plate_plan_to_db
from cali.sqlmodel._model import AnalysisResult

exp = Experiment(
    id=0,
    name="Debug Experiment",
    description="Debugging settings issue",
    created_at=datetime.now(),
    database_name="cali_debug.db",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    labels_path="tests/test_data/evoked/evk_labels",
    analysis_path="analysis_results",
)

data = TensorstoreZarrReader(exp.data_path)
exp.plate = useq_plate_plan_to_db(
    (plate_plan := data.sequence.stage_positions),
    exp,
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
)

settings = AnalysisSettings(threads=4)
print(f"Created settings: id={settings.id}, dff_window={settings.dff_window}")

analysis = AnalysisRunner()
analysis.run(exp, settings, global_position_indices=[0])

# Don't access settings.id after run (it's detached)

# Check database
engine = create_engine(f"sqlite:///{exp.db_path}")
with Session(engine) as session:
    all_settings = session.exec(select(AnalysisSettings)).all()
    print(f"\nAfter first run - Settings in DB: {len(all_settings)}")
    for s in all_settings:
        print(f"  ID {s.id}: threads={s.threads}, dff_window={s.dff_window}")

    all_results = session.exec(select(AnalysisResult)).all()
    print(f"\nAfter first run - Results in DB: {len(all_results)}")
    for r in all_results:
        print(f"  Result ID {r.id}: settings_id={r.analysis_settings}")

print("\n" + "=" * 60 + "\n")

settings1 = AnalysisSettings(threads=4, dff_window=150)
print(f"Created settings1: id={settings1.id}, dff_window={settings1.dff_window}")

analysis.run(exp, settings1, global_position_indices=[0])

# Don't access settings1.id after run (it's detached)

# Check database again
with Session(engine) as session:
    all_settings = session.exec(select(AnalysisSettings)).all()
    print(f"\nAfter second run - Settings in DB: {len(all_settings)}")
    for s in all_settings:
        print(f"  ID {s.id}: threads={s.threads}, dff_window={s.dff_window}")

    all_results = session.exec(select(AnalysisResult)).all()
    print(f"\nAfter second run - Results in DB: {len(all_results)}")
    for r in all_results:
        print(f"  Result ID {r.id}: settings_id={r.analysis_settings}")
