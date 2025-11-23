from pathlib import Path

from sqlmodel import create_engine, Session, select

from cali.runner import CaliRunner
from cali.analysis import AnalysisRunner
from cali.detection import DetectionRunner
from cali.sqlmodel import AnalysisSettings, Experiment, CaliResult
from cali.sqlmodel._model import DetectionSettings
from cali.sqlmodel._visualize_experiment import print_all_analysis_results


def table(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        # Get all AnalysisResults
        results = session.exec(select(CaliResult).order_by(CaliResult.id)).all()

        # Get experiment name
        from cali.sqlmodel._model import Experiment

        experiment = session.exec(select(Experiment)).first()
        experiment_name = experiment.name if experiment else "Unknown"
        db_name = Path(db_path).name

        print("\n" + "=" * 140)
        print(
            f"ALL ANALYSIS RESULTS - TABLE OF RUNS | Database: {db_name} | Experiment: {experiment_name}"
        )
        print("=" * 140)
        print(
            f"{'ID':<5} {'Created At':<20} {'Experiment ID':<15} {'Detection ID':<15} {'Analysis ID':<15} {'Positions':<30}"
        )
        print("-" * 140)

        for result in results:
            created_at = result.created_at.strftime("%Y-%m-%d %H:%M:%S")
            experiment_id = str(result.experiment) if result.experiment else "None"
            detection_id = (
                str(result.detection_settings) if result.detection_settings else "None"
            )
            analysis_id = (
                str(result.analysis_settings) if result.analysis_settings else "None"
            )
            positions = (
                str(result.positions_analyzed) if result.positions_analyzed else "None"
            )

            print(
                f"{result.id:<5} {created_at:<20} {experiment_id:<15} {detection_id:<15} {analysis_id:<15} {positions:<30}"
            )



cali = CaliRunner()

data_path="tests/test_data/evoked/evk.tensorstore.zarr"

experiment = Experiment.create_from_data(
    data_path=data_path,
    name="New Experiment",
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
)

# -------------------------------------------------------
# Run 1: d1 only (no analysis)
cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
    global_position_indices=[0],
    overwrite=True,
)
table(cali.database_path)


# -------------------------------------------------------
# Run 2: d1 + a1 (reuses d1 detection, adds analysis)
cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
    analysis_settings=AnalysisSettings(dff_window=150),
    global_position_indices=[0],
)
table(cali.database_path)

# -------------------------------------------------------
# Run 3: d1 + a1 again (should skip both detection and analysis)
cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
    analysis_settings=AnalysisSettings(dff_window=150),
    global_position_indices=[0],
)
table(cali.database_path)

# -------------------------------------------------------
# Run 4: d1 + a2 (reuses d1 detection, new analysis)
cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
    analysis_settings=AnalysisSettings(dff_window=100),
    global_position_indices=[0],
)
table(cali.database_path)

# -------------------------------------------------------
# Run 5: d2 + a1 (new detection, reuses a1)
cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(
        method="cellpose", model_type="cpsam", diameter=30
    ),
    analysis_settings=AnalysisSettings(dff_window=150),
    global_position_indices=[0],
)
table(cali.database_path)

# -------------------------------------------------------
# Run 6: d2 + a2 (reuses d2, reuses a2)
cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(
        method="cellpose", model_type="cpsam", diameter=30
    ),
    analysis_settings=AnalysisSettings(dff_window=100),
    global_position_indices=[0],
)
table(cali.database_path)


# -------------------------------------------------------
# Run 7: d1 a1 (reuses d1 detection, reuses a1 analysis) - overwrite run 2
cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
    analysis_settings=AnalysisSettings(dff_window=150),
    global_position_indices=[0],
)
table(cali.database_path)

# -------------------------------------------------------
# Run 8: d3 only (new detection, no analysis)
cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(
        method="cellpose", model_type="cpsam", diameter=35
    ),
    analysis_settings=None,
    global_position_indices=[0],
)
table(cali.database_path)


engine = create_engine(f"sqlite:///{cali.database_path}")
print_all_analysis_results(engine, show_settings=False)
