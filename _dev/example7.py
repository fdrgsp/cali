from pathlib import Path

from sqlmodel import create_engine, Session, select

from cali.runner import CaliRunner
from cali.analysis import AnalysisRunner
from cali.detection import DetectionRunner
from cali.sqlmodel import AnalysisSettings, Experiment, CaliResult
from cali.sqlmodel._model import DetectionSettings
from cali.sqlmodel._visualize_experiment import print_cali_results


def table(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        # Get all CaliResults
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
    name="New Experiment",
    data_path=data_path,
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
)
cali.run(
    experiment,
    data_path,
    detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
    # analysis_settings=AnalysisSettings(dff_window=150),
    global_position_indices=[0],
    overwrite=True,
)
table(cali.database_path)

# -------------------------------------------------------
experiment1 = Experiment.create_from_data(
    name="Another Experiment",
    data_path=data_path,
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
)
cali.run(
    experiment1,
    data_path,
    detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
    # analysis_settings=AnalysisSettings(dff_window=150),
    global_position_indices=[0],
    database_name="another_experiment",  # Different database name for different experiment
)
table(cali.database_path)

engine = create_engine(f"sqlite:///{cali.database_path}")
print_cali_results(engine, show_settings=False)
