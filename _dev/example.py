from pathlib import Path
from time import time

from sqlmodel import Session, create_engine, select

from cali._constants import EVOKED
from cali.analysis import AnalysisRunner
from cali.detection import DetectionRunner
from cali.sqlmodel import AnalysisSettings, DetectionSettings, Experiment
from cali.sqlmodel._model import AnalysisResult
from cali.sqlmodel._visualize_experiment import (
    print_all_analysis_results,
    print_experiment_tree_from_engine,
)


def _table(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        # Get all AnalysisResults
        results = session.exec(select(AnalysisResult).order_by(AnalysisResult.id)).all()

        print("\n" + "=" * 100)
        print("ALL ANALYSIS RESULTS - TABLE OF RUNS")
        print("=" * 100)
        print(
            f"{'ID':<5} {'Created At':<20} {'Detection ID':<15} {'Analysis ID':<15} {'Positions':<15}"
        )
        print("-" * 100)

        for result in results:
            created_at = result.created_at.strftime("%Y-%m-%d %H:%M:%S")
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
                f"{result.id:<5} {created_at:<20} {detection_id:<15} {analysis_id:<15} {positions:<15}"
            )

        print("=" * 100)
        print(f"\nTotal AnalysisResults: {len(results)}")


exp = Experiment.create_from_data(
    name="New Experiment",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    # analysis_path="tests/test_data/evoked/database/",
    analysis_path="/Users/fdrgsp/Desktop/cali_test",
    database_name="testcalidb",
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

d_settings_1 = DetectionSettings(method="cellpose", model_type="cpsam")
d_settings_2 = DetectionSettings(method="cellpose", model_type="cpsam", diameter=50)

a_settings_1 = AnalysisSettings(threads=4, dff_window=100)
a_settings_2 = AnalysisSettings(threads=4, dff_window=150)


# RUN1 D1 A1
detection.run(exp, d_settings_1, global_position_indices=[0])
analysis.run(exp, a_settings_1, d_settings_1, global_position_indices=[0])
# TODO: assert
# det id = 1 and analysis id = 1
_table(exp.db_path)

# RUN2 D1 A2
analysis.run(exp, a_settings_2, d_settings_1, global_position_indices=[0])
# TODO: assert
# det id = 1 and analysis id = 2
_table(exp.db_path)

# RUN3 D2 A2
detection.run(exp, d_settings_2, global_position_indices=[0])
analysis.run(exp, a_settings_2, d_settings_2, global_position_indices=[0])
# TODO: assert
# det id = 2 and analysis id = 2
_table(exp.db_path)

# RUN4 D2 A1
analysis.run(exp, a_settings_1, d_settings_2, global_position_indices=[0])
# TODO: assert
# det id = 2 and analysis id = 1
_table(exp.db_path)

import time
time.sleep(3)

# RUN5 D1 A2 (overwrite RUN 2)
detection.run(exp, d_settings_1, global_position_indices=[0])
analysis.run(exp, a_settings_2, d_settings_1, global_position_indices=[0])
# TODO: assert
# det id = 1 and analysis id = 2
_table(exp.db_path)



# Visualize the complete experiment tree with analysis results
# engine = create_engine(f"sqlite:///{exp.db_path}")
# print_all_analysis_results(
#     engine,
#     experiment_name=None,
#     show_settings=False,
#     max_experiment_level="roi",
# )

# print("---" * 10)

# print_experiment_tree_from_engine(
#     "New Experiment",
#     engine,
#     max_level="roi",
#     show_analysis_results=True,
#     show_settings=True,
# )
