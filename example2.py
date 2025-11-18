from sqlmodel import create_engine

from cali._constants import EVOKED
from cali.sqlmodel import Experiment
from cali.sqlmodel._util import save_experiment_to_database
from cali.sqlmodel._visualize_experiment import (
    print_all_analysis_results,
    print_experiment_tree,
)

exp = Experiment.create_from_data(
    name="New Experiment",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    analysis_path="/Users/fdrgsp/Desktop/cali_test",
    database_name="evk.tensorstore.zarr.db",
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
    experiment_type=EVOKED,
)

save_experiment_to_database(exp, overwrite=True)

engine = create_engine(f"sqlite:///{exp.db_path}")
print_all_analysis_results(
    engine,
    experiment_name=None,
    show_settings=True,
    max_experiment_level="roi",
)
print_experiment_tree(exp)
