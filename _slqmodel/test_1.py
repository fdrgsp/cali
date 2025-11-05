from pathlib import Path

import useq
from migrate_json_to_db import load_analysis_from_json, save_experiment_to_db

analysis_dir = Path(
    # "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont_analysis"
    # "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis"
    "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt_output"
)
plate = useq.WellPlate.from_str("96-well")
experiment = load_analysis_from_json(analysis_dir, useq_plate=plate)
experiment_name = experiment.name

from visualize_experiment import print_model_tree

db_path = Path("db_from_json.db")
session = save_experiment_to_db(experiment, db_path, overwrite=True, keep_session=True)
print_model_tree(experiment, max_level="well")


# from sqlmodel import create_engine
# from visualize_experiment import print_experiment_tree
# experiment = load_analysis_from_json(analysis_dir, useq_plate=plate)
# experiment_name = experiment.name
# db_path = Path("db_from_json.db")
# session = save_experiment_to_db(experiment, db_path, overwrite=True)
# engine = create_engine(f"sqlite:///{db_path}")
# print_experiment_tree(experiment_name, engine)
