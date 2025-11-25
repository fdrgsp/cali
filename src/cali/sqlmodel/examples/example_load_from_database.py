"""Example script to load an experiment from a database and print its tree structure."""

from sqlalchemy import create_engine

from cali.sqlmodel import print_experiment_tree
from cali.sqlmodel._model import Experiment
from cali.sqlmodel._visualize_experiment import print_cali_results

database_path = "tests/test_data/evoked/results.cali"
exp = Experiment.load_from_db(database_path)
assert exp is not None
print_experiment_tree(exp)


engine = create_engine(f"sqlite:///{database_path}")
print_cali_results(engine, show_settings=False)
