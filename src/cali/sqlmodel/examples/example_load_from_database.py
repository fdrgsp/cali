"""Example script to load an experiment from a database and print its tree structure."""

from cali.sqlmodel import load_experiment_from_db, print_experiment_tree

database_path = "tests/test_data/evoked/evk_analysis/cali.db"
exp = load_experiment_from_db(database_path)
assert exp is not None
print_experiment_tree(exp)
