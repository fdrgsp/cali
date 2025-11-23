from rich import print

from cali.sqlmodel import Experiment


exp = Experiment.load_from_db("tests/test_data/evoked/results.cali")
print(exp)
