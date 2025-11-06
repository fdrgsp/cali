from pathlib import Path

from qtpy.QtWidgets import QApplication
from rich import print
from sqlmodel import Session, create_engine, select

from cali._plate_viewer._plate_viewer_with_sqlmodel import PlateViewer
from cali.sqlmodel import Experiment


def load_experiment_from_database(database_path: str | Path) -> Experiment | None:
    """Load the experiment from the given database path."""
    try:
        engine = create_engine(f"sqlite:///{database_path}")
        session = Session(engine)
        result = session.exec(select(Experiment))
        experiment = result.first()
        return experiment
    except Exception as e:
        print(f"Error loading experiment: {e}")
        return None


# database_path = (
#     "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis/cali.db"
# )
# exp = load_experiment_from_database(database_path)
# print(exp)


app = QApplication([])

pl = PlateViewer()
pl.show()

app.exec()
