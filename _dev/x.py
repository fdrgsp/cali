from cali._plate_viewer._plate_viewer_with_sqlmodel import PlateViewer
from qtpy.QtWidgets import QApplication

# database_path = (
#     "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis/cali.db"
# )
# exp = load_experiment_from_db(database_path)
# print(exp)


app = QApplication([])

pl = PlateViewer()
pl.show()

app.exec()
