"""Quick visual test to verify multi-well x-axis labels are not clipped."""

from qtpy.QtWidgets import QApplication
from sqlmodel import create_engine

from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
from cali.plot._main_plot import plot_multi_well_data

# Create app
app = QApplication([])

# Create widget
widget = _MultilWellGraphWidget(None)
widget.resize(800, 600)

# Connect to test database
db_path = "tests/test_data/multi_pos/result_2pos.cali"
engine = create_engine(f"sqlite:///{db_path}")

widget.database_path = db_path
widget.engine = engine
widget.run_id = 1

# Plot a multi-well bar plot
plot_name = "Calcium Peaks Amplitude"
plot_multi_well_data(widget, plot_name, engine, run_id=1)

widget.setWindowTitle(f"Test: {plot_name} - Check x-axis labels at bottom")
widget.show()

print("Visual test window opened.")
print("Check that x-axis labels are fully visible (not clipped at bottom).")
print("Close the window to exit.")

app.exec()
