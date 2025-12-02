"""Test that setting fov and run_id doesn't produce warnings."""

import logging
from sqlmodel import create_engine
from qtpy.QtWidgets import QApplication
import sys

from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget

# Capture warnings
logging.basicConfig(level=logging.WARNING)

app = QApplication.instance() or QApplication(sys.argv)

db_path = "tests/test_data/2pos/result_2pos.cali"
engine = create_engine(f"sqlite:///{db_path}")

print("Creating widget and setting run_id and fov...")
widget = _SingleWellGraphWidget(None)  # type: ignore[arg-type]
widget.database_path = db_path
widget.engine = engine

# This should NOT produce "Analysis '' not found in registry" warnings
widget.run_id = 1
widget.fov = "B5_0000"

print("✅ No warnings expected - if you see any above, the fix didn't work!")

engine.dispose(close=True)
