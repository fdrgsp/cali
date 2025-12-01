"""Test the combo box enabling/disabling with actual database."""

from sqlmodel import create_engine

from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
from cali.plot._main_plot import PipelineStage

# Test database
db_path = "tests/test_data/2pos/result_2pos.cali"
engine = create_engine(f"sqlite:///{db_path}")

print("=" * 80)
print("TESTING COMBO BOX WITH REAL DATA")
print("=" * 80)

# Create widget (need QApplication)
from qtpy.QtWidgets import QApplication
import sys

app = QApplication.instance() or QApplication(sys.argv)

widget = _SingleWellGraphWidget(None)
widget.database_path = db_path
widget.engine = engine

print("\n1. Initial state (no FOV or run_id):")
has_det, has_ext, has_ana = widget._check_pipeline_stage_availability()
print(f"   Detection: {has_det}, Extraction: {has_ext}, Analysis: {has_ana}")
print(f"   FOV: '{widget._fov}', Run ID: {widget._run_id}")

print("\n2. Set run_id = 1:")
widget.run_id = 1
has_det, has_ext, has_ana = widget._check_pipeline_stage_availability()
print(f"   Detection: {has_det}, Extraction: {has_ext}, Analysis: {has_ana}")
print(f"   FOV: '{widget._fov}', Run ID: {widget._run_id}")

print("\n3. Set fov = 'B5_0000' (actual FOV name in database):")
widget.fov = "B5_0000"
has_det, has_ext, has_ana = widget._check_pipeline_stage_availability()
print(f"   Detection: {has_det}, Extraction: {has_ext}, Analysis: {has_ana}")
print(f"   FOV: '{widget._fov}', Run ID: {widget._run_id}")

# Check combo box items
print("\n4. Check combo box items:")
model = widget._combo.model()
enabled_count = 0
disabled_count = 0

from qtpy.QtCore import Qt

for i in range(model.rowCount()):
    item = model.item(i)
    is_section = item.data(Qt.ItemDataRole.UserRole + 1)
    if is_section or item.text() == "None":
        continue
    
    if item.flags() & Qt.ItemFlag.ItemIsEnabled:
        enabled_count += 1
    else:
        disabled_count += 1

print(f"   Enabled: {enabled_count}, Disabled: {disabled_count}")

if disabled_count > 0 and has_det and has_ext and has_ana:
    print("\n❌ PROBLEM: Items are disabled even though all pipeline stages are complete!")
    print("\nShowing first 5 disabled items:")
    count = 0
    for i in range(model.rowCount()):
        if count >= 5:
            break
        item = model.item(i)
        is_section = item.data(Qt.ItemDataRole.UserRole + 1)
        if is_section or item.text() == "None":
            continue
        if not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            print(f"   - {item.text()}")
            count += 1
else:
    print("\n✅ Items are correctly enabled!")

engine.dispose(close=True)
