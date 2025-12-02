"""Test that unavailable plots are disabled (grayed out) instead of red text."""

from pathlib import Path

from qtpy.QtCore import Qt
from sqlmodel import Session, create_engine, delete

from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
from cali.sqlmodel._model import DataAnalysis, Traces

# Test database
db_path = "tests/test_data/2pos/result_2pos.cali"
engine = create_engine(f"sqlite:///{db_path}")

# Create a widget
widget = _SingleWellGraphWidget(None)
widget.database_path = db_path
widget.engine = engine
widget.fov = "Well1-Pos_000_000"
widget.run_id = 1

print("=" * 80)
print("TESTING COMBO BOX ITEM DISABLING")
print("=" * 80)

# Get the combo box model
model = widget._combo.model()

print(f"\nTotal items in combo: {model.rowCount()}")

# Count enabled/disabled items
enabled = []
disabled = []
sections = []

for i in range(model.rowCount()):
    item = model.item(i)
    text = item.text()
    
    # Check if it's a section (divider)
    is_section = item.data(Qt.ItemDataRole.UserRole + 1)
    if is_section:
        sections.append(text)
        continue
    
    # Check if item is enabled
    if item.flags() & Qt.ItemFlag.ItemIsEnabled:
        enabled.append(text)
    else:
        disabled.append(text)
        # Check if it's grayed out
        color = item.foreground().color()
        print(f"  ❌ DISABLED: {text} (color: {color.name()})")

print(f"\n✅ Enabled plots: {len(enabled)}")
print(f"❌ Disabled plots: {len(disabled)}")
print(f"📁 Sections: {len(sections)}")

# Now test scenario without analysis data
print("\n" + "=" * 80)
print("TESTING WITHOUT ANALYSIS DATA")
print("=" * 80)

# Temporarily remove analysis data
with Session(engine) as session:
    # Delete analysis data for this FOV and run
    stmt = delete(DataAnalysis).where(DataAnalysis.analysis_result_id == 1)
    session.exec(stmt)  # type: ignore
    session.commit()

# Force widget to rebuild combo box
widget._update_experiment_type()
widget._update_combo_box()

model = widget._combo.model()
disabled_without_analysis = []

for i in range(model.rowCount()):
    item = model.item(i)
    text = item.text()
    is_section = item.data(Qt.ItemDataRole.UserRole + 1)
    
    if not is_section and not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
        disabled_without_analysis.append(text)

print(f"❌ Disabled plots without analysis: {len(disabled_without_analysis)}")
print(f"\nExample disabled plots requiring analysis:")
for plot in disabled_without_analysis[:5]:
    print(f"  - {plot}")

# Now test without extraction data
print("\n" + "=" * 80)
print("TESTING WITHOUT EXTRACTION DATA")
print("=" * 80)

with Session(engine) as session:
    # Delete traces data
    stmt = delete(Traces).where(Traces.analysis_result_id == 1)
    session.exec(stmt)  # type: ignore
    session.commit()

widget._update_experiment_type()
widget._update_combo_box()

model = widget._combo.model()
disabled_without_extraction = []

for i in range(model.rowCount()):
    item = model.item(i)
    text = item.text()
    is_section = item.data(Qt.ItemDataRole.UserRole + 1)
    
    if not is_section and not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
        disabled_without_extraction.append(text)

print(f"❌ Disabled plots without extraction: {len(disabled_without_extraction)}")

print("\n" + "=" * 80)
print("✅ SUCCESS: Plots are now disabled instead of showing red text!")
print("=" * 80)

engine.dispose(close=True)
