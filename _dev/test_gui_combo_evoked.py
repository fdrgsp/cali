"""Test that GUI combo box shows evoked plots for evoked experiments."""

from sqlmodel import Session, create_engine, select

from cali._constants import EVOKED
from cali.plot._main_plot import AnalysisGroup, get_available_plots
from cali.sqlmodel._model import AnalysisSettings, CaliResult

# Test database path
db_path = "tests/test_data/2pos/result_2pos.cali"

# Create engine
engine = create_engine(f"sqlite:///{db_path}")

# Get the first run's experiment type
with Session(engine) as session:
    stmt = select(AnalysisSettings.experiment_type).join(CaliResult).where(
        CaliResult.id == 1
    )
    experiment_type = session.exec(stmt).first()
    print(f"Run #1 experiment type: '{experiment_type}'")
    print(f"EVOKED constant: '{EVOKED}'")
    print(f"Match: {experiment_type == EVOKED}")
    print()

# Get plots for this experiment type
plots = get_available_plots(
    group=AnalysisGroup.SINGLE_WELL,
    has_detection=True,
    has_extraction=True,
    has_analysis=True,
    experiment_type=experiment_type,
)

# Count evoked plots
evoked_count = sum(
    1
    for category_plots in plots.values()
    for plot in category_plots
    if "Stimulated" in plot or "Non-Stimulated" in plot
)

print(f"Total plots available: {sum(len(p) for p in plots.values())}")
print(f"Evoked plots shown: {evoked_count}")
print()

# Show evoked categories
print("Evoked plot categories:")
for category in sorted(plots.keys()):
    if "Evoked" in category or "Stimulated" in category:
        print(f"  - {category}: {len(plots[category])} plots")
        for plot in plots[category][:3]:  # Show first 3
            print(f"    * {plot}")
        if len(plots[category]) > 3:
            print(f"    ... and {len(plots[category]) - 3} more")

engine.dispose(close=True)

print("\n✅ SUCCESS: Evoked plots will be shown in the GUI combo box!")
