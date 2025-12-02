"""Test that evoked plots are properly filtered by experiment type."""

from cali._constants import EVOKED
from cali.plot._main_plot import AnalysisGroup, get_available_plots

# Test 1: Spontaneous experiment (no experiment_type or "spontaneous")
print("=" * 80)
print("TEST 1: Spontaneous Experiment (experiment_type=None)")
print("=" * 80)
spontaneous_plots = get_available_plots(
    group=AnalysisGroup.SINGLE_WELL,
    has_detection=True,
    has_extraction=True,
    has_analysis=True,
    experiment_type=None,  # Show all plots
)

# Count evoked plots (should include all plots)
evoked_count = sum(
    1
    for plots in spontaneous_plots.values()
    for plot in plots
    if "Stimulated" in plot or "Non-Stimulated" in plot
)
total_count = sum(len(plots) for plots in spontaneous_plots.values())

print(f"Total plots available: {total_count}")
print(f"Evoked plots shown: {evoked_count}")
print("\nEvoked plot categories:")
for category in sorted(spontaneous_plots.keys()):
    if "Evoked" in category or "Stimulated" in category:
        print(f"  - {category}: {len(spontaneous_plots[category])} plots")

# Test 2: Evoked experiment (experiment_type="evoked")
print("\n" + "=" * 80)
print(f"TEST 2: Evoked Experiment (experiment_type=EVOKED='{EVOKED}')")
print("=" * 80)
evoked_plots = get_available_plots(
    group=AnalysisGroup.SINGLE_WELL,
    has_detection=True,
    has_extraction=True,
    has_analysis=True,
    experiment_type=EVOKED,
)

evoked_count = sum(
    1
    for plots in evoked_plots.values()
    for plot in plots
    if "Stimulated" in plot or "Non-Stimulated" in plot
)
total_count = sum(len(plots) for plots in evoked_plots.values())

print(f"Total plots available: {total_count}")
print(f"Evoked plots shown: {evoked_count}")
print("\nEvoked plot categories:")
for category in sorted(evoked_plots.keys()):
    if "Evoked" in category or "Stimulated" in category:
        print(f"  - {category}: {len(evoked_plots[category])} plots")
        for plot in evoked_plots[category]:
            print(f"    * {plot}")

# Test 3: Check filtering works (evoked plots excluded for spontaneous)
print("\n" + "=" * 80)
print("TEST 3: Verify evoked plots are excluded for spontaneous experiments")
print("=" * 80)
spontaneous_explicit = get_available_plots(
    group=AnalysisGroup.SINGLE_WELL,
    has_detection=True,
    has_extraction=True,
    has_analysis=True,
    experiment_type="Spontaneous",  # Different from EVOKED
)

evoked_count_spont = sum(
    1
    for plots in spontaneous_explicit.values()
    for plot in plots
    if "Stimulated" in plot or "Non-Stimulated" in plot
)

print(f"Evoked plots in spontaneous experiment: {evoked_count_spont}")
print(f"✅ PASS" if evoked_count_spont == 0 else "❌ FAIL")

# Test 4: Verify the new correlation/synchrony plots exist
print("\n" + "=" * 80)
print("TEST 4: Verify new stimulated/non-stimulated correlation plots")
print("=" * 80)
expected_plots = [
    "Stimulated Calcium Peaks Synchrony",
    "Stimulated Calcium Peaks Cross-Correlation",
    "Stimulated Inferred Spikes Synchrony",
    "Stimulated Inferred Spikes Cross-Correlation",
    "Non-Stimulated Calcium Peaks Synchrony",
    "Non-Stimulated Calcium Peaks Cross-Correlation",
    "Non-Stimulated Inferred Spikes Synchrony",
    "Non-Stimulated Inferred Spikes Cross-Correlation",
]

all_plots_flat = [
    plot for plots in evoked_plots.values() for plot in plots
]

missing = [p for p in expected_plots if p not in all_plots_flat]
if missing:
    print(f"❌ FAIL - Missing plots: {missing}")
else:
    print(f"✅ PASS - All 8 new correlation/synchrony plots found!")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"When experiment_type=None: {sum(len(p) for p in spontaneous_plots.values())} plots")
print(f"When experiment_type='evoked': {sum(len(p) for p in evoked_plots.values())} plots")
print(f"When experiment_type='spontaneous': {sum(len(p) for p in spontaneous_explicit.values())} plots")
