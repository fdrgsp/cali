"""Demonstrate the combo box disabling logic without GUI."""

from cali.plot._main_plot import ANALYSIS_PRODUCTS, AnalysisGroup, PipelineStage

# Count plots by pipeline stage requirement
by_stage = {
    PipelineStage.DETECTION: [],
    PipelineStage.EXTRACTION: [],
    PipelineStage.ANALYSIS: [],
}

for product in ANALYSIS_PRODUCTS:
    if product.group == AnalysisGroup.SINGLE_WELL:
        by_stage[product.pipeline_stage].append(product.name)

print("=" * 80)
print("PLOT AVAILABILITY BY PIPELINE STAGE")
print("=" * 80)

print(f"\n📍 DETECTION-only plots (always available if ROIs exist): {len(by_stage[PipelineStage.DETECTION])}")
for plot in by_stage[PipelineStage.DETECTION][:3]:
    print(f"  ✓ {plot}")
if len(by_stage[PipelineStage.DETECTION]) > 3:
    print(f"  ... and {len(by_stage[PipelineStage.DETECTION]) - 3} more")

print(f"\n📊 EXTRACTION plots (require traces): {len(by_stage[PipelineStage.EXTRACTION])}")
for plot in by_stage[PipelineStage.EXTRACTION][:3]:
    print(f"  ✓ {plot}")
if len(by_stage[PipelineStage.EXTRACTION]) > 3:
    print(f"  ... and {len(by_stage[PipelineStage.EXTRACTION]) - 3} more")

print(f"\n📈 ANALYSIS plots (require full pipeline): {len(by_stage[PipelineStage.ANALYSIS])}")
for plot in by_stage[PipelineStage.ANALYSIS][:5]:
    print(f"  ✓ {plot}")
if len(by_stage[PipelineStage.ANALYSIS]) > 5:
    print(f"  ... and {len(by_stage[PipelineStage.ANALYSIS]) - 5} more")

print("\n" + "=" * 80)
print("HOW IT WORKS IN THE GUI")
print("=" * 80)

scenarios = [
    ("After Detection Only", True, False, False),
    ("After Detection + Extraction", True, True, False),
    ("After Full Pipeline (Detection + Extraction + Analysis)", True, True, True),
]

for scenario_name, has_det, has_ext, has_ana in scenarios:
    print(f"\n{scenario_name}:")
    print(f"  • Detection plots: {'✅ ENABLED' if has_det else '❌ DISABLED'}")
    print(f"  • Extraction plots: {'✅ ENABLED' if (has_det and has_ext) else '❌ DISABLED'}")
    print(f"  • Analysis plots: {'✅ ENABLED' if (has_det and has_ext and has_ana) else '❌ DISABLED'}")

print("\n" + "=" * 80)
print("✅ IMPROVED USER EXPERIENCE")
print("=" * 80)
print("""
BEFORE: Combo text was red - unclear which plots are available
AFTER:  Unavailable plots are grayed out and not clickable

Benefits:
• Clear visual indication of what's available
• Prevents clicking on plots that will fail
• Users can see full list of plots (roadmap of what's coming)
• No confusing red text - standard disabled UI pattern
""")
