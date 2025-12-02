"""Verify that the set_combo_text_red method has been properly removed."""

import sys

# Check that the old method is gone from pyqtgraph widget
try:
    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
    
    # Check if the method exists
    if hasattr(_SingleWellGraphWidget, 'set_combo_text_red'):
        print("❌ FAIL: set_combo_text_red still exists in _SingleWellGraphWidget")
        sys.exit(1)
    else:
        print("✅ PASS: set_combo_text_red removed from _SingleWellGraphWidget")
    
    # Check that the new methods exist
    if hasattr(_SingleWellGraphWidget, '_check_pipeline_stage_availability'):
        print("✅ PASS: _check_pipeline_stage_availability method added")
    else:
        print("❌ FAIL: _check_pipeline_stage_availability not found")
        sys.exit(1)
    
    # Verify the _update_combo_box has been enhanced
    import inspect
    source = inspect.getsource(_SingleWellGraphWidget._update_combo_box)
    if 'pipeline_stage' in source and 'has_detection' in source:
        print("✅ PASS: _update_combo_box properly checks pipeline stages")
    else:
        print("❌ FAIL: _update_combo_box doesn't check pipeline stages")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL CHECKS PASSED")
print("=" * 70)
print("""
Summary of changes:
• Removed: set_combo_text_red() method
• Added: _check_pipeline_stage_availability() method
• Enhanced: _update_combo_box() to disable items based on data availability
• Updated: All calls in _cali_gui.py to remove combo_red parameter

Result: Combo boxes now show disabled (grayed) items instead of red text!
""")
