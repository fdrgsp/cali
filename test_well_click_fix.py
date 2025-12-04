"""Manual test to verify well clicking behavior.

This script tests that:
1. Clicking on a well with no data clears plots
2. Clicking on a well with data shows plots
3. No need to click on "last used well" first
"""

from qtpy.QtWidgets import QApplication

from cali.gui import CaliGui

if __name__ == "__main__":
    app = QApplication([])

    # Create GUI with test database
    gui = CaliGui()
    gui._database_path = "tests/test_data/multi_pos/result_2pos.cali"
    gui._data_path = "tests/test_data/multi_pos/evk.tensorstore.zarr"
    gui._initialize_from_database(gui._database_path, gui._data_path)

    print("\n" + "=" * 80)
    print("MANUAL TEST: Well Click Behavior")
    print("=" * 80)
    print("\nTest Steps:")
    print("1. Select 'Calcium Raw Traces' in a graph widget combo")
    print("2. Click on a well with data (e.g., B5) - plot should display")
    print("3. Click on a well with NO data - plot should clear")
    print(
        "4. Click back on a well WITH data (e.g., B5) - plot should display IMMEDIATELY"
    )
    print("   (without needing to click on the 'last used well' first)")
    print("\nIf step 4 works correctly, the fix is successful!")
    print("=" * 80 + "\n")

    gui.show()
    app.exec()
