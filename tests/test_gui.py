from pytestqt.qtbot import QtBot

from cali.gui import CaliGui


def test_launch_gui(qtbot: QtBot) -> None:
    """Test launching the Cali GUI."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui.show()
