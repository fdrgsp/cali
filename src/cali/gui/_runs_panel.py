"""Panel widget for displaying analysis and detection runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fonticon_mdi6 import MDI6
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon

if TYPE_CHECKING:
    from pathlib import Path

    from cali.sqlmodel._model import AnalysisResult


class _RunsPanel(QGroupBox):
    """Panel that displays analysis and detection runs.

    This widget displays a list of all analysis runs stored in the database.

    Signals
    -------
    runSelected : int
        Emitted when a run is selected, passes the AnalysisResult ID
    """

    runSelected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Analysis Runs", parent=parent)

        # Panel properties
        self._database_path: Path | None = None

        # Setup UI
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # List widget for runs
        self._runs_list = QListWidget()
        self._runs_list.setAlternatingRowColors(True)
        self._runs_list.itemClicked.connect(self._on_run_clicked)
        self._runs_list.setToolTip(
            "Click on a run to load its analysis and detection settings"
        )

        layout.addWidget(self._runs_list)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def set_database_path(self, db_path: Path | None) -> None:
        """Set the database path and reload runs.

        Parameters
        ----------
        db_path : Path | None
            Path to the database file
        """
        self._database_path = db_path
        self.refresh_runs()

    def refresh_runs(self) -> None:
        """Refresh the list of runs from the database."""
        self._runs_list.clear()

        if self._database_path is None or not self._database_path.exists():
            return

        from cali.sqlmodel._model import AnalysisResult

        try:
            # Load all analysis results ordered by creation time (most recent first)
            results = AnalysisResult.load_from_database(self._database_path)

            if not isinstance(results, list):
                results = [results]

            for result in results:
                self._add_run_item(result)

        except Exception as e:
            print(f"Error loading runs: {e}")

    def _add_run_item(self, result: AnalysisResult) -> None:
        """Add a run item to the list.

        Parameters
        ----------
        result : AnalysisResult
            The analysis result to add
        """
        # Format the display text
        created_at = result.created_at.strftime("%Y-%m-%d %H:%M:%S")
        positions = len(result.positions_analyzed or [])

        # Get method from detection settings if available
        method = "N/A"
        if hasattr(result, "detection_settings_obj") and result.detection_settings_obj:
            method = result.detection_settings_obj.method.capitalize()

        item_text = (
            f"Run #{result.id} - {created_at}\n"
            f"  Method: {method} | Positions: {positions}"
        )

        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, result.id)

        # Add icon based on status
        item.setIcon(icon(MDI6.check_circle, color="green"))

        self._runs_list.addItem(item)

    def _on_run_clicked(self, item: QListWidgetItem) -> None:
        """Handle run item click.

        Parameters
        ----------
        item : QListWidgetItem
            The clicked item
        """
        run_id = item.data(Qt.ItemDataRole.UserRole)
        if run_id is not None:
            self.runSelected.emit(run_id)


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main_window = QWidget()
    main_window.setGeometry(100, 100, 800, 600)
    main_window.setWindowTitle("Runs Panel Test")

    layout = QVBoxLayout(main_window)
    runs_panel = _RunsPanel(main_window)
    layout.addWidget(runs_panel)

    main_window.show()

    sys.exit(app.exec())
