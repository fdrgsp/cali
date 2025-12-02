"""Dialog for selecting a run when multiple compatible runs exist."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from cali.sqlmodel._model import CaliResult


class RunSelectionDialog(QDialog):
    """Dialog for selecting a run when ambiguity is detected.

    This dialog is shown when multiple runs exist with the same detection settings
    but different extraction/analysis settings, and the user needs to disambiguate
    which run to add new detected positions to.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        runs: list[CaliResult] | None = None,
        message: str = "",
    ) -> None:
        """Initialize the run selection dialog.

        Parameters
        ----------
        parent : QWidget | None
            Parent widget
        runs : list[CaliResult] | None
            List of compatible runs to choose from
        message : str
            Message explaining why disambiguation is needed
        """
        super().__init__(parent)
        self.setWindowTitle("⚠️ Multiple Compatible Runs Found")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        self._runs = runs or []
        self._selected_run: CaliResult | None = None

        layout = QVBoxLayout(self)

        # Explanation label
        explanation = QLabel(message)
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Instruction label
        instruction = QLabel(
            "\nPlease select which run the new detected positions should be added to:"
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        # List widget for displaying runs
        self._runs_list = QListWidget()
        self._runs_list.setAlternatingRowColors(True)

        # Populate the list with run information
        for run in self._runs:
            item_text = self._format_run_display(run)
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, run.id)
            self._runs_list.addItem(item)

        # Connect double-click to accept
        self._runs_list.itemDoubleClicked.connect(self.accept)

        layout.addWidget(self._runs_list)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _format_run_display(self, run: CaliResult) -> str:
        """Format a run for display in the list.

        Parameters
        ----------
        run : CaliResult
            The run to format

        Returns
        -------
        str
            Formatted display string
        """
        parts = [f"Run ID {run.id}:"]

        # Detection info
        parts.append(f"  Detection: ID {run.detection_settings_id}")

        # Extraction info
        if run.extraction_settings_id is not None:
            parts.append(f"  Extraction: ID {run.extraction_settings_id}")
        else:
            parts.append("  Extraction: None")

        # Analysis info
        if run.analysis_settings_id is not None:
            parts.append(f"  Analysis: ID {run.analysis_settings_id}")
        else:
            parts.append("  Analysis: None")

        # Position info
        positions_info = []
        if run.positions_detected:
            positions_info.append(f"detected={run.positions_detected}")
        if run.positions_extracted:
            positions_info.append(f"extracted={run.positions_extracted}")
        if run.positions_analyzed:
            positions_info.append(f"analyzed={run.positions_analyzed}")

        if positions_info:
            parts.append(f"  Positions: {', '.join(positions_info)}")

        return "\n".join(parts)

    def get_selected_run_id(self) -> int | None:
        """Get the ID of the selected run.

        Returns
        -------
        int | None
            The ID of the selected run, or None if no run was selected
        """
        selected_items = self._runs_list.selectedItems()
        if selected_items:
            run_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
            return int(run_id) if run_id is not None else None
        return None

    @staticmethod
    def select_run(
        parent: QWidget | None,
        runs: list[CaliResult],
        message: str,
    ) -> int | None:
        """Show the dialog and return the selected run ID.

        Parameters
        ----------
        parent : QWidget
            Parent widget
        runs : list[CaliResult]
            List of compatible runs to choose from
        message : str
            Message explaining why disambiguation is needed

        Returns
        -------
        int | None
            The ID of the selected run, or None if cancelled
        """
        dialog = RunSelectionDialog(parent, runs, message)
        if dialog.exec():
            return dialog.get_selected_run_id()
        return None
