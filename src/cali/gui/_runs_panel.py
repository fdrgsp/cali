"""Panel widget for displaying analysis and detection runs."""

from __future__ import annotations

from pathlib import Path

from fonticon_mdi6 import MDI6
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from sqlmodel import select
from superqt.fonticon import icon

from cali._constants import RED
from cali.logger import cali_logger
from cali.sqlmodel._model import AnalysisResult, DetectionSettings


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

        # Database path
        self._database_path: Path | None = None

        # Layout
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

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()  # Push buttons to the right

        # Delete selected run button
        self._delete_btn = QPushButton("Delete Selected")
        self._delete_btn.setIcon(icon(MDI6.delete, color=RED))
        self._delete_btn.setToolTip("Delete the selected run from the database")
        self._delete_btn.clicked.connect(self._delete_selected_run)
        self._delete_btn.setEnabled(False)  # Disabled by default
        buttons_layout.addWidget(self._delete_btn)

        # Clear all runs button
        self._clear_all_btn = QPushButton("Delete All")
        self._clear_all_btn.setIcon(icon(MDI6.delete_forever, color=RED))
        self._clear_all_btn.setToolTip("Delete all runs from the database")
        self._clear_all_btn.clicked.connect(self._clear_all_runs)
        buttons_layout.addWidget(self._clear_all_btn)

        layout.addLayout(buttons_layout)

        # Connect selection change to enable/disable delete button
        self._runs_list.itemSelectionChanged.connect(self._on_selection_changed)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def clear(self) -> None:
        """Clear the runs list."""
        self._runs_list.clear()

    def database_path(self) -> Path | None:
        """Get the current database path.

        Returns
        -------
        Path | None
            Path to the database file or None if not set
        """
        return self._database_path

    def set_database_path(self, db_path: Path | str) -> None:
        """Set the database path and reload runs.

        Parameters
        ----------
        db_path : Path | None
            Path to the database file
        """
        if isinstance(db_path, str):
            db_path = Path(db_path)

        self._database_path = db_path
        self.refresh_runs()

    def refresh_runs(self) -> None:
        """Refresh the list of runs from the database."""
        self.clear()

        if self._database_path is None or not self._database_path.exists():
            return

        try:
            # Load all analysis results ordered by creation time (most recent first)
            results = AnalysisResult.load_from_database(self._database_path)

            if not isinstance(results, list):
                results = [results]

            for result in sorted(results, key=lambda r: r.created_at):
                self._add_run_item(result)

        except Exception as e:
            cali_logger.error(f"Error loading runs: {e}")

    def _add_run_item(self, result: AnalysisResult) -> None:
        """Add a run item to the list.

        Parameters
        ----------
        result : AnalysisResult
            The analysis result to add
        """
        if self._database_path is None:
            return

        # Format the display text
        created_at = result.created_at.strftime("%Y-%m-%d %H:%M:%S")

        # Get detection settings id and method
        d_id = result.detection_settings
        d_settings = DetectionSettings.load_from_database(self._database_path, id=d_id)
        assert isinstance(d_settings, DetectionSettings)  # cannot be a list here

        item_text = (
            f"Run #{result.id} - {created_at}\n"
            f"  ✅ Detection ID: {d_id} - Method: {d_settings.method}\n"
        )
        analysis_icon = "❌" if result.analysis_settings is None else "✅"
        item_text += f"  {analysis_icon} Analysis ID: {result.analysis_settings}"

        item = QListWidgetItem(item_text)
        item.setIcon(icon(MDI6.run_fast))
        item.setData(Qt.ItemDataRole.UserRole, result.id)

        self._runs_list.addItem(item)

    def _on_selection_changed(self) -> None:
        """Handle selection change to enable/disable delete button."""
        has_selection = len(self._runs_list.selectedItems()) > 0
        self._delete_btn.setEnabled(has_selection)

    def _delete_selected_run(self) -> None:
        """Delete the selected run from the database."""
        current_item = self._runs_list.currentItem()
        if current_item is None:
            return

        run_id = current_item.data(Qt.ItemDataRole.UserRole)
        if run_id is None:
            return

        # Confirm deletion
        reply = QMessageBox.warning(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete Run #{run_id}?\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._delete_run_from_database(run_id)
            self.refresh_runs()
            cali_logger.info(f"🚮 Deleted Run #{run_id} from database.")

    def _clear_all_runs(self) -> None:
        """Delete all runs from the database."""
        if self._runs_list.count() == 0:
            return

        # Confirm clearing all
        reply = QMessageBox.warning(
            self,
            "Confirm Clear All",
            "Are you sure you want to delete ALL runs from the database?\n\n"
            "This will permanently delete all analysis results and detection "
            "settings.\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._clear_all_from_database()
            self.refresh_runs()
            cali_logger.info("🚮 Deleted ALL runs from database.")

    def _delete_run_from_database(self, run_id: int) -> None:
        """Delete a specific run from the database.

        Parameters
        ----------
        run_id : int
            The ID of the AnalysisResult to delete
        """
        if self._database_path is None:
            return

        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(f"sqlite:///{self._database_path}")
            with Session(engine) as session:
                # Delete the analysis result (this should cascade to related data)
                result = session.get(AnalysisResult, run_id)
                if result:
                    session.delete(result)
                    session.commit()
            engine.dispose(close=True)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to delete run: {e}")

    def _clear_all_from_database(self) -> None:
        """Delete all runs from the database."""
        if self._database_path is None:
            return

        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(f"sqlite:///{self._database_path}")
            with Session(engine) as session:
                # Delete all analysis results (this should cascade to related data)
                results = session.exec(select(AnalysisResult)).all()
                for result in results:
                    session.delete(result)
                session.commit()
            engine.dispose(close=True)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to clear all runs: {e}")

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
