"""Panel widget for displaying analysis and detection runs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fonticon_mdi6 import MDI6
from qtpy.QtCore import QEvent, QObject, Qt, Signal
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
from superqt.utils import signals_blocked

from cali._constants import RED
from cali.logger import cali_logger
from cali.sqlmodel._model import AnalysisSettings, CaliResult, DetectionSettings

if TYPE_CHECKING:
    from sqlmodel import Session


class _RunsPanel(QGroupBox):
    """Panel that displays analysis and detection runs.

    This widget displays a list of all analysis runs stored in the database.

    Signals
    -------
    runSelected : int
        Emitted when a run is selected, passes the CaliResult ID
    settingsChanged : None
        Emitted when detection settings may have changed (e.g., after deletion)
    """

    runSelected = Signal(int)
    settingsDeleted = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Cali Runs", parent=parent)

        # Database path
        self._database_path: Path | None = None

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # List widget for runs
        self._runs_list = QListWidget()
        self._runs_list.setAlternatingRowColors(True)
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
        self._runs_list.itemClicked.connect(self._on_item_clicked)

        # Allow deselecting by clicking empty area in list
        self._runs_list.viewport().installEventFilter(self)

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
            from sqlalchemy import desc
            from sqlmodel import Session, create_engine, select

            engine = create_engine(f"sqlite:///{self._database_path}")
            with Session(engine) as session:
                # Join CaliResult with DetectionSettings to avoid N+1 queries
                # Order by created_at descending (most recent first)
                stmt = (
                    select(CaliResult, DetectionSettings)
                    .where(CaliResult.detection_settings == DetectionSettings.id)
                    .order_by(desc(CaliResult.created_at))  # type: ignore
                )
                results = session.exec(stmt).all()

                for result, detection_settings in results:
                    self._add_run_item(result, detection_settings)

            engine.dispose(close=True)

        except Exception as e:
            cali_logger.error(f"Error loading runs: {e}")

    def _add_run_item(
        self, result: CaliResult, detection_settings: DetectionSettings
    ) -> None:
        """Add a run item to the list.

        Parameters
        ----------
        result : CaliResult
            The analysis result to add
        detection_settings : DetectionSettings
            The detection settings associated with the result
        """
        # Format the display text
        created_at = result.created_at.strftime("%Y-%m-%d %H:%M:%S")

        d_id = result.detection_settings
        item_text = (
            f"Run #{result.id} - {created_at}\n"
            f"  ✅ Detection ID: {d_id} ({detection_settings.method})\n"
        )
        
        # Extraction status
        extraction_icon = "❌" if result.extraction_settings is None else "✅"
        item_text += f"  {extraction_icon} Extraction ID: {result.extraction_settings}\n"
        
        # Analysis status
        analysis_icon = "❌" if result.analysis_settings is None else "✅"
        item_text += f"  {analysis_icon} Analysis ID: {result.analysis_settings}"

        item = QListWidgetItem(item_text)
        # item.setIcon(icon(MDI6.run_fast))
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
            self.settingsDeleted.emit()  # Notify that settings may have changed
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
            self.settingsDeleted.emit()  # Notify that settings may have changed
            cali_logger.info("🚮 Deleted ALL runs from database.")

    def _delete_run_from_database(self, run_id: int) -> None:
        """Delete a specific run from the database with smart cascading.

        Deletes the run and cleans up orphaned settings and ROIs:
        - Deletes DetectionSettings if no other run uses them (and their ROIs)
        - Deletes AnalysisSettings if no other run uses them

        Parameters
        ----------
        run_id : int
            The ID of the CaliResult to delete
        """
        if self._database_path is None:
            return

        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(f"sqlite:///{self._database_path}")
            with Session(engine) as session:
                # Get the result before deleting to capture its settings IDs
                result = session.get(CaliResult, run_id)
                if not result:
                    return

                detection_id = result.detection_settings
                analysis_id = result.analysis_settings

                # Delete the analysis result (cascades to Traces via relationship)
                session.delete(result)
                session.commit()

                # Clean up orphaned settings and ROIs
                self._cleanup_orphaned_data(session, detection_id, analysis_id)

            engine.dispose(close=True)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to delete run: {e}")

    def _clear_all_from_database(self) -> None:
        """Delete all runs from the database with smart cascading.

        Deletes all runs and cleans up all orphaned settings and ROIs.
        """
        if self._database_path is None:
            return

        try:
            from sqlmodel import Session, create_engine, delete, select

            from cali.sqlmodel._model import ROI

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            with Session(engine) as session:
                # Collect all detection and analysis settings before deleting
                # We only need the IDs
                stmt = select(
                    CaliResult.detection_settings, CaliResult.analysis_settings
                )
                rows = session.exec(stmt).all()

                det_ids = {r[0] for r in rows if r[0] is not None}
                ana_ids = {r[1] for r in rows if r[1] is not None}

                # Delete all analysis results (cascades to Traces)
                session.exec(delete(CaliResult))

                # Clean up orphaned settings and ROIs
                if det_ids:
                    # Delete ROIs associated with these detection settings
                    session.exec(
                        delete(ROI).where(ROI.detection_settings_id.in_(det_ids))  # type: ignore
                    )
                    # Delete DetectionSettings
                    session.exec(
                        delete(DetectionSettings).where(
                            DetectionSettings.id.in_(det_ids)  # type: ignore
                        )
                    )

                if ana_ids:
                    # Delete AnalysisSettings
                    session.exec(
                        delete(AnalysisSettings).where(
                            AnalysisSettings.id.in_(ana_ids)  # type: ignore
                        )
                    )

                session.commit()

            engine.dispose(close=True)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to clear all runs: {e}")

    def _cleanup_orphaned_data(
        self,
        session: Session,
        detection_id: int | None,
        analysis_id: int | None,
    ) -> None:
        """Clean up orphaned settings and ROIs after deleting a run.

        Parameters
        ----------
        session : Session
            Database session
        detection_id : int | None
            Detection settings ID to check
        analysis_id : int | None
            Analysis settings ID to check
        """
        from cali.sqlmodel._model import (
            FOV,
            ROI,
            AnalysisSettings,
            DetectionSettings,
        )

        # Check if DetectionSettings are orphaned (not used by any other run)
        if detection_id is not None:
            other_runs_using_detection = session.exec(
                select(CaliResult).where(CaliResult.detection_settings == detection_id)
            ).first()

            if not other_runs_using_detection:
                # No other runs use this detection - delete the settings and ROIs
                cali_logger.info(
                    f"🧹 Cleaning up orphaned DetectionSettings #{detection_id}"
                )

                # Delete all ROIs with this detection_settings_id. These ROIs are
                # deleted even if their FOV contains ROIs from other detections
                # (This will cascade to delete Traces, DataAnalysis, and Masks)
                rois_to_delete = session.exec(
                    select(ROI).where(ROI.detection_settings_id == detection_id)
                ).all()

                roi_count = len(rois_to_delete)
                fov_ids = {roi.fov_id for roi in rois_to_delete}

                for roi in rois_to_delete:
                    session.delete(roi)

                # Delete the detection settings
                detection_settings = session.get(DetectionSettings, detection_id)
                if detection_settings:
                    session.delete(detection_settings)

                session.commit()

                cali_logger.info(
                    f"  Deleted {roi_count} ROIs from {len(fov_ids)} FOV(s)"
                )

                # Clean up empty FOVs
                # (only delete FOVs that have NO ROIs left from any detection)
                for fov_id in fov_ids:
                    fov = session.get(FOV, fov_id)
                    if fov:
                        # Refresh to get updated relationships
                        session.refresh(fov)
                        if not fov.rois:
                            cali_logger.info(f"  Deleting empty FOV {fov.name}")
                            session.delete(fov)

                session.commit()

        # Check if AnalysisSettings are orphaned (not used by any other run)
        if analysis_id is not None:
            other_runs_using_analysis = session.exec(
                select(CaliResult).where(CaliResult.analysis_settings == analysis_id)
            ).first()

            if not other_runs_using_analysis:
                # No other runs use this analysis - delete the settings
                cali_logger.info(
                    f"🧹 Cleaning up orphaned AnalysisSettings #{analysis_id}"
                )
                analysis_settings = session.get(AnalysisSettings, analysis_id)
                if analysis_settings:
                    session.delete(analysis_settings)
                    session.commit()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle run item click.

        Parameters
        ----------
        item : QListWidgetItem
            The clicked item
        """
        run_id = item.data(Qt.ItemDataRole.UserRole)
        if run_id is not None:
            self.runSelected.emit(run_id)

    def get_selected_run_id(self) -> int | None:
        """Get the ID of the currently selected run.

        Returns
        -------
        int | None
            CaliResult ID of the selected run, or None if no run selected
        """
        current_item = self._runs_list.currentItem()
        if current_item is None:
            return None

        return current_item.data(Qt.ItemDataRole.UserRole)

    def get_selected_detection_settings_id(self) -> int | None:
        """Get the detection settings ID from the currently selected run.

        Returns
        -------
        int | None
            Detection settings ID of the selected run, or None if no run selected
        """
        current_item = self._runs_list.currentItem()
        if current_item is None or self._database_path is None:
            return None

        run_id = current_item.data(Qt.ItemDataRole.UserRole)
        if self._database_path is None:
            return None

        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            with Session(engine) as session:
                result = session.get(CaliResult, run_id)
                detection_id = result.detection_settings if result else None
            engine.dispose(close=True)
            return detection_id
        except Exception as e:
            cali_logger.error(f"Failed to get detection settings ID: {e}")
            return None

    def get_detection_settings_ids(self) -> list[int]:
        """Get all unique detection settings IDs from database.

        Queries the DetectionSettings table directly to find all available
        detection settings, regardless of whether they're in a CaliResult.

        Returns
        -------
        list[int]
            Sorted list of unique detection settings IDs
        """
        if self._database_path is None:
            return []

        try:
            from sqlmodel import Session, create_engine, select

            from cali.sqlmodel._model import DetectionSettings

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            with Session(engine) as session:
                # Get all detection settings IDs directly from the table
                stmt = select(DetectionSettings.id)
                results = session.exec(stmt).all()
                ids = {r for r in results if r is not None}
            engine.dispose(close=True)
            return sorted(ids)
        except Exception as e:
            cali_logger.error(f"Failed to get detection settings IDs: {e}")
            return []

    def get_extraction_settings_ids(self) -> list[int]:
        """Get all unique extraction settings IDs from database.

        Queries the ExtractionSettings table directly to find all available
        extraction settings, regardless of whether they're in a CaliResult.

        Returns
        -------
        list[int]
            Sorted list of unique extraction settings IDs
        """
        if self._database_path is None:
            return []

        try:
            from sqlmodel import Session, create_engine, select

            from cali.sqlmodel._model import ExtractionSettings

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            with Session(engine) as session:
                # Get all extraction settings IDs directly from the table
                stmt = select(ExtractionSettings.id)
                results = session.exec(stmt).all()
                ids = {r for r in results if r is not None}
            engine.dispose(close=True)
            return sorted(ids)
        except Exception as e:
            cali_logger.error(f"Failed to get extraction settings IDs: {e}")
            return []

    def get_analysis_settings_ids(self) -> list[int]:
        """Get all unique analysis settings IDs from runs.

        Returns
        -------
        list[int]
            Sorted list of unique analysis settings IDs
        """
        if self._database_path is None:
            return []

        try:
            from sqlmodel import Session, create_engine, select

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            with Session(engine) as session:
                # Get all unique analysis settings IDs
                stmt = select(CaliResult.analysis_settings).distinct()
                results = session.exec(stmt).all()
                ids = {r for r in results if r is not None}
            engine.dispose(close=True)
            return sorted(ids)
        except Exception as e:
            cali_logger.error(f"Failed to get analysis settings IDs: {e}")
            return []

    def highlight_run_by_settings(
        self,
        detection_id: int | None,
        extraction_id: int | None,
        analysis_id: int | None,
    ) -> None:
        """Highlight the run that matches detection, extraction, and analysis settings.

        If no exact match is found, deselect all runs.

        Parameters
        ----------
        detection_id : int | None
            Detection settings ID to match
        extraction_id : int | None
            Extraction settings ID to match (None for detection-only runs)
        analysis_id : int | None
            Analysis settings ID to match (None for runs without analysis)
        """
        if self._database_path is None:
            return

        try:
            from sqlalchemy import desc
            from sqlmodel import Session, create_engine, select

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            with Session(engine) as session:
                # Find run with matching settings
                query = select(CaliResult)
                if detection_id is not None:
                    query = query.where(CaliResult.detection_settings == detection_id)
                if extraction_id is not None:
                    query = query.where(CaliResult.extraction_settings == extraction_id)
                if analysis_id is not None:
                    query = query.where(CaliResult.analysis_settings == analysis_id)

                # Order by created_at desc and take first
                query = query.order_by(desc(CaliResult.created_at))  # type: ignore
                matching_run = session.exec(query).first()

            engine.dispose(close=True)

            # Find and select the matching item in the list
            if matching_run:
                for i in range(self._runs_list.count()):
                    item = self._runs_list.item(i)
                    if item and item.data(Qt.ItemDataRole.UserRole) == matching_run.id:
                        with signals_blocked(self._runs_list):
                            self._runs_list.setCurrentItem(item)
                        return

            # No match found - deselect all
            self._runs_list.clearSelection()

        except Exception as e:
            cali_logger.error(f"Failed to highlight run by settings: {e}")

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Filter events to allow deselecting by clicking empty area in list.

        Parameters
        ----------
        a0 : QObject | None
            The object that received the event
        a1 : QEvent | None
            The event to filter

        Returns
        -------
        bool
            True if event was handled, False otherwise
        """
        if (
            a0 == self._runs_list.viewport()
            and a1
            and a1.type() == QEvent.Type.MouseButtonPress
        ):
            # Check if click is on empty area
            item = self._runs_list.itemAt(a1.pos())  # type: ignore
            if item is None:
                # Clicked on white area - deselect all
                self._runs_list.clearSelection()
        return super().eventFilter(a0, a1)
