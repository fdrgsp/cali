"""Panel widget for displaying analysis and detection runs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from qtpy.QtCore import QEvent, QObject, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from sqlalchemy import func
from sqlmodel import select
from superqt import QIconifyIcon
from superqt.utils import signals_blocked

from cali._constants import RED
from cali.logger import cali_logger
from cali.sqlmodel._model import AnalysisSettings, CaliResult, DetectionSettings

if TYPE_CHECKING:
    from sqlmodel import Session


class _DetectionSummary(NamedTuple):
    """Summary info for a DetectionSettings row, used in dialogs and the saved list."""

    detection_id: int
    method: str
    model_type: str | None
    run_count: int
    roi_count: int
    fov_count: int

    def label(self) -> str:
        """Render a single-line description for UI."""
        head = f"Detection #{self.detection_id} — {self.method}"
        if self.model_type:
            head += f" / {self.model_type}"
        run_word = "run" if self.run_count == 1 else "runs"
        roi_word = "ROI" if self.roi_count == 1 else "ROIs"
        fov_word = "FOV" if self.fov_count == 1 else "FOVs"
        return (
            f"{head}  ({self.run_count} {run_word}, "
            f"{self.roi_count} {roi_word} across {self.fov_count} {fov_word})"
        )


class _DetectionKeepDialog(QDialog):
    """Dialog asking the user which detections to keep when deleting all runs."""

    def __init__(
        self, summaries: list[_DetectionSummary], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Delete All")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "All runs will be deleted.\n"
                "Tick any segmentations you want to keep — unticked ones will also "
                "be deleted along with their ROIs."
            )
        )
        self._checkboxes: dict[int, QCheckBox] = {}
        for summary in summaries:
            cb = QCheckBox(summary.label())
            self._checkboxes[summary.detection_id] = cb
            layout.addWidget(cb)
        if not summaries:
            layout.addWidget(QLabel("(no detection settings to keep)"))

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def kept_detection_ids(self) -> set[int]:
        return {did for did, cb in self._checkboxes.items() if cb.isChecked()}


class _RunsPanel(QGroupBox):
    """Panel that displays analysis and detection runs.

    This widget displays a list of all analysis runs stored in the database,
    plus a section listing orphan ("saved") segmentations — DetectionSettings
    rows kept around without an associated CaliResult.

    Signals
    -------
    runSelected : int
        Emitted when a run is selected, passes the CaliResult ID
    segmentationSelected : int
        Emitted when a saved (orphan) segmentation is selected, passes the
        DetectionSettings ID
    settingsDeleted : None
        Emitted when settings may have changed (e.g. after deletion)
    """

    runSelected = Signal(int)
    segmentationSelected = Signal(int)
    settingsDeleted = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("cali runs", parent=parent)

        # Database path
        self._database_path: Path | None = None

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Splitter so the user can resize runs vs saved segmentations
        self._splitter = QSplitter(Qt.Orientation.Vertical)

        # Runs list
        self._runs_list = QListWidget()
        self._runs_list.setAlternatingRowColors(True)
        self._runs_list.setToolTip(
            "Click on a run to load its analysis and detection settings"
        )
        self._splitter.addWidget(self._runs_list)

        # Saved segmentations section
        saved_container = QWidget()
        saved_layout = QVBoxLayout(saved_container)
        saved_layout.setContentsMargins(0, 0, 0, 0)
        saved_layout.setSpacing(2)
        _SAVED_SEGS_TOOLTIP = (
            "Each segmentation is normally stored as part of a run.\n"
            "When you delete a run, cali asks whether you also want to delete\n"
            "the segmentation (ROIs + masks) that was produced by that run.\n"
            "If you choose to keep it, it is preserved here — independent of\n"
            "any run — so you can inspect its labels or reuse it later.\n\n"
            "Click an entry to load its detection settings into the Detection\n"
            "tab and preview its labels in the image viewer."
        )
        saved_segs_label = QLabel("Saved Segmentations")
        saved_segs_label.setToolTip(_SAVED_SEGS_TOOLTIP)
        saved_layout.addWidget(saved_segs_label)
        self._saved_segs_list = QListWidget()
        self._saved_segs_list.setAlternatingRowColors(True)
        self._saved_segs_list.setToolTip(_SAVED_SEGS_TOOLTIP)
        saved_layout.addWidget(self._saved_segs_list)
        self._splitter.addWidget(saved_container)
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 1)

        layout.addWidget(self._splitter)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()  # Push buttons to the right

        # Delete selected button (works for either list)
        self._delete_btn = QPushButton("Delete Selected")
        self._delete_btn.setIcon(QIconifyIcon("mdi:delete", color=RED))
        self._delete_btn.setToolTip(
            "Delete the selected run (if it's the only run left, you will be asked if "
            "you want to keep the segmentation)."
        )
        self._delete_btn.clicked.connect(self._delete_selected)
        self._delete_btn.setEnabled(False)
        buttons_layout.addWidget(self._delete_btn)

        # Clear all button
        self._delete_all_btn = QPushButton("Delete All")
        self._delete_all_btn.setIcon(QIconifyIcon("mdi:delete-forever", color=RED))
        self._delete_all_btn.setToolTip(
            "Delete all runs (you will be asked if you want to keep any segmentations)."
        )
        self._delete_all_btn.clicked.connect(self._clear_all_runs)
        buttons_layout.addWidget(self._delete_all_btn)

        layout.addLayout(buttons_layout)

        # Selection wiring — clicking one list deselects the other
        self._runs_list.itemSelectionChanged.connect(self._on_runs_selection_changed)
        self._runs_list.itemClicked.connect(self._on_run_item_clicked)
        self._saved_segs_list.itemSelectionChanged.connect(
            self._on_saved_segs_selection_changed
        )
        self._saved_segs_list.itemClicked.connect(self._on_saved_seg_clicked)

        # Allow deselecting by clicking empty area in either list
        self._runs_list.viewport().installEventFilter(self)
        self._saved_segs_list.viewport().installEventFilter(self)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    # ------------------------------------------------------------------ public API

    def clear(self) -> None:
        """Clear both lists."""
        self._runs_list.clear()
        self._saved_segs_list.clear()

    def database_path(self) -> Path | None:
        """Get the current database path."""
        return self._database_path

    def set_database_path(self, db_path: Path | str) -> None:
        """Set the database path and reload runs."""
        if isinstance(db_path, str):
            db_path = Path(db_path)
        self._database_path = db_path
        self.refresh_runs()

    def refresh_runs(self) -> None:
        """Refresh the runs list and the saved-segmentations list."""
        self.clear()

        if self._database_path is None or not self._database_path.exists():
            return

        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(f"sqlite:///{self._database_path}")
            with Session(engine) as session:
                # Runs
                stmt = (
                    select(CaliResult, DetectionSettings)
                    .where(CaliResult.detection_settings_id == DetectionSettings.id)
                    .order_by(CaliResult.created_at)
                )
                for result, detection_settings in session.exec(stmt).all():
                    self._add_run_item(result, detection_settings)

                # Saved (orphan) segmentations: DetectionSettings without any CaliResult
                used_ids_stmt = select(CaliResult.detection_settings_id).where(
                    CaliResult.detection_settings_id.is_not(None)  # type: ignore[union-attr]
                )
                orphan_stmt = (
                    select(DetectionSettings)
                    .where(DetectionSettings.id.not_in(used_ids_stmt))  # type: ignore[union-attr]
                    .order_by(DetectionSettings.id)
                )
                for d_settings in session.exec(orphan_stmt).all():
                    summary = self._build_summary(session, d_settings)
                    self._add_saved_seg_item(summary)

            engine.dispose(close=True)

        except Exception as e:
            cali_logger.error(f"Error loading runs: {e}")

    def select_run_by_index(self, idx: int, block_signals: bool = False) -> None:
        """Select a run by its index in the list."""
        if 0 <= idx < self._runs_list.count():
            item = self._runs_list.item(idx)
            if item:
                self._runs_list.setCurrentItem(item)
                if not block_signals:
                    self._on_run_item_clicked(item)

    def select_run_by_id(self, run_id: int, block_signals: bool = False) -> None:
        """Select a run by its CaliResult ID."""
        for i in range(self._runs_list.count()):
            item = self._runs_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == run_id:
                self._runs_list.setCurrentItem(item)
                if not block_signals:
                    self._on_run_item_clicked(item)
                return

    def get_run_id_by_index(self, idx: int) -> int | None:
        """Get the CaliResult ID of the run at the given index."""
        if 0 <= idx < self._runs_list.count():
            item = self._runs_list.item(idx)
            if item:
                return item.data(Qt.ItemDataRole.UserRole)  # type: ignore
        return None

    def get_selected_run_id(self) -> int | None:
        """Get the ID of the currently selected run, or None if no run selected."""
        current_item = self._runs_list.currentItem()
        if current_item is None or not current_item.isSelected():
            return None
        return current_item.data(Qt.ItemDataRole.UserRole)  # type: ignore

    def get_selected_detection_settings_id(self) -> int | None:
        """Get the detection settings ID currently in focus.

        Returns the detection_settings_id from the selected run, or — if no run
        is selected — the detection_settings_id of the selected saved
        segmentation. Returns None if neither list has a selection.
        """
        # Saved seg has priority only when no run is selected
        run_item = self._runs_list.currentItem()
        if run_item is not None and run_item.isSelected():
            run_id = run_item.data(Qt.ItemDataRole.UserRole)
            if run_id is None or self._database_path is None:
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
                    detection_id = result.detection_settings_id if result else None
                engine.dispose(close=True)
                return detection_id
            except Exception as e:
                cali_logger.error(f"Failed to get detection settings ID: {e}")
                return None

        seg_item = self._saved_segs_list.currentItem()
        if seg_item is not None and seg_item.isSelected():
            return seg_item.data(Qt.ItemDataRole.UserRole)  # type: ignore

        return None

    def get_selected_saved_segmentation_id(self) -> int | None:
        """Get the DetectionSettings ID of the selected saved segmentation, if any."""
        item = self._saved_segs_list.currentItem()
        if item is None or not item.isSelected():
            return None
        return item.data(Qt.ItemDataRole.UserRole)  # type: ignore

    def get_detection_settings_ids(self) -> list[int]:
        """Get all unique detection settings IDs from the database."""
        return self._fetch_ids(DetectionSettings.id)

    def get_extraction_settings_ids(self) -> list[int]:
        """Get all unique extraction settings IDs from the database."""
        from cali.sqlmodel._model import ExtractionSettings

        return self._fetch_ids(ExtractionSettings.id)

    def get_analysis_settings_ids(self) -> list[int]:
        """Get all unique analysis settings IDs from runs."""
        if self._database_path is None:
            return []
        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            with Session(engine) as session:
                stmt = select(CaliResult.analysis_settings_id).distinct()
                ids = {r for r in session.exec(stmt).all() if r is not None}
            engine.dispose(close=True)
            return sorted(ids)
        except Exception as e:
            cali_logger.error(f"Failed to get analysis settings IDs: {e}")
            return []

    def get_run_ids(self) -> list[int]:
        """Get all run IDs from the database."""
        if self._database_path is None:
            return []
        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            try:
                with Session(engine) as session:
                    stmt = select(CaliResult.id)
                    ids = [r for r in session.exec(stmt).all() if r is not None]
                return sorted(ids)
            finally:
                engine.dispose(close=True)
        except Exception as e:
            cali_logger.error(f"Failed to get run IDs: {e}")
            return []

    def highlight_run_by_settings(
        self,
        detection_id: int | None,
        extraction_id: int | None,
        analysis_id: int | None,
    ) -> None:
        """Highlight the run that matches detection, extraction, and analysis settings.

        If no exact match is found, deselect all runs.
        """
        if self._database_path is None:
            return

        try:
            from sqlalchemy import desc
            from sqlmodel import Session, create_engine

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            with Session(engine) as session:
                query = select(CaliResult)
                if detection_id is not None:
                    query = query.where(
                        CaliResult.detection_settings_id == detection_id
                    )
                if extraction_id is not None:
                    query = query.where(
                        CaliResult.extraction_settings_id == extraction_id
                    )
                if analysis_id is not None:
                    query = query.where(CaliResult.analysis_settings_id == analysis_id)

                query = query.order_by(desc(CaliResult.created_at))
                matching_run = session.exec(query).first()

            engine.dispose(close=True)

            if matching_run:
                for i in range(self._runs_list.count()):
                    item = self._runs_list.item(i)
                    if item and item.data(Qt.ItemDataRole.UserRole) == matching_run.id:
                        with signals_blocked(self._runs_list):
                            self._runs_list.setCurrentItem(item)
                        return

            self._runs_list.clearSelection()

        except Exception as e:
            cali_logger.error(f"Failed to highlight run by settings: {e}")

    # ------------------------------------------------------------- internal helpers

    def _fetch_ids(self, column: Any) -> list[int]:
        """Fetch a sorted, unique list of non-null IDs from a single column."""
        if self._database_path is None:
            return []
        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            try:
                with Session(engine) as session:
                    ids = {
                        r for r in session.exec(select(column)).all() if r is not None
                    }
            finally:
                engine.dispose(close=True)
            return sorted(ids)
        except Exception as e:
            cali_logger.error(f"Failed to fetch IDs: {e}")
            return []

    def _build_summary(
        self, session: Session, d_settings: DetectionSettings
    ) -> _DetectionSummary:
        """Compute counts (runs / ROIs / FOVs) for a DetectionSettings row."""
        from cali.sqlmodel._model import ROI

        did = d_settings.id
        assert did is not None

        run_count = session.exec(
            select(func.count())
            .select_from(CaliResult)
            .where(CaliResult.detection_settings_id == did)
        ).one()
        roi_count = session.exec(
            select(func.count())
            .select_from(ROI)
            .where(ROI.detection_settings_id == did)
        ).one()
        fov_count = session.exec(
            select(func.count(func.distinct(ROI.fov_id))).where(
                ROI.detection_settings_id == did
            )
        ).one()

        return _DetectionSummary(
            detection_id=did,
            method=d_settings.method,
            model_type=d_settings.model_type,
            run_count=int(run_count or 0),
            roi_count=int(roi_count or 0),
            fov_count=int(fov_count or 0),
        )

    def _all_detection_summaries(self) -> list[_DetectionSummary]:
        """Build summaries for every DetectionSettings row in the DB."""
        if self._database_path is None:
            return []
        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            try:
                with Session(engine) as session:
                    rows = session.exec(
                        select(DetectionSettings).order_by(DetectionSettings.id)
                    ).all()
                    summaries = [self._build_summary(session, d) for d in rows]
            finally:
                engine.dispose(close=True)
            return summaries
        except Exception as e:
            cali_logger.error(f"Failed to compute detection summaries: {e}")
            return []

    def _count_runs_using_detection(self, detection_id: int) -> int:
        """Count CaliResults referencing the given detection_settings_id."""
        if self._database_path is None:
            return 0
        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            try:
                with Session(engine) as session:
                    count = session.exec(
                        select(func.count())
                        .select_from(CaliResult)
                        .where(CaliResult.detection_settings_id == detection_id)
                    ).one()
            finally:
                engine.dispose(close=True)
            return int(count or 0)
        except Exception as e:
            cali_logger.error(f"Failed to count runs using detection: {e}")
            return 0

    def _add_run_item(
        self, result: CaliResult, detection_settings: DetectionSettings
    ) -> None:
        """Add a run item to the list."""
        created_at = result.created_at.strftime("%Y-%m-%d %H:%M:%S")
        last_modified = result.last_modified.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        d_id = result.detection_settings_id
        item_text = (
            f"Run #{result.id} - {created_at}\n"
            f"  📝 Modified: {last_modified}\n"
            f"  ✅ Detection ID: {d_id} ({detection_settings.method})\n"
        )

        # Extraction status with incomplete indicator
        extraction_icon = "❌" if result.extraction_settings_id is None else "✅"
        extraction_incomplete = ""
        if result.extraction_settings_id is not None:
            detected = set(result.positions_detected or [])
            extracted = set(result.positions_extracted or [])
            if detected and extracted and len(detected) != len(extracted):
                extraction_incomplete = " ⚠️"

        item_text += (
            f"  {extraction_icon} Extraction ID: {result.extraction_settings_id}"
            f"{extraction_incomplete}\n"
        )

        # Analysis status with incomplete indicator
        analysis_icon = "❌" if result.analysis_settings_id is None else "✅"
        analysis_incomplete = ""
        if result.analysis_settings_id is not None:
            detected = set(result.positions_detected or [])
            analyzed = set(result.positions_analyzed or [])
            if detected and analyzed and len(detected) != len(analyzed):
                analysis_incomplete = " ⚠️"

        item_text += (
            f"  {analysis_icon} Analysis ID: {result.analysis_settings_id}"
            f"{analysis_incomplete}"
        )

        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, result.id)

        item.setToolTip(
            f"Run #{result.id}\n"
            f"Created: {created_at}\n"
            f"Last Modified: {last_modified}\n"
            f"Detection Settings ID: {d_id}\n"
            f"Extraction Settings ID: {result.extraction_settings_id}\n"
            f"Analysis Settings ID: {result.analysis_settings_id}\n"
            f"Positions Detected: {len(result.positions_detected or [])}\n"
            f"Positions Extracted: {len(result.positions_extracted or [])}\n"
            f"Positions Analyzed: {len(result.positions_analyzed or [])}\n"
        )

        self._runs_list.addItem(item)

    def _add_saved_seg_item(self, summary: _DetectionSummary) -> None:
        """Add an orphan-detection item to the saved segmentations list."""
        item = QListWidgetItem(summary.label())
        item.setData(Qt.ItemDataRole.UserRole, summary.detection_id)
        item.setToolTip(
            f"Saved segmentation\n"
            f"Detection Settings ID: {summary.detection_id}\n"
            f"Method: {summary.method}\n"
            f"Model: {summary.model_type}\n"
            f"ROIs: {summary.roi_count} across {summary.fov_count} FOV(s)"
        )
        self._saved_segs_list.addItem(item)

    # ------------------------------------------------------------------ selection

    def _on_runs_selection_changed(self) -> None:
        has_runs = len(self._runs_list.selectedItems()) > 0
        if has_runs:
            with signals_blocked(self._saved_segs_list):
                self._saved_segs_list.clearSelection()
        self._update_delete_button()

    def _on_saved_segs_selection_changed(self) -> None:
        has_segs = len(self._saved_segs_list.selectedItems()) > 0
        if has_segs:
            with signals_blocked(self._runs_list):
                self._runs_list.clearSelection()
        self._update_delete_button()

    def _update_delete_button(self) -> None:
        has_selection = bool(self._runs_list.selectedItems()) or bool(
            self._saved_segs_list.selectedItems()
        )
        self._delete_btn.setEnabled(has_selection)

    def _on_run_item_clicked(self, item: QListWidgetItem) -> None:
        run_id = item.data(Qt.ItemDataRole.UserRole)
        if run_id is not None:
            self.runSelected.emit(run_id)

    def _on_saved_seg_clicked(self, item: QListWidgetItem) -> None:
        detection_id = item.data(Qt.ItemDataRole.UserRole)
        if detection_id is not None:
            self.segmentationSelected.emit(detection_id)

    # -------------------------------------------------------------------- delete

    def _delete_selected(self) -> None:
        """Delete whichever item (run or saved segmentation) is currently selected."""
        if self._saved_segs_list.selectedItems():
            self._delete_selected_saved_segmentation()
        else:
            self._delete_selected_run()

    def _delete_selected_run(self) -> None:
        """Delete the selected run, asking about segmentation when relevant."""
        current_item = self._runs_list.currentItem()
        if current_item is None:
            return

        run_id = current_item.data(Qt.ItemDataRole.UserRole)
        if run_id is None:
            return

        # If this run is the only one using its detection, give the user a choice
        detection_id = self._get_detection_id_for_run(run_id)
        keep_detection = False
        if (
            detection_id is not None
            and self._count_runs_using_detection(detection_id) == 1
        ):
            choice = self._ask_keep_or_delete_segmentation(run_id, detection_id)
            if choice is None:
                return  # cancelled
            keep_detection = choice
        else:
            reply = QMessageBox.warning(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete Run #{run_id}?\n\n"
                "This action cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._delete_run_from_database(run_id, keep_detection=keep_detection)
        self.refresh_runs()
        self.settingsDeleted.emit()
        if keep_detection:
            cali_logger.info(
                f"🚮 Deleted Run #{run_id}; kept segmentation #{detection_id}."
            )
        else:
            cali_logger.info(f"🚮 Deleted Run #{run_id} from database.")

    def _delete_selected_saved_segmentation(self) -> None:
        """Delete an orphan segmentation (DetectionSettings + its ROIs/Masks)."""
        item = self._saved_segs_list.currentItem()
        if item is None:
            return
        detection_id = item.data(Qt.ItemDataRole.UserRole)
        if detection_id is None:
            return

        reply = QMessageBox.warning(
            self,
            "Delete Saved Segmentation",
            f"Delete saved segmentation #{detection_id}?\n\n"
            "All ROIs and masks created by this detection will be permanently "
            "removed.\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._delete_detection_data(detection_id)
        self.refresh_runs()
        self.settingsDeleted.emit()
        cali_logger.info(f"🚮 Deleted saved segmentation #{detection_id}.")

    def _ask_keep_or_delete_segmentation(
        self, run_id: int, detection_id: int
    ) -> bool | None:
        """Ask the user whether to keep the segmentation or delete it too.

        Returns
        -------
        bool | None
            True to keep, False to delete, None if the user cancelled.
        """
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Confirm Deletion")
        msg.setText(
            f"Run #{run_id} is the only run using Detection #{detection_id}.\n\n"
            "Keep the segmentation (ROIs + masks) for future use, or delete "
            "everything?"
        )
        keep_btn = msg.addButton("Keep segmentation", QMessageBox.ButtonRole.AcceptRole)
        delete_btn = msg.addButton(
            "Delete everything", QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(cancel_btn)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is cancel_btn:
            return None
        if clicked is keep_btn:
            return True
        if clicked is delete_btn:
            return False
        return None

    def _clear_all_runs(self) -> None:
        """Delete all runs, with a per-detection keep/delete dialog."""
        if self._runs_list.count() == 0 and self._saved_segs_list.count() == 0:
            return

        summaries = self._all_detection_summaries()
        dialog = _DetectionKeepDialog(summaries, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        kept_ids = dialog.kept_detection_ids()
        self._clear_all_from_database(keep_detection_ids=kept_ids)
        self.refresh_runs()
        self.settingsDeleted.emit()
        if kept_ids:
            cali_logger.info(
                f"🚮 Deleted all runs; kept segmentations: {sorted(kept_ids)}."
            )
        else:
            cali_logger.info("🚮 Deleted ALL runs from database.")

    # -------------------------------------------------------- DB delete operations

    def _get_detection_id_for_run(self, run_id: int) -> int | None:
        if self._database_path is None:
            return None
        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(f"sqlite:///{self._database_path}")
            try:
                with Session(engine) as session:
                    result = session.get(CaliResult, run_id)
                    detection_id = result.detection_settings_id if result else None
            finally:
                engine.dispose(close=True)
            return detection_id
        except Exception as e:
            cali_logger.error(f"Failed to look up detection for run {run_id}: {e}")
            return None

    def _delete_run_from_database(
        self, run_id: int, *, keep_detection: bool = False
    ) -> None:
        """Delete a run, optionally preserving its detection segmentation.

        Parameters
        ----------
        run_id : int
            The CaliResult ID to delete.
        keep_detection : bool
            If True, leave the DetectionSettings row + ROIs + masks in place
            even if no other run references them (they become an orphan
            "saved segmentation").
        """
        if self._database_path is None:
            return

        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(f"sqlite:///{self._database_path}")
            try:
                with Session(engine) as session:
                    result = session.get(CaliResult, run_id)
                    if not result:
                        return

                    detection_id = result.detection_settings_id
                    analysis_id = result.analysis_settings_id

                    # Delete the analysis result (cascades to Traces via FK)
                    session.delete(result)
                    session.commit()

                    # Clean up orphaned settings (and ROIs unless keep_detection)
                    self._cleanup_orphaned_data(
                        session,
                        detection_id,
                        analysis_id,
                        keep_detection=keep_detection,
                    )
            finally:
                engine.dispose(close=True)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to delete run: {e}")

    def _clear_all_from_database(
        self, keep_detection_ids: set[int] | None = None
    ) -> None:
        """Delete all runs; optionally preserve specific detection segmentations.

        Parameters
        ----------
        keep_detection_ids : set[int] | None
            DetectionSettings IDs to preserve (along with their ROIs/Masks).
            All other detections — and their ROIs — will be deleted.
            Extraction and analysis settings are always deleted (run-scoped).
        """
        if self._database_path is None:
            return

        keep = set(keep_detection_ids or set())

        try:
            from sqlmodel import Session, create_engine, delete

            from cali.sqlmodel._model import ROI, ExtractionSettings

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            try:
                with Session(engine) as session:
                    # Drop all runs (cascades to Traces, DataAnalysis, FOVAnalysis
                    # via FK)
                    session.exec(delete(CaliResult))
                    # Run-scoped settings always go
                    session.exec(delete(ExtractionSettings))
                    session.exec(delete(AnalysisSettings))
                    session.commit()

                    # Detection cleanup: per-detection so we can spare kept ones
                    all_detection_ids = [
                        did
                        for did in session.exec(select(DetectionSettings.id)).all()
                        if did is not None
                    ]
                    for did in all_detection_ids:
                        if did in keep:
                            continue
                        self._delete_detection_data(did, session=session)

                    # Untagged ROIs (detection_settings_id is NULL) are unreachable
                    # without a run, so wipe them too.
                    session.exec(
                        delete(ROI).where(ROI.detection_settings_id.is_(None))  # type: ignore[union-attr]
                    )
                    session.commit()

                    self._delete_empty_fovs(session)
            finally:
                engine.dispose(close=True)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to clear all runs: {e}")

    def _delete_detection_data(
        self, detection_id: int, session: Session | None = None
    ) -> None:
        """Delete a DetectionSettings row plus all its ROIs/Masks and any empty FOVs.

        If ``session`` is provided, the work happens in that session and the
        caller owns the lifecycle. Otherwise a one-off engine/session is opened.
        """
        from cali.sqlmodel._model import ROI, Mask

        def _do(s: Session) -> None:
            rois = s.exec(
                select(ROI).where(ROI.detection_settings_id == detection_id)
            ).all()
            fov_ids = {roi.fov_id for roi in rois}
            mask_ids = {roi.roi_mask_id for roi in rois if roi.roi_mask_id is not None}

            for roi in rois:
                s.delete(roi)
            s.flush()

            for mask_id in mask_ids:
                mask = s.get(Mask, mask_id)
                if mask is not None:
                    s.delete(mask)

            d_settings = s.get(DetectionSettings, detection_id)
            if d_settings is not None:
                s.delete(d_settings)

            s.commit()
            self._delete_empty_fovs(s, fov_ids)

        if session is not None:
            _do(session)
            return

        if self._database_path is None:
            return
        try:
            from sqlmodel import Session, create_engine

            engine = create_engine(f"sqlite:///{self._database_path}")
            try:
                with Session(engine) as s:
                    _do(s)
            finally:
                engine.dispose(close=True)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to delete segmentation: {e}")

    def _cleanup_orphaned_data(
        self,
        session: Session,
        detection_id: int | None,
        analysis_id: int | None,
        *,
        keep_detection: bool = False,
    ) -> None:
        """Clean up settings/ROIs orphaned by a single-run deletion.

        Parameters
        ----------
        session : Session
            Active session (caller owns transaction lifecycle).
        detection_id, analysis_id : int | None
            Settings IDs from the just-deleted run.
        keep_detection : bool
            If True, never delete the detection (even if orphaned).
        """
        # Detection + ROIs
        if detection_id is not None and not keep_detection:
            other_runs_using_detection = session.exec(
                select(CaliResult).where(
                    CaliResult.detection_settings_id == detection_id
                )
            ).first()
            if not other_runs_using_detection:
                cali_logger.info(
                    f"🧹 Cleaning up orphaned DetectionSettings #{detection_id}"
                )
                self._delete_detection_data(detection_id, session=session)

        # Analysis settings
        if analysis_id is not None:
            other_runs_using_analysis = session.exec(
                select(CaliResult).where(CaliResult.analysis_settings_id == analysis_id)
            ).first()
            if not other_runs_using_analysis:
                cali_logger.info(
                    f"🧹 Cleaning up orphaned AnalysisSettings #{analysis_id}"
                )
                analysis_settings = session.get(AnalysisSettings, analysis_id)
                if analysis_settings:
                    session.delete(analysis_settings)
                    session.commit()

    def _delete_empty_fovs(
        self, session: Session, fov_ids: set[int] | None = None
    ) -> None:
        """Delete FOVs that no longer have any ROIs.

        If ``fov_ids`` is None, scan every FOV.
        """
        from cali.sqlmodel._model import FOV

        if fov_ids is None:
            candidates = session.exec(select(FOV)).all()
        else:
            candidates = [
                fov for fov_id in fov_ids if (fov := session.get(FOV, fov_id))
            ]

        for fov in candidates:
            session.refresh(fov)
            if not fov.rois:
                cali_logger.info(f"  Deleting empty FOV {fov.name}")
                session.delete(fov)

        session.commit()

    # ------------------------------------------------------------------ events

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Allow deselecting by clicking empty area in either list."""
        if a1 and a1.type() == QEvent.Type.MouseButtonPress:
            if a0 is self._runs_list.viewport():
                if self._runs_list.itemAt(a1.pos()) is None:
                    self._runs_list.clearSelection()
            elif a0 is self._saved_segs_list.viewport():
                if self._saved_segs_list.itemAt(a1.pos()) is None:
                    self._saved_segs_list.clearSelection()
        return bool(super().eventFilter(a0, a1))
