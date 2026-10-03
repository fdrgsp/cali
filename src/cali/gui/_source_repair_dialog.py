"""Explicit extraction selection and verified comparison for legacy source repair."""

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from sqlmodel import Session, col, select

from cali.sqlmodel import (
    CaliResult,
    create_cali_engine,
    preview_legacy_result_source,
)


class _SourceRepairDialog(QDialog):
    """Compare a user-selected extraction before enabling source repair."""

    def __init__(
        self, database_path: Path, result_id: int, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._database_path = database_path
        self._result_id = result_id
        self._verified_source_id: int | None = None
        self.setWindowTitle(f"Repair source for Run #{result_id}")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Choose the extraction that produced this run's stored traces. "
            "cali will compare calcium and spike values, timing, units, cached "
            "metric settings, and ROI ordering before enabling repair.\n\n"
            "Applying a verified selection restores source links and records your "
            "choice in the audit. Stored scientific values are preserved. "
            "Dependent runs need their own source selection."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self._sources_combo = QComboBox()
        self._sources_combo.addItem("Select an extraction run...", None)
        layout.addWidget(self._sources_combo)
        self._comparison_label = QLabel("Select a source to compare.")
        self._comparison_label.setTextFormat(Qt.TextFormat.PlainText)
        self._comparison_label.setWordWrap(True)
        self._comparison_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._comparison_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        apply_btn = buttons.addButton(
            "Apply source repair", QDialogButtonBox.ButtonRole.AcceptRole
        )
        assert apply_btn is not None
        self._apply_btn = apply_btn
        self._apply_btn.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._sources_combo.currentIndexChanged.connect(self._compare_source)
        try:
            self._load_sources()
        except Exception as error:
            self._comparison_label.setText(
                f"Could not load extraction sources: {error}"
            )

    def _load_sources(self) -> None:
        if not self._database_path.exists():
            raise ValueError("The database is no longer available.")
        engine = create_cali_engine(f"sqlite:///{self._database_path}")
        try:
            with Session(engine) as session:
                result: CaliResult | None = session.get(CaliResult, self._result_id)
                if result is None:
                    raise ValueError("The selected run is no longer available.")
                sources = session.exec(
                    select(CaliResult)
                    .where(
                        CaliResult.experiment == result.experiment,
                        CaliResult.detection_settings_id
                        == result.detection_settings_id,
                        CaliResult.extraction_settings_id
                        == result.extraction_settings_id,
                        col(CaliResult.source_extraction_result_id)
                        == col(CaliResult.id),
                        col(CaliResult.id) != self._result_id,
                    )
                    .order_by(col(CaliResult.created_at), col(CaliResult.id))
                ).all()
                for source in sources:
                    resolution = source.legacy_trace_resolution or ""
                    if (
                        not source.positions_extracted
                        or resolution.startswith("unresolved")
                        or resolution == "multiple_sources"
                    ):
                        continue
                    positions = ", ".join(map(str, source.positions_extracted))
                    self._sources_combo.addItem(
                        f"Run #{source.id} · {source.created_at:%Y-%m-%d %H:%M:%S} "
                        f"· positions {positions}",
                        source.id,
                    )
        finally:
            engine.dispose(close=True)
        if self._sources_combo.count() == 1:
            self._comparison_label.setText(
                "No resolved extraction has matching detection and extraction "
                "settings. A fresh extraction and analysis is required."
            )

    def _compare_source(self) -> None:
        self._verified_source_id = None
        self._apply_btn.setEnabled(False)
        source_id = self._sources_combo.currentData()
        if source_id is None:
            self._comparison_label.setText("Select a source to compare.")
            return
        try:
            if not self._database_path.exists():
                raise ValueError("The database is no longer available.")
            engine = create_cali_engine(f"sqlite:///{self._database_path}")
            try:
                with Session(engine) as session:
                    preview = preview_legacy_result_source(
                        session, self._result_id, source_id
                    )
            finally:
                engine.dispose(close=True)
        except Exception as error:
            self._comparison_label.setText(
                f"This source cannot repair the stored result.\n\n{error}"
            )
            return
        outputs = ", ".join(
            f"{method.upper()} ({units})" for method, units in preview.spike_outputs
        )
        self._comparison_label.setText(
            f"Verified match: Run #{preview.result_id} → extraction "
            f"Run #{preview.source_result_id}\n"
            f"Positions: {', '.join(map(str, preview.positions)) or 'unknown'}\n"
            f"Matching ROI traces: {preview.trace_count}\n"
            f"ROI spike metrics: {preview.roi_metric_count}; "
            f"FOV spike metrics: {preview.fov_metric_count}\n"
            f"Stored spike outputs: {outputs or 'none'}\n"
            "Retained frame counts: "
            f"{', '.join(map(str, preview.retained_frame_counts)) or 'unknown'}\n"
            f"Source start frames (0-based): "
            f"{', '.join(map(str, preview.source_start_frames))}\n\n"
            "Calcium and spike payloads, frame coordinates, units, cached metric "
            "settings, and FOV ordering passed repair validation."
        )
        self._verified_source_id = source_id
        self._apply_btn.setEnabled(True)

    def selected_source_id(self) -> int | None:
        """Return only the explicitly selected, successfully verified source."""
        return self._verified_source_id

    def accept(self) -> None:
        """Accept only after the selected source has passed verification."""
        if self._verified_source_id is not None:
            super().accept()
