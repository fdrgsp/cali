from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fonticon_mdi6 import MDI6
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon
from superqt.utils import signals_blocked

from cali._constants import GREEN, RED
from cali.gui._util import (
    _BrowseWidget,
    _ChoosePositionsWidget,
    create_divider_line,
    parse_lineedit_text,
)

if TYPE_CHECKING:
    from cali.sqlmodel._model import DetectionSettings

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed

CUSTOM_MODEL_PATH = (
    Path(__file__).parent.parent
    / "detection"
    / "cellpose_models"
    / "cp3_img8_epoch7000_py"
)


@dataclass(frozen=True)
class CellposeSettings:
    model_type: str = "cpsam"
    model_path: str | None = None
    diameter: float | None = None
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    min_size: int = 10
    normalize: bool = True
    batch_size: int = 8


@dataclass(frozen=True)
class CaimanSettings: ...


class _SelectModelPath(_BrowseWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "Custom Model",
        tooltip: str = "Choose the path to the custom Cellpose model.",
    ) -> None:
        super().__init__(parent, label, "", tooltip, is_dir=False)

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select the {self._label_text}.",
            "",
            "",
        )
        if path:
            self._path.setText(path)


class _CellposeDetectionWidget(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setTitle("Cellpose")
        self.setCheckable(True)

        # MODEL SELECTION WIDGETS -----------------------------------------------------
        self._model_wdg = QWidget(self)
        model_wdg_layout = QHBoxLayout(self._model_wdg)
        model_wdg_layout.setContentsMargins(0, 0, 0, 0)
        model_wdg_layout.setSpacing(5)
        self._models_combo_label = QLabel("Model Type:")
        self._models_combo_label.setSizePolicy(*FIXED)
        self._models_combo = QComboBox()
        self._models_combo.addItems(["cpsam", "cyto3", "custom"])
        self._models_combo.currentTextChanged.connect(self._on_model_combo_changed)
        model_wdg_layout.addWidget(self._models_combo_label)
        model_wdg_layout.addWidget(self._models_combo, 1)

        self._browse_custom_model = _SelectModelPath(self)
        self._browse_custom_model.setValue(CUSTOM_MODEL_PATH)
        self._browse_custom_model.hide()

        # DIAMETER WIDGETS ------------------------------------------------------------
        self._diameter_wdg = QWidget(self)
        self._diameter_wdg.setToolTip(
            "Set the diameter of the cells. Leave 0 for automatic detection."
        )
        diameter_layout = QHBoxLayout(self._diameter_wdg)
        diameter_layout.setContentsMargins(0, 0, 0, 0)
        diameter_layout.setSpacing(5)
        self._diameter_label = QLabel("Diameter:")
        self._diameter_label.setSizePolicy(*FIXED)
        self._diameter_spin = QDoubleSpinBox(self)
        self._diameter_spin.setSpecialValueText("Auto")
        self._diameter_spin.setRange(0, 1000)
        self._diameter_spin.setValue(0)
        diameter_layout.addWidget(self._diameter_label)
        diameter_layout.addWidget(self._diameter_spin)

        # CELLPOSE THRESHOLDS ---------------------------------------------------------
        self._cellprob_wdg = QWidget(self)
        self._cellprob_wdg.setToolTip(
            "Cell probability threshold (all pixels > threshold are used "
            "for dynamics). Lower values detect more masks. Default is 0.0 ("
            "cellpose default)."
        )
        prob_layout = QHBoxLayout(self._cellprob_wdg)
        prob_layout.setContentsMargins(0, 0, 0, 0)
        prob_layout.setSpacing(5)
        self._cellprob_label = QLabel("Cell Probability Threshold:")
        self._cellprob_label.setSizePolicy(*FIXED)
        self._cellprob_threshold_spin = QDoubleSpinBox(self)
        self._cellprob_threshold_spin.setRange(-6.0, 6.0)
        self._cellprob_threshold_spin.setValue(0.0)
        self._cellprob_threshold_spin.setSingleStep(0.1)
        prob_layout.addWidget(self._cellprob_label)
        prob_layout.addWidget(self._cellprob_threshold_spin)

        self._flow_wdg = QWidget(self)
        self._flow_wdg.setToolTip(
            "Flow error threshold (all cells with errors below threshold are kept). "
            "Higher values detect more masks."
        )
        flow_layout = QHBoxLayout(self._flow_wdg)
        flow_layout.setContentsMargins(0, 0, 0, 0)
        flow_layout.setSpacing(5)
        self._flow_label = QLabel("Flow Threshold:")
        self._flow_label.setSizePolicy(*FIXED)
        self._flow_threshold_spin = QDoubleSpinBox(self)
        self._flow_threshold_spin.setRange(0.0, 3.0)
        self._flow_threshold_spin.setValue(0.4)
        self._flow_threshold_spin.setSingleStep(0.1)
        self._flow_threshold_spin.setToolTip(
            "Flow error threshold (all cells with errors below threshold are kept). "
            "Higher values detect more masks. Default is 0.4 (cellpose default)."
        )
        flow_layout.addWidget(self._flow_label)
        flow_layout.addWidget(self._flow_threshold_spin)

        # MIN SIZE WIDGET -------------------------------------------------------------
        self._min_size_wdg = QWidget(self)
        self._min_size_wdg.setToolTip(
            "Minimum number of pixels for a mask to be kept. Masks smaller than "
            "this will be removed as they are likely artifacts or debris. "
            "Default is 15 pixels. Set to 1 to keep all masks."
        )
        min_size_layout = QHBoxLayout(self._min_size_wdg)
        min_size_layout.setContentsMargins(0, 0, 0, 0)
        min_size_layout.setSpacing(5)
        self._min_size_label = QLabel("Min Mask Size (px):")
        self._min_size_label.setSizePolicy(*FIXED)
        self._min_size_spin = QSpinBox()
        self._min_size_spin.setRange(1, 10000)
        self._min_size_spin.setValue(15)
        min_size_layout.addWidget(self._min_size_label)
        min_size_layout.addWidget(self._min_size_spin)

        # NORMALIZE CHECKBOX ----------------------------------------------------------
        self._normalize_wdg = QWidget(self)
        self._normalize_wdg.setToolTip(
            "Normalize images before segmentation. "
            "This rescales pixel values to 0-1 range using 1st and 99th percentiles.\n"
            "By default, this is enabled (cellpose default)."
        )
        normalize_layout = QHBoxLayout(self._normalize_wdg)
        normalize_layout.setContentsMargins(0, 0, 0, 0)
        normalize_layout.setSpacing(5)
        self._normalize_label = QLabel("Normalize Images:")
        self._normalize_label.setSizePolicy(*FIXED)
        self._normalize_checkbox = QCheckBox()
        self._normalize_checkbox.setChecked(True)
        normalize_layout.addWidget(self._normalize_label)
        normalize_layout.addWidget(self._normalize_checkbox)
        normalize_layout.addStretch(1)

        # BATCH SIZE WIDGET -----------------------------------------------------------
        self._batch_wdg = QWidget(self)
        self._batch_wdg.setToolTip(
            "Number of images to process per batch. Higher values are faster "
            "but use more memory."
        )
        batch_layout = QHBoxLayout(self._batch_wdg)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(5)
        self._batch_label = QLabel("Batch Size:")
        self._batch_label.setSizePolicy(*FIXED)
        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(1, 32)
        self._batch_size_spin.setValue(8)
        batch_layout.addWidget(self._batch_label)
        batch_layout.addWidget(self._batch_size_spin)

        # LAYOUT ----------------------------------------------------------------------
        cp_wdg_layout = QVBoxLayout(self)
        cp_wdg_layout.setContentsMargins(10, 10, 10, 10)
        cp_wdg_layout.setSpacing(5)

        cp_wdg_layout.addWidget(create_divider_line("Select Cellpose Model"))
        cp_wdg_layout.addWidget(self._model_wdg)
        cp_wdg_layout.addWidget(self._browse_custom_model)
        cp_wdg_layout.addWidget(create_divider_line("Cellpose Parameters"))
        cp_wdg_layout.addWidget(self._diameter_wdg)
        cp_wdg_layout.addWidget(self._cellprob_wdg)
        cp_wdg_layout.addWidget(self._flow_wdg)
        cp_wdg_layout.addWidget(self._min_size_wdg)
        cp_wdg_layout.addWidget(self._normalize_wdg)
        cp_wdg_layout.addWidget(create_divider_line("Batch Processing"))
        cp_wdg_layout.addWidget(self._batch_wdg)

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> CellposeSettings:
        """Return the current Cellpose parameters as a CellposeData object."""
        model_type = self._models_combo.currentText()
        model_path = self._browse_custom_model.value() if model_type == "custom" else ""
        diameter = self._diameter_spin.value()
        cellprob_threshold = self._cellprob_threshold_spin.value()
        flow_threshold = self._flow_threshold_spin.value()
        min_size = self._min_size_spin.value()
        normalize = self._normalize_checkbox.isChecked()
        batch_size = self._batch_size_spin.value()

        return CellposeSettings(
            model_type=model_type,
            model_path=model_path,
            diameter=None if diameter == 0 else diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
            normalize=normalize,
            batch_size=batch_size,
        )

    def setValue(self, value: CellposeSettings) -> None:
        """Set the Cellpose parameters from a CellposeData object."""
        self._models_combo.setCurrentText(value.model_type)
        if value.model_type == "custom" and value.model_path is not None:
            self._browse_custom_model.setValue(value.model_path)
        self._diameter_spin.setValue(0 if value.diameter is None else value.diameter)
        self._cellprob_threshold_spin.setValue(value.cellprob_threshold)
        self._flow_threshold_spin.setValue(value.flow_threshold)
        self._min_size_spin.setValue(value.min_size)
        self._normalize_checkbox.setChecked(value.normalize)
        self._batch_size_spin.setValue(value.batch_size)

    # PRIVATE METHODS -----------------------------------------------------------------

    def _on_model_combo_changed(self, text: str) -> None:
        """Show or hide the custom model path widget."""
        if text == "custom":
            self._browse_custom_model.show()
        else:
            self._browse_custom_model.hide()


class _CaimanDetectionWidget(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setTitle("CaImAn")
        self.setCheckable(True)

        caiman_wdg_layout = QVBoxLayout(self)
        caiman_wdg_layout.setContentsMargins(10, 10, 10, 10)
        caiman_wdg_layout.setSpacing(5)

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> CaimanSettings:
        """Return the current CaImAn parameters."""
        ...

    def setValue(self, value: CaimanSettings) -> None:
        """Set the CaImAn parameters."""
        ...


class _RunDetectionWidget(QWidget):
    """Widget to display the progress of the analysis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # progress bar
        self._progress_bar = QProgressBar(self)
        self._progress_pos_label = QLabel()
        self._elapsed_time_label = QLabel("00:00:00")

        # buttons
        self._run_btn = QPushButton("Run")
        self._run_btn.setSizePolicy(*FIXED)
        self._run_btn.setIcon(icon(MDI6.play, color=GREEN))
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
        self._cancel_btn.setIcon(QIcon(icon(MDI6.stop, color=RED)))

        # main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        main_layout.addWidget(self._run_btn)
        main_layout.addWidget(self._cancel_btn)
        main_layout.addWidget(self._progress_bar)
        main_layout.addWidget(self._progress_pos_label)
        main_layout.addWidget(self._elapsed_time_label)

    def progress_bar_maximum(self) -> int:
        """Return the maximum value of the progress bar."""
        return cast("int", self._progress_bar.maximum())

    def set_progress_bar_label(self, text: str) -> None:
        """Update the progress label with elapsed time."""
        self._progress_pos_label.setText(text)

    def set_progress_bar_range(self, minimum: int, maximum: int) -> None:
        """Set the range of the progress bar."""
        self._progress_bar.setRange(minimum, maximum)

    def reset_progress_bar(self) -> None:
        """Reset the progress bar and elapsed time label."""
        self._progress_bar.reset()
        self._progress_bar.setValue(0)
        self._progress_pos_label.setText("[0/0]")
        self._elapsed_time_label.setText("00:00:00")

    def set_time_label(self, elapsed_time: str) -> None:
        """Update the elapsed time label."""
        self._elapsed_time_label.setText(elapsed_time)

    def update_progress_bar_plus_one(self) -> None:
        """Automatically update the progress bar value and label.

        The value is incremented by 1 each time this method is called.
        """
        value = self._progress_bar.value() + 1
        self._progress_bar.setValue(value)
        self._progress_pos_label.setText(f"[{value}/{self._progress_bar.maximum()}]")

    def reset(self) -> None:
        """Reset the widget to default values."""
        self.reset_progress_bar()


class _DetectionGUI(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # MAIN WIDGET -----------------------------------------------------------------
        group_wdg = QGroupBox(self)
        group_layout = QVBoxLayout(group_wdg)
        group_layout.setContentsMargins(10, 10, 10, 10)
        group_layout.setSpacing(5)

        # CELLPOSE WIDGET -------------------------------------------------------------
        self._cellpose_wdg = _CellposeDetectionWidget(self)
        self._cellpose_wdg.setChecked(True)

        # CAIMAN WIDGET ---------------------------------------------------------------
        self._caiman_wdg = _CaimanDetectionWidget(self)
        self._caiman_wdg.setChecked(False)

        # SCROLL AREA WIDGET ---------------------------------------------------------
        detection_scroll_area = QScrollArea()
        detection_scroll_area.setWidgetResizable(True)
        detection_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        detection_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        # add cellpose and caiman widgets to scroll area
        group_layout.addWidget(self._cellpose_wdg)
        group_layout.addWidget(self._caiman_wdg)
        group_layout.addStretch(1)
        detection_scroll_area.setWidget(group_wdg)

        # BOTTOM WIDGET ---------------------------------------------------------------
        self._positions_wdg = _ChoosePositionsWidget(self)
        self._run_detection_wdg = _RunDetectionWidget(self)
        run_wdg = QGroupBox(self)
        run_layout = QVBoxLayout(run_wdg)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(5)
        run_layout.addWidget(create_divider_line("Positions to Analyze"))
        run_layout.addWidget(self._positions_wdg)
        run_layout.addWidget(self._run_detection_wdg)

        # CONNECTIONS -----------------------------------------------------------------
        self._cellpose_wdg.toggled.connect(self._on_detection_method_toggled)
        self._caiman_wdg.toggled.connect(self._on_detection_method_toggled)

        # MAIN LAYOUT -----------------------------------------------------------------
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(15)
        main_layout.addWidget(detection_scroll_area)
        main_layout.addWidget(run_wdg)

        # STYLING ---------------------------------------------------------------------
        cp = self._cellpose_wdg
        fixed_lbl_width = cp._cellprob_label.sizeHint().width()
        cp._models_combo_label.setMinimumWidth(fixed_lbl_width)
        cp._browse_custom_model._label.setMinimumWidth(fixed_lbl_width)
        cp._diameter_label.setMinimumWidth(fixed_lbl_width)
        cp._cellprob_label.setMinimumWidth(fixed_lbl_width)
        cp._flow_label.setMinimumWidth(fixed_lbl_width)
        cp._min_size_label.setMinimumWidth(fixed_lbl_width)
        cp._batch_label.setMinimumWidth(fixed_lbl_width)
        cp._normalize_label.setMinimumWidth(fixed_lbl_width)
        self._positions_wdg._pos_lbl.setMinimumWidth(fixed_lbl_width)

    # PROPERTIES ----------------------------------------------------------------------

    @property
    def run(self):  # noqa: ANN202
        """Signal emitted when the run button is clicked."""
        return self._run_detection_wdg._run_btn.clicked

    @property
    def cancel(self):  # noqa: ANN202
        """Signal emitted when the cancel button is clicked."""
        return self._run_detection_wdg._cancel_btn.clicked

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> CellposeSettings | CaimanSettings:
        """Return the detection parameters of the selected method."""
        if self._cellpose_wdg.isChecked():
            return self._cellpose_wdg.value()
        else:  # caiman is selected
            return self._caiman_wdg.value()

    def setValue(self, value: CellposeSettings | CaimanSettings) -> None:
        """Set the detection parameters of the selected method."""
        if isinstance(value, CellposeSettings):
            self._cellpose_wdg.setValue(value)
            with signals_blocked(self._cellpose_wdg):
                self._cellpose_wdg.setChecked(True)
            with signals_blocked(self._caiman_wdg):
                self._caiman_wdg.setChecked(False)
        elif isinstance(value, CaimanSettings):
            self._caiman_wdg.setValue(value)
            with signals_blocked(self._caiman_wdg):
                self._caiman_wdg.setChecked(True)
            with signals_blocked(self._cellpose_wdg):
                self._cellpose_wdg.setChecked(False)
        else:
            raise TypeError(
                "Value must be an instance of CellposeSettings or" "CaimanSettings."
            )

    def positions(self) -> list[int]:
        """Get the positions to analyze."""
        return parse_lineedit_text(self._positions_wdg.value())

    def set_positions(self, positions: list[int]) -> None:
        """Set the positions to analyze."""
        self._positions_wdg.setValue(",".join(map(str, positions)))

    def enable(self, enabled: bool) -> None:
        """Enable or disable the detection GUI."""
        self._cellpose_wdg.setEnabled(enabled)
        self._caiman_wdg.setEnabled(enabled)
        self._positions_wdg.setEnabled(enabled)

    def reset(self) -> None:
        """Reset the detection GUI to default values."""
        self._cellpose_wdg.setValue(CellposeSettings())
        self._caiman_wdg.setValue(CaimanSettings())
        with signals_blocked(self._cellpose_wdg):
            self._cellpose_wdg.setChecked(True)
        with signals_blocked(self._caiman_wdg):
            self._caiman_wdg.setChecked(False)
        self._positions_wdg.setValue("")
        self._run_detection_wdg.reset()

    def to_model_settings(self) -> tuple[list[int], DetectionSettings]:
        """Convert current GUI settings to AnalysisSettings model.

        Returns
        -------
        tuple[list[int], AnalysisSettings]
            A tuple containing the list of positions to analyze and the
            AnalysisSettings model instance.
        """
        from datetime import datetime

        from cali.sqlmodel import DetectionSettings

        settings = self.value()

        if isinstance(settings, CellposeSettings):
            model_type = (
                settings.model_path
                if settings.model_type == "custom" and settings.model_path is not None
                else settings.model_type
            )
            settings = DetectionSettings(
                created_at=datetime.now(),
                method="cellpose",
                model_type=model_type,
                diameter=None if settings.diameter == 0 else settings.diameter,
                cellprob_threshold=settings.cellprob_threshold,
                flow_threshold=settings.flow_threshold,
                min_size=settings.min_size,
                normalize=settings.normalize,
                batch_size=settings.batch_size,
                # + caiman defaults
            )
        else:  #  caiman
            settings = DetectionSettings(
                created_at=datetime.now(),
                method="caiman",
                # + cellpose defaults
            )

        return self.positions(), settings

    # PRIVATE METHODS -----------------------------------------------------------------

    def _on_detection_method_toggled(self, checked: bool) -> None:
        """Make checkable group boxes behave like radio buttons."""
        sender = cast("QGroupBox", self.sender())
        if not checked:
            # Prevent unchecking - at least one must be selected
            with signals_blocked(sender):
                sender.setChecked(True)
            return

        # When one is checked, uncheck the other
        if sender is self._cellpose_wdg:
            with signals_blocked(self._caiman_wdg):
                self._caiman_wdg.setChecked(False)
        elif sender is self._caiman_wdg:
            with signals_blocked(self._cellpose_wdg):
                self._cellpose_wdg.setChecked(False)
            self._cellpose_wdg.blockSignals(False)

    def update_progress_label(self, elapsed_time: str) -> None:
        """Update the progress label with elapsed time."""
        self._run_detection_wdg.set_time_label(elapsed_time)
